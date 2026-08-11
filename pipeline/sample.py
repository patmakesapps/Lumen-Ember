"""The canonical training sample + deterministic JSONL I/O.

Wire format is OpenAI-style chat/messages. We deliberately do NOT store
Glimmer's rendered ATEM string — the rendering is done at train time by the
model's own chat template (see pipeline/glimmer.py). Storing structured
messages keeps the dataset portable if the template is revised, and lets the
same rows feed both the 30B and any future variant.

    {
      "id": "<stable sha256 prefix>",
      "messages": [ {role, content, tool_calls?, tool_call_id?, name?}, ... ],
      "tools":    [ {"type":"function","function":{name, description, parameters}} ],
      "provenance": {source, license, extraction_method, repo_sha},
      "meta":     {stage, tags, weight, ...}
    }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from pipeline.provenance import Provenance
from pipeline import secrets_scan

VALID_ROLES = {"system", "user", "assistant", "tool"}


class SampleInvalid(ValueError):
    """Raised when a sample violates the wire contract."""


@dataclass
class Sample:
    messages: list[dict]
    provenance: Provenance
    tools: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def stable_id(self) -> str:
        """Content-addressed id. Idempotent stages can dedupe on this."""
        payload = json.dumps(
            {"messages": self.messages, "tools": self.tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> dict:
        return {
            "id": self.stable_id(),
            "messages": self.messages,
            "tools": self.tools,
            "provenance": self.provenance.to_dict(),
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(sample: dict, *, require_provenance: bool = True) -> None:
    """Structural validation of a serialized sample. Raises SampleInvalid."""
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise SampleInvalid(f"{sample.get('id','?')}: messages must be a non-empty list")

    if require_provenance:
        prov = sample.get("provenance") or {}
        missing = [
            k for k in ("source", "license", "extraction_method", "repo_sha")
            if not prov.get(k)
        ]
        if missing:
            raise SampleInvalid(
                f"{sample.get('id','?')}: provenance missing {missing}"
            )

    seen_call_ids: set[str] = set()
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise SampleInvalid(f"message[{i}] is not an object")
        role = m.get("role")
        if role not in VALID_ROLES:
            raise SampleInvalid(f"message[{i}]: bad role {role!r}")

        if role == "assistant" and m.get("tool_calls"):
            # Muse Glimmer supports ONE tool call per turn and does not support
            # parallel calls (per Meta's prompting guide). The chat template
            # will happily render several joined by <|eom|>, so this is not
            # caught downstream — it has to be rejected here.
            if len(m["tool_calls"]) > 1:
                raise SampleInvalid(
                    f"message[{i}]: {len(m['tool_calls'])} tool_calls in one "
                    f"assistant turn. Glimmer does not support parallel tool "
                    f"calls — split into sequential turns with a tool result "
                    f"between them."
                )
            for j, tc in enumerate(m["tool_calls"]):
                fn = (tc or {}).get("function") or {}
                if not fn.get("name"):
                    raise SampleInvalid(f"message[{i}].tool_calls[{j}]: missing name")
                args = fn.get("arguments")
                # THE critical rule — Glimmer's chat template calls
                # raise_exception() when arguments is not a mapping. A
                # JSON-string here is a train-time crash, not a warning.
                if not isinstance(args, dict):
                    raise SampleInvalid(
                        f"message[{i}].tool_calls[{j}]: arguments must be a dict, "
                        f"got {type(args).__name__}. Glimmer's ATEM template "
                        f"cannot parse a JSON string."
                    )
                if tc.get("id"):
                    seen_call_ids.add(tc["id"])

        if role == "tool":
            if not (m.get("name") or m.get("tool_call_id")):
                raise SampleInvalid(
                    f"message[{i}]: tool message needs name or tool_call_id "
                    f"(the template resolves the name from prior tool_calls)"
                )
            if m.get("content") is None:
                raise SampleInvalid(f"message[{i}]: tool message has null content")

    # A tool result with an id that was never called renders as a bare id.
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid and tcid not in seen_call_ids and not m.get("name"):
                raise SampleInvalid(
                    f"message[{i}]: tool_call_id {tcid!r} matches no prior call"
                )


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def write_jsonl(
    path: Path,
    samples: Iterable[Sample | dict],
    *,
    secrets_on_hit: str = "fail",
    validate_rows: bool = True,
) -> int:
    """Write samples deterministically. Returns the row count.

    Idempotent by construction: same inputs => byte-identical file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    dropped = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            row = s.to_dict() if isinstance(s, Sample) else s
            if validate_rows:
                validate(row)
            if not secrets_scan.gate(
                row, on_hit=secrets_on_hit, where=f"{path.name}"
            ):
                dropped += 1
                continue
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            n += 1
    if dropped:
        print(f"  [secrets] dropped {dropped} row(s) from {path.name}")
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in read_jsonl(path))
