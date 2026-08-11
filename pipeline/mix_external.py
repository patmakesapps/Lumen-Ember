"""Stage 5 — mixture assembly, split, and MIXTURE.md.

Produces `data/final/train.jsonl`, `val.jsonl`, and `MIXTURE.md` documenting
exact counts and licences for commercial compliance.

External corpora are OPT-IN (`--external`). Path A — the pipeline-validation
training run — deliberately runs without them: they are multi-GB downloads, and
the cross-model eval showed tool-call schema validity is already 100% on every
model tested, so their only remaining job is preventing catastrophic forgetting
rather than teaching the format.

Licence handling: every external dataset's licence is read from the live Hub
card at download time and compared against what we expect. A mismatch is a
hard failure, not a warning — this dataset is being assembled for commercial
use and a silently relicensed source must stop the build.

Run:
    python -m pipeline.mix_external                    # first-party only (Path A)
    python -m pipeline.mix_external --external         # + Glaive/Hermes/ToolACE
    python -m pipeline.mix_external --external --external-limit 2000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

from pipeline.config import (CATEGORY_WEIGHTS, FINAL, STAGED, TARGET_MIXTURE,
                             TRAINED_MODEL_NAME, rng)
from pipeline.glimmer import coerce_arguments
from pipeline.provenance import Provenance
from pipeline.sample import read_jsonl, validate, write_jsonl, SampleInvalid

VAL_FRACTION = 0.06
MAX_UPSAMPLE = 3

# What each external corpus is expected to be licensed as. Verified against the
# live card at download time; mismatch aborts.
EXTERNAL_SOURCES = {
    "glaiveai/glaive-function-calling-v2": "apache-2.0",
    "NousResearch/hermes-function-calling-v1": "apache-2.0",
    "Team-ACE/ToolACE": "apache-2.0",
    "HuggingFaceTB/smoltalk": "apache-2.0",
}


class LicenseMismatch(RuntimeError):
    """Raised when a Hub card's licence differs from what we recorded."""


@dataclass
class Segment:
    name: str
    rows: list[dict]

    def __len__(self) -> int:
        return len(self.rows)


# --------------------------------------------------------------------------
# First-party segments
# --------------------------------------------------------------------------

_SEGMENT_OF_FILE = {
    "trajectories": "trajectories",
    "trajectories_smoke": "trajectories",
    "static_approval": "static_extraction",
    "static_commits": "static_extraction",
    "static_design": "static_extraction",
    "static_docs_qa": "static_extraction",
    "static_tool_call": "static_extraction",
    "static_tool_impl": "static_extraction",
    "analysis_compare": "static_extraction",
    "analysis_concerns": "static_extraction",
    "analysis_crate_role": "static_extraction",
    "analysis_layering": "static_extraction",
}


def load_first_party(source: str) -> dict[str, list[dict]]:
    """Load staged rows grouped into mixture segments."""
    segments: dict[str, list[dict]] = defaultdict(list)
    path = STAGED / f"{source}.jsonl"
    if path.exists():
        for row in read_jsonl(path):
            stage = str((row.get("meta") or {}).get("stage", ""))
            seg = "trajectories" if stage.startswith("3") else "static_extraction"
            segments[seg].append(row)
        return segments

    for f in sorted(STAGED.glob("*.jsonl")):
        if f.stem in ("filtered", "train", "val"):
            continue
        seg = _SEGMENT_OF_FILE.get(f.stem)
        if seg is None:
            continue
        segments[seg].extend(read_jsonl(f))
    return segments


def apply_weights(rows: list[dict]) -> list[dict]:
    """Upsample by category weight.

    Boundary rows repeat because they target the only gap the baseline eval
    actually found; see docs/FINDINGS.md §3.
    """
    out = []
    for row in rows:
        meta = row.get("meta") or {}
        weight = meta.get("weight")
        if weight is None:
            weight = CATEGORY_WEIGHTS.get(
                meta.get("category") or meta.get("extractor") or "", 1.0)
        repeats = max(1, min(MAX_UPSAMPLE, int(round(float(weight)))))
        for i in range(repeats):
            if i == 0:
                out.append(row)
            else:
                clone = json.loads(json.dumps(row))
                clone.setdefault("meta", {})["upsample_index"] = i
                out.append(clone)
    return out


# --------------------------------------------------------------------------
# External corpora
# --------------------------------------------------------------------------

def _verify_license(dataset_id: str, expected: str) -> str:
    """Read the licence off the live Hub card. Hard-fail on mismatch."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for --external "
            "(pip install -r requirements.txt)") from e

    info = HfApi().dataset_info(dataset_id)
    tags = list(getattr(info, "tags", []) or [])
    card = getattr(info, "card_data", None)
    found = None
    if card is not None:
        found = getattr(card, "license", None) or (
            card.get("license") if isinstance(card, dict) else None)
    if not found:
        for tag in tags:
            if tag.startswith("license:"):
                found = tag.split(":", 1)[1]
                break
    if isinstance(found, list):
        found = found[0] if found else None
    found = str(found or "").strip().lower()

    if found != expected.lower():
        raise LicenseMismatch(
            f"{dataset_id}: card says licence {found!r}, expected "
            f"{expected!r}. Refusing to include it — resolve before building "
            f"a dataset intended for commercial use."
        )
    return found


def _to_chat(example: dict) -> list[dict] | None:
    """Best-effort conversion of a Hub row to our message format."""
    msgs = (example.get("messages") or example.get("conversations")
            or example.get("conversation"))
    if not isinstance(msgs, list) or not msgs:
        return None
    role_map = {"human": "user", "gpt": "assistant", "system": "system",
                "tool": "tool", "function": "tool", "user": "user",
                "assistant": "assistant"}
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = role_map.get(str(m.get("role") or m.get("from") or "").lower())
        if role is None:
            return None
        entry: dict = {"role": role}
        content = m.get("content", m.get("value"))
        entry["content"] = "" if content is None else str(content)
        if m.get("tool_calls"):
            calls = m["tool_calls"]
            if not isinstance(calls, list) or not calls:
                return None
            # Glimmer supports ONE call per turn; split rather than drop.
            for tc in calls:
                try:
                    fixed = coerce_arguments(tc)
                except Exception:
                    return None
                out.append({"role": "assistant", "content": entry["content"],
                            "tool_calls": [fixed]})
                entry["content"] = ""
            continue
        if role == "tool" and not (m.get("name") or m.get("tool_call_id")):
            entry["name"] = "tool"
        out.append(entry)
    return out or None


def load_external(limit_per_source: int, seed_name: str = "external") -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "datasets is required for --external "
            "(pip install -r requirements.txt)") from e

    rows: list[dict] = []
    for dataset_id, expected in EXTERNAL_SOURCES.items():
        license_id = _verify_license(dataset_id, expected)
        print(f"  {dataset_id}: licence {license_id} verified")
        ds = load_dataset(dataset_id, split="train", streaming=True)
        kept = 0
        for example in ds:
            if kept >= limit_per_source:
                break
            messages = _to_chat(example)
            if not messages:
                continue
            row = {
                "messages": messages,
                "tools": example.get("tools") or [],
                "provenance": Provenance(
                    source=dataset_id,
                    license=license_id,
                    extraction_method="mix_external:hub-stream",
                    repo_sha=str(getattr(
                        __import__("huggingface_hub").HfApi()
                        .dataset_info(dataset_id), "sha", "unknown")),
                ).to_dict(),
                "meta": {"stage": "5", "extractor": "external",
                         "segment": ("general_instruct" if "smoltalk" in dataset_id
                                     else "external_tool_calling"),
                         "weight": 1.0, "tags": ["external"]},
            }
            try:
                validate(row)
            except SampleInvalid:
                continue
            row["id"] = f"ext-{abs(hash(json.dumps(messages, sort_keys=True))):016x}"
            rows.append(row)
            kept += 1
        print(f"    kept {kept}")
    return rows


# --------------------------------------------------------------------------
# Split + report
# --------------------------------------------------------------------------

def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic, stratified by segment × category.

    Splitting on the ORIGINAL sample id keeps upsampled clones together — a
    row appearing in both train and val would leak, and weight-3 rows are
    exactly the ones most likely to be split across the boundary.
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta") or {}
        buckets[(meta.get("segment") or "first_party",
                 meta.get("category") or meta.get("extractor") or "?")].append(row)

    val_ids: set[str] = set()
    for key, group in sorted(buckets.items()):
        ids = sorted({r.get("id", "") for r in group})
        r = rng(f"split:{key}")
        n_val = max(1, int(round(len(ids) * VAL_FRACTION))) if len(ids) > 8 else 0
        val_ids.update(r.sample(ids, n_val) if n_val else [])

    train = [r for r in rows if r.get("id") not in val_ids]
    val = [r for r in rows if r.get("id") in val_ids]
    # Val keeps one copy of each row; upsampling is a training-side concern.
    seen: set[str] = set()
    deduped_val = []
    for row in val:
        rid = row.get("id", "")
        if rid in seen:
            continue
        seen.add(rid)
        deduped_val.append(row)
    return train, deduped_val


def write_mixture_md(train: list[dict], val: list[dict], external_used: bool) -> str:
    def tally(rows, key):
        c = Counter()
        for r in rows:
            meta = r.get("meta") or {}
            c[meta.get(key) or meta.get("extractor") or "?"] += 1
        return c

    seg_train = tally(train, "segment")
    licences = Counter()
    sources = Counter()
    for r in train + val:
        prov = r.get("provenance") or {}
        licences[prov.get("license", "?")] += 1
        sources[prov.get("source", "?")] += 1

    total = len(train)
    lines = [
        f"# MIXTURE — {TRAINED_MODEL_NAME}",
        "",
        f"Base model: `meta-models/Muse-Glimmer-30B` (Apache-2.0)  ",
        f"Train rows: **{len(train):,}**  ·  Val rows: **{len(val):,}**",
        "",
        "## Segments (train, after weighting)",
        "",
        "| Segment | Rows | Share | Target |",
        "|---|---:|---:|---:|",
    ]
    for seg, n in seg_train.most_common():
        target = TARGET_MIXTURE.get(seg)
        lines.append(f"| {seg} | {n:,} | {n / max(1, total) * 100:.1f}% | "
                     f"{'' if target is None else f'{target * 100:.0f}%'} |")

    lines += [
        "",
        "## Licences",
        "",
        "| Licence | Rows |",
        "|---|---:|",
    ]
    for lic, n in licences.most_common():
        lines.append(f"| `{lic}` | {n:,} |")

    lines += [
        "",
        "## Sources",
        "",
        "| Source | Rows | Licence |",
        "|---|---:|---|",
    ]
    lic_of = {}
    for r in train + val:
        prov = r.get("provenance") or {}
        lic_of[prov.get("source", "?")] = prov.get("license", "?")
    for src, n in sources.most_common():
        lines.append(f"| `{src}` | {n:,} | {lic_of.get(src, '?')} |")

    lines += [
        "",
        "## Compliance notes",
        "",
        "- Every row carries `{source, license, extraction_method, repo_sha}`;",
        "  `pipeline.sample.validate` refuses rows missing any of them.",
        "- First-party licences are verified against the LICENSE file on disk",
        "  at every `pipeline.doctor` run (LumaKit MIT, grok-build-fork",
        "  Apache-2.0, LumaBot MIT).",
        "- `grok-build-fork` contributes architecture *analysis* only — no raw",
        "  source. Ported (`openai/codex`, `sst/opencode`), ratatui-derived, and",
        "  vendored `third_party/` code is excluded per the repo's own NOTICE",
        "  files, and excluded from every line count cited in a sample.",
        "- External corpora have their licence read from the live Hugging Face",
        "  card at download time and compared to the expected value; a mismatch",
        "  aborts the build.",
        "- No secrets: `pipeline.secrets_scan` runs over the fully serialised",
        "  sample and hard-fails first-party stages.",
        "",
    ]
    if not external_used:
        lines += [
            "> **External corpora are not included in this build.** The",
            "> cross-model eval measured 100% tool-call schema validity on every",
            "> model tested, so external tool-calling data is no longer teaching",
            "> the format; its remaining role is preventing catastrophic",
            "> forgetting. Re-run with `--external` to include it.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 — mixture and split")
    ap.add_argument("--source", default="filtered",
                    help="staged file stem to mix (default: filtered)")
    ap.add_argument("--external", action="store_true",
                    help="download and include external corpora (multi-GB)")
    ap.add_argument("--external-limit", type=int, default=1500,
                    help="max rows per external source")
    ap.add_argument("--no-weights", action="store_true")
    args = ap.parse_args()

    segments = load_first_party(args.source)
    rows: list[dict] = []
    for seg, seg_rows in segments.items():
        for row in seg_rows:
            row.setdefault("meta", {})["segment"] = seg
        rows.extend(seg_rows)
    print(f"first-party: {len(rows)} rows "
          f"({ {k: len(v) for k, v in segments.items()} })")

    external_used = False
    if args.external:
        ext = load_external(args.external_limit)
        rows.extend(ext)
        external_used = True
        print(f"external: {len(ext)} rows")

    if not args.no_weights:
        before = len(rows)
        rows = apply_weights(rows)
        print(f"weighting: {before} -> {len(rows)} rows "
              f"(boundary categories upsampled)")

    train, val = stratified_split(rows)

    # Val must not contain a row whose id also appears in train.
    train_ids = {r.get("id") for r in train}
    leaked = [r for r in val if r.get("id") in train_ids]
    if leaked:
        print(f"  ERROR: {len(leaked)} val rows also present in train")
        return 1

    FINAL.mkdir(parents=True, exist_ok=True)
    n_train = write_jsonl(FINAL / "train.jsonl", train, secrets_on_hit="drop")
    n_val = write_jsonl(FINAL / "val.jsonl", val, secrets_on_hit="drop")
    (FINAL / "MIXTURE.md").write_text(
        write_mixture_md(train, val, external_used), encoding="utf-8")

    print(f"\n  train.jsonl  {n_train:,}")
    print(f"  val.jsonl    {n_val:,}")
    print(f"  MIXTURE.md   written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
