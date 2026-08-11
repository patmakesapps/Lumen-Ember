"""Stage 1 — static extraction from LumaKit.

Six extractors, each independently runnable and idempotent:

  tool_impl   1A  registered tool -> its real implementation source
  tool_call   1B  tool schema    -> a valid call for a task (with distractors)
  commits     1C  commit message <-> diff, both directions
  docs_qa     1D  README/docs sections -> grounded behaviour Q&A
  approval    1E  approval-boundary behaviour (overweighted)
  design      1F  framework-design principles and anti-patterns

Everything here is offline and deterministic — no teacher endpoint. Natural
phrasing and open-ended variety are stage 3's job; stage 1's job is exact,
verifiable grounding in the repo as it actually is.

Run:
    python -m pipeline.extract_static                  # all extractors
    python -m pipeline.extract_static --only approval  # one
    python -m pipeline.extract_static --show 3         # print first N samples
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pipeline import provenance
from pipeline.authoring import AUTHOR_SYSTEM, reflow as _reflow
from pipeline.config import LUMAKIT_REPO, STAGED, is_denied_path, rng
from pipeline.lumakit_probe import probe, scoped_system_prompt
from pipeline.registry_introspect import dump_registry
from pipeline.sample import Sample, write_jsonl

# --------------------------------------------------------------------------
# Shared system prompts
# --------------------------------------------------------------------------

# For agent-behaviour samples we use LumaKit's REAL system prompt so training
# matches inference exactly.
def agent_system(profile: str = "default") -> str:
    return probe()["prompts"][profile]


MAX_DIFF_LINES = 900          # p90 of this repo's minable commits is ~844
MAX_SECTION_CHARS = 4000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(LUMAKIT_REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def _tool_block(names: list[str], registry_by_name: dict) -> list[dict]:
    """OpenAI function-tool blocks for a specific subset of the registry."""
    out = []
    for n in names:
        t = registry_by_name.get(n)
        if not t:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"] or {"type": "object", "properties": {}},
            },
        })
    return out


# Share of tool-bearing samples that carry the complete 107-tool block, so the
# real (crowded, ~17.5k-token) inference condition is represented. The rest use
# a narrowed block — see lumakit_probe.scoped_system_prompt for the reasoning.
FULL_BLOCK_RATE = 0.12


def _tool_context(
    target: str,
    all_names: list[str],
    by_name: dict,
    stream: str,
    *,
    k: int = 10,
) -> tuple[list[dict], str]:
    """Return (tools_block, system_prompt) that AGREE with each other.

    The system prompt names its own tool list inline, so a narrowed block with
    the full 107-name prompt would train the model to call tools it was never
    offered. Prompt and block are always derived from the same name set.
    """
    r = rng(f"toolctx:{stream}:{target}")
    if r.random() < FULL_BLOCK_RATE:
        names = list(all_names)
    else:
        # Bias toward same-group tools: the hard selection decisions are
        # between neighbours (find_definition vs read_symbol vs read_file),
        # not between a git tool and a robot motor.
        group = by_name[target]["group"]
        same = [n for n in all_names if by_name[n]["group"] == group and n != target]
        other = [n for n in all_names if by_name[n]["group"] != group]
        n_same = min(len(same), max(2, k // 2))
        picked = r.sample(same, n_same) + r.sample(other, min(len(other), k - n_same))
        names = sorted({target, *picked})

    return _tool_block(names, by_name), scoped_system_prompt(names)


def _call(name: str, args: dict, cid: str = "call_1") -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def _assistant(content: str | None, calls: list[dict] | None = None) -> dict:
    """LumaKit's prompt asks for a status line alongside tool calls, so our
    assistant turns carry both — matching the framework's own convention."""
    m: dict = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = calls
    return m


# ==========================================================================
# 1A — tool schema -> implementation
# ==========================================================================

@dataclass
class ToolSource:
    tool_name: str
    factory_src: str
    impl_src: str
    module: str


def _scan_tool_sources() -> dict[str, ToolSource]:
    """AST-map every registered tool name to its factory + execute function."""
    found: dict[str, ToolSource] = {}
    tools_root = LUMAKIT_REPO / "tools"

    for path in sorted(tools_root.rglob("*.py")):
        if path.name == "__init__.py" or is_denied_path(path):
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        top_funcs = {
            n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not (node.name.startswith("get_") and node.name.endswith("_tool")):
                continue

            # Find `return {... 'name': <str>, 'execute': <ref> ...}`
            for ret in ast.walk(node):
                if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Dict):
                    continue
                d = ret.value
                tool_name = exec_ref = None
                for k, v in zip(d.keys, d.values):
                    if not isinstance(k, ast.Constant):
                        continue
                    if k.value == "name" and isinstance(v, ast.Constant):
                        tool_name = v.value
                    elif k.value == "execute":
                        if isinstance(v, ast.Name):
                            exec_ref = v.id
                        elif isinstance(v, ast.Attribute):
                            exec_ref = v.attr
                if not tool_name:
                    continue

                factory_src = ast.get_source_segment(src, node) or ""
                impl_src = ""
                impl_node = top_funcs.get(exec_ref) if exec_ref else None
                # Inner closures (code_intel) live inside the factory already.
                if impl_node is not None and impl_node is not node:
                    impl_src = ast.get_source_segment(src, impl_node) or ""

                found[tool_name] = ToolSource(
                    tool_name=tool_name,
                    factory_src=factory_src,
                    impl_src=impl_src,
                    module=path.relative_to(LUMAKIT_REPO).as_posix(),
                )
                break
    return found


def extract_tool_impl() -> Iterator[Sample]:
    reg = dump_registry()
    by_name = {t["name"]: t for t in reg["tools"]}
    sources = _scan_tool_sources()

    for name in sorted(sources):
        ts = sources[name]
        meta = by_name.get(name)
        if not meta:
            continue
        body = ts.factory_src + (("\n\n" + ts.impl_src) if ts.impl_src else "")
        if len(body) < 120:          # trivial stubs teach nothing
            continue

        schema = json.dumps(meta["inputSchema"], indent=2, sort_keys=True)
        required = (meta["inputSchema"] or {}).get("required", [])
        gate = ""
        if meta["always_confirm"]:
            gate = (
                "\n\nThis capability is arbitrary-execution or destructive, so it must sit "
                "behind an unconditional approval prompt rather than a command denylist."
            )

        user = (
            f"Add a tool to this agent framework: {meta['description'].rstrip('.')}.\n\n"
            f"It should be registered in the `{meta['group']}` group and expose this "
            f"input schema:\n\n```json\n{schema}\n```\n\n"
            f"Follow the framework's conventions: a `get_*_tool()` factory returning a dict "
            f"with `name`, `description`, `inputSchema`, and `execute`, and a private "
            f"implementation function that returns a plain dict."
            + (f" Required inputs: {', '.join(required)}." if required else "")
            + gate
        )

        answer = (
            f"```python\n# {ts.module}\n{body.strip()}\n```\n\n"
            f"The factory is picked up automatically by "
            f"`ToolRegistry.load_tools_from_folder()`, which scans for `get_*_tool` "
            f"names and infers the `{meta['group']}` group from the directory. Validation "
            f"against `inputSchema` happens centrally in `ToolRegistry.execute()`, so "
            f"`{name}` never has to re-check types itself."
        )

        yield Sample(
            messages=[
                {"role": "system", "content": AUTHOR_SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
            provenance=provenance.lumakit("extract_static:tool_impl:ast"),
            meta={"stage": "1A", "extractor": "tool_impl", "tool": name,
                  "group": meta["group"], "weight": 1.0,
                  "tags": ["tool-authoring", "framework-conventions"]},
        )


# ==========================================================================
# 1B — schema -> valid call
# ==========================================================================

_VALUES: dict[str, list] = {
    "path": ["core/approval_policy.py", "README.md", "tools/repo/read_file.py"],
    "file_path": ["agent.py", "core/task_runner.py"],
    "directory": ["core", "tools/repo"],
    "dir": ["tools", "docs"],
    "pattern": ["ALWAYS_CONFIRM_TOOLS", "def execute", "load_tools_from_folder"],
    "query": ["approval policy", "how tools are registered"],
    "command": ["pytest -q", "python -m pytest tests/test_approval_policy.py"],
    "content": ["# notes\n\nfirst line\n"],
    "message": ["Add approval gate for delete_file"],
    "name": ["lumen-ember"],
    "symbol": ["ToolRegistry", "tool_always_requires_approval"],
    "reason": ["Run the approval-policy tests to confirm the gate still fires"],
    "url": ["https://example.com/docs"],
    "text": ["Remember that the approval prompt is the control, not command matching"],
    "limit": [20], "max_results": [10], "timeout": [120],
    "line": [42], "start": [1], "end": [40],
    "recursive": [True], "dry_run": [True],
    # Added after review: these were falling through to the "example" fallback
    # and producing calls whose arguments were meaningless.
    "patch": ["*** Begin Patch\n*** Update File: core/approval_policy.py\n"
              "@@\n-ALWAYS_CONFIRM_TOOLS = frozenset({\n"
              "+ALWAYS_CONFIRM_TOOLS = frozenset({  # gated regardless of toggle\n"
              "*** End Patch"],
    "old_string": ["require_tool_approvals"],
    "new_string": ["require_tool_approvals  # see approval_policy"],
    "code": ["print(len(registry.tools))"],
    "title": ["Tighten the approval gate"],
    "description": ["Confirm delete_file still prompts when safe mode is on"],
    "task": ["Review the approval policy module and summarise the two policies"],
    "prompt": ["Summarise how approvals differ between interactive and autonomous runs"],
    "key": ["deployment-notes"],
    "value": ["Approval prompt is the control, not command matching"],
    "search": ["approval"], "term": ["approval"], "needle": ["frozenset"],
    "branch": ["main"], "remote": ["origin"], "ref": ["HEAD"],
    # speed is normalised 0.0-1.0, not RPM or percent — the registry rejects
    # 40 with "'speed' must be <= 1.0". Caught by stage 4's schema gate.
    "selector": ["#submit"], "seconds": [3], "speed": [0.4], "distance": [50],
    "angle": [90], "direction": ["forward"], "count": [3], "index": [0],
    "chat_id": ["chat_001"], "task_id": ["task_001"], "project": ["lumen-ember"],
    "group_id": ["group_001"], "photo_id": ["photo_001"],
}

# Params we are willing to leave as a generic fallback because the real value
# genuinely is free-form and the schema carries the meaning.
_JUNK_TOLERANT = {"extra", "options", "metadata", "context", "payload", "data"}


def _is_junk(args: dict, schema: dict) -> bool:
    """True if the synthesized args are placeholders rather than real values."""
    required = list((schema or {}).get("required") or [])
    if not required:
        return False
    for key in required:
        if key in _JUNK_TOLERANT:
            continue
        v = args.get(key)
        if isinstance(v, str) and v in ("example", ""):
            return True
    return False

_TYPE_FALLBACK = {
    "string": "example", "integer": 5, "number": 2.5,
    "boolean": True, "array": [], "object": {},
}


def _synth_args(schema: dict, stream: str, *, optional: bool = False) -> dict | None:
    """Build a schema-conformant argument dict deterministically."""
    props = (schema or {}).get("properties") or {}
    required = list((schema or {}).get("required") or [])
    if not props:
        return {}
    r = rng(f"args:{stream}")
    args: dict = {}
    keys = list(required)
    if optional:
        extra = [k for k in props if k not in required]
        if extra:
            keys = keys + [r.choice(sorted(extra))]

    for key in keys:
        spec = props.get(key) or {}
        typ = spec.get("type")
        if spec.get("enum"):
            args[key] = spec["enum"][0]
            continue
        if key in _VALUES:
            cand = [v for v in _VALUES[key]
                    if typ is None
                    or (typ == "string" and isinstance(v, str))
                    or (typ in ("integer", "number") and isinstance(v, (int, float))
                        and not isinstance(v, bool))
                    or (typ == "boolean" and isinstance(v, bool))]
            if cand:
                args[key] = cand[r.randrange(len(cand))]
                continue
        if typ == "array":
            item_t = ((spec.get("items") or {}).get("type")) or "string"
            args[key] = [_TYPE_FALLBACK.get(item_t, "example")]
            continue
        if typ in _TYPE_FALLBACK:
            v = _TYPE_FALLBACK[typ]
            if typ in ("integer", "number"):
                lo, hi = spec.get("minimum"), spec.get("maximum")
                if lo is not None:
                    v = max(v, lo)
                if hi is not None:
                    v = min(v, hi)
            args[key] = v
            continue
        args[key] = "example"

    return args


def _task_phrasing(tool: dict, args: dict, variant: int) -> str:
    desc = tool["description"].rstrip(".")
    primary = next((f"`{v}`" for v in args.values() if isinstance(v, str) and v), "")
    forms = [
        f"{desc}. Target: {primary}." if primary else f"{desc}.",
        f"I need you to {desc[0].lower() + desc[1:]}"
        + (f" — {primary}." if primary else "."),
        f"Can you handle this: {desc[0].lower() + desc[1:]}"
        + (f" ({primary})" if primary else "") + "?",
    ]
    return forms[variant % len(forms)]


def extract_tool_calls() -> Iterator[Sample]:
    reg = dump_registry()
    by_name = {t["name"]: t for t in reg["tools"]}
    all_names = sorted(by_name)

    for name in all_names:
        tool = by_name[name]
        # Approval-gated tools get their behaviour taught in 1E, where the
        # correct answer is not "just call it".
        if tool["always_confirm"]:
            continue
        for variant in range(3):
            args = _synth_args(
                tool["inputSchema"], f"{name}:{variant}", optional=(variant == 2)
            )
            if args is None:
                continue
            # A call whose arguments are all placeholder junk teaches the model
            # to emit junk. Drop those rather than ship them.
            if _is_junk(args, tool["inputSchema"]):
                continue
            tools, sys_prompt = _tool_context(
                name, all_names, by_name, f"call:{variant}", k=10
            )
            yield Sample(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": _task_phrasing(tool, args, variant)},
                    _assistant(
                        f"Calling `{name}` now.",
                        [_call(name, args)],
                    ),
                ],
                tools=tools,
                provenance=provenance.lumakit("extract_static:tool_call:schema-synth"),
                meta={"stage": "1B", "extractor": "tool_call", "tool": name,
                      "variant": variant, "n_tools_offered": len(tools),
                      "full_block": len(tools) == len(all_names),
                      "weight": 1.0, "tags": ["tool-call", "tool-selection"]},
            )


# ==========================================================================
# 1C — commit mining
# ==========================================================================

_SKIP_SUBJECT = re.compile(
    r"\b(merge|bump|typo|fmt|format|lint|whitespace|wip)\b|typo|formatting", re.I
)
_BINARY_EXT = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".lock")


def _clean_diff(diff: str) -> tuple[str, list[str]]:
    """Drop secret-bearing and binary files; truncate very large diffs."""
    chunks, current, dropped = [], [], []
    keep = True
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            if current and keep:
                chunks.append("\n".join(current))
            current, keep = [line], True
            target = line.split(" b/")[-1] if " b/" in line else line
            if is_denied_path(target) or target.endswith(_BINARY_EXT):
                keep = False
                dropped.append(target)
            continue
        if keep:
            current.append(line)
    if current and keep:
        chunks.append("\n".join(current))

    out = "\n".join(chunks)
    lines = out.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        out = "\n".join(lines[:MAX_DIFF_LINES]) + (
            f"\n… [diff truncated: {len(lines) - MAX_DIFF_LINES} more lines]"
        )
    return out, dropped


def _minable_commits() -> list[tuple[str, str, str]]:
    shas = _git("rev-list", "--no-merges", "HEAD").split()
    out = []
    for sha in shas:
        subject = _git("log", "-1", "--pretty=%s", sha).strip()
        if _SKIP_SUBJECT.search(subject) or len(subject.split()) <= 5:
            continue
        diff = _git("show", "--pretty=", "-M", "--no-color", sha)
        cleaned, _dropped = _clean_diff(diff)
        if len(cleaned.splitlines()) <= 10 or not cleaned.strip():
            continue
        out.append((sha, subject, cleaned))
    return out


def extract_commits() -> Iterator[Sample]:
    commits = _minable_commits()
    r = rng("commits:forward")
    # diff -> message is the well-posed direction and gets every commit.
    # message -> diff is underspecified (a one-line subject cannot determine a
    # 200-line patch), so it is deliberately the minority at ~1:2.
    forward_idx = set(r.sample(range(len(commits)), max(1, len(commits) // 3)))

    for i, (sha, subject, diff) in enumerate(commits):
        yield Sample(
            messages=[
                {"role": "system", "content": AUTHOR_SYSTEM},
                {"role": "user", "content":
                    "Write a concise, specific commit message for this diff. "
                    "Describe the behaviour change, not the file list.\n\n"
                    f"```diff\n{diff}\n```"},
                {"role": "assistant", "content": subject},
            ],
            provenance=provenance.lumakit("extract_static:commits:diff_to_message"),
            meta={"stage": "1C", "extractor": "commits", "direction": "diff_to_message",
                  "commit": sha[:12], "weight": 1.0, "tags": ["commit-message"]},
        )
        if i in forward_idx:
            yield Sample(
                messages=[
                    {"role": "system", "content": AUTHOR_SYSTEM},
                    {"role": "user", "content":
                        f"In this agent framework, implement the following change:\n\n"
                        f"{subject}\n\nShow the diff."},
                    {"role": "assistant", "content": f"```diff\n{diff}\n```"},
                ],
                provenance=provenance.lumakit("extract_static:commits:message_to_diff"),
                meta={"stage": "1C", "extractor": "commits", "direction": "message_to_diff",
                      "commit": sha[:12], "weight": 0.5, "tags": ["patch-generation"]},
            )


# ==========================================================================
# 1D — docs -> behaviour Q&A
# ==========================================================================

_QUESTION_FORMS = (
    "In this framework, how does {topic} work?",
    "Explain {topic} — what's the behaviour and why is it designed that way?",
    "A user is building a similar agent harness and asks about {topic}. What should they know?",
)


def _md_sections(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections, heading, buf = [], None, []
    for line in text.splitlines():
        if re.match(r"^##\s+\S", line):
            if heading and buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("#").strip(), []
        elif heading:
            buf.append(line)
    if heading and buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if len(b) > 200]


def extract_docs_qa() -> Iterator[Sample]:
    paths = [LUMAKIT_REPO / "README.md", *sorted((LUMAKIT_REPO / "docs").glob("*.md"))]
    for path in paths:
        if not path.exists() or is_denied_path(path):
            continue
        rel = path.relative_to(LUMAKIT_REPO).as_posix()
        for idx, (heading, body) in enumerate(_md_sections(path)):
            body = body[:MAX_SECTION_CHARS]
            r = rng(f"docs:{rel}:{heading}")
            form = _QUESTION_FORMS[r.randrange(len(_QUESTION_FORMS))]
            yield Sample(
                messages=[
                    {"role": "system", "content": AUTHOR_SYSTEM},
                    {"role": "user", "content": form.format(topic=heading)},
                    {"role": "assistant", "content": body},
                ],
                provenance=provenance.lumakit(f"extract_static:docs_qa:{rel}"),
                meta={"stage": "1D", "extractor": "docs_qa", "doc": rel,
                      "heading": heading, "section_index": idx, "weight": 1.0,
                      "tags": ["framework-behaviour", "docs"]},
            )


# ==========================================================================
# 1E — approval boundaries  (highest-value; overweighted)
# ==========================================================================

APPROVAL_WEIGHT = 3.0

_TASK_FOR_TOOL = {
    "delete_file": ("Delete core/telegram_speech.py — we're dropping voice support.",
                    {"path": "core/telegram_speech.py"}),
    "git_add": ("Stage everything I've changed.", {"paths": ["."]}),
    "git_commit": ("Commit this with the message 'tighten approval policy'.",
                   {"message": "tighten approval policy"}),
    "git_push": ("Push my branch up.", {}),
    "execute_shell": ("Run the approval policy tests.",
                      {"command": "pytest tests/test_approval_policy.py -q",
                       "reason": "Verify the approval gate still fires after the change"}),
    "execute_python": ("Run a quick script to count the registered tools.",
                       {"code": "print(len(registry.tools))"}),
    "run_command": ("Run the test suite.",
                    {"command": "pytest -q",
                     "reason": "Check nothing regressed after the approval change"}),
    "lumabot_reboot": ("Reboot the robot, it's acting weird.", {}),
    "lumabot_poweroff": ("Shut the robot down for the night.", {}),
    "lumabot_start_autonomy": ("Let the robot roam around the house for a bit.", {}),
}


def _args_for(tool: dict, hand: dict, stream: str) -> dict:
    """Hand-authored args win; synthesis only fills genuinely missing required
    fields.

    Fixed after review: the original `_synth_args(...) or hand` inverted this,
    so a user asking to delete `telegram_speech.py` produced a call targeting
    `approval_policy.py`. Task text and arguments disagreeing is worse than a
    generic sample — it teaches the model to ignore the request.
    """
    args = dict(hand or {})
    schema = tool.get("inputSchema") or {}
    required = list(schema.get("required") or [])
    missing = [k for k in required if k not in args]
    if missing:
        synth = _synth_args(schema, stream) or {}
        for k in missing:
            if k in synth:
                args[k] = synth[k]
    return args


AUTONOMOUS_NOTE = ("\n\nYou are running inside an autonomous task. No human is "
                   "present to approve tool actions.")


def extract_approval() -> Iterator[Sample]:
    reg = dump_registry()
    by_name = {t["name"]: t for t in reg["tools"]}
    policy = reg["policy"]
    all_names = sorted(by_name)

    def emit(messages, tools, shape, **extra):
        return Sample(
            messages=messages, tools=tools,
            provenance=provenance.lumakit(f"extract_static:approval:{shape}"),
            meta={"stage": "1E", "extractor": "approval", "shape": shape,
                  "weight": APPROVAL_WEIGHT,
                  "tags": ["approval-boundary", "safety"], **extra},
        )

    gated = [t for t in policy["ALWAYS_CONFIRM_TOOLS"] if t in by_name]

    # -- (i) safe mode ON + always-confirm tool: call it WITH a reason and stop.
    #        Three phrasings each so the behaviour isn't tied to one wording.
    for tool_name in gated:
        task, hand = _TASK_FOR_TOOL.get(tool_name, (f"Please run {tool_name}.", {}))
        args = _args_for(by_name[tool_name], hand, f"approval:{tool_name}")
        for variant, lead in enumerate((
            "This needs your approval before it runs — `{t}` is gated regardless of "
            "the approvals toggle. Sending the request now; I'll wait for your "
            "decision rather than assume it went through.",
            "Before I do this: `{t}` always asks first, so you'll get a prompt. "
            "I'm not treating it as done until the tool comes back successful.",
            "Queuing `{t}` for your approval — it's an arbitrary-execution or "
            "destructive tool, so the prompt fires even with approvals turned off.",
        )):
            tools, sys_p = _tool_context(
                tool_name, all_names, by_name, f"approval:{tool_name}:{variant}", k=9
            )
            yield emit(
                [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": task},
                    _assistant(lead.format(t=tool_name), [_call(tool_name, args)]),
                ],
                tools, "interactive_requires_approval",
                tool=tool_name, variant=variant,
            )

    # -- (vi/vii) denial: acknowledge, do NOT retry, do NOT claim success.
    for tool_name in gated:
        task, hand = _TASK_FOR_TOOL.get(tool_name, (f"Please run {tool_name}.", {}))
        args = _args_for(by_name[tool_name], hand, f"denial:{tool_name}")
        tools, sys_p = _tool_context(
            tool_name, all_names, by_name, f"denial:{tool_name}", k=8
        )
        yield emit(
            [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": task},
                _assistant("Requesting approval for this one.",
                           [_call(tool_name, args, "call_d1")]),
                {"role": "tool", "tool_call_id": "call_d1", "name": tool_name,
                 "content": json.dumps({"success": False,
                                        "error": "User declined the tool action.",
                                        "toolName": tool_name})},
                _assistant(
                    f"Understood — you declined, so `{tool_name}` did not run and "
                    f"nothing changed. I won't retry it. If you want a different "
                    f"approach, tell me what to adjust and I'll propose that instead."
                ),
            ],
            tools, "denied_no_retry_no_fake_completion", tool=tool_name,
        )

    # -- (viii) APPROVAL GRANTED: proceed, then report only what the result says.
    #           The mirror of the denial case — without it we teach hesitancy.
    for tool_name in gated:
        task, hand = _TASK_FOR_TOOL.get(tool_name, (f"Please run {tool_name}.", {}))
        args = _args_for(by_name[tool_name], hand, f"granted:{tool_name}")
        tools, sys_p = _tool_context(
            tool_name, all_names, by_name, f"granted:{tool_name}", k=8
        )
        yield emit(
            [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": task},
                _assistant("Asking for approval first.",
                           [_call(tool_name, args, "call_g1")]),
                {"role": "tool", "tool_call_id": "call_g1", "name": tool_name,
                 "content": json.dumps({"success": True,
                                        "data": {"ok": True, "tool": tool_name}})},
                _assistant(
                    f"Approved and done — `{tool_name}` completed successfully. "
                    f"That's confirmed from the tool result, not assumed."
                ),
            ],
            tools, "approved_then_proceed", tool=tool_name,
        )

    # -- (ix) ONE-SHOT GRANT semantics: approval covers the command that was
    #         approved, not the next one. Grounded in
    #         tests/test_task_approvals.py::test_grant_does_not_match_different_command.
    for i, (first, second) in enumerate((
        ("pytest -q", "git push origin main"),
        ("ls -la", "rm -rf build/"),
        ("pytest tests/test_approval_policy.py -q", "git commit -am 'wip'"),
    )):
        tools, sys_p = _tool_context(
            "execute_shell", all_names, by_name, f"oneshot:{i}", k=8
        )
        yield emit(
            [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": f"Run `{first}`, and after that run `{second}`."},
                _assistant(f"Starting with `{first}`.",
                           [_call("execute_shell",
                                  {"command": first,
                                   "reason": "First command the user asked for"},
                                  "call_o1")]),
                {"role": "tool", "tool_call_id": "call_o1", "name": "execute_shell",
                 "content": json.dumps({"success": True,
                                        "data": {"stdout": "ok", "returncode": 0}})},
                _assistant(
                    f"`{first}` finished. `{second}` needs its own approval — the "
                    f"grant you just gave covers that exact command, not the next "
                    f"one, so I'm requesting it separately rather than reusing it.",
                    [_call("execute_shell",
                           {"command": second,
                            "reason": "Second command the user asked for"},
                           "call_o2")]),
            ],
            tools, "one_shot_grant_does_not_carry", pair=i,
        )

    # -- (ii) autonomous task + refused tool.
    for tool_name in [t for t in policy["AUTONOMOUS_REFUSED_TOOLS"] if t in by_name]:
        task, _ = _TASK_FOR_TOOL.get(
            tool_name, (f"Run {tool_name} as part of this task.", {})
        )
        tools, sys_p = _tool_context(
            tool_name, all_names, by_name, f"autonomous:{tool_name}", k=8
        )
        yield emit(
            [
                {"role": "system", "content": sys_p + AUTONOMOUS_NOTE},
                {"role": "user", "content": task},
                _assistant(
                    f"I can't do that from inside an autonomous task. `{tool_name}` "
                    f"requires explicit human approval, and there's nobody here to ask, "
                    f"so the task runner refuses it outright rather than running it "
                    f"unattended.\n\nI've stopped before calling it. Re-run this "
                    f"interactively and I'll request approval, or tell me what you'd "
                    f"like me to do with the rest of the task."
                ),
            ],
            tools, "autonomous_refusal", tool=tool_name,
        )

    # -- (iv) protected shell command inside an autonomous task.
    for cmd in ("git add -A && git commit -m 'wip'", "rm -rf build/",
                "git push origin main", "del /f /q dist\\*",
                "unlink core/approval_policy.py"):
        tools, sys_p = _tool_context(
            "execute_shell", all_names, by_name, f"shellrefuse:{cmd}", k=8
        )
        yield emit(
            [
                {"role": "system", "content": sys_p + AUTONOMOUS_NOTE},
                {"role": "user", "content": f"Run this for me: {cmd}"},
                _assistant(
                    f"Not from an autonomous task. `{cmd}` is a repo-mutating or "
                    f"destructive command, and the task runner screens those out "
                    f"because approving them needs a human. I haven't run it.\n\n"
                    f"Run it interactively and I'll ask you to approve it there."
                ),
            ],
            tools, "autonomous_shell_refusal", command=cmd,
        )

    # -- (x) NOT every shell command is gated. Counterpart to the above, or we
    #        teach blanket shell refusal inside tasks.
    for cmd, why in (("pytest -q", "Run the test suite"),
                     ("ls -la core", "List the core module"),
                     ("git status --short", "Check the working tree state")):
        tools, sys_p = _tool_context(
            "execute_shell", all_names, by_name, f"shellok:{cmd}", k=8
        )
        yield emit(
            [
                {"role": "system", "content": sys_p + AUTONOMOUS_NOTE},
                {"role": "user", "content": f"Run this for me: {cmd}"},
                _assistant(
                    f"Running `{cmd}` — it's read-only, so it doesn't hit the "
                    f"approval screen.",
                    [_call("execute_shell", {"command": cmd, "reason": why})]),
            ],
            tools, "autonomous_shell_allowed", command=cmd,
        )

    # -- (iii) role-scoped denial on a shared surface. All owner-only tools.
    for tool_name in sorted(policy["OWNER_ONLY_TOOLS"]):
        if tool_name not in by_name:
            continue
        tools, sys_p = _tool_context(
            tool_name, all_names, by_name, f"role:{tool_name}", k=8
        )
        yield emit(
            [
                {"role": "system", "content":
                    sys_p + "\n\nInterface: Telegram. The active user is a "
                    "trusted non-owner."},
                {"role": "user", "content": f"Use {tool_name} for me."},
                _assistant(
                    f"'{tool_name}' is not permitted for your role. Ask the owner to "
                    f"run this, or to change your role with /role."
                ),
            ],
            tools, "role_denial", tool=tool_name,
        )

    # -- (v) NEGATIVE CONTROLS. Without these we train a model that asks
    #        permission to read a file, which is worse than useless.
    safe = [n for n in all_names
            if not by_name[n]["always_confirm"]
            and not by_name[n]["owner_only"]
            and by_name[n]["group"] in ("repo", "code_intel", "memory", "struqt")]
    r2 = rng("approval:negative")
    for tool_name in r2.sample(safe, min(30, len(safe))):
        tool = by_name[tool_name]
        args = _synth_args(tool["inputSchema"], f"neg:{tool_name}") or {}
        if _is_junk(args, tool["inputSchema"]):
            continue
        tools, sys_p = _tool_context(
            tool_name, all_names, by_name, f"neg:{tool_name}", k=8
        )
        yield emit(
            [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": _task_phrasing(tool, args, 1)},
                _assistant(f"On it — running `{tool_name}`.", [_call(tool_name, args)]),
            ],
            tools, "read_only_no_approval_needed",
            tool=tool_name, note="negative control",
        )

    # -- (xi) PARTIAL COMPLETION honesty: a run that was partly blocked must be
    #         reported as partly blocked, not summarised as success.
    tools, sys_p = _tool_context("git_status", all_names, by_name, "partial", k=9)
    yield emit(
        [
            {"role": "system", "content": sys_p + AUTONOMOUS_NOTE},
            {"role": "user", "content":
                "Check what changed, then stage and commit it with a sensible message."},
            _assistant("Checking the working tree first.",
                       [_call("git_status", {}, "call_p1")]),
            {"role": "tool", "tool_call_id": "call_p1", "name": "git_status",
             "content": json.dumps({"success": True, "data": {
                 "modified": ["core/approval_policy.py", "tests/test_approval_policy.py"]}})},
            _assistant(
                "Here's where that lands:\n\n"
                "**Done:** two files are modified — `core/approval_policy.py` and "
                "`tests/test_approval_policy.py`.\n\n"
                "**Not done:** staging and committing. `git_add` and `git_commit` "
                "require explicit approval and can't run inside an autonomous task, "
                "so I stopped rather than working around it.\n\n"
                "Nothing has been staged or committed. Run this interactively and "
                "I'll request approval for both steps."
            ),
        ],
        tools, "partial_completion_reported_honestly",
    )


# ==========================================================================
# 1F — design principles and anti-patterns
# ==========================================================================

DESIGN_WEIGHT = 2.5


def _regex_census() -> dict:
    """Measured, not asserted — the numbers go into the samples as evidence."""
    files = [p for p in LUMAKIT_REPO.rglob("*.py") if ".git" not in p.parts]
    pat = re.compile(r"re\.(compile|search|match|sub|findall|finditer|fullmatch|split)\b")
    loc = sites = 0
    touched = 0
    tool_sites = tool_files_touched = 0
    n_tool_files = 0
    for p in files:
        t = p.read_text(encoding="utf-8", errors="replace")
        loc += len(t.splitlines())
        n = len(pat.findall(t))
        sites += n
        if n:
            touched += 1
        rel = p.relative_to(LUMAKIT_REPO).as_posix()
        if rel.startswith("tools/"):
            n_tool_files += 1
            tool_sites += n
            if n:
                tool_files_touched += 1
    return {"files": len(files), "loc": loc, "sites": sites, "touched": touched,
            "per_1k": round(sites / loc * 1000, 2), "tool_sites": tool_sites,
            "tool_files_touched": tool_files_touched, "n_tool_files": n_tool_files}


def extract_design() -> Iterator[Sample]:
    c = _regex_census()

    def emit(user: str, answer: str, shape: str):
        return Sample(
            messages=[
                {"role": "system", "content": AUTHOR_SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": _reflow(answer)},
            ],
            provenance=provenance.lumakit(f"extract_static:design:{shape}"),
            meta={"stage": "1F", "extractor": "design", "shape": shape,
                  "weight": DESIGN_WEIGHT,
                  "tags": ["framework-design", "anti-pattern", "intent-reasoning"]},
        )

    yield emit(
        "I'm building an agent framework. I want it to block dangerous shell "
        "commands — help me write the regex denylist for things like rm -rf, "
        "git push --force, dd, mkfs, and so on.",
        f"""
        Don't build the denylist. It's the most common structural mistake in
        agent harnesses, and it fails for a reason no amount of pattern
        tuning fixes: the space of destructive commands is unbounded and
        trivially obfuscated. `rm -rf /` is caught; `python -c "import
        shutil; shutil.rmtree('/')"`, `find . -delete`, an alias, a shell
        variable, a base64'd payload, or a script you wrote two steps ago are
        not. You end up maintaining a filter that blocks the literal examples
        in your test suite and nothing an actual failure mode would look like
        — while giving everyone the impression the system is safe.

        Put the boundary at the capability, not the string:

        ```python
        # Tools that must never run without explicit human approval,
        # regardless of the global approvals setting.
        ALWAYS_CONFIRM_TOOLS = frozenset({{
            "delete_file", "git_add", "git_commit", "git_push",
            "execute_shell", "execute_python", "run_command",
        }})

        def tool_always_requires_approval(tool_name, tool_inputs):
            if not safe_mode_enabled():
                return False
            return tool_name in ALWAYS_CONFIRM_TOOLS
        ```

        The reasoning: a tool that runs arbitrary commands is *unconditionally*
        gated because you cannot inspect its argument and know what it does.
        The approval prompt is the control, not command matching. That
        property holds no matter what string the model generates, which is
        exactly what a denylist can't promise.

        Two corollaries worth building in from the start:

        1. **One policy module, imported by every surface.** If the interactive
           agent and the autonomous task runner each carry their own copy of
           the rules, they drift, and the drift is always in the unsafe
           direction. Put the frozensets in one file and import it in both.
        2. **Where no human exists, refuse instead of guessing.** An autonomous
           task has nobody to prompt, so gated tools are refused outright
           rather than silently allowed.

        If you still want a command screen for autonomous runs, keep it — but
        label it honestly as a best-effort filter, not a security boundary, so
        nobody downstream mistakes it for the actual control.
        """,
        "denylist_is_unwinnable",
    )

    yield emit(
        "How should my agent decide which tool to call? I was going to write a "
        "router that regex-matches the user's message for keywords like 'read', "
        "'commit', 'search' and dispatches from there.",
        f"""
        Don't write the router. A phrase parser in front of a model is a cheat
        that feels like control and buys you a worse agent.

        Concretely, what a keyword router costs you: "can you check what's in
        the config and then stage it" has to hit two tools in order — your
        regex sees `check` and `stage` and has to invent a precedence rule.
        "undo that" has no keyword at all. "look at the approval policy" means
        read a file; "look at the living room" means take a photo. The
        disambiguating information is *intent*, which is the one thing the
        model is better at than your pattern table — and you've placed the
        pattern table where it can overrule the model.

        The structure that works: expose tools as schemas, describe selection
        policy in the system prompt, and let the model reason:

        ```
        Interpret the user's natural-language intent yourself and call the
        appropriate structured tool; there is no phrase parser.
        ```

        Then make selection learnable by writing *policy*, not patterns:

        - "For project overview questions, call inspect_project first."
        - "When the user asks where a function is implemented and wants to see
          its body, don't stop at find_definition — follow it with read_symbol."
        - "Prefer the dedicated git tools over raw shell for status and diffs."

        Those are the same decisions the regex router encoded, expressed where
        they can compose with context instead of fighting it.

        As a sanity check on how little pattern-matching a well-built harness
        actually needs: LumaKit is {c['loc']:,} lines across {c['files']}
        Python files and contains {c['sites']} `re.*` call sites total
        ({c['per_1k']} per 1k lines), in {c['touched']} files. Inside
        `tools/` — the entire dispatch surface — there are {c['tool_sites']}
        call sites across {c['tool_files_touched']} of {c['n_tool_files']}
        files, and every one of them parses output (patch hunks, git porcelain,
        source symbols). None of them route intent.
        """,
        "no_phrase_parser",
    )

    yield emit(
        "My agent keeps retrying the same failing tool call with slightly "
        "different arguments until it runs out of turns. How do I fix that?",
        """
        That loop is a design symptom, not a model defect: the tool told the
        agent *that* it failed without telling it *what is true now*, so the
        only move left is to guess again.

        Fix it at the tool boundary — make failures observational:

        ```python
        return {
            "success": False,
            "blocked_reason": "target_not_found",
            "recovery_snapshot": {
                "interactive_elements": [...],
                "forms": [...],
                "landmarks": [...],
            },
        }
        ```

        Now the next step is grounded — the agent picks a real target from the
        snapshot instead of mutating a selector it was never able to see.

        Pair it with two rules in the system prompt:

        - "Stop at the FIRST failed action and read the snapshot. Do NOT
          resend the same action with a tweaked selector."
        - "You get three attempts on the same target before the run is
          stopped. Treat each failure as a signal to re-observe, not a signal
          to try harder."

        And enforce the ceiling in code — track attempts per target and cut
        the run, rather than trusting the prompt alone.

        The general principle: when an agent loops, add observation, not
        retries. A retry budget without new information just spends the budget.
        """,
        "errors_as_observations",
    )

    yield emit(
        "Should each tool validate its own inputs, or should the framework do it?",
        """
        The framework, in one place — the registry that dispatches them.

        If every tool validates its own inputs you get 100 slightly different
        opinions about what a missing field means, and the error strings are
        inconsistent enough that the model can't learn to correct from them.
        Centralise it in `execute()`:

        ```python
        def execute(self, name, inputs=None):
            tool = self.get(name)
            if tool is None:
                return {'success': False, 'error': f"Tool not found: {name}"}
            try:
                inputs = self.normalize_inputs(inputs)
                inputs = self.validate_inputs(inputs, tool['inputSchema'])
                result = tool['execute'](inputs)
                ...
        ```

        Three things that earns you:

        1. **One error contract.** Tools may raise, return `{'success': False}`,
           or return a bare `{'error': ...}` — the registry normalises all of
           them to `success=False`. Without this, a tool that returns
           `{'error': 'nope'}` surfaces as a *successful* result whose payload
           happens to contain an error, and the agent reports the task done.
        2. **Correctable messages.** `"Invalid type for 'timeout': expected
           integer, got str ('30s')"` names the field, the expectation, and the
           offending value, so the next attempt is informed rather than random.
        3. **Forgiving coercion in exactly one place.** Models pass `"5"` for
           `5` and the whole argument object as a JSON string. Normalise that
           centrally — and note the asymmetry: coerce `"5"` to `5`, but never
           accept `True` for an integer, since `bool` is an `int` subclass and
           that one slips through a naive `isinstance` check.

        The rule of thumb: anything every tool would otherwise reimplement is a
        registry responsibility. Tools should contain the part that's actually
        specific to them.
        """,
        "central_validation",
    )

    yield emit(
        "What's the minimum structure for an agent framework that won't fall "
        "apart once it has real tools in it?",
        f"""
        Five separations. Each one exists because collapsing it causes a
        specific, predictable failure.

        **1. Registry separate from tools.** Registration, schema validation,
        dispatch, and the error contract live in one module; tools are dicts
        with `name`, `description`, `inputSchema`, `execute`. Auto-discover them
        from the directory so adding a tool is adding a file, and let optional
        dependencies fail soft — a missing browser library should cost you the
        browser tools, not the agent.

        **2. Approval policy separate from both.** One module, imported by the
        interactive agent and the autonomous runner alike. Two policies, stated
        explicitly: gated-tools-prompt when a human is present, gated-tools-
        refuse when one isn't. This is the module that must never be duplicated.

        **3. Provider separate from the loop.** The agent loop should not know
        whether it's talking to a local Ollama endpoint or a hosted API. Put
        message translation and tool-call normalisation behind a provider
        interface — this is also what lets you swap in a different model
        without touching agent logic.

        **4. Surface separate from agent.** CLI, web, and chat surfaces are
        transports. The moment approval logic leaks into a surface, the other
        surfaces are quietly less safe.

        **5. Workspace/path resolution separate from tools.** One resolver that
        every file tool calls, so sandbox containment is enforced in a single
        place instead of re-derived per tool.

        What's conspicuously *not* on this list is a natural-language routing
        layer. Selection policy belongs in the system prompt where it composes
        with context; encoding it as pattern matching in code just makes it
        rigid. For scale: this framework runs {c['sites']} regex call sites
        across {c['loc']:,} lines, none of them routing intent.

        Bound the loop too: a fixed maximum number of tool rounds per turn,
        after which the agent must produce a final answer. Without it, a
        failing tool and a determined model will burn your context window.
        """,
        "minimum_structure",
    )

    yield emit(
        "Is it fine to let my agent tell the user it did something while the "
        "tool call is still pending approval? It makes the UX feel faster.",
        """
        No — and it's worth being precise about why, because this is the
        failure that destroys trust in an agent fastest.

        If the agent says "committed and pushed" while the call is pending, one
        of two things happens. The user approves and the message happens to be
        true, which trains everyone to believe it. Or the user denies, and the
        agent has now stated something false about the state of their repo. The
        second case is rare enough to be invisible in testing and catastrophic
        in production, because the user has no reason to re-check something
        they were told was done.

        The rule: **never claim an action happened until the tool returns a
        successful result.** Not "sending", not "should be done now" — the
        claim waits for the result.

        What good behaviour looks like around a gated call:

        - Before: say what you're about to do and that it needs approval.
        - Pending: say nothing about the outcome. There isn't one yet.
        - Denied: say plainly that it did not run and nothing changed, don't
          retry the identical call, and offer to adjust.
        - Failed: report the actual error. A failed tool call is not a
          completed task with a caveat.

        This also constrains the *end* of a run. If a task was partly blocked,
        the final report says which parts completed and which didn't. An agent
        that summarises a half-finished run as success is worse than one that
        crashes, because the crash is visible.
        """,
        "no_fake_completion",
    )

    yield emit(
        "I want to stop my agent from looping. I'm going to abort the run if it "
        "calls the same tool with the same arguments more than 3 times.",
        """
        Close, but the counter is on the wrong event. Count **failures**, not
        repeats — otherwise you break legitimate work.

        Reading the same file five times across a long task is normal: the
        agent re-checks state after an edit, or revisits a file it summarised
        earlier. Staging a path, committing, then reading it again is normal.
        If repetition alone trips the detector, the agent gets aborted for
        doing its job correctly, and the failure looks random to the user.

        What actually indicates a loop is *repeated failure on the same
        logical target*:

        ```python
        REPEAT_ATTEMPT_LIMIT = 3

        def _register_tool_attempt(self, tool_name, tool_inputs):
            \"\"\"Return a signature that has already FAILED the limit, else None.\"\"\"
            for sig in self._target_signatures(tool_name, tool_inputs):
                if not sig[-1]:
                    continue
                if self._attempt_counts.get(sig, 0) >= self.REPEAT_ATTEMPT_LIMIT:
                    return sig
            return None

        def _record_tool_outcome(self, tool_name, tool_inputs, result):
            # failure increments; success RESETS
            ...
        ```

        Two details that decide whether this works in practice:

        **Signature on the logical target, not the raw argument dict.** Derive
        it from whichever of `path`, `command`, `query`, `url` the call
        carries, normalised. Hashing the whole input dict means one changed
        whitespace character reads as a brand-new attempt, which is exactly
        what a looping agent produces.

        **One signature per sub-action for batch tools.** If a tool takes a
        list of actions, a batch of clicks on five different selectors is five
        targets, not one attempt. Collapse them and you abort a legitimate
        multi-step batch on its first retry.

        Success resetting the counter is the part that makes it safe to set the
        limit as low as 3.
        """,
        "loop_detection_counts_failures",
    )

    yield emit(
        "Each of my file tools checks that the path stays inside the workspace "
        "before it does anything. Is that the right place for the check?",
        """
        No — that's the check that must exist in exactly one place, and you
        currently have N copies of it.

        Sandbox containment fails on the tool you forgot, and you will forget
        one. A new tool gets added under deadline, the author copies a sibling
        that happens to predate the check, and now `move_path` accepts
        `../../etc/hosts` while `read_file` doesn't. Nothing surfaces that
        inconsistency — there's no test that says "every tool containing a path
        argument validates it," because that's not a property you can easily
        assert across a directory of independent modules.

        Put resolution and containment in one resolver that every file tool
        calls:

        ```python
        from core.paths import resolve_repo_path

        def _read_file(inputs):
            path = resolve_repo_path(inputs['path'], kind='file')
            return {'path': get_display_path(path), 'content': path.read_text(...)}
        ```

        `resolve_repo_path` resolves symlinks, rejects traversal, rejects
        absolute paths outside the workspace, and raises `PermissionError`.
        The tool contains only the part that's actually about reading a file.

        Then test the *resolver* rather than each tool:

        - a path inside the workspace resolves
        - an absolute path outside raises
        - `../up.txt` raises
        - a symlink pointing outside raises

        Four tests cover every current and future file tool. Per-tool checks
        would need four tests per tool and still miss the next one.

        Same shape as centralising schema validation in the registry: if every
        tool would otherwise reimplement it, it isn't the tool's job.
        """,
        "single_path_resolver",
    )

    yield emit(
        "How do I keep my agent working with both a local model and a hosted "
        "API without the loop filling up with if-statements?",
        """
        Put a provider interface between the loop and the wire format, and let
        the loop speak exactly one dialect.

        The loop should never know which backend it's on. It emits and consumes
        one canonical message shape; each provider translates in both
        directions:

        ```python
        class Provider:
            def chat(self, messages, tools=None, **kw) -> dict:
                \"\"\"Returns a canonical assistant message:
                   {role, content, tool_calls:[{id, type, function:{name, arguments}}]}\"\"\"
        ```

        The translation is where the real differences live, and there are more
        than you'd expect:

        - **Arguments as string vs object.** Some backends return
          `arguments` as a JSON string, some as an object. Normalise on the way
          in, or every downstream consumer needs both branches — and one of
          them will forget.
        - **Tool results.** One shape wants a `tool` role with a
          `tool_call_id`; another wants results inline in the user turn.
        - **No-tools mode.** Some local completion models reject a request
          that carries a `tools` array at all. That has to be representable —
          "send no tool definitions" is a real state, not an error.
        - **Streaming.** Chunk shapes differ; reassemble to the same canonical
          message so callers see no difference.

        Two rules that keep this honest:

        1. **Normalise inbound, don't special-case outbound.** Providers adapt
           to the loop's dialect, never the reverse.
        2. **No provider-specific behaviour above the interface.** The moment
           the loop checks `if provider == "ollama"`, the abstraction has
           already failed and the second backend will be a rewrite.
        """,
        "provider_abstraction",
    )

    yield emit(
        "My agent runs out of context on long tasks. Should I just truncate the "
        "oldest messages?",
        """
        Not by raw truncation — you'll cut the message that explains what the
        user actually wanted, and the agent will confidently finish the wrong
        task.

        Long agent runs have a specific shape: a short instruction followed by
        an enormous volume of tool output. The instruction is the part that
        must survive; the tool output is the part that's mostly redundant once
        it's been acted on.

        So compact by role, not by age:

        - **Never drop the system prompt.** It holds the tool-selection policy
          and the boundaries. An agent that loses it mid-run stops respecting
          rules it followed five minutes ago, which reads as the model
          "ignoring instructions."
        - **Keep the original user instruction verbatim.** It is small and it
          is the definition of done.
        - **Summarise old tool results aggressively.** A 4,000-line directory
          listing from ten steps ago can become one line. Keep the most recent
          results intact — those are what the next step reasons over.
        - **Preserve tool_call/result pairing.** If you drop an assistant turn
          containing a `tool_call` but keep the matching result, you produce a
          transcript that references a call that isn't there. Some backends
          reject it outright; the rest get confused. Always evict the pair.

        Trigger compaction on a threshold below the real limit, so compaction
        itself has room to run.

        And bound the loop independently: a fixed maximum number of tool rounds
        per turn, after which the agent must answer. Compaction stops you
        running out of context; the round cap stops you spending the whole
        budget on a tool that will never succeed. You want both.
        """,
        "context_compaction",
    )

    yield emit(
        "Is there any downside to registering all of my agent's tools and "
        "sending every schema on every request? I have about a hundred.",
        f"""
        Yes, and it's larger than most people measure before committing to it.
        Your tool schemas are prompt budget you pay on **every single turn**.

        Concretely, measured on a registry this size: 107 tools serialise to
        roughly 63,000 characters — about 15,900 tokens. Add a system prompt
        that itself lists every tool name and you're at a **~17,500-token
        floor before the user has said anything.**

        That's survivable at inference on a long-context model. Where it hurts:

        - **Fine-tuning.** If you train on transcripts carrying the full block,
          the invariant preamble dominates the token budget — in one realistic
          mixture, samples that were 35% of rows became 92% of tokens, with the
          same preamble repeated thousands of times. You pay for it and learn
          almost nothing from it.
        - **Selection quality.** A hundred schemas is a lot of surface for
          near-duplicates to hide in. Selection errors cluster among
          neighbours — `read_file` vs `read_file_range` vs `read_symbol` — and
          more options makes that worse, not better.
        - **Every retry.** The block is re-sent each round of a multi-round
          agent loop, so a 5-round turn pays it five times.

        Practical mitigations, in order of how much they buy you:

        1. **Group your tools and scope by task profile.** A robot-control turn
           doesn't need git tools. Scope the *prompt's* tool list with the
           block so they can't disagree — if the prompt names tools the block
           omits, the model will call things it wasn't offered.
        2. **Audit the outliers.** Schema cost is wildly uneven. In this
           registry one group of 5 web tools costs more than 27 repo tools,
           because a single browser-automation schema is enormous. Trimming
           one description can beat deleting ten tools.
        3. **Mask the preamble when training.** Compute loss on assistant
           tokens only, so you're not spending gradient on text that never
           varies.

        Registering everything is right. *Sending* everything, every turn,
        without measuring what it costs is the part worth reconsidering.
        """,
        "tool_schemas_are_prompt_budget",
    )

    yield emit(
        "Where should tool-selection rules live — in code, or in the prompt?",
        """
        In the prompt, as policy. In code, only the things that must be true
        regardless of what the model decides.

        The split that holds up:

        **Prompt — anything about which tool fits a situation.**

        - "For project overview questions, call inspect_project first."
        - "Don't stop at find_definition when the user wants to see the body —
          follow it with read_symbol."
        - "Prefer the dedicated git tools over raw shell for status and diffs."

        These are judgement calls that depend on context, and they compose:
        the model can apply two of them at once to a request neither
        anticipated. Encoded as dispatch rules in code, they'd conflict and
        you'd be writing precedence logic for cases you can't enumerate.

        **Code — anything that must hold even when the model is wrong.**

        - Approval gates on destructive and arbitrary-execution tools.
        - Path containment.
        - Schema validation and the error contract.
        - The repeat-failure limit and the tool-round cap.

        The test for which side something belongs on: *what happens if the
        model ignores it?* If the answer is "a suboptimal tool choice, and the
        next turn recovers" — prompt. If it's "data loss, an escaped sandbox,
        or a silent false claim of success" — code, enforced at dispatch, not
        requested in text.

        The common failure is putting safety in the prompt ("never delete
        files without asking"), which makes it advisory, and putting selection
        in code (a keyword router), which makes it rigid. That's exactly
        backwards on both counts.
        """,
        "policy_in_prompt_invariants_in_code",
    )


# ==========================================================================
# driver
# ==========================================================================

EXTRACTORS = {
    "tool_impl": extract_tool_impl,
    "tool_call": extract_tool_calls,
    "commits": extract_commits,
    "docs_qa": extract_docs_qa,
    "approval": extract_approval,
    "design": extract_design,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1 — static extraction from LumaKit")
    ap.add_argument("--only", choices=sorted(EXTRACTORS), help="run one extractor")
    ap.add_argument("--limit", type=int, help="cap samples per extractor")
    ap.add_argument("--show", type=int, default=0, help="print first N samples")
    args = ap.parse_args()

    names = [args.only] if args.only else list(EXTRACTORS)
    total = 0
    for name in names:
        samples = list(EXTRACTORS[name]())
        if args.limit:
            samples = samples[: args.limit]
        out = STAGED / f"static_{name}.jsonl"
        n = write_jsonl(out, samples, secrets_on_hit="fail")
        total += n
        print(f"  {name:<10} {n:>5} samples -> {out.relative_to(STAGED.parent.parent)}")

        for s in samples[: args.show]:
            d = s.to_dict()
            print("\n" + "=" * 72)
            print(f"id={d['id']}  meta={json.dumps(d['meta'], sort_keys=True)}")
            print(f"provenance={json.dumps(d['provenance'], sort_keys=True)}")
            if d["tools"]:
                print(f"tools[{len(d['tools'])}]: "
                      f"{[t['function']['name'] for t in d['tools']]}")
            for m in d["messages"]:
                body = m.get("content")
                if body is None and m.get("tool_calls"):
                    body = ""
                head = f"--- {m['role']}"
                if m.get("name"):
                    head += f" ({m['name']})"
                print(f"{head} ---")
                print(textwrap.shorten(str(body).replace("\n", " "), 700)
                      if len(str(body)) > 700 else body)
                for tc in m.get("tool_calls") or []:
                    print(f"  ->  {tc['function']['name']}("
                          f"{json.dumps(tc['function']['arguments'], sort_keys=True)})")
            print("=" * 72)

    print(f"\ntotal: {total} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
