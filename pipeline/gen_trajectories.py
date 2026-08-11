"""Stage 3 — synthetic trajectory generation. The core asset.

Runs LumaKit for real, in a sandboxed throwaway workspace, against a
configurable teacher endpoint, and logs each full episode (every turn, tool
call, tool result, approval decision, and the final report) as one training
sample.

Isolation, in layers:

  * temp workspace seeded with a fixture project; the agent's workspace root
    is set there, so path containment keeps file tools inside it
  * HOME/USERPROFILE redirected to a throwaway dir, so LumaKit builds a fresh
    ~/.lumakit and cannot read or write the developer's real config, memory
    DB, chat store, or web session token
  * safe mode ON and approvals ON, written into the isolated runtime config
  * the LumaBot daemon is an in-process mock (pipeline.mock_lumabot); no
    hardware, and failure modes are injected deterministically
  * no network egress beyond the model endpoint

Approximation worth stating plainly: autonomous episodes run the interactive
`Agent` with an autonomous-context system suffix and an approval hook that
denies gated tools, rather than driving `core/task_runner.py` itself. The
resulting *behaviour* under test is the same — a gated tool cannot execute and
the model must report that — but it is not the task runner's own code path.
Recorded in provenance as `gen_trajectories:autonomous-approximated`.

Run:
    python -m pipeline.gen_trajectories --smoke            # 5 episodes
    python -m pipeline.gen_trajectories --limit 20
    python -m pipeline.gen_trajectories                    # all seed tasks
    python -m pipeline.gen_trajectories --smoke --dry-run  # plumbing only, $0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pipeline import provenance
from pipeline.config import (LUMAKIT_REPO, STAGED, isolated_lumakit_env, rng)
from pipeline.glimmer import coerce_arguments
from pipeline.sample import Sample, SampleInvalid, validate, write_jsonl
from pipeline.seed_tasks import (TaskSpec, WORKSPACE_FILES, all_tasks,
                                 smoke_tasks)
from pipeline.teacher import api_key, base_url, teacher_model

OUT_PATH = STAGED / "trajectories.jsonl"

AUTONOMOUS_NOTE = ("\n\nYou are running inside an autonomous task. No human is "
                   "present to approve tool actions.")
TELEGRAM_NOTE = ("\n\nInterface: Telegram. The active user is a trusted "
                 "non-owner.")

# Which injected daemon fault each robot-ish task wants. Keyed on task id
# prefix so seed_tasks stays free of harness detail.
_FAULT_FOR_TASK = {
    "err_003": "obstacle",
    "err_004": "camera_unavailable",
    "err_005": "motors_not_ready",
    "err_026": "obstacle",
    "err_027": "autonomy_unavailable",
}


# ---------------------------------------------------------------------------
# The per-episode driver (runs inside the LumaKit checkout, one subprocess)
# ---------------------------------------------------------------------------

_DRIVER = r"""
import contextlib, io, json, os, sys
from pathlib import Path

spec = json.loads(os.environ["LUMEN_EPISODE"])
events = []
_buf = io.StringIO()

with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    from core.app_runtime_config import save_app_runtime_config
    save_app_runtime_config({
        "llm_provider": spec["provider"],
        "provider_models": {spec["provider"]: spec["model"]},
        "provider_fallback_models": {},
        "require_tool_approvals": True,
        "tools_enabled": True,
        "safe_mode": True,
    })

    from core.display import DisplayHooks
    from agent import Agent

    pending = {"name": None, "inputs": None}
    approvals = {"n": 0}
    policy = spec["approval"]

    def show_tool_call(name, inputs):
        pending["name"], pending["inputs"] = name, inputs
        events.append({"type": "tool_call", "name": name, "inputs": inputs})

    def show_tool_result(result):
        events.append({"type": "tool_result",
                       "success": bool((result or {}).get("success", True))})

    def confirm(prompt):
        approvals["n"] += 1
        if policy == "approve":
            decision = True
        elif policy == "deny":
            decision = False
        elif policy == "approve_first_deny_second":
            decision = approvals["n"] == 1
        else:
            decision = True
        events.append({"type": "approval", "prompt": prompt,
                       "tool": pending["name"], "approved": decision,
                       "index": approvals["n"]})
        return decision

    agent = Agent(display=DisplayHooks(
        show_tool_call=show_tool_call,
        show_tool_result=show_tool_result,
        confirm=confirm,
        status=lambda *_a, **_k: None,
        stream_delta=lambda *_a, **_k: False,
        stream_end=lambda *_a, **_k: None,
    ))
    agent.set_workspace_root(Path(spec["workspace"]))
    if spec.get("profile"):
        agent.set_runtime_profile(spec["profile"])

    # Inject the surface/autonomy context into the SYSTEM message through
    # LumaKit's own hook, which rebuilds messages[0]. Without this the agent
    # never learns it is running unattended and an "autonomous" episode is
    # byte-identical to an interactive one.
    if spec.get("context_note"):
        agent.apply_runtime_overrides(context_instructions=spec["context_note"])

    error = None
    try:
        agent.ask_llm(spec["prompt"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    messages = []
    for m in getattr(agent, "messages", []) or []:
        entry = {k: v for k, v in m.items()
                 if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
        messages.append(entry)

sys.stdout.write(json.dumps({
    "messages": messages,
    "events": events,
    "error": error,
    "log": _buf.getvalue()[-2000:],
}))
"""


# ---------------------------------------------------------------------------
# Episode -> Sample
# ---------------------------------------------------------------------------

def _split_parallel_calls(messages: list[dict]) -> list[dict]:
    """Glimmer supports ONE tool call per assistant turn.

    LumaKit's loop iterates `for tool_call in tool_calls`, so it can emit
    several at once. Rewrite those into sequential assistant/tool pairs,
    preserving order, rather than dropping the episode.
    """
    by_id = {}
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            by_id[m["tool_call_id"]] = m

    out: list[dict] = []
    emitted_results: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant" and len(m.get("tool_calls") or []) > 1:
            calls = m["tool_calls"]
            for i, tc in enumerate(calls):
                out.append({
                    "role": "assistant",
                    # Keep the status text on the first split turn only.
                    "content": m.get("content") if i == 0 else None,
                    "tool_calls": [tc],
                })
                result = by_id.get(tc.get("id"))
                if result is not None:
                    out.append(result)
                    emitted_results.add(tc.get("id"))
            continue
        if (m.get("role") == "tool"
                and m.get("tool_call_id") in emitted_results):
            continue          # already emitted beside its split call
        out.append(m)
    return out


WORKSPACE_PLACEHOLDER = "/workspace"
HOME_PLACEHOLDER = "/home/agent"

# Any absolute path pointing at one of this harness's throwaway directories,
# in either slash style and with either single or JSON-escaped separators.
_TEMP_PATH_RE = re.compile(
    r"[A-Za-z]:(?:\\\\|\\|/)[^\"'\s,)\]}]*?lumen-ember-(?:ws|home)-[A-Za-z0-9_]+"
    r"(?:(?:\\\\|\\|/)[^\"'\s,)\]}]*)?"
)

# The bare directory *name*, with no path attached. `get_project_tree` labels
# its output with the workspace basename, so the drive-prefixed pattern above
# never sees it.
_TEMP_NAME_RE = re.compile(r"lumen-ember-(ws|home)-[A-Za-z0-9_]+")


def _path_scrubber(workspace: str, home: str):
    """Replace machine-specific absolute paths with stable placeholders.

    Tool results echo real paths back — `workspace_context` returns `cwd`,
    `home`, and `data_dir` verbatim — so without this the dataset would carry
    the developer's username and a fresh temp directory that changes every
    run. Same failure as the system-prompt leak, one layer down: it breaks
    reproducibility AND teaches the model somebody's directory layout.
    """
    pairs: list[tuple[str, str]] = []
    for raw, placeholder in ((workspace, WORKSPACE_PLACEHOLDER),
                             (home, HOME_PLACEHOLDER)):
        if not raw:
            continue
        variants = {raw}
        try:                                  # long form vs 8.3 short form
            variants.add(str(Path(raw).resolve()))
        except OSError:
            pass
        for v in list(variants):
            variants.update({v.replace("\\", "/"), v.replace("\\", "\\\\")})
        pairs.extend((v, placeholder) for v in variants)
    # Belt and braces: the 8.3 short form and the bare username can appear via
    # paths we did not construct ourselves.
    user = os.environ.get("USERNAME") or ""
    for extra in filter(None, (user, (user[:6].upper() + "~1") if user else "")):
        pairs.append((extra, "user"))
    pairs.sort(key=lambda p: -len(p[0]))

    def scrub(text: str | None) -> str | None:
        if not text:
            return text
        for needle, replacement in pairs:
            if needle and needle in text:
                text = text.replace(needle, replacement)
        # Safety net for any harness path we did not construct ourselves
        # (resolved differently, embedded in an error string, etc.).
        text = _TEMP_PATH_RE.sub(
            lambda m: (WORKSPACE_PLACEHOLDER if "-ws-" in m.group(0)
                       else HOME_PLACEHOLDER), text)
        text = _TEMP_NAME_RE.sub(
            lambda m: "workspace" if m.group(1) == "ws" else "home", text)
        return text

    return scrub


def _normalise(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        entry: dict = {"role": role}
        if role == "assistant" and m.get("tool_calls"):
            entry["content"] = m.get("content")
            entry["tool_calls"] = [coerce_arguments(tc) for tc in m["tool_calls"]]
        elif role == "tool":
            entry["content"] = m.get("content")
            if m.get("tool_call_id"):
                entry["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                entry["name"] = m["name"]
        else:
            entry["content"] = m.get("content") or ""
        out.append(entry)
    return out


def episode_to_sample(task: TaskSpec, payload: dict, model: str) -> Sample | None:
    messages = _normalise(_split_parallel_calls(payload.get("messages") or []))
    if len(messages) < 2:
        return None

    scrub = _path_scrubber(payload.get("workspace") or "", payload.get("home") or "")
    for m in messages:
        m["content"] = scrub(m.get("content"))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, dict):
                fn["arguments"] = {
                    k: (scrub(v) if isinstance(v, str) else v)
                    for k, v in args.items()
                }

    events = payload.get("events") or []
    approvals = [e for e in events if e["type"] == "approval"]
    tool_calls = [e for e in events if e["type"] == "tool_call"]

    method = ("gen_trajectories:autonomous-approximated" if task.autonomous
              else "gen_trajectories:interactive")

    sample = Sample(
        messages=messages,
        provenance=provenance.synthetic(method, model),
        meta={
            "stage": "3",
            "extractor": "trajectories",
            "task_id": task.id,
            "category": task.category,
            "approval_policy": task.approval,
            "autonomous": task.autonomous,
            "surface": task.surface,
            "profile": task.profile,
            "n_tool_calls": len(tool_calls),
            "n_approvals": len(approvals),
            "approvals_denied": sum(1 for a in approvals if not a["approved"]),
            "tools_used": sorted({e["name"] for e in tool_calls if e.get("name")}),
            "expect": task.expect,
            "teacher_model": model,
            # Boundary episodes carry the dataset's highest weight; see
            # docs/FINDINGS.md §3 for why this is where the headroom is.
            "weight": 3.0 if task.category in ("approval_gate", "should_refuse")
                      else 2.0 if task.category == "error_recovery" else 1.0,
            "tags": ["trajectory", task.category],
        },
    )
    try:
        validate(sample.to_dict())
    except SampleInvalid as e:
        sample.meta["invalid"] = str(e)[:200]
        return None
    return sample


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _seed_workspace(task: TaskSpec) -> Path:
    # .resolve() because Windows mkdtemp hands back the 8.3 short form
    # (C:\Users\PATRIC~1\...) while LumaKit stores the long form, and a
    # literal path replacement against the wrong variant silently matches
    # nothing.
    root = Path(tempfile.mkdtemp(prefix="lumen-ember-ws-")).resolve()
    for rel in (task.seed_files or ()):
        content = WORKSPACE_FILES.get(rel)
        if content is None:
            continue
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    return root


def run_episode(task: TaskSpec, *, model: str, timeout: float = 300.0) -> dict:
    from pipeline.mock_lumabot import MockDaemon

    workspace = _seed_workspace(task)
    fault = _FAULT_FOR_TASK.get(task.id, "none")
    if task.category == "error_recovery" and "robot" in task.expect.lower():
        fault = fault if fault != "none" else "unreachable"

    env = isolated_lumakit_env()
    env["LLM_PROVIDER"] = "openai"
    env["LLM_BASE_URL"] = base_url()
    env["LLM_API_KEY"] = api_key() or ""
    env["LLM_MODEL"] = model
    # An autonomous run has nobody to ask, so every gated tool is denied
    # regardless of what the task spec says. Leaving these on "approve" would
    # have produced episodes where an unattended agent gets waved through —
    # the exact behaviour the dataset exists to correct.
    approval = "deny" if task.autonomous else task.approval

    context_note = ""
    if task.autonomous:
        context_note = AUTONOMOUS_NOTE.strip()
    elif task.surface == "telegram_trusted":
        context_note = TELEGRAM_NOTE.strip()

    env["LUMEN_EPISODE"] = json.dumps({
        "prompt": task.prompt,
        "approval": approval,
        "context_note": context_note,
        "workspace": str(workspace),
        "profile": task.profile,
        "provider": "openai",
        "model": model,
    })

    try:
        with MockDaemon(fault) as bot_url:
            env["LUMABOT_URL"] = bot_url
            proc = subprocess.run(
                [sys.executable, "-c", _DRIVER],
                cwd=str(LUMAKIT_REPO), env=env,
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        if proc.returncode != 0:
            return {"messages": [], "events": [],
                    "error": f"driver exit {proc.returncode}: {proc.stderr[-600:]}"}
        payload = json.loads(proc.stdout)
        # Paths the scrubber needs; the driver cannot know the parent's view.
        payload["workspace"] = str(workspace)
        payload["home"] = env.get("USERPROFILE", "")
        return payload
    except subprocess.TimeoutExpired:
        return {"messages": [], "events": [], "error": f"timeout after {timeout}s"}
    except json.JSONDecodeError as e:
        return {"messages": [], "events": [],
                "error": f"driver emitted non-JSON: {e}"}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 — trajectory generation")
    ap.add_argument("--smoke", action="store_true", help="5 tasks, end to end")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--category")
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate plumbing without calling the endpoint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = args.model or teacher_model()
    tasks = smoke_tasks() if args.smoke else all_tasks()
    if args.category:
        tasks = [t for t in tasks if t.category == args.category]
    if args.limit:
        tasks = tasks[: args.limit]

    if args.dry_run:
        ok = 0
        for t in tasks:
            ws = _seed_workspace(t)
            files = sorted(p.relative_to(ws).as_posix() for p in ws.rglob("*")
                           if p.is_file())
            shutil.rmtree(ws, ignore_errors=True)
            env = isolated_lumakit_env()
            assert env["USERPROFILE"] != os.environ.get("USERPROFILE")
            ok += 1
            print(f"  ok {t.id:<20} {t.category:<15} files={len(files)} "
                  f"approval={t.approval} autonomous={t.autonomous}")
        print(f"\ndry run: {ok}/{len(tasks)} episodes constructible, endpoint untouched")
        return 0

    if not api_key():
        print("No API key. Put OPENROUTER_API_KEY in .env, or use --dry-run.")
        return 1

    out_path = Path(args.out) if args.out else OUT_PATH
    print(f"Stage 3 — {len(tasks)} episodes | model={model} | out={out_path.name}")

    samples, failures = [], []
    started = time.monotonic()
    for i, task in enumerate(tasks, 1):
        payload = run_episode(task, model=model, timeout=args.timeout)
        sample = episode_to_sample(task, payload, model) if not payload.get("error") else None
        if sample is None:
            failures.append((task.id, payload.get("error") or "unusable episode"))
            mark = "F"
        else:
            samples.append(sample)
            mark = "."
        print(mark, end="", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(tasks)}")
    print()

    n = write_jsonl(out_path, samples, secrets_on_hit="fail")
    elapsed = time.monotonic() - started
    print(f"\n  {n} episodes -> {out_path}")
    print(f"  failures: {len(failures)}  |  {elapsed / 60:.1f} min")
    for tid, why in failures[:10]:
        print(f"    {tid}: {why[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
