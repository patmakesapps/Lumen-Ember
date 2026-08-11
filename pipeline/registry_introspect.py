"""Load LumaKit's real ToolRegistry and dump every tool schema to JSON.

This is the ground truth for three later stages:
  - stage 1 generates schema->implementation and schema->call samples from it
  - stage 3 hands these schemas to the teacher as the `tools` block
  - stage 4 validates every emitted tool call against them

We import LumaKit in a *subprocess* rooted at the repo so its `load_dotenv()`
and optional-dependency probing cannot leak into this process, and so a tool
module that misbehaves at import can't take the pipeline down.

Run:  python -m pipeline.registry_introspect
Out:  data/raw/tool_schemas.json   (idempotent; stable key order)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pipeline.config import LUMAKIT_REPO, RAW

OUT_PATH = RAW / "tool_schemas.json"

# Driver runs inside the LumaKit checkout with it on sys.path.
_DRIVER = r"""
import json, sys, io, contextlib

# Tool modules print/warn on missing optional deps; keep stdout clean for JSON.
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    from pathlib import Path
    import tempfile

    from tool_registry import ToolRegistry
    from core import approval_policy as ap
    from tools.code_intel.code_index import LazyCodeIndex

    # Mirror Agent.__init__ exactly: code_intel tools are factories that take a
    # CodeIndex, so the folder loader skips them and they are registered from
    # LazyCodeIndex.get_tools(). Building the registry any other way would
    # produce a tool list the real agent never sees.
    reg = ToolRegistry()
    reg.load_tools_from_folder(skip_dirs={'code_intel'})

    # Lazy index over a throwaway root: get_tools() only needs the schemas, and
    # pointing at an empty dir keeps introspection from scanning a real tree.
    _tmp_root = Path(tempfile.mkdtemp(prefix='lumen-ember-introspect-'))
    for tool in LazyCodeIndex(root=_tmp_root, storage_manager=None).get_tools():
        reg.register(tool, group='code_intel')

    tools = []
    for name, t in sorted(reg.tools.items()):
        tools.append({
            'name': name,
            'description': t.get('description', ''),
            'inputSchema': t.get('inputSchema', {}),
            'group': t.get('group', 'general'),
            'llm_exposed': bool(t.get('llm_exposed', True)),
            'always_confirm': name in ap.ALWAYS_CONFIRM_TOOLS,
            'autonomous_refused': name in ap.AUTONOMOUS_REFUSED_TOOLS,
            'owner_only': name in ap.OWNER_ONLY_TOOLS,
            'limited_denied': name in ap.LIMITED_DENIED_TOOLS,
        })

    policy = {
        'ALWAYS_CONFIRM_TOOLS': sorted(ap.ALWAYS_CONFIRM_TOOLS),
        'AUTONOMOUS_REFUSED_TOOLS': sorted(ap.AUTONOMOUS_REFUSED_TOOLS),
        'OWNER_ONLY_TOOLS': sorted(ap.OWNER_ONLY_TOOLS),
        'LIMITED_DENIED_TOOLS': sorted(ap.LIMITED_DENIED_TOOLS),
        'PROTECTED_SHELL_COMMAND_RE': ap.PROTECTED_SHELL_COMMAND_RE.pattern,
        'COMMAND_TOOLS': sorted(ap._COMMAND_TOOLS),
        'ROLES': list(ap.VALID_ROLES),
    }

payload = {'tools': tools, 'policy': policy, 'import_log': _buf.getvalue()[-4000:]}
sys.stdout.write(json.dumps(payload))
"""


def dump_registry(*, force: bool = False) -> dict:
    """Return the registry payload, refreshing the cached JSON if needed."""
    if OUT_PATH.exists() and not force:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))

    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd=str(LUMAKIT_REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to load LumaKit registry (exit {proc.returncode}).\n"
            f"stderr:\n{proc.stderr[-3000:]}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Registry driver did not emit JSON: {e}\n"
            f"stdout head:\n{proc.stdout[:1500]}"
        ) from e

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def openai_tools(payload: dict | None = None, *, exposed_only: bool = True) -> list[dict]:
    """Registry -> OpenAI function-tool blocks (what the chat template wants)."""
    payload = payload or dump_registry()
    out = []
    for t in payload["tools"]:
        if exposed_only and not t["llm_exposed"]:
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


def by_name(payload: dict | None = None) -> dict[str, dict]:
    payload = payload or dump_registry()
    return {t["name"]: t for t in payload["tools"]}


def _main() -> int:
    payload = dump_registry(force="--force" in sys.argv)
    tools = payload["tools"]
    groups: dict[str, int] = {}
    for t in tools:
        groups[t["group"]] = groups.get(t["group"], 0) + 1

    print(f"Wrote {OUT_PATH}")
    print(f"  tools:            {len(tools)}")
    print(f"  llm_exposed:      {sum(1 for t in tools if t['llm_exposed'])}")
    print(f"  always_confirm:   {sum(1 for t in tools if t['always_confirm'])}")
    print(f"  autonomous_refused: {sum(1 for t in tools if t['autonomous_refused'])}")
    print(f"  owner_only:       {sum(1 for t in tools if t['owner_only'])}")
    print(f"  groups:           " + ", ".join(
        f"{g}={n}" for g, n in sorted(groups.items(), key=lambda kv: -kv[1])
    ))
    log = payload.get("import_log", "").strip()
    if log:
        skipped = [ln for ln in log.splitlines() if "skipping" in ln]
        if skipped:
            print(f"  skipped modules (missing optional deps): {len(skipped)}")
            for ln in skipped[:8]:
                print(f"    - {ln.strip()[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
