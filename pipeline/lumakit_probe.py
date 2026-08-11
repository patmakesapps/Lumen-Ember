"""Extract LumaKit's *real* runtime artifacts, so training matches inference.

Specifically the system prompts produced by `Agent.build_system_prompt()` for
each runtime profile. Writing our own approximation would train the model
against a prompt it will never actually see.

Runs in a subprocess rooted at the LumaKit checkout for the same reasons as
`registry_introspect`. Tool use is forced ON explicitly — the flag is
persisted in the developer's ~/.lumakit config, and extraction must not depend
on local machine state.

Run:  python -m pipeline.lumakit_probe
Out:  data/raw/system_prompts.json
"""

from __future__ import annotations

import json
import subprocess
import sys

from pipeline.config import LUMAKIT_REPO, RAW, isolated_lumakit_env

OUT_PATH = RAW / "system_prompts.json"

# Canonical stand-in for the extraction machine's workspace path, so prompts
# are byte-identical across runs and machines.
WORKSPACE_PLACEHOLDER = "/workspace"

_DRIVER = r"""
import contextlib, io, json, os, sys, tempfile
from pathlib import Path

# HOME/USERPROFILE point at a throwaway dir (config.isolated_lumakit_env), so
# LumaKit builds a pristine data dir and falls back to DEFAULT_CONFIG. Tools
# are therefore already on, and nothing here can write to the developer's real
# ~/.lumakit. An earlier version called set_tools_enabled(True), which
# persists to disk and silently flipped that setting in the live install.
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    from core.app_runtime_config import tools_enabled
    from core.paths import get_data_dir
    assert tools_enabled(), 'isolated data dir should default tools on'
    _data_dir = str(get_data_dir())

    from agent import Agent
    a = Agent()
    # Stable path, not mkdtemp: the default prompt embeds "Current working
    # directory: <root>", so a random root makes the prompt — and therefore
    # every sample id derived from it — different on every run.
    # .resolve() matters on Windows: gettempdir() can hand back the 8.3 short
    # form (C:\Users\PATRIC~1\...) while set_workspace_root stores the long
    # form, and then the parent's path substitution silently matches nothing.
    _ws = (Path(tempfile.gettempdir()) / 'lumen-ember-probe-workspace').resolve()
    _ws.mkdir(parents=True, exist_ok=True)
    a.set_workspace_root(_ws)

    prompts = {}
    for profile in (None, 'lumabot', 'lumabot_remote'):
        a.set_runtime_profile(profile)
        prompts[profile or 'default'] = a.build_system_prompt()

    a.set_runtime_profile(None)
    meta = {'max_tool_rounds': a.MAX_TOOL_ROUNDS,
            'tool_count': len(a.registry.tools),
            'data_dir': _data_dir,
            'workspace_root': str(_ws)}

sys.stdout.write(json.dumps({'prompts': prompts, 'meta': meta}))
"""


def probe(*, force: bool = False) -> dict:
    if OUT_PATH.exists() and not force:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))

    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd=str(LUMAKIT_REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=isolated_lumakit_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LumaKit probe failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    payload = json.loads(proc.stdout)

    # Normalise the workspace path out of the prompts. It is an absolute path
    # on the extraction machine, so leaving it in would (a) make the dataset
    # differ per machine and (b) train the model on somebody's temp directory.
    # A neutral placeholder is both reproducible and better data.
    root = payload["meta"].get("workspace_root")
    if root:
        for variant in {root, root.replace("\\", "/"), root.replace("\\", "\\\\")}:
            payload["prompts"] = {
                k: v.replace(variant, WORKSPACE_PLACEHOLDER)
                for k, v in payload["prompts"].items()
            }
        payload["meta"]["workspace_root"] = WORKSPACE_PLACEHOLDER
        leaked = [k for k, v in payload["prompts"].items() if root in v]
        if leaked:
            raise RuntimeError(
                f"workspace path survived normalisation in prompts {leaked} — "
                f"the dataset would not be reproducible. root={root!r}"
            )

    # The isolated HOME is a fresh mkdtemp per run, so recording it verbatim
    # would make this file differ on every run for a purely diagnostic field.
    payload["meta"]["data_dir"] = "<isolated>"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def system_prompt(profile: str = "default") -> str:
    return probe()["prompts"][profile]


_TOOLS_LINE = "Your tools: "


def scoped_system_prompt(names: list[str]) -> str:
    """The real default prompt with its tool-name list narrowed to `names`.

    Why this exists: the full 107-tool block plus this prompt is a ~17.5k-token
    floor per sample. At the target mixture that makes LumaKit rows 35% of the
    corpus but 92% of its tokens, repeating one invariant preamble ~7,000
    times. Training on a narrowed set keeps the token budget honest; the model
    still meets the full block at inference, where Glimmer's 131k context makes
    it a non-issue.

    This mirrors LumaKit's own mechanism rather than inventing one: the
    `lumabot` profile already renders `Your tools:` from
    `registry.list(groups=...)`. We generalise the same substitution to an
    explicit name set, leaving every behavioural rule in the prompt untouched.

    A minority of samples keep the full list (see FULL_BLOCK_RATE) so the
    crowded real condition is represented too.
    """
    prompt = system_prompt("default")
    out = []
    replaced = False
    for line in prompt.splitlines():
        if line.startswith(_TOOLS_LINE):
            out.append(_TOOLS_LINE + ", ".join(sorted(names)))
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise RuntimeError(
            f"Could not find {_TOOLS_LINE!r} in LumaKit's system prompt — the "
            f"prompt format changed; re-check pipeline/lumakit_probe.py"
        )
    return "\n".join(out)


if __name__ == "__main__":
    p = probe(force="--force" in sys.argv)
    print(f"Wrote {OUT_PATH}")
    for k, v in sorted(p["prompts"].items()):
        print(f"  {k:16} {len(v):>6} chars")
    print(f"  MAX_TOOL_ROUNDS = {p['meta']['max_tool_rounds']}, "
          f"tools = {p['meta']['tool_count']}")
