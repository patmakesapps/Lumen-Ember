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

from pipeline.config import LUMAKIT_REPO, RAW

OUT_PATH = RAW / "system_prompts.json"

_DRIVER = r"""
import contextlib, io, json, os, sys, tempfile
from pathlib import Path

# Determinism + safety: the default system prompt splices in an identity block
# built from LUMI_EMAIL_ADDRESS and an on-disk identity file described as
# "accounts, credentials, site logins". Neither may ever reach a sample, and
# both would make extraction depend on the developer's machine.
os.environ.pop('LUMI_EMAIL_ADDRESS', None)

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    # Deterministic: never inherit the local ~/.lumakit tool switch.
    from core.app_runtime_config import set_tools_enabled
    set_tools_enabled(True)

    from agent import Agent
    a = Agent()
    a.set_workspace_root(Path(tempfile.mkdtemp(prefix='lumen-ember-probe-')))

    prompts = {}
    for profile in (None, 'lumabot', 'lumabot_remote'):
        a.set_runtime_profile(profile)
        prompts[profile or 'default'] = a.build_system_prompt()

    a.set_runtime_profile(None)
    meta = {'max_tool_rounds': a.MAX_TOOL_ROUNDS, 'tool_count': len(a.registry.tools)}

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
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LumaKit probe failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    payload = json.loads(proc.stdout)
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
