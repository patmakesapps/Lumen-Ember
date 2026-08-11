"""Structural facts about grok-build-fork — the grounding for stage 2.

Emits crate names, first-party dependency edges, sizes, and module layout.
Deliberately NO source code: stage 2 produces architecture analysis, and the
fork is used for reasoning about structure, not as a code corpus.

Exclusions are taken from the repo's own notices rather than guessed:

  third_party/                              vendored upstream (mermaid-to-svg, …)
  xai-grok-tools/src/implementations/codex/     ported from openai/codex
  xai-grok-tools/src/implementations/opencode/  ported from sst/opencode
  xai-ratatui-inline/                       forked from ratatui's Terminal

Note the trap: `xai-ratatui-textarea` sounds like the same case but its NOTICE
states it is "first-party code maintained by xAI ... not a wholesale
redistribution of third-party source". Excluding by name pattern would drop
first-party work and keep ported work. The notices are the authority.

Ported directories are excluded from LOC counts too, so every number stage 2
cites describes first-party code only.

Run:  python -m pipeline.grok_structure
Out:  data/raw/grok_structure.json
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from pipeline.config import GROK_REPO, RAW

OUT_PATH = RAW / "grok_structure.json"

# Whole crates that are ports/forks of upstream projects.
EXCLUDED_CRATES = {"xai-ratatui-inline"}

# Directories inside otherwise-first-party crates that hold ported code.
EXCLUDED_SUBPATHS = (
    "src/implementations/codex/",
    "src/implementations/opencode/",
)


def _is_ported(rel_posix: str) -> bool:
    return any(seg in rel_posix for seg in EXCLUDED_SUBPATHS)


def _is_test_file(rel_posix: str) -> bool:
    """Rust keeps tests beside implementation, often in huge `tests.rs` modules.

    Counting them as implementation would badly overstate crate sizes — in this
    repo the single largest files under `src/` include 273 KB and 212 KB test
    modules. Stage 2 cites implementation size, so the split has to be real.
    """
    name = rel_posix.rsplit("/", 1)[-1]
    return (
        name == "tests.rs"
        or name.endswith("_test.rs")
        or name.endswith("_tests.rs")
        or "/tests/" in rel_posix
    )


def _crate_stats(crate_dir: Path) -> tuple[int, int, int, list[str]]:
    """(impl_loc, test_loc, n_impl_files, module_dirs) — first-party Rust only."""
    src = crate_dir / "src"
    impl_loc = test_loc = files = 0
    mods: set[str] = set()
    if not src.exists():
        return 0, 0, 0, []
    for rs in src.rglob("*.rs"):
        rel = rs.relative_to(crate_dir).as_posix()
        if _is_ported(rel):
            continue
        n = len(rs.read_text(encoding="utf-8", errors="replace").splitlines())
        if _is_test_file(rel):
            test_loc += n
            continue
        files += 1
        impl_loc += n
        parent = rs.parent.relative_to(src).as_posix()
        if parent and parent != ".":
            mods.add(parent.split("/")[0])
    return impl_loc, test_loc, files, sorted(mods)


def scan() -> dict:
    crates: dict[str, dict] = {}

    for cargo in sorted((GROK_REPO / "crates").glob("*/*/Cargo.toml")):
        try:
            data = tomllib.loads(cargo.read_text(encoding="utf-8", errors="replace"))
        except tomllib.TOMLDecodeError:
            continue
        pkg = data.get("package") or {}
        name = pkg.get("name")
        if not name or name in EXCLUDED_CRATES:
            continue

        crate_dir = cargo.parent
        deps = sorted(
            d for d in (data.get("dependencies") or {})
            if d.startswith("xai-") and d not in EXCLUDED_CRATES and d != name
        )
        loc, test_loc, nfiles, mods = _crate_stats(crate_dir)
        crates[name] = {
            "name": name,
            "path": crate_dir.relative_to(GROK_REPO).as_posix(),
            "group": crate_dir.parent.name,
            "description": (pkg.get("description") or "").strip(),
            "deps": deps,
            "loc": loc,
            "test_loc": test_loc,
            "files": nfiles,
            "modules": mods,
            "has_ported_code": name == "xai-grok-tools",
        }

    # Reverse edges + layering depth (longest path to a leaf, cycles clipped).
    for c in crates.values():
        c["dependents"] = []
    for name, c in crates.items():
        for d in c["deps"]:
            if d in crates:
                crates[d]["dependents"].append(name)
    for c in crates.values():
        c["dependents"].sort()

    depth_cache: dict[str, int] = {}

    def depth(name: str, seen: frozenset[str] = frozenset()) -> int:
        if name in depth_cache:
            return depth_cache[name]
        if name in seen or name not in crates:
            return 0
        d = 0
        for dep in crates[name]["deps"]:
            if dep in crates:
                d = max(d, 1 + depth(dep, seen | {name}))
        depth_cache[name] = d
        return d

    for name, c in crates.items():
        c["depth"] = depth(name)

    return {
        "repo": "grok-build-fork",
        "crates": crates,
        "totals": {
            "n_crates": len(crates),
            "loc": sum(c["loc"] for c in crates.values()),
            "test_loc": sum(c["test_loc"] for c in crates.values()),
            "files": sum(c["files"] for c in crates.values()),
            "max_depth": max((c["depth"] for c in crates.values()), default=0),
        },
        "exclusions": {
            "crates": sorted(EXCLUDED_CRATES),
            "subpaths": list(EXCLUDED_SUBPATHS),
            "trees": ["third_party/"],
        },
    }


def load(*, force: bool = False) -> dict:
    if OUT_PATH.exists() and not force:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    payload = scan()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _main() -> int:
    p = load(force="--force" in sys.argv)
    t = p["totals"]
    print(f"Wrote {OUT_PATH}")
    print(f"  first-party crates: {t['n_crates']}")
    print(f"  implementation LOC: {t['loc']:,} across {t['files']:,} .rs files")
    print(f"  test LOC (split out): {t['test_loc']:,} "
          f"({t['test_loc'] / max(1, t['loc'] + t['test_loc']) * 100:.0f}% of Rust under src/)")
    print(f"  max dependency depth: {t['max_depth']}")
    cs = p["crates"]
    print("\n  most depended-on:")
    for n, c in sorted(cs.items(), key=lambda kv: -len(kv[1]["dependents"]))[:10]:
        print(f"    {len(c['dependents']):>3} dependents  {n}  ({c['loc']:,} loc)")
    print("\n  largest:")
    for n, c in sorted(cs.items(), key=lambda kv: -kv[1]["loc"])[:10]:
        print(f"    {c['loc']:>7,} loc  {n}  (depth {c['depth']}, "
              f"{len(c['deps'])} first-party deps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
