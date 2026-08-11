"""Shared paths, seeds, and model identity for every pipeline stage.

Determinism contract: every stage that samples, shuffles, or splits pulls its
RNG from `rng(name)` here. Same inputs + same seed => byte-identical outputs.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

# The base model we fine-tune from.
BASE_MODEL_30B = "meta-models/Muse-Glimmer-30B"

# The trained artifact. Used for Axolotl/Unsloth output dirs, eval report
# headers, and MIXTURE.md.
TRAINED_MODEL_NAME = "Lumen-Ember-30B"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES = PROJECT_ROOT / "sources"
LUMAKIT_REPO = SOURCES / "LumaKit"
GROK_REPO = SOURCES / "grok-build-fork"
LUMABOT_REPO = SOURCES / "LumaBot"
GLIMMER_REF = SOURCES / "_glimmer_ref"

DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
STAGED = DATA / "staged"
FINAL = DATA / "final"

CONFIGS = PROJECT_ROOT / "configs"

for _d in (RAW, STAGED, FINAL, CONFIGS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

GLOBAL_SEED = 20260811


def rng(name: str) -> random.Random:
    """A stable, independent RNG per logical stream.

    Deriving from a name (not a shared global) means adding a new stage never
    shifts the draws of an existing one.
    """
    return random.Random(f"{GLOBAL_SEED}:{name}")


# --------------------------------------------------------------------------
# Teacher endpoint (stages 3 and 4)
# --------------------------------------------------------------------------

TEACHER_BASE_URL = os.environ.get("TEACHER_BASE_URL", "http://localhost:11434/v1")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "muse-glimmer:30b")

# --------------------------------------------------------------------------
# Sources that must never be read by any extractor.
# --------------------------------------------------------------------------

# Secret-bearing or secret-shaped files. `.env.example` is excluded even though
# it ships placeholder values — the placeholders are shaped like real keys and
# teaching the model to emit them is pure downside.
GLOBAL_PATH_DENYLIST = (
    ".env",
    ".env.example",
    ".env.local",
    "config.env",
    "secrets.json",
    "credentials.json",
    "token.json",
    "id_rsa",
    ".pem",
    ".p12",
    ".pfx",
    ".keystore",
)

# grok-build-fork: third-party code is out of scope for extraction entirely.
GROK_EXCLUDED_PATHS = (
    "third_party/",
    "THIRD-PARTY-NOTICES",
    "THIRD_PARTY_NOTICES.md",
    "NOTICE",
    "Cargo.lock",
)


def isolated_lumakit_env() -> dict:
    """Env for LumaKit subprocesses that cannot touch the real ~/.lumakit.

    `core.paths.get_data_dir()` hardcodes `Path.home() / ".lumakit"` with no
    override, and that directory is shared with the developer's *installed*
    LumaKit. It holds `app_runtime_config.json` (tool switch, safe mode,
    approvals), the memory DB, chat store, code index, and a web session
    token. Reading it makes extraction depend on local machine state; writing
    it mutates a live application — `set_tools_enabled()` persists to disk, so
    an innocent-looking probe silently flips a setting in the real install.

    Redirecting HOME/USERPROFILE at a throwaway directory gives LumaKit a
    pristine data dir, so it falls back to DEFAULT_CONFIG (tools_enabled=True,
    safe_mode=True) — deterministic, and inert with respect to the user's
    machine. Stage 3 uses the same isolation for its sandboxed episodes.
    """
    env = dict(os.environ)
    home = tempfile.mkdtemp(prefix="lumen-ember-home-")
    env["HOME"] = home
    env["USERPROFILE"] = home
    # Windows expanduser falls back to HOMEDRIVE+HOMEPATH if USERPROFILE is
    # absent; drop them so there is no path back to the real profile.
    env.pop("HOMEDRIVE", None)
    env.pop("HOMEPATH", None)
    # Never let the developer's identity leak into an extracted system prompt.
    env.pop("LUMI_EMAIL_ADDRESS", None)
    return env


def is_denied_path(path: str | Path) -> bool:
    """True if this path must not be read by any extractor."""
    p = str(path).replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    for bad in GLOBAL_PATH_DENYLIST:
        if name == bad or name.endswith(bad):
            return True
    return False
