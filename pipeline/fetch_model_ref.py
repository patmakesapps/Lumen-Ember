"""Fetch Muse Glimmer's template/config reference files (tiny, no weights).

Idempotent: skips files already present unless --force. Downloads ~90 KB
total — the model weights are never touched by this pipeline.

Run:  python -m pipeline.fetch_model_ref [--force]
"""

from __future__ import annotations

import sys
import urllib.request

from pipeline.config import GLIMMER_REF, BASE_MODEL_30B

FILES = (
    "chat_template.jinja",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)


def fetch(force: bool = False) -> int:
    GLIMMER_REF.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{BASE_MODEL_30B}/raw/main/"
    for name in FILES:
        dest = GLIMMER_REF / name
        if dest.exists() and not force:
            print(f"  skip  {name} ({dest.stat().st_size} bytes, already present)")
            continue
        try:
            with urllib.request.urlopen(base + name, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"  ok    {name} ({len(data)} bytes)")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(fetch(force="--force" in sys.argv))
