"""Pull train/val from the private HF dataset onto the pod.

Jupyter's file upload corrupted a 10.9 MB JSONL (arrived at 20.8 MB with
merged lines), so the Hub is the transfer path. Needs HF_TOKEN in the env.

    python configs/fetch_data.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = "patmakesapps/lumen-ember-data"
DEST = Path("data/final")
EXPECTED = {"train.jsonl": 10_914_813, "val.jsonl": 686_007}


def main() -> int:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        target = DEST / name
        if target.exists():
            target.unlink()

    snapshot_download(REPO, repo_type="dataset",
                      local_dir=str(DEST), token=token)

    ok = True
    for name, expected_size in EXPECTED.items():
        path = DEST / name
        if not path.exists():
            print(f"  MISSING {name}")
            ok = False
            continue
        size = path.stat().st_size
        bad = 0
        rows = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
        match = "ok" if size == expected_size else f"MISMATCH (expected {expected_size:,})"
        print(f"  {name}: {size:,} bytes {match} | {rows} rows | {bad} parse errors")
        if size != expected_size or bad:
            ok = False

    print("\nDATA OK" if ok else "\nDATA BAD — do not train")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
