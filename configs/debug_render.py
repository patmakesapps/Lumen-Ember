"""Print what the pod's processor actually renders, and where the markers are.

Cheap: loads the tokenizer/processor only, no model weights, no GPU.

    python configs/debug_render.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE = "meta-models/Muse-Glimmer-30B"


def main() -> int:
    from transformers import AutoTokenizer
    try:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
        print(f"loaded AutoProcessor: {type(tok).__name__}")
    except Exception as e:
        print(f"AutoProcessor failed ({e}); falling back to AutoTokenizer")
        tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        print(f"loaded AutoTokenizer: {type(tok).__name__}")

    rows = []
    with open("data/final/train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    with_calls = [r for r in rows
                  if any(m.get("tool_calls") for m in r["messages"])]
    without = [r for r in rows
               if not any(m.get("tool_calls") for m in r["messages"])]
    print(f"rows: {len(rows)}  with tool_calls: {len(with_calls)}  "
          f"without: {len(without)}")

    for label, sample in (("WITH tool_calls", with_calls[0]),
                          ("WITHOUT tool_calls", without[0])):
        text = tok.apply_chat_template(
            sample["messages"], tools=sample.get("tools") or None,
            tokenize=False, add_generation_prompt=False)
        print("\n" + "=" * 70)
        print(f"{label}  (extractor={sample['meta'].get('extractor')})")
        print("=" * 70)
        print(f"length: {len(text):,} chars")
        starts = [m.start() for m in re.finditer(re.escape("<|start|>"), text)]
        print(f"'<|start|>' occurrences: {len(starts)}")
        roles = [text[s:s + 40].replace("\n", " ") for s in starts]
        for r in roles:
            print(f"   {r!r}")
        print(f"contains '<|start|>assistant': "
              f"{'<|start|>assistant' in text}")
        print(f"contains 'atem:invoke': {'atem:invoke' in text}")
        print("\n--- tail 400 chars ---")
        print(text[-400:])

    # And what the text tokenizer does to a chunk.
    text_tok = getattr(tok, "tokenizer", tok)
    print("\n" + "=" * 70)
    print(f"text tokenizer: {type(text_tok).__name__}")
    probe = "<|start|>assistant to=read_file<|message|>hello"
    try:
        ids = text_tok(probe, add_special_tokens=False).input_ids
        print(f"tokenize({probe!r}) -> {len(ids)} ids: {ids[:12]}")
    except Exception as e:
        print(f"tokenize FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
