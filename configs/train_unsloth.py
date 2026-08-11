"""Lumen-Ember-30B — QLoRA training script (Unsloth, 1x H100 80GB).

Run this ON the RunPod pod. See the RunPod runbook in README.md.

    python configs/train_unsloth.py \
        --train data/final/train.jsonl \
        --val   data/final/val.jsonl \
        --out   outputs/Lumen-Ember-30B

Why Unsloth and not Axolotl: Axolotl does not support `muse_glimmer` (its
supported-model list at time of writing tops out at Shieldstral / Gemma 4 /
Qwen3.5). Unsloth shipped a day-0 Muse Glimmer fine-tuning guide. An Axolotl
config is included alongside this file for when support lands, clearly marked
as untested.

Every hyperparameter choice is justified inline. Where a value came from a
measurement in this repo, the measurement is named.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — each value explained.
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-models/Muse-Glimmer-30B"
OUTPUT_NAME = "Lumen-Ember-30B"

CONFIG = dict(
    # -- Quantisation -------------------------------------------------------
    # 4-bit NF4. A 30B in bf16 is ~60 GB of weights alone; quantised it is
    # ~18-20 GB, leaving an 80 GB card room for activations at long sequence
    # length. The base stays frozen, so quantisation error is not compounded
    # by weight updates — only the adapter trains, in higher precision.
    load_in_4bit=True,

    # -- LoRA ---------------------------------------------------------------
    # r=32 / alpha=64 (2:1). Enough capacity for a behavioural change without
    # the overfitting risk of r=64+ on a small corpus. This dataset is ~1k
    # unique rows; capacity is not the binding constraint, data is.
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.0,          # Unsloth's fused path is fastest at exactly 0
    # All attention AND MLP projections. Attention-only adapters move style;
    # MLP is where a lot of behavioural//factual routing lives, and the target
    # behaviour here (refusing to act when unattended) is a disposition, not a
    # surface format. Costs little at r=32.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",       # attention
        "gate_proj", "up_proj", "down_proj",          # MLP
    ],
    use_gradient_checkpointing="unsloth",
    random_state=20260811,

    # -- Sequence ------------------------------------------------------------
    # Measured on data/final/train.jsonl: median 2,350 tok, p90 5,049,
    # max 17,702. 8192 covers 93.8% of rows intact; 4096 would truncate 14.5%,
    # and truncation on an agent transcript usually removes the final report —
    # the exact tokens that teach honest completion.
    #
    # Unsloth's guide suggests starting at 4096. Drop to 4096 if this OOMs;
    # the cost is losing the long multi-round episodes.
    max_seq_length=8192,
    # Packing concatenates short samples to fill the window. With a median of
    # 2,350 tokens against an 8192 window, packing is roughly a 3x throughput
    # win. Requires correct EOS handling so packed samples don't bleed.
    packing=True,

    # -- Optimisation --------------------------------------------------------
    num_train_epochs=2,        # small corpus; 3+ starts memorising
    learning_rate=1e-4,        # standard QLoRA band (1e-4..2e-4) for r=32
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # effective batch 16
    optim="adamw_8bit",        # optimiser states in 8-bit; saves ~2 GB
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    seed=20260811,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    report_to="none",
)

# Loss is computed on assistant tokens only. This matters more here than in a
# typical SFT run: the system preamble is ~17.5k tokens of tool schemas that
# never vary, and training on it would spend most of the gradient budget
# reproducing an invariant block. See docs/FINDINGS.md §1.3.
TRAIN_ON_RESPONSES_ONLY = True


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/final/train.jsonl")
    ap.add_argument("--val", default="data/final/val.jsonl")
    ap.add_argument("--out", default=f"outputs/{OUTPUT_NAME}")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--max-seq", type=int, default=CONFIG["max_seq_length"])
    ap.add_argument("--epochs", type=float, default=CONFIG["num_train_epochs"])
    ap.add_argument("--dry-run", action="store_true",
                    help="render the dataset through the chat template and stop")
    ap.add_argument("--push", default=None, help="HF repo id to push the adapter to")
    args = ap.parse_args()

    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    print(f"Loading {args.base} in 4-bit ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq,
        dtype=None,                      # auto: bf16 on H100
        load_in_4bit=CONFIG["load_in_4bit"],
    )

    # Muse Glimmer is multimodal. We train text only, so the perception encoder
    # stays frozen — per Unsloth's guidance, and because nothing in this
    # dataset is an image.
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
        use_gradient_checkpointing=CONFIG["use_gradient_checkpointing"],
        random_state=CONFIG["random_state"],
    )

    def render(row: dict) -> dict:
        """Render one sample through Glimmer's own chat template.

        Never hand-format ATEM. Meta's prompting guide is explicit that
        manually formatting the prompt without apply_chat_template degrades
        output, and the template is the only thing that knows the current
        tool-catalog and reasoning-strength layout.
        """
        text = tokenizer.apply_chat_template(
            row["messages"],
            tools=row.get("tools") or None,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_rows = load_rows(Path(args.train))
    val_rows = load_rows(Path(args.val))
    print(f"  train {len(train_rows):,}   val {len(val_rows):,}")

    train_ds = Dataset.from_list(train_rows).map(
        render, remove_columns=Dataset.from_list(train_rows).column_names)
    val_ds = Dataset.from_list(val_rows).map(
        render, remove_columns=Dataset.from_list(val_rows).column_names)

    if args.dry_run:
        sample = train_ds[0]["text"]
        print("\n--- rendered sample (first 1500 chars) ---")
        print(sample[:1500])
        lens = [len(tokenizer(t["text"]).input_ids) for t in train_ds.select(
            range(min(200, len(train_ds))))]
        lens.sort()
        print(f"\ntoken length over {len(lens)} sampled rows: "
              f"median={lens[len(lens) // 2]:,} "
              f"p90={lens[int(len(lens) * 0.9)]:,} max={lens[-1]:,}")
        over = sum(1 for n in lens if n > args.max_seq)
        print(f"rows exceeding max_seq_length={args.max_seq}: {over}/{len(lens)}")
        return 0

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=args.out,
            dataset_text_field="text",
            max_seq_length=args.max_seq,
            packing=CONFIG["packing"],
            num_train_epochs=args.epochs,
            learning_rate=CONFIG["learning_rate"],
            lr_scheduler_type=CONFIG["lr_scheduler_type"],
            warmup_ratio=CONFIG["warmup_ratio"],
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            optim=CONFIG["optim"],
            weight_decay=CONFIG["weight_decay"],
            max_grad_norm=CONFIG["max_grad_norm"],
            bf16=CONFIG["bf16"],
            seed=CONFIG["seed"],
            logging_steps=CONFIG["logging_steps"],
            eval_strategy=CONFIG["eval_strategy"],
            eval_steps=CONFIG["eval_steps"],
            save_strategy=CONFIG["save_strategy"],
            save_steps=CONFIG["save_steps"],
            save_total_limit=CONFIG["save_total_limit"],
            report_to=CONFIG["report_to"],
        ),
    )

    if TRAIN_ON_RESPONSES_ONLY:
        # Mask everything before the assistant turn. Glimmer's turn markers.
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start|>user<|message|>",
            response_part="<|start|>assistant",
        )

    stats = trainer.train()
    print(stats)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved -> {args.out}")

    if args.push:
        token = os.environ.get("HF_TOKEN")
        model.push_to_hub(args.push, token=token)
        tokenizer.push_to_hub(args.push, token=token)
        print(f"pushed -> https://huggingface.co/{args.push}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
