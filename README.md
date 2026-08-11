# Lumen-Ember

![Lumen-Ember](lumen-ember-hero.png)

Dataset pipeline + training config for QLoRA-tuning **Meta Muse Glimmer 30B**
into **Lumen-Ember-30B** — a LumaKit-native local agent model that emits
well-formed LumaKit tool calls, respects approval boundaries, and survives
multi-turn agent loops.

This repo builds the **dataset and configs**. It does not run training.

> **Secrets:** `.gitignore` excludes `huggingface.txt`, `.env*`, `*.token`,
> `*.pem` and similar before anything is staged. `pipeline.doctor` verifies the
> secrets gate on every run, and its own test vectors are assembled at runtime
> so this repo never contains a literal key pattern. `sources/` and `data/` are
> ignored too — the checkouts are independent repos, and every stage is
> deterministic, so the data regenerates byte-identically from the pipeline
> plus the pinned source SHAs. The dataset belongs on the Hugging Face Hub,
> not in git history.

---

## Status

| Stage | Module | State |
|---|---|---|
| 0 | `pipeline.config` / `provenance` / `secrets_scan` / `sample` / `glimmer` / `registry_introspect` / `lumakit_probe` / `doctor` | ✅ built |
| 1 | `pipeline.extract_static` — 729 samples | ✅ built |
| 2 | `pipeline.extract_analysis` — 49 samples | ✅ built |
| 3 | `pipeline.gen_trajectories` — harness built, smoke-tested | ✅ built |
| 4 | `pipeline.filter` | ✅ built |
| 5 | `pipeline.mix_external` — 964 train / 44 val | ✅ built |
| 6 | `configs/train_unsloth.py` + `pipeline.eval` | ✅ built, trained |

Supporting modules: `pipeline.atem` (parses Glimmer's ATEM tool-call syntax back
into structured calls — hosted APIs do this for you, self-hosting does not),
`pipeline.local_backend` (`--backend local`, scores a locally-loaded
base+adapter), `configs/fetch_data.py` (pulls train/val from the Hub with an
integrity check), `configs/debug_render.py` (prints what the pod's processor
actually renders — tokenizer only, no GPU).

**Baseline measured before training** (51 held-out probes, ~$0.12/run):

| Model | Schema validity | Tool selection | Autonomous refusal | Negative controls |
|---|---:|---:|---:|---:|
| `meta/muse-glimmer-30b` | **100.0%** | 74.3% | **10.0%** | 83.3% |
| `qwen/qwen3.5-35b-a3b` | **100.0%** | 88.6% | **10.0%** | 83.3% |
| `anthropic/claude-sonnet-5` | **100.0%** | 85.7% | **30.0%** | 100.0% |

Tool-call formatting is already solved on every model tested — that objective
was dropped. The measured gap is autonomous boundary discipline, and it is not
specific to Muse Glimmer. Full write-up in [`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## Verified facts about the base model

Checked **2026-08-11** against the released artifacts, not from memory. The
model is one day old; everything below came from the actual repo files, now
cached in `sources/_glimmer_ref/`.

| Field | Value |
|---|---|
| Repo | `meta-models/Muse-Glimmer-30B` |
| License | Apache-2.0 |
| `model_type` | `muse_glimmer` |
| `architectures` | `MuseGlimmerForConditionalGeneration` (multimodal: text + perception encoder) |
| Vocab | 202,048 (200k BPE + 2,048 special) |
| Context | 131,072 |
| BOS / EOS / PAD | `<\|begin_of_text\|>` / `<\|end_of_text\|>` / `<\|finetune_right_pad\|>` |
| Chat template | "Onyx ATEM" — `<\|start\|>role<\|message\|>…<\|eot\|>`, tool calls as `<atem:function_calls>` XML |
| Sizes released | 30B only (+ GGUF, ExecuTorch, and a 3B *drafter*). **No 8B variant exists.** |

**Five consequences that shape this pipeline:**

1. **Tool calls are XML-inline, not JSON.** We still store OpenAI-style
   structured messages and let `apply_chat_template` render ATEM at train
   time — a template revision becomes a re-render, not a dataset rebuild.
   Meta's guide is explicit that manually formatting the prompt without
   `apply_chat_template` degrades output.
2. **`tool_call.function.arguments` must be a `dict`.** The template calls
   `raise_exception()` on a JSON string. Glaive/Hermes/ToolACE all ship
   strings, so stage 5 must coerce. Enforced in `pipeline.sample.validate`.
3. **One tool call per assistant turn — no parallel calls.** Per Meta's
   prompting guide. The template does *not* enforce this (it renders several
   joined by `<|eom|>`), so reading the template alone gives the opposite
   impression. Enforced in `pipeline.sample.validate`; stages 3 and 5 must
   split multi-call turns into sequential ones.
4. **~17,500-token floor per full-registry sample.** The 107-tool block is
   ~15,900 tokens and the tool catalog renders *into the system turn*. At the
   target mixture that makes LumaKit rows 35% of samples but **92% of
   tokens**. Tool-bearing samples therefore use a narrowed block with a
   matching system prompt; ~12% keep the full block. See
   `lumakit_probe.scoped_system_prompt`.
5. **Axolotl does not support `muse_glimmer`.** Unsloth ships a day-0
   fine-tuning guide, so Unsloth is the primary trainer. See `configs/`.

`reasoning_strength` is rendered into the system turn and accepts
`low | medium | high | xhigh` (default `high`); prior reasoning traces go in
`reasoning_content` on assistant messages.

---

## Setup

```powershell
cd "C:\Lumen Ember"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sources (already cloned into `sources/`):

```powershell
git clone https://github.com/patmakesapps/LumaKit          sources\LumaKit
git clone https://github.com/patmakesapps/grok-build-fork  sources\grok-build-fork
git clone https://github.com/patmakesapps/LumaBot          sources\LumaBot
```

| Source | License | Use |
|---|---|---|
| LumaKit | MIT (verified) | stages 1, 3 — primary |
| grok-build-fork | Apache-2.0 (verified, first-party crates only) | stage 2 |
| LumaBot | MIT (verified) | stage 3 — robot daemon + simulator |

LumaBot originally shipped with no LICENSE file, which defaults to all rights
reserved. MIT was added upstream in `7e11fa0` to match LumaKit's license and
copyright holder, and `pipeline.doctor` now verifies all three.

---

## Running the pipeline

Every stage is independently runnable and idempotent — rerunning a stage with
unchanged inputs produces a byte-identical output file.

### Stage 0 — spine and preflight

```powershell
python -m pipeline.fetch_model_ref          # ~90 KB of template/config only, no weights
python -m pipeline.registry_introspect      # -> data/raw/tool_schemas.json  (107 tools)
python -m pipeline.doctor                   # full preflight; exit 0 = green
```

`doctor` verifies: both checkouts present, licenses still MIT/Apache-2.0,
Glimmer reference files match the assumptions in `pipeline/glimmer.py`,
LumaKit's registry imports and yields 107 tools, the secrets gate catches
known key formats without firing on placeholders, and the sample validator
rejects JSON-string tool arguments.

### Stage 1 — static extraction from LumaKit

```powershell
python -m pipeline.lumakit_probe               # -> data/raw/system_prompts.json
python -m pipeline.extract_static              # all six extractors
python -m pipeline.extract_static --only approval --show 3   # review samples
```

| Extractor | Samples | What it teaches |
|---|---:|---|
| `tool_impl` | 105 | registered tool → its real implementation source |
| `tool_call` | 228 | schema → a valid call, chosen against 10 offered tools |
| `commits` | 165 | diff → message (2:1 over message → diff) |
| `docs_qa` | 92 | README/docs section → grounded behaviour Q&A |
| `approval` | 127 | approval boundaries, 11 shapes, `weight 3.0` |
| `design` | 12 | framework-design principles + anti-patterns, `weight 2.5` |
| **total** | **729** | |

Design notes worth knowing before editing this stage:

- **Prompt and tool block always agree.** LumaKit's system prompt names its
  tool list inline, so a narrowed block with the full 107-name prompt would
  train the model to call tools it was never offered. Both derive from one
  name set. Verified: 0 disagreements, 0 calls to un-offered tools.
- **Hand-authored arguments win over synthesis.** An earlier version inverted
  this and produced a sample where the user asked to delete
  `telegram_speech.py` and the call targeted `approval_policy.py`.
- **Junk-argument samples are dropped** (63 of them) rather than shipped with
  placeholder values like `patch: "example"`.
- **Negative controls are mandatory.** Without samples where read-only tools
  proceed without asking, we train a model that begs permission to read a file.

### Stage 2 — architecture analysis from grok-build-fork

```powershell
python -m pipeline.grok_structure      # -> data/raw/grok_structure.json
python -m pipeline.extract_analysis    # 49 samples
```

No raw Rust. Every sample is analysis prose grounded in *measured* structure —
crate boundaries, first-party dependency edges, graph depth, size.

| Extractor | n | What it teaches |
|---|---:|---|
| `concerns` | 6 | separating workspace / tools / shell / sandbox / config / MCP |
| `crate_role` | 34 | per-crate role, its edges, what it must never depend on |
| `layering` | 6 | dependency-direction reasoning from real edges |
| `compare` | 3 | 73-crate Rust workspace vs single-package Python framework |

**Licensing boundary.** `pipeline.grok_structure` takes exclusions from the
repo's own notices, not from name patterns:

- `third_party/` — vendored upstream
- `xai-grok-tools/src/implementations/codex/` — ported from openai/codex
- `xai-grok-tools/src/implementations/opencode/` — ported from sst/opencode
- `xai-ratatui-inline/` — forked from ratatui's Terminal

The trap: `xai-ratatui-textarea` sounds like the same case, but its NOTICE
states it is *"first-party code maintained by xAI … not a wholesale
redistribution."* A name-pattern rule would drop first-party work and keep
ported work. Ported directories are excluded from line counts too, so every
number cited in a sample describes first-party code only (73 crates,
1,099,162 impl LOC, with 151,711 test LOC split out).

### Stage 3 — synthetic trajectories

```powershell
python -m pipeline.gen_trajectories --smoke --dry-run   # plumbing only, $0
python -m pipeline.gen_trajectories --smoke             # 5 episodes end to end
python -m pipeline.gen_trajectories                     # all 282 seed tasks
```

Runs LumaKit for real in a sandboxed workspace: temp workspace with path
containment, `HOME` redirected away from your real `~/.lumakit`, safe mode and
approvals forced on, and an in-process mock LumaBot daemon with deterministic
fault injection (409 obstacle-safety, 409 motors-not-ready, 503
camera-unavailable, unreachable).

### Stage 4 — filter and judge

```powershell
python -m pipeline.filter --all --no-judge     # free gates only
python -m pipeline.filter --all                # + LLM judge, keep >= 4
```

Gates in cost order: structural → schema (against the real registry) → length →
eval decontamination → near-dedup → judge. Set `JUDGE_MODEL` in `.env` to
something *other* than `TEACHER_MODEL`; the filter warns if you don't.

### Stage 5 — mixture and split

```powershell
python -m pipeline.mix_external                        # first-party only
python -m pipeline.mix_external --external             # + Glaive/Hermes/ToolACE
```

Emits `data/final/{train,val}.jsonl` and `MIXTURE.md`. External corpora are
opt-in and multi-GB; each one's licence is read from the live Hub card at
download time and a mismatch aborts the build.

### Stage 6 — training

See the RunPod runbook below, and `configs/train_unsloth.py`.

---

## RunPod runbook

**Trainer: Unsloth.** Axolotl does not support `muse_glimmer`
(`configs/axolotl_lumen_ember_30b.yml` is written and reviewed but marked
unusable until support lands).

### Pod

| | |
|---|---|
| GPU | 1× H100 80GB (A100 80GB works; expect ~1.5× the time) |
| Template | RunPod PyTorch 2.x / CUDA 12.x |
| Volume | **50 GB minimum**, mounted at `/workspace` |
| Ports | 8888 (Jupyter) — no inference port needed for training |

The container is ephemeral and the volume is not. Base weights, checkpoints and
the adapter all live on the volume, or you lose them when the pod stops.

### Volume layout

```
/workspace
├── Lumen-Ember/            # this repo
│   ├── configs/train_unsloth.py
│   └── data/final/{train,val}.jsonl
├── hf-cache/               # HF_HOME — base weights land here (~60 GB)
└── outputs/Lumen-Ember-30B/  # adapter, ~200-400 MB
```

### Launch

```bash
# 1. environment
curl -fsSL https://unsloth.ai/install.sh | sh
export HF_HOME=/workspace/hf-cache
export HF_TOKEN=<your token>

# 2. code + data  (data/ is gitignored, so copy it up separately)
cd /workspace && git clone https://github.com/patmakesapps/Lumen-Ember
#    from your laptop:  scp data/final/*.jsonl root@<pod>:/workspace/Lumen-Ember/data/final/

# 3. VERIFY BEFORE TRAINING — renders the dataset through Glimmer's own chat
#    template and reports token lengths. Catches a broken template or an
#    oversized corpus in ~2 minutes instead of 40.
cd /workspace/Lumen-Ember
python configs/train_unsloth.py --dry-run

# 4. train
python configs/train_unsloth.py \
    --train data/final/train.jsonl \
    --val   data/final/val.jsonl \
    --out   /workspace/outputs/Lumen-Ember-30B \
    --push  <hf-user>/Lumen-Ember-30B-adapter
```

### Expected

| | Path A (validation run) | Full run |
|---|---|---|
| Rows | 964 train / 44 val | ~2–3k train |
| Tokens (2 epochs) | ~5.6M | ~10–20M |
| VRAM | ~55–70 GB at `seq_len 8192`, packing on | same |
| Wall clock | **30–60 min** | 2–4 h |
| Cost | **~$3–5** | ~$8–12 |

If it OOMs: drop `--max-seq 4096` (loses 14.5% of rows to truncation), then
`gradient_accumulation_steps` 16 → 8.

### Resume

Checkpoints land in `--out` every 100 steps (`save_total_limit=3`). To resume
after a pod restart, point `--out` at the same directory — HF `Trainer` picks
up the latest checkpoint. Push the adapter to the Hub as soon as training ends;
the volume survives a stop but not an account-level accident.

### After training

```powershell
# on your laptop, against the adapter served anywhere OpenAI-compatible
python -m pipeline.eval --label lumen-ember --model <adapter-endpoint>
```

Compare `data/final/eval-lumen-ember.md` to `eval-base.md`. The number that
matters is **autonomous refusal**, measured at 10.0% on the base model across
two identical runs (see `docs/FINDINGS.md`).

---

## Layout

```
pipeline/          stages + shared spine
  config.py            paths, seeds, model identity, path denylist
  provenance.py        {source, license, extraction_method, repo_sha}
  secrets_scan.py      hard-fail secrets gate
  sample.py            canonical sample schema + deterministic JSONL I/O
  glimmer.py           verified chat-template facts + ATEM contract
  registry_introspect.py  loads LumaKit's real ToolRegistry
  doctor.py            stage 0 self-check
data/raw/          extracted-but-unprocessed artifacts
data/staged/       per-stage sample output
data/final/        train.jsonl, val.jsonl, MIXTURE.md
configs/           Unsloth/Axolotl training configs
sources/           read-only checkouts + _glimmer_ref/
```

---

## Safety invariants

- **No secrets, ever.** `pipeline.secrets_scan` runs on the fully serialized
  sample — system prompts, tool arguments, tool results, assistant text. First-party
  stages *hard-fail*; external corpora drop the row. `.env` and `.env.example`
  are on a global path denylist and are stripped from mined diffs (they appear
  in 14 otherwise-minable commits).
- **First-party only from the fork.** `third_party/`, `THIRD-PARTY-NOTICES`,
  and crate-local notice files are excluded from all extraction. The fork is
  used for architecture-analysis prose only — never raw code dumps.
- **Provenance is mandatory.** `sample.validate` refuses any row missing
  `{source, license, extraction_method, repo_sha}`.
- **Determinism.** All shuffles/splits draw from `pipeline.config.rng(name)`,
  seeded per logical stream so adding a stage never perturbs an existing one.
