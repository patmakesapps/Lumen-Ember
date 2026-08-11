"""Muse Glimmer chat-template facts, verified against the released artifacts.

Verified 2026-08-11 against meta-models/Muse-Glimmer-30B:
  - config.json         -> model_type "muse_glimmer",
                           architectures ["MuseGlimmerForConditionalGeneration"]
  - tokenizer_config.json -> vocab 202,048; bos <|begin_of_text|>;
                           eos <|end_of_text|>; pad <|finetune_right_pad|>
  - chat_template.jinja -> the "Onyx ATEM" protocol, reproduced below

Do not hand-render ATEM anywhere in this pipeline. We store OpenAI-style
structured messages and let `tokenizer.apply_chat_template` do the rendering,
so a template revision by Meta is a re-render, not a dataset rebuild.

Template shape (for reviewers — this is what the Jinja emits):

    <|start|>system<|message|>{system}
    <NL><NL>Reasoning strength: {high|medium|low}.
    <NL><NL>{tool definitions, JSONSchema, one per line}
    <NL><NL># Valid recipients: "self", "<ns>.*", ..., "user".<|eot|>
    <|start|>user<|message|>{text}<|eot|>
    <|start|>assistant to=self<|message|>{reasoning_content}<|eom|>
    <|start|>assistant to={tool_name}<|message|><atem:function_calls>
      <atem:invoke name="{tool_name}">
        <atem:parameter name="{k}">{v}</atem:parameter>
      </atem:invoke>
    </atem:function_calls><|eom|  or  |eot|>
    <|start|>tool {name}<|message|><tool_output name="{name}">
    {result}
    </tool_output><|eot|>
    <|start|>assistant to=user<|message|>{final answer}<|eot|>

Consequences the pipeline must respect:

  1. tool_call.function.arguments MUST be a dict. The template explicitly
     calls raise_exception() otherwise. External corpora (Glaive, Hermes,
     ToolACE) store it as a JSON *string* — stage 5 must coerce.
  2. ONE tool call per assistant turn. Meta's prompting guide is explicit:
     "Muse Glimmer supports one tool call per turn. It does not support
     parallel tool calls; return each tool result before asking the model to
     select the next tool." The template does NOT enforce this — it renders
     several calls as separate `<|start|>assistant to=...` blocks joined by
     <|eom|> — so reading the template alone gives the opposite impression.
     Enforced in `pipeline.sample.validate`. Matters for stage 3 (LumaKit's
     loop iterates `for tool_call in tool_calls`) and stage 5 (Hermes and
     ToolACE both ship parallel-call rows).
  3. Assistant reasoning goes in `reasoning_content`, rendered to
     `to=self` and terminated with <|eom|>. It is a first-class channel,
     not a prose convention.
  4. Tool namespacing is derived from `name.split('.')[0]`. LumaKit tool
     names are flat, so each tool becomes its own namespace. See
     NAMESPACE_NOTE below.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pipeline.config import GLIMMER_REF, BASE_MODEL_30B

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
PAD_TOKEN = "<|finetune_right_pad|>"

TURN_START = "<|start|>"
TURN_MESSAGE = "<|message|>"
END_OF_MESSAGE = "<|eom|>"   # more from the same role follows
END_OF_TURN = "<|eot|>"      # turn boundary

MODEL_TYPE = "muse_glimmer"
ARCHITECTURE = "MuseGlimmerForConditionalGeneration"
VOCAB_SIZE = 202_048
MAX_POSITION_EMBEDDINGS = 131_072

# The template's own knowledge-cutoff default; pinned so renders are
# reproducible rather than dependent on strftime_now() at train time.
KNOWLEDGE_CUTOFF = "2026-01-04"
CURRENT_DATE = "2026-08-11"

# Reasoning strength is rendered into the system turn by the template.
# Values per Meta's prompting guide; the template defaults to "high".
REASONING_STRENGTHS = ("low", "medium", "high", "xhigh")
DEFAULT_REASONING_STRENGTH = "high"

# Hard model constraint, not a template constraint. See note 2 above.
MAX_TOOL_CALLS_PER_TURN = 1

NAMESPACE_NOTE = """\
LumaKit tool names are flat (`read_file`, `execute_shell`), so Glimmer's
namespace derivation makes each tool its own namespace and the system header
lists `"read_file.*", "execute_shell.*", ...`. We keep flat names on purpose:
the trained model must emit exactly the names LumaKit's ToolRegistry can
dispatch. Namespacing by registry `group` (repo.read_file) would read better
in the header but would require a translation shim at inference time.\
"""


class TemplateContractError(RuntimeError):
    """Raised when a sample would crash Glimmer's chat template."""


def assert_renderable(messages: list[dict]) -> None:
    """Cheap pre-flight for the template's hard failure mode.

    `tokenizer.apply_chat_template` raises deep inside Jinja with a message
    that does not name the offending row; this surfaces it with context.
    """
    for i, m in enumerate(messages):
        for j, tc in enumerate(m.get("tool_calls") or []):
            args = ((tc or {}).get("function") or {}).get("arguments")
            if not isinstance(args, dict):
                raise TemplateContractError(
                    f"messages[{i}].tool_calls[{j}].function.arguments is "
                    f"{type(args).__name__}, must be dict — the ATEM template "
                    f"raises on non-mapping arguments."
                )


def coerce_arguments(tool_call: dict) -> dict:
    """Normalize a tool_call so `arguments` is a dict.

    External corpora store arguments as a JSON string. This is the single
    conversion point; stage 5 routes every external row through it.
    """
    tc = dict(tool_call)
    fn = dict(tc.get("function") or {})
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError as e:
            raise TemplateContractError(
                f"tool_call arguments is an unparseable JSON string: {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise TemplateContractError(
                f"tool_call arguments parsed to {type(parsed).__name__}, need dict"
            )
        fn["arguments"] = parsed
    elif args is None:
        fn["arguments"] = {}
    elif not isinstance(args, dict):
        raise TemplateContractError(
            f"tool_call arguments is {type(args).__name__}, need dict"
        )
    tc["function"] = fn
    return tc


@lru_cache(maxsize=1)
def local_template() -> str:
    """The chat template as downloaded from the model repo."""
    path = GLIMMER_REF / "chat_template.jinja"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run: python -m pipeline.fetch_model_ref"
        )
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def local_config() -> dict:
    return json.loads((GLIMMER_REF / "config.json").read_text(encoding="utf-8"))


def verify_reference_files() -> dict:
    """Confirm the downloaded reference matches what this module assumes."""
    cfg = local_config()
    found_type = cfg.get("model_type")
    found_arch = (cfg.get("architectures") or [None])[0]
    problems = []
    if found_type != MODEL_TYPE:
        problems.append(f"model_type {found_type!r} != {MODEL_TYPE!r}")
    if found_arch != ARCHITECTURE:
        problems.append(f"architecture {found_arch!r} != {ARCHITECTURE!r}")
    tmpl = local_template()
    if "atem:function_calls" not in tmpl:
        problems.append("chat template no longer contains ATEM blocks")
    return {
        "base_model": BASE_MODEL_30B,
        "model_type": found_type,
        "architecture": found_arch,
        "template_bytes": len(tmpl),
        "problems": problems,
    }
