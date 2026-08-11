"""Run the eval against a locally-loaded model + LoRA adapter.

Exposes the same `.chat()` surface as `pipeline.teacher.TeacherClient`, so
`pipeline.eval` scores a self-hosted adapter with no other changes.

Why this exists: a hosted API returns OpenAI-style `tool_calls` because the
provider parses the model's output for you. Loading the model yourself gives
raw text in Glimmer's ATEM syntax, so `pipeline.atem` parses it back. Without
that step a self-hosted eval scores zero tool calls and looks like a
catastrophic regression that isn't real.

Runs ON the pod (model weights are cached there), not from a laptop.

    python -m pipeline.eval --label lumen-ember \
        --backend local --adapter patmakesapps/Lumen-Ember-30B-adapter
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pipeline import atem
from pipeline.teacher import Usage

DEFAULT_MAX_NEW_TOKENS = 1200   # Glimmer reasons before answering; Meta's guide
                                # warns that clipping this truncates mid-reasoning


@dataclass
class LocalClient:
    model: str                       # adapter repo/path, or base for a baseline run
    base: str | None = None          # override the adapter's recorded base
    max_seq_length: int = 8192
    load_in_4bit: bool = True
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    usage: Usage = field(default_factory=Usage)
    mode: str = "local"

    _model: object = None
    _tok: object = None
    _text_tok: object = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from unsloth import FastLanguageModel

        print(f"Loading {self.model} (4bit={self.load_in_4bit}) ...")
        model, tok = FastLanguageModel.from_pretrained(
            model_name=self.model,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        self._model, self._tok = model, tok
        # Glimmer is multimodal, so this is a MuseGlimmerProcessor, not a
        # tokenizer. Calling a processor with a bare string makes it try to
        # base64-decode the text as image data ("Invalid base64-encoded
        # string"). apply_chat_template lives on the processor; plain
        # tokenizing and decoding must go to the inner text tokenizer.
        self._text_tok = getattr(tok, "tokenizer", tok)
        print(f"  loaded ({type(tok).__name__}"
              f"{' -> ' + type(self._text_tok).__name__ if self._text_tok is not tok else ''})")

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_strength: str | None = None,
    ) -> dict:
        self._ensure_loaded()
        import torch

        text = self._tok.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._text_tok(text, return_tensors="pt").to(self._model.device)
        prompt_len = inputs.input_ids.shape[-1]

        started = time.monotonic()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._text_tok.pad_token_id or self._text_tok.eos_token_id,
            )
        elapsed = time.monotonic() - started

        new_tokens = out[0][prompt_len:]
        completion = self._text_tok.decode(new_tokens, skip_special_tokens=False)

        self.usage.calls += 1
        self.usage.input_tokens += int(prompt_len)
        self.usage.output_tokens += int(new_tokens.shape[-1])
        self.usage.seconds += elapsed

        return atem.to_message(completion)
