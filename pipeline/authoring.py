"""Shared helpers for the long-form authoring stages (1F design, stage 2).

Kept separate so `extract_analysis` doesn't have to import `extract_static`.
"""

from __future__ import annotations

import re
import textwrap

# For framework-authoring samples the model is not acting as Lumi; it is
# helping someone build a harness. Different job, different prompt.
AUTHOR_SYSTEM = (
    "You are an expert on agent-framework architecture. You help engineers "
    "design and implement local agent harnesses: tool registries, approval "
    "and safety boundaries, agent loops, and provider abstractions. You "
    "answer with working code in the conventions of the codebase at hand, "
    "and you explain the reasoning behind structural choices."
)


def reflow(text: str) -> str:
    """Unwrap soft line breaks in prose, leaving code blocks and lists intact.

    Long-form answers are authored as indented triple-quoted strings, so the
    source's hard wraps would otherwise survive into the training text and
    teach the model to emit ragged mid-sentence line breaks.
    """
    out: list[str] = []
    in_code = False
    para: list[str] = []

    def flush():
        if para:
            out.append(" ".join(s.strip() for s in para))
            para.clear()

    for line in textwrap.dedent(text).strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if not stripped:
            flush()
            out.append("")
            continue
        # Structural lines keep their own break.
        if re.match(r"^\s*(?:[-*+]\s|\d+\.\s|#{1,6}\s|>|\|)", line):
            flush()
            out.append(line)
            continue
        para.append(line)
    flush()

    result = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", result).strip()
