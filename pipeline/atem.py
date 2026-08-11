"""Parse Muse Glimmer's ATEM tool-call syntax back into structured calls.

Hosted APIs (OpenRouter) do this for you and hand back OpenAI-style
`tool_calls`. When you serve the model yourself — or generate locally — you
get raw text and have to parse it, or every self-hosted eval silently scores
zero tool calls.

The format is defined by the model's own chat_template.jinja:

    <atem:function_calls>
    <atem:invoke name="read_file">
    <atem:parameter name="path">core/approval_policy.py</atem:parameter>
    <atem:parameter name="recursive">true</atem:parameter>
    <atem:parameter name="opts">{"depth": 2}</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

Serialisation rules the template uses, which this inverts:
  booleans   -> the bare words true / false
  null       -> the bare word null
  dict/list  -> JSON
  everything else -> written as-is, so numbers arrive as bare digits and
  strings arrive unquoted and unescaped

The template itself notes the output "is not expected to be valid XML and is
parsed with regular expressions", so a regex parser is the intended reading,
not a shortcut. Parameter values may contain newlines and quotes, hence the
non-greedy DOTALL match on the closing tag rather than an XML parser.
"""

from __future__ import annotations

import json
import re

_INVOKE_RE = re.compile(
    r"<atem:invoke\s+name=\"(?P<name>[^\"]+)\"\s*>(?P<body>.*?)</atem:invoke>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<atem:parameter\s+name=\"(?P<key>[^\"]+)\"\s*>(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)

# Turn markers, so plain-text answers can be cleaned up.
_SPECIAL_RE = re.compile(
    r"<\|(?:begin_of_text|end_of_text|start|message|eot|eom|finetune_right_pad)\|>"
)
# Must be applied BEFORE special tokens are stripped: `<|message|>` is what
# terminates the recipient. Remove the markers first and `to=user<|message|>I
# won't...` collapses to `to=userI won't...`, where a greedy [^\s<]+ swallows
# the first letter of the actual answer.
# Not anchored to ^: a generation can contain several turns (reasoning to=self,
# then to=user), and anchoring only strips the first. The lookahead keeps it
# safe to apply globally — it only matches a recipient that is immediately
# followed by the message marker.
_RECIPIENT_RE = re.compile(r"\s*to=[^\s<]+(?=<\|message\|>)")


def _coerce(raw: str):
    """Invert the template's value serialisation.

    Numbers deliberately stay strings unless they parse as JSON scalars —
    LumaKit's registry coerces '5' to 5 for integer fields, so leaving them
    as text exercises the same forgiving path a real call would take.
    """
    text = raw.strip()
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "null":
        return None
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw


def parse_calls(text: str) -> list[dict]:
    """Extract tool calls as OpenAI-style dicts. Empty list if none."""
    calls = []
    for i, match in enumerate(_INVOKE_RE.finditer(text or "")):
        args = {
            p.group("key"): _coerce(p.group("value"))
            for p in _PARAM_RE.finditer(match.group("body"))
        }
        calls.append({
            "id": f"call_{i + 1}",
            "type": "function",
            "function": {"name": match.group("name"), "arguments": args},
        })
    return calls


def strip_markup(text: str) -> str:
    """Plain assistant text with ATEM blocks and turn markers removed."""
    without_calls = re.sub(
        r"<atem:function_calls>.*?</atem:function_calls>", "", text or "", flags=re.DOTALL)
    cleaned = _RECIPIENT_RE.sub("", without_calls)   # before stripping markers
    cleaned = _SPECIAL_RE.sub("", cleaned)
    return cleaned.strip()


def to_message(text: str) -> dict:
    """Raw generation -> an assistant message shaped like an API response."""
    calls = parse_calls(text)
    message: dict = {"role": "assistant", "content": strip_markup(text) or None}
    if calls:
        message["tool_calls"] = calls
    return message


if __name__ == "__main__":
    sample = (
        ' to=read_file<|message|><atem:function_calls>\n'
        '<atem:invoke name="read_file">\n'
        '<atem:parameter name="path">core/approval_policy.py</atem:parameter>\n'
        '<atem:parameter name="recursive">true</atem:parameter>\n'
        '<atem:parameter name="limit">20</atem:parameter>\n'
        '<atem:parameter name="opts">{"depth": 2}</atem:parameter>\n'
        '</atem:invoke>\n</atem:function_calls><|eot|>'
    )
    print(json.dumps(to_message(sample), indent=2))
    print()
    print(json.dumps(to_message(
        " to=user<|message|>I won't delete that without approval.<|eot|>"), indent=2))
