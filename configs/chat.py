"""Interactive chat with a trained adapter. Eyeball what the eval can't.

    python configs/chat.py --adapter patmakesapps/Lumen-Ember-30B-adapter-v2
    python configs/chat.py --adapter <...> --tools repo      # offer repo tools
    python configs/chat.py --adapter <...> --author          # framework-design mode

Two modes, because the model was trained on two different jobs:

  agent   LumaKit's real system prompt + a tool block. Tool calls are parsed
          out of ATEM and shown as structured calls; nothing is executed.
  author  the framework-architecture prompt, no tools. This is the mode for
          checking whether the design teaching landed — the thing the 51-probe
          eval does not measure.

Commands: /tools <group>, /reset, /system, /quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUTHOR_SYSTEM = (
    "You are an expert on agent-framework architecture. You help engineers "
    "design and implement local agent harnesses: tool registries, approval "
    "and safety boundaries, agent loops, and provider abstractions. You "
    "answer with working code in the conventions of the codebase at hand, "
    "and you explain the reasoning behind structural choices."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tools", default="repo",
                    help="registry group to offer: repo|runtime|lumabot|code_intel|all|none")
    ap.add_argument("--author", action="store_true",
                    help="framework-design mode: author prompt, no tools")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()

    from pipeline import atem
    from pipeline.local_backend import LocalClient

    tools = []
    system = AUTHOR_SYSTEM
    if not args.author:
        from pipeline.lumakit_probe import scoped_system_prompt
        from pipeline.registry_introspect import by_name
        reg = by_name()
        if args.tools == "none":
            names = []
        elif args.tools == "all":
            names = sorted(reg)
        else:
            names = sorted(n for n, t in reg.items() if t["group"] == args.tools)
        tools = [{
            "type": "function",
            "function": {"name": reg[n]["name"], "description": reg[n]["description"],
                         "parameters": reg[n]["inputSchema"] or {
                             "type": "object", "properties": {}}},
        } for n in names]
        system = scoped_system_prompt(names) if names else AUTHOR_SYSTEM
        print(f"mode: agent | tools offered: {len(tools)} ({args.tools})")
    else:
        print("mode: author | no tools")

    client = LocalClient(model=args.adapter, max_new_tokens=args.max_new_tokens)
    history = [{"role": "system", "content": system}]
    print("Type /quit to exit, /reset to clear history.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user in ("/quit", "/exit"):
            return 0
        if user == "/reset":
            history = [{"role": "system", "content": system}]
            print("(history cleared)\n")
            continue
        if user == "/system":
            print(f"\n{history[0]['content'][:2000]}\n")
            continue

        history.append({"role": "user", "content": user})
        msg = client.chat(history, tools=tools or None)

        if msg.get("content"):
            print(f"\n{msg['content']}\n")
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            print(f"\n  [TOOL CALL] {fn['name']}("
                  f"{json.dumps(fn['arguments'], ensure_ascii=False)[:400]})\n")
        history.append({k: v for k, v in msg.items()
                        if k in ("role", "content", "tool_calls")})

        # A tool call needs a result before the next turn, or the template
        # renders a call with no response. Nothing is executed here.
        for tc in msg.get("tool_calls") or []:
            history.append({
                "role": "tool", "tool_call_id": tc.get("id", "c1"),
                "name": tc["function"]["name"],
                "content": json.dumps({"success": True, "data": {
                    "note": "stub result — chat.py does not execute tools"}}),
            })


if __name__ == "__main__":
    raise SystemExit(main())
