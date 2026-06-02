#!/usr/bin/env python3
"""
CLI entry point for the memory-persistent conversational agent.

Usage:
    python main.py                          # new session
    python main.py --session <id>           # resume a session (same memories)
    python main.py --list-memories          # show all stored memories
    python main.py --forget <memory-id>     # deactivate a memory by id/prefix
    python main.py --db path/to/mem.db      # custom database path
"""

import argparse
import os
import sys
import uuid

import anthropic

from agent import Agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Conversational agent with persistent memory across sessions"
    )
    p.add_argument("--session", default=None, help="Session ID (default: new UUID)")
    p.add_argument("--db", default="memories.db", help="Memory database path")
    p.add_argument("--model", default="claude-sonnet-4-6", help="Claude model")
    p.add_argument(
        "--list-memories", action="store_true", help="List memories and exit"
    )
    p.add_argument(
        "--forget", metavar="ID", help="Deactivate memory by id/prefix and exit"
    )
    return p.parse_args()


def print_memories(agent: Agent) -> None:
    memories = agent.get_memories()
    if not memories:
        print("No memories stored.")
        return
    print(f"\n{len(memories)} stored memories:\n")
    for m in memories:
        star = "*" * min(int(m.importance), 10)
        print(f"  [{m.id[:8]}] [{m.category:10s}] {star:<10} {m.content}")
    print()


def repl(agent: Agent) -> None:
    mem_count = agent.store.count_active()
    print(f"Session : {agent.session_id}")
    print(f"Memories: {mem_count} stored  |  db: {agent.store.db_path}")
    print("Commands: 'memories' — list  |  'forget <id>' — remove  |  'quit' — exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        low = user_input.lower()

        if low in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if low == "memories":
            print_memories(agent)
            continue

        if low.startswith("forget "):
            mem_id = user_input.split(None, 1)[1].strip()
            if agent.forget(mem_id):
                print(f"[forgotten: {mem_id}]\n")
            else:
                print(f"[not found: {mem_id}]\n")
            continue

        print("Assistant: ", end="", flush=True)
        try:
            for chunk in agent.chat(user_input):
                print(chunk, end="", flush=True)
            print("\n")
        except anthropic.APIError as exc:
            print(f"\n[API error: {exc}]\n")
        except KeyboardInterrupt:
            print("\n[interrupted]\n")


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    session_id = args.session or str(uuid.uuid4())
    agent = Agent(
        session_id=session_id,
        db_path=args.db,
        model=args.model,
        api_key=api_key,
    )

    if args.list_memories:
        print_memories(agent)
        return

    if args.forget:
        if agent.forget(args.forget):
            print(f"Forgotten: {args.forget}")
        else:
            print(f"Not found: {args.forget}", file=sys.stderr)
            sys.exit(1)
        return

    repl(agent)


if __name__ == "__main__":
    main()
