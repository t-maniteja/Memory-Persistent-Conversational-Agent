#!/usr/bin/env python3
"""
Automated demo: shows memory persisting and being recalled across two sessions.

Session 1 — user establishes facts, preferences, and a decision.
Session 2 — fresh agent instance, same DB, different session ID. 
            The agent should reflect what it learned in session 1 without being
            told again.

Run:
    export ANTHROPIC_API_KEY="..."
    python demo.py
"""

import os
import sys
import time
import uuid
from pathlib import Path

from agent import Agent

DEMO_DB = "demo_memories.db"

SESSION_1_MESSAGES = [
    """
I'm building a multiplayer AI-native IDE called Neander.

The goal is to let business stakeholders and engineers collaborate in the same workspace.
Instead of requirements moving through tickets and handoffs, we want business intent to flow directly into implementation.

I'm especially interested in agent workflows that can translate product requirements into technical plans.
""",

    """
One of my engineering principles is to optimize for iteration speed early.

I usually avoid infrastructure complexity until there's evidence it's needed.
I'd rather start with a simple system that everyone understands than prematurely optimize for scale.
""",

    """
We recently debated Postgres versus SQLite for local workspace state.

We chose SQLite because deployment simplicity and local-first reliability mattered more than horizontal scalability.
If constraints change later, we'll migrate.
""",

    """
I strongly prefer concise code and explicit data flow.

When reviewing code, I usually reject abstractions that hide control flow or make debugging harder.
""",

    """
A future feature we're considering is collaborative architecture planning.

Multiple users and AI agents should be able to propose designs, critique tradeoffs, and converge on implementation plans together.
"""
]

SESSION_2_MESSAGES = [
    """
How would you design error handling for a new feature in my product?
""",

    """
What's a reasonable persistence strategy for workspace state?
""",

    """
We're discussing how AI agents should participate in architecture reviews.
Any thoughts?
""",

    """
One of our engineers wants to introduce several layers of abstraction to reduce duplication.

How would you evaluate that proposal?
"""
]
SEPARATOR = "─" * 60


def stream_and_collect(agent: Agent, message: str) -> str:
    print(f"  User      : {message}")
    print("  Assistant : ", end="", flush=True)
    chunks = []
    for chunk in agent.chat(message):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print("\n")
    return "".join(chunks)


def wait_for_extraction(agent: Agent, timeout: float = 8.0) -> None:
    """Poll until extraction thread finishes (indicated by lock being free)."""
    deadline = time.time() + timeout
    time.sleep(1.0)  # Give the thread a moment to start
    while time.time() < deadline:
        if agent._extraction_lock.acquire(blocking=False):
            agent._extraction_lock.release()
            break
        time.sleep(0.3)


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Clean slate
    if Path(DEMO_DB).exists():
        Path(DEMO_DB).unlink()

    session_1_id = str(uuid.uuid4())
    session_2_id = str(uuid.uuid4())

    # ----------------------------------------------------------------
    # Session 1
    # ----------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print(f"SESSION 1  [{session_1_id[:8]}...]")
    print(f"{SEPARATOR}\n")

    agent1 = Agent(session_id=session_1_id, db_path=DEMO_DB, api_key=api_key)

    for msg in SESSION_1_MESSAGES:
        stream_and_collect(agent1, msg)

    print("  [waiting for memory extraction to complete...]\n")
    wait_for_extraction(agent1)

    memories = agent1.store.get_all_active()
    print(f"  Memories stored: {len(memories)}")
    for m in memories:
        print(f"    [{m.category:10s}] (importance={m.importance:.0f}) {m.content}")

    # ----------------------------------------------------------------
    # Session 2 — fresh agent, same DB
    # ----------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print(
        f"SESSION 2  [{session_2_id[:8]}...]  (fresh agent instance, same memory store)"
    )
    print(f"{SEPARATOR}\n")

    agent2 = Agent(session_id=session_2_id, db_path=DEMO_DB, api_key=api_key)
    print(
        f"  Memory store has {agent2.store.count_active()} memories at session start.\n"
    )

    for msg in SESSION_2_MESSAGES:
        stream_and_collect(agent2, msg)

    print(f"\n{SEPARATOR}")
    print("WHAT TO OBSERVE")
    print(SEPARATOR)

    print("  • Agent remembers Neander and its collaborative IDE vision.")
    print("  • It recalls the SQLite decision and the reasoning behind it.")
    print("  • It adapts recommendations to a preference for simple systems.")
    print("  • It remembers a dislike of unnecessary abstractions.")
    print("  • It builds on previously discussed AI-agent collaboration ideas.")
    print("  • It uses context from a previous session without conversation history.")

    print(f"\n  Demo DB: {DEMO_DB}  (delete to reset)")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
