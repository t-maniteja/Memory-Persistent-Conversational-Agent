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
    "Hey! Quick context: I'm a senior Go engineer, 8 years in. Currently building a CLI tool "
    "called 'flux' that manages Kubernetes deployments for my startup.",
    "I really hate boilerplate and verbose code. Keep things idiomatic and concise — "
    "no 6-line error checks when two will do.",
    "We just decided to use SQLite for local state in flux rather than a full Postgres instance. "
    "Performance requirements don't justify it at this stage.",
]

SESSION_2_MESSAGES = [
    "Hi there! Can you help me think about how to structure error handling in my project?",
    "What's the best approach for local state persistence given our setup?",
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
    print("  • Agent 2 knows you're a Go engineer without being told.")
    print("  • It keeps error handling idiomatic and concise.")
    print("  • When asked about persistence, it references the SQLite decision.")
    print("  • It mentions 'flux' when contextually relevant.")
    print(f"\n  Demo DB: {DEMO_DB}  (delete to reset)")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
