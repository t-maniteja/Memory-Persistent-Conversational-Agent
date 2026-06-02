import threading
from typing import Iterator

import anthropic
from anthropic.types import MessageParam

from .extractor import MemoryExtractor
from .memory_store import Memory, MemoryStore

_BASE_SYSTEM = """\
You are a helpful AI assistant with persistent memory across conversations.

You have access to memories from previous sessions with this user. Use them naturally:
- Don't re-ask for information you already know.
- Match the user's stated preferences and working style.
- Build on past context and decisions.

Be conversational. Don't announce "I remember that..." unless it's genuinely useful to acknowledge continuity.\
"""


class Agent:
    def __init__(
        self,
        session_id: str,
        db_path: str = "memories.db",
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        memory_limit: int = 12,
        min_importance: float = 3.0,
    ) -> None:
        self.session_id: str = session_id
        self.model: str = model
        self.memory_limit: int = memory_limit
        self.min_importance: float = min_importance

        self._client: anthropic.Anthropic = anthropic.Anthropic(api_key=api_key)
        self.store: MemoryStore = MemoryStore(db_path)
        self._extractor: MemoryExtractor = MemoryExtractor(self._client, model)

        self.conversation: list[MessageParam] = []
        self._extraction_lock: threading.Lock = threading.Lock()
        self._extracted_up_to: int = 0

    def chat(self, user_message: str) -> Iterator[str]:
        self.conversation.append({"role": "user", "content": user_message})

        relevant = self.store.search(
            user_message,
            limit=self.memory_limit,
            min_importance=self.min_importance,
        )

        full_chunks: list[str] = []
        with self._client.messages.stream(
            model=self.model,
            max_tokens=2048,
            system=_build_system_prompt(relevant),
            messages=self.conversation,
        ) as stream:
            for chunk in stream.text_stream:
                full_chunks.append(chunk)
                yield chunk

        self.conversation.append({"role": "assistant", "content": "".join(full_chunks)})

        # Only extract turns the extractor hasn't seen yet — prevents re-processing
        # earlier turns and creating near-duplicate memories.
        new_turns = list(self.conversation[self._extracted_up_to:])
        self._extracted_up_to = len(self.conversation)

        threading.Thread(
            target=self._extract_and_store,
            args=(new_turns, list(relevant)),
            daemon=True,
        ).start()

    def get_memories(self) -> list[Memory]:
        return self.store.get_all_active()

    def forget(self, memory_id: str) -> bool:
        resolved = self.store.resolve_id_prefix(memory_id)
        if not resolved:
            return False
        self.store.deactivate(resolved)
        return True

    def reset_conversation(self) -> None:
        self.conversation.clear()
        self._extracted_up_to = 0

    def _extract_and_store(
        self,
        conversation: list[MessageParam],
        existing_memories: list[Memory],
    ) -> None:
        if not self._extraction_lock.acquire(blocking=False):
            return

        try:
            result = self._extractor.extract(conversation, existing_memories, self.session_id)

            for memory in result.to_add:
                self.store.add(memory)
            for old_id, new_memory in result.to_supersede:
                self.store.supersede(old_id, new_memory)
            for memory_id in result.to_deactivate:
                self.store.deactivate(memory_id)

            if not result.is_empty:
                parts = []
                if result.to_add:
                    parts.append(f"+{len(result.to_add)}")
                if result.to_supersede:
                    parts.append(f"~{len(result.to_supersede)}")
                if result.to_deactivate:
                    parts.append(f"-{len(result.to_deactivate)}")
                print(f"\n[memory: {', '.join(parts)} | total: {self.store.count_active()}]", flush=True)
        finally:
            self._extraction_lock.release()


def _build_system_prompt(memories: list[Memory]) -> str:
    if not memories:
        return _BASE_SYSTEM
    lines = "\n".join(f"- {m.to_context_str()}" for m in memories)
    return f"{_BASE_SYSTEM}\n\nREMEMBERED FROM PAST SESSIONS:\n{lines}"
