import json
import re
from dataclasses import dataclass
from typing import Any, cast, get_args

import anthropic
from anthropic.types import TextBlock

from .memory_store import Memory, MemoryCategory

_SENSITIVE_PATTERNS = [
    re.compile(r'(?i)(password|passwd|secret|token|api[_\s-]?key)\s*[:=]\s*\S{6,}'),
    re.compile(r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b'),
    re.compile(r'\b[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b'),
    re.compile(r'(?i)(bearer|sk-|pk-|ghp_|xox[baprs]-)[A-Za-z0-9_\-]{10,}'),
]

_SYSTEM_PROMPT = """You are a memory extraction system for an AI assistant. Analyze conversations and extract what's worth remembering in future sessions.

## What to store

Return memories that help an assistant work better with this specific user long-term:

- **preference**: How the user likes things done — coding style, tools, language, tone, workflow
- **fact**: Who they are, their role, projects, tech stack, constraints, environment
- **decision**: Explicit choices or commitments made ("we decided to use PostgreSQL")
- **context**: Current projects, goals, or broader work situation

## What NOT to store

- Passwords, API keys, tokens, SSN, credit card numbers, or any secrets
- Transient one-off queries (weather, single lookups) with no lasting relevance
- Information the user said is temporary or hypothetical ("for now", "just testing")
- Information already captured in existing_memories (check before adding)
- Generic greetings, pleasantries, or filler with no informational content

## Importance scale (1–10)

- 1–3: Vague, uncertain, or rarely useful
- 4–6: Useful context (general preferences, secondary projects)
- 7–9: High-value (explicit stated preference, key decision, central project)
- 10: Critical — should always shape responses

## Conflict handling

If new information contradicts an existing memory, put the old id in `memories_to_supersede.old_id` with the updated content. If an existing memory is simply wrong/outdated with nothing to replace it, put the id in `memories_to_deactivate`.

## Output format

Return ONLY valid JSON — no markdown fences, no prose:

{
  "memories_to_add": [
    {"content": "...", "category": "preference|fact|decision|context", "importance": 1-10}
  ],
  "memories_to_supersede": [
    {"old_id": "...", "content": "...", "category": "...", "importance": 1-10}
  ],
  "memories_to_deactivate": ["full-uuid-or-prefix"]
}

Return empty lists if nothing warrants storage. Never invent information not present in the conversation."""


@dataclass
class ExtractionResult:
    to_add: list[Memory]
    to_supersede: list[tuple[str, Memory]]
    to_deactivate: list[str]

    @property
    def is_empty(self) -> bool:
        return not (self.to_add or self.to_supersede or self.to_deactivate)


class MemoryExtractor:
    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6") -> None:
        self.client = client
        self.model = model

    def extract(
        self,
        conversation: list[anthropic.types.MessageParam],
        existing_memories: list[Memory],
        session_id: str,
    ) -> ExtractionResult:
        if not conversation:
            return ExtractionResult([], [], [])

        user_content = _build_prompt(conversation, existing_memories)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            block = response.content[0]
            if not isinstance(block, TextBlock):
                return ExtractionResult([], [], [])
            return self._parse(block.text.strip(), session_id, existing_memories)
        except Exception as exc:
            print(f"\n[memory extraction error: {exc}]")
            return ExtractionResult([], [], [])

    def _parse(self, raw: str, session_id: str, existing: list[Memory]) -> ExtractionResult:
        raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE).strip('`').strip()

        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            return ExtractionResult([], [], [])

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return ExtractionResult([], [], [])

        to_add: list[Memory] = []
        to_supersede: list[tuple[str, Memory]] = []
        to_deactivate: list[str] = []

        for item in data.get("memories_to_add", []):
            mem = _parse_item(item, session_id)
            if mem:
                to_add.append(mem)

        for item in data.get("memories_to_supersede", []):
            old_id = _resolve_id(str(item.get("old_id", "")), existing)
            if old_id:
                mem = _parse_item(item, session_id)
                if mem:
                    to_supersede.append((old_id, mem))

        for raw_id in data.get("memories_to_deactivate", []):
            resolved = _resolve_id(str(raw_id), existing)
            if resolved:
                to_deactivate.append(resolved)

        return ExtractionResult(to_add, to_supersede, to_deactivate)


def _build_prompt(
    conversation: list[anthropic.types.MessageParam],
    existing: list[Memory],
) -> str:
    existing_str = "(none)" if not existing else "\n".join(
        f"  [{m.id[:8]}] [{m.category}] (importance={m.importance:.0f}) {m.content}"
        for m in existing
    )
    convo_str = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in conversation
    )
    return (
        f"EXISTING MEMORIES:\n{existing_str}\n\n"
        f"CONVERSATION TO ANALYZE:\n{convo_str}\n\n"
        "Extract what's worth remembering from this conversation."
    )


_VALID_CATEGORIES = get_args(MemoryCategory)


def _parse_item(item: dict[str, Any], session_id: str) -> Memory | None:
    content = str(item.get("content", "")).strip()
    if not content or len(content) < 8:
        return None

    raw_cat = str(item.get("category", "fact"))
    category = cast(MemoryCategory, raw_cat if raw_cat in _VALID_CATEGORIES else "fact")

    raw_imp = item.get("importance", 5)
    importance = float(raw_imp) if isinstance(raw_imp, (int, float)) else 5.0

    if _is_sensitive(content):
        return None

    return Memory(
        content=content,
        category=category,
        importance=max(1.0, min(10.0, importance)),
        session_id=session_id,
    )


def _resolve_id(candidate: str, existing: list[Memory]) -> str | None:
    if not candidate:
        return None
    for m in existing:
        if m.id == candidate or m.id.startswith(candidate):
            return m.id
    return None


def _is_sensitive(text: str) -> bool:
    return any(p.search(text) for p in _SENSITIVE_PATTERNS)
