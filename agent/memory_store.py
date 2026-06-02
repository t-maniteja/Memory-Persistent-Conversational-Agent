import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, cast

MemoryCategory = Literal["preference", "fact", "decision", "context"]

_STOP_WORDS = {
    "the", "is", "in", "it", "of", "to", "a", "an", "and", "or", "for",
    "on", "at", "by", "my", "me", "we", "us", "be", "do", "go", "so",
    "if", "as", "but", "no", "up", "he", "she", "i", "you",
}

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    category      TEXT NOT NULL CHECK(category IN ('preference','fact','decision','context')),
    importance    REAL NOT NULL CHECK(importance BETWEEN 1 AND 10),
    created_at    REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count  INTEGER DEFAULT 0,
    session_id    TEXT DEFAULT '',
    is_active     INTEGER DEFAULT 1,
    superseded_by TEXT REFERENCES memories(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id       UNINDEXED,
    content,
    category UNINDEXED,
    content='memories',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, id, content, category)
    VALUES (new.rowid, new.id, new.content, new.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, id, content, category)
    VALUES ('delete', old.rowid, old.id, old.content, old.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, id, content, category)
    VALUES ('delete', old.rowid, old.id, old.content, old.category);
    INSERT INTO memories_fts(rowid, id, content, category)
    VALUES (new.rowid, new.id, new.content, new.category);
END;

CREATE INDEX IF NOT EXISTS idx_mem_active_importance
    ON memories(is_active, importance DESC);

CREATE INDEX IF NOT EXISTS idx_mem_category
    ON memories(category, is_active);
"""


@dataclass
class Memory:
    content: str
    category: MemoryCategory
    importance: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    session_id: str = ""
    is_active: bool = True
    superseded_by: str | None = None

    def to_context_str(self) -> str:
        return f"[{self.category}] {self.content}"


class MemoryStore:
    def __init__(self, db_path: str = "memories.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(_INIT_SQL)
            conn.commit()
        finally:
            conn.close()

    def add(self, memory: Memory) -> str:
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO memories
                        (id, content, category, importance, created_at,
                         last_accessed, access_count, session_id, is_active, superseded_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.id, memory.content, memory.category, memory.importance,
                        memory.created_at, memory.last_accessed, memory.access_count,
                        memory.session_id, int(memory.is_active), memory.superseded_by,
                    ),
                )
        finally:
            conn.close()
        return memory.id

    def supersede(self, old_id: str, new_memory: Memory) -> str:
        new_id = self.add(new_memory)
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    "UPDATE memories SET is_active=0, superseded_by=? WHERE id=?",
                    (new_id, old_id),
                )
        finally:
            conn.close()
        return new_id

    def deactivate(self, memory_id: str) -> None:
        conn = self._conn()
        try:
            with conn:
                conn.execute("UPDATE memories SET is_active=0 WHERE id=?", (memory_id,))
        finally:
            conn.close()

    def search(
        self,
        query: str,
        limit: int = 12,
        min_importance: float = 3.0,
    ) -> list[Memory]:
        """
        Hybrid retrieval: FTS5 keyword match merged with top high-importance memories.

        Pure keyword search misses cases where the query has no lexical overlap with a
        stored memory that's still relevant — e.g. "help me with error handling" won't
        match "Senior Go engineer" even though that context shapes the answer. Pinning
        the top few high-importance memories ensures core user facts always surface.
        """
        safe_query = _to_fts_query(query)
        conn = self._conn()
        try:
            if safe_query:
                keyword_rows = cast(list[sqlite3.Row], conn.execute(
                    """
                    SELECT m.* FROM memories m
                    WHERE m.rowid IN (
                        SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?
                    )
                    AND m.is_active = 1
                    AND m.importance >= ?
                    ORDER BY m.importance DESC, m.last_accessed DESC
                    LIMIT ?
                    """,
                    (safe_query, min_importance, limit - 4),
                ).fetchall())
            else:
                keyword_rows = cast(list[sqlite3.Row], [])

            # Always include the top high-importance memories regardless of keyword match.
            pinned_rows = cast(list[sqlite3.Row], conn.execute(
                """
                SELECT * FROM memories
                WHERE is_active=1 AND importance >= 7.0
                ORDER BY importance DESC, last_accessed DESC
                LIMIT 4
                """,
            ).fetchall())

            # Merge, preserving keyword results first, deduplicating by id.
            keyword_mems = [_row_to_memory(r) for r in keyword_rows]
            pinned_mems = [_row_to_memory(r) for r in pinned_rows]
            seen = {m.id for m in keyword_mems}
            memories = (keyword_mems + [m for m in pinned_mems if m.id not in seen])[:limit]

            if memories:
                now = time.time()
                conn.executemany(
                    "UPDATE memories SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                    [(now, m.id) for m in memories],
                )
                conn.commit()
        finally:
            conn.close()

        return memories

    def get_by_category(self, category: MemoryCategory, limit: int = 20) -> list[Memory]:
        conn = self._conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE category=? AND is_active=1
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
                """,
                (category, limit),
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_memory(r) for r in rows]

    def get_all_active(self) -> list[Memory]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM memories WHERE is_active=1 ORDER BY importance DESC",
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_memory(r) for r in rows]

    def get_by_id(self, memory_id: str) -> "Memory | None":
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        finally:
            conn.close()
        return _row_to_memory(row) if row else None

    def count_active(self) -> int:
        conn = self._conn()
        try:
            result = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE is_active=1"
            ).fetchone()[0]
        finally:
            conn.close()
        return result

    def resolve_id_prefix(self, prefix: str) -> str | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT id FROM memories WHERE id=? OR id LIKE ? LIMIT 1",
                (prefix, f"{prefix}%"),
            ).fetchone()
        finally:
            conn.close()
        return row["id"] if row else None


def _to_fts_query(text: str) -> str:
    words = re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())
    words = [w for w in words if w not in _STOP_WORDS]
    if not words:
        return ""
    return " OR ".join(f'"{w}"' for w in words[:10])


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        category=row["category"],
        importance=row["importance"],
        created_at=row["created_at"],
        last_accessed=row["last_accessed"],
        access_count=row["access_count"],
        session_id=row["session_id"],
        is_active=bool(row["is_active"]),
        superseded_by=row["superseded_by"],
    )
