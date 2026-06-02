import os
import tempfile

import pytest

from agent.memory_store import Memory, MemoryStore, _to_fts_query


@pytest.fixture
def db_file():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def store(db_file):
    return MemoryStore(db_file)


def mem(**kwargs) -> Memory:
    defaults = {"content": "User prefers Python 3.11", "category": "preference", "importance": 7.0}
    defaults.update(kwargs)
    return Memory(**defaults)


class TestCRUD:
    def test_add_and_fetch(self, store):
        m = mem(content="User is a data scientist")
        store.add(m)
        fetched = store.get_by_id(m.id)
        assert fetched is not None
        assert fetched.content == m.content

    def test_deactivate_hides_from_queries(self, store):
        m = mem(content="User prefers dark mode")
        store.add(m)
        store.deactivate(m.id)
        assert store.get_by_id(m.id).is_active is False
        assert store.count_active() == 0

    def test_count_active(self, store):
        store.add(mem(content="one"))
        store.add(mem(content="two"))
        assert store.count_active() == 2


class TestSupersede:
    def test_old_deactivated_new_active(self, store):
        old = mem(content="User uses vim")
        store.add(old)
        new = mem(content="User switched to neovim")
        new_id = store.supersede(old.id, new)

        assert store.get_by_id(old.id).is_active is False
        assert store.get_by_id(old.id).superseded_by == new_id
        assert store.get_by_id(new_id).is_active is True

    def test_only_new_memory_appears_in_search(self, store):
        old = mem(content="User uses vim editor")
        store.add(old)
        store.supersede(old.id, mem(content="User switched to neovim editor"))

        results = store.search("editor")
        assert len(results) == 1
        assert "neovim" in results[0].content


class TestSearch:
    def test_basic_keyword_match(self, store):
        store.add(mem(content="User is a TypeScript engineer"))
        assert len(store.search("TypeScript")) == 1

    def test_importance_filter(self, store):
        store.add(mem(content="high value fact", importance=8.0))
        store.add(mem(content="low value fact", importance=2.0))
        results = store.search("fact", min_importance=5.0)
        assert len(results) == 1
        assert results[0].content == "high value fact"

    def test_inactive_excluded(self, store):
        m = mem(content="User likes Rust")
        store.add(m)
        store.deactivate(m.id)
        assert not any(r.id == m.id for r in store.search("Rust"))

    def test_increments_access_count(self, store):
        m = mem(content="User loves pytest")
        store.add(m)
        store.search("pytest")
        assert store.get_by_id(m.id).access_count == 1

    def test_empty_query_falls_back_to_importance_rank(self, store):
        store.add(mem(content="alpha", importance=9.0))
        store.add(mem(content="beta", importance=4.0))
        results = store.search("", min_importance=3.0)
        assert results[0].importance >= results[-1].importance

    def test_respects_limit(self, store):
        for i in range(20):
            store.add(mem(content=f"Python memory {i}", importance=5.0))
        assert len(store.search("Python", limit=5)) <= 5


class TestPersistence:
    def test_memories_survive_reopen(self, db_file):
        store1 = MemoryStore(db_file)
        m = mem(content="User is building a CLI tool called flux")
        store1.add(m)

        store2 = MemoryStore(db_file)
        results = store2.search("flux")
        assert len(results) == 1
        assert results[0].id == m.id

    def test_deactivation_survives_reopen(self, db_file):
        store1 = MemoryStore(db_file)
        m = mem(content="outdated preference")
        store1.add(m)
        store1.deactivate(m.id)

        assert MemoryStore(db_file).count_active() == 0

    def test_supersede_survives_reopen(self, db_file):
        store1 = MemoryStore(db_file)
        old = mem(content="old preference")
        store1.add(old)
        store1.supersede(old.id, mem(content="new preference"))

        store2 = MemoryStore(db_file)
        assert store2.get_by_id(old.id).is_active is False
        assert store2.count_active() == 1


class TestIdResolution:
    def test_full_id(self, store):
        m = mem()
        store.add(m)
        assert store.resolve_id_prefix(m.id) == m.id

    def test_prefix(self, store):
        m = mem()
        store.add(m)
        assert store.resolve_id_prefix(m.id[:8]) == m.id

    def test_unknown_prefix_returns_none(self, store):
        assert store.resolve_id_prefix("xxxxxxxx") is None


class TestFTSQuery:
    def test_keywords_included(self):
        q = _to_fts_query("TypeScript Python engineer")
        assert '"typescript"' in q
        assert '"python"' in q

    def test_stop_words_excluded(self):
        q = _to_fts_query("the user is a developer")
        assert '"the"' not in q and '"is"' not in q
        assert '"developer"' in q

    def test_empty_returns_empty(self):
        assert _to_fts_query("") == ""
