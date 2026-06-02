import unittest.mock as mock

import pytest

from agent.extractor import (
    ExtractionResult,
    MemoryExtractor,
    _is_sensitive,
    _parse_item,
    _resolve_id,
)
from agent.memory_store import Memory


def make_extractor() -> MemoryExtractor:
    return MemoryExtractor(mock.MagicMock(), model="claude-sonnet-4-6")


def make_memory(id_suffix: str = "0001") -> Memory:
    return Memory(
        id=f"aaaabbbb-cccc-dddd-eeee-ffff{id_suffix:>08}",
        content="User uses vim",
        category="preference",
        importance=7.0,
    )


class TestSensitiveDataFilter:
    def test_rejects_credential_patterns(self):
        assert _is_sensitive("password: hunter2")
        assert _is_sensitive("api_key=sk-abc123defghijklm")
        assert _is_sensitive("token=Bearer eyJhbGciOiJIUzI1NiJ9.x.y")

    def test_rejects_ssn(self):
        assert _is_sensitive("my SSN is 123-45-6789")

    def test_rejects_credit_card(self):
        assert _is_sensitive("card 4111 1111 1111 1111")

    def test_rejects_api_tokens(self):
        assert _is_sensitive("ghp_abcdefghijklmnopqrstuvwxyz12")

    def test_passes_safe_content(self):
        assert not _is_sensitive("User prefers pytest over unittest")
        assert not _is_sensitive("User is building a FastAPI service")

    def test_password_in_context_not_flagged(self):
        assert not _is_sensitive("User wants to improve password management UX")


class TestParseItem:
    def test_valid_item(self):
        m = _parse_item({"content": "User prefers Go", "category": "preference", "importance": 8}, "s")
        assert m is not None
        assert m.category == "preference"
        assert m.importance == 8.0

    def test_clamps_importance(self):
        assert _parse_item({"content": "something notable", "importance": 99}, "s").importance == 10.0
        assert _parse_item({"content": "something notable", "importance": -5}, "s").importance == 1.0

    def test_unknown_category_defaults_to_fact(self):
        m = _parse_item({"content": "some information", "category": "unknown", "importance": 5}, "s")
        assert m.category == "fact"

    def test_short_content_rejected(self):
        assert _parse_item({"content": "ok", "importance": 5}, "s") is None

    def test_sensitive_content_rejected(self):
        assert _parse_item({"content": "api_key=sk-supersecret12345678", "importance": 5}, "s") is None


class TestExtractorParsing:
    def test_parse_adds_memories(self):
        e = make_extractor()
        raw = '{"memories_to_add": [{"content": "User is a senior Go engineer", "category": "fact", "importance": 8}], "memories_to_supersede": [], "memories_to_deactivate": []}'
        result = e._parse(raw, "sess", [])
        assert len(result.to_add) == 1
        assert result.to_add[0].content == "User is a senior Go engineer"

    def test_parse_supersedes_by_prefix(self):
        e = make_extractor()
        existing = [make_memory()]
        raw = f'{{"memories_to_add": [], "memories_to_supersede": [{{"old_id": "{existing[0].id[:8]}", "content": "User switched to neovim", "category": "preference", "importance": 7}}], "memories_to_deactivate": []}}'
        result = e._parse(raw, "sess", existing)
        assert len(result.to_supersede) == 1
        old_id, new_mem = result.to_supersede[0]
        assert old_id == existing[0].id
        assert "neovim" in new_mem.content

    def test_parse_deactivates(self):
        e = make_extractor()
        existing = [make_memory()]
        raw = f'{{"memories_to_add": [], "memories_to_supersede": [], "memories_to_deactivate": ["{existing[0].id[:8]}"]}}'
        result = e._parse(raw, "sess", existing)
        assert existing[0].id in result.to_deactivate

    def test_parse_strips_markdown_fences(self):
        e = make_extractor()
        raw = '```json\n{"memories_to_add": [{"content": "User prefers short answers", "category": "preference", "importance": 6}], "memories_to_supersede": [], "memories_to_deactivate": []}\n```'
        result = e._parse(raw, "sess", [])
        assert len(result.to_add) == 1

    def test_malformed_json_returns_empty(self):
        assert make_extractor()._parse("not json", "sess", []).is_empty

    def test_filters_sensitive_in_parsed_output(self):
        e = make_extractor()
        raw = '{"memories_to_add": [{"content": "api_key=sk-supersecretkey123456", "category": "fact", "importance": 5}], "memories_to_supersede": [], "memories_to_deactivate": []}'
        assert make_extractor()._parse(raw, "sess", []).is_empty


class TestExtractionResult:
    def test_is_empty(self):
        assert ExtractionResult([], [], []).is_empty

    def test_not_empty_with_additions(self):
        m = Memory(content="fact", category="fact", importance=5.0)
        assert not ExtractionResult([m], [], []).is_empty
