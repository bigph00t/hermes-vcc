"""Tests for hermes_vcc.recovery — post-compression recovery tool."""

import json
from pathlib import Path

import pytest

from hermes_vcc.archive import archive_before_compression
from hermes_vcc.recovery import handle_vcc_recover


def _create_archive(messages, archive_dir, session_id="test-session", cycle=1):
    """Helper: create an archive and return (session_dir, archive_dir)."""
    session_dir = archive_before_compression(
        messages=messages,
        session_id=session_id,
        archive_dir=archive_dir,
        compression_cycle=cycle,
    )
    return session_dir


class TestListNoArchives:
    """Returns helpful message when no archives exist."""

    def test_list_no_archives(self, tmp_path):
        # archive_dir exists but is empty
        archive_dir = tmp_path / "vcc_archives"
        archive_dir.mkdir()

        result = handle_vcc_recover(
            action="list",
            archive_dir=archive_dir,
        )
        assert "No session directories" in result

    def test_list_no_archive_dir(self, tmp_path):
        """Returns helpful message when archive_dir does not exist."""
        result = handle_vcc_recover(
            action="list",
            archive_dir=tmp_path / "nonexistent",
        )
        assert "No VCC archive directory" in result


class TestListWithArchives:
    """Returns formatted cycle list from existing archives."""

    def test_list_with_archives(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir, cycle=1)
        _create_archive(basic_conversation, archive_dir, cycle=2)

        result = handle_vcc_recover(
            action="list",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "VCC Archives" in result
        assert "2 cycles" in result
        assert "cycle_1" in result
        assert "cycle_2" in result


class TestOverview:
    """Returns .min.txt content for a cycle."""

    def test_overview_returns_min_txt(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="overview",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "Overview" in result
        assert "cycle_1" in result
        # .min.txt should contain some conversation content
        assert len(result) > 20


class TestSearch:
    """Regex search over the .txt transcript."""

    def test_search_finds_matches(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="search",
            query="Paris",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "Paris" in result
        assert "match" in result.lower()
        # Should show line numbers
        assert ">>" in result  # match highlight prefix

    def test_search_no_matches(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="search",
            query="xyzzy_nonexistent_pattern",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "No matches" in result

    def test_search_invalid_regex(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="search",
            query="[invalid(regex",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "Invalid regex" in result

    def test_search_missing_query(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="search",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "requires" in result.lower()


class TestRead:
    """Reads specific line ranges from .txt transcript."""

    def test_read_line_range(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="read",
            query="1-5",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "cycle_1" in result
        assert "lines 1-5" in result
        # Should contain line numbers
        assert "1:" in result

    def test_read_invalid_range(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="read",
            query="not-a-range",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "Invalid line range" in result

    def test_read_out_of_bounds(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="read",
            query="99999-100000",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "exceeds" in result.lower()


class TestUnknownAction:
    """Returns error for unknown action."""

    def test_unknown_action(self, basic_conversation, archive_dir, vcc_py_path):
        _create_archive(basic_conversation, archive_dir)

        result = handle_vcc_recover(
            action="delete",
            archive_dir=archive_dir,
            session_id="test-session",
        )
        assert "Unknown action" in result
        assert "delete" in result
