"""End-to-end roundtrip tests: archive -> recover via overview + search + read."""

import json
from pathlib import Path

import pytest

from hermes_vcc.archive import archive_before_compression
from hermes_vcc.recovery import handle_vcc_recover


class TestFullPipeline:
    """Create messages -> archive -> recover via overview + search + read."""

    def test_full_pipeline(self, basic_conversation, archive_dir, vcc_py_path):
        session_id = "roundtrip-session"
        archive_before_compression(
            messages=basic_conversation,
            session_id=session_id,
            archive_dir=archive_dir,
            compression_cycle=1,
        )

        # Overview should return .min.txt content
        overview = handle_vcc_recover(
            action="overview",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "Overview" in overview
        assert "cycle_1" in overview
        assert len(overview) > 50

        # Search should find content from the conversation
        search = handle_vcc_recover(
            action="search",
            query="Paris",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "Paris" in search
        assert ">>" in search  # match highlight

        # Read should return specific lines
        read = handle_vcc_recover(
            action="read",
            query="1-10",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "cycle_1" in read
        assert "1:" in read  # line numbers present


class TestMultiCycle:
    """Archive twice, recover from both cycles."""

    def test_multi_cycle(self, basic_conversation, archive_dir, vcc_py_path):
        session_id = "multi-roundtrip"

        # Cycle 1 with basic conversation
        archive_before_compression(
            messages=basic_conversation,
            session_id=session_id,
            archive_dir=archive_dir,
            compression_cycle=1,
        )

        # Cycle 2 with modified conversation (add a message)
        extended = basic_conversation + [
            {"role": "user", "content": "What about the capital of Spain?"},
            {"role": "assistant", "content": "The capital of Spain is Madrid."},
        ]
        archive_before_compression(
            messages=extended,
            session_id=session_id,
            archive_dir=archive_dir,
            compression_cycle=2,
        )

        # List should show both cycles
        listing = handle_vcc_recover(
            action="list",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "2 cycles" in listing
        assert "cycle_1" in listing
        assert "cycle_2" in listing

        # Overview of cycle 1
        overview_1 = handle_vcc_recover(
            action="overview",
            archive_id="1",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "cycle_1" in overview_1

        # Overview of cycle 2 (default = latest)
        overview_2 = handle_vcc_recover(
            action="overview",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "cycle_2" in overview_2

        # Search cycle 2 for Madrid (only in the extended conversation)
        search_2 = handle_vcc_recover(
            action="search",
            query="Madrid",
            archive_id="2",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "Madrid" in search_2

        # Search cycle 1 for Madrid should find nothing
        search_1 = handle_vcc_recover(
            action="search",
            query="Madrid",
            archive_id="1",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "No matches" in search_1


class TestToolHeavyRoundtrip:
    """Tool-heavy sessions with 10+ tool calls round-trip correctly."""

    def test_tool_heavy_roundtrip(self, tool_heavy_session, archive_dir, vcc_py_path):
        session_id = "tool-heavy-roundtrip"

        # The tool_heavy_session has 3 tool calls; extend with more
        extra_calls = []
        for i in range(7):
            call_id = f"call_extra_{i}"
            extra_calls.extend([
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": f"Tool_{i}",
                                "arguments": json.dumps({"arg": f"value_{i}"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": f"Result from tool {i}: success",
                    "tool_call_id": call_id,
                },
            ])

        messages = tool_heavy_session + extra_calls

        archive_before_compression(
            messages=messages,
            session_id=session_id,
            archive_dir=archive_dir,
            compression_cycle=1,
        )

        # Overview (.min.txt) should exist and have content
        overview = handle_vcc_recover(
            action="overview",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "cycle_1" in overview
        assert len(overview) > 100

        # Search for tool names from the original session
        search_read = handle_vcc_recover(
            action="search",
            query="Read",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "Read" in search_read

        # Search for one of the extra tool results
        search_extra = handle_vcc_recover(
            action="search",
            query="Tool_3",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "Tool_3" in search_extra

        # Read the full transcript and verify it has substantial content
        read_all = handle_vcc_recover(
            action="read",
            query="1-200",
            archive_dir=archive_dir,
            session_id=session_id,
        )
        assert "cycle_1" in read_all
        # Should have many lines given 10+ tool calls
        line_count = read_all.count("\n")
        assert line_count > 20
