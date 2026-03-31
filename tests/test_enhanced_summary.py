"""Tests for hermes_vcc.enhanced_summary — VCC-enhanced compressor."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_vcc.enhanced_summary import (
    _messages_to_serialized_text,
    compile_to_brief,
    generate_vcc_enhanced_summary,
)


class TestCompileToBrief:
    """compile_to_brief returns .min.txt content."""

    def test_compile_to_brief(self, basic_conversation, vcc_py_path):
        result = compile_to_brief(basic_conversation)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compile_to_brief_empty(self):
        result = compile_to_brief([])
        assert result is None


class TestGenerateEnhancedSummary:
    """generate_vcc_enhanced_summary calls original_summary_fn with enriched input."""

    def test_generate_enhanced_summary(self, basic_conversation, vcc_py_path):
        mock_fn = MagicMock(return_value="summary result")

        result = generate_vcc_enhanced_summary(
            messages_to_summarize=basic_conversation,
            original_summary_fn=mock_fn,
        )

        assert result == "summary result"
        mock_fn.assert_called_once()

        # The enriched input should contain the structural skeleton header
        call_arg = mock_fn.call_args[0][0]
        assert "STRUCTURAL SKELETON" in call_arg
        assert "FULL CONVERSATION TURNS" in call_arg

    def test_generate_enhanced_summary_fallback(self, basic_conversation):
        """If VCC fails, still calls original_summary_fn with plain turns."""
        mock_fn = MagicMock(return_value="plain summary")

        # Patch import_vcc at the source (hermes_vcc.utils) so the local
        # import inside generate_vcc_enhanced_summary picks it up.
        with patch(
            "hermes_vcc.utils.import_vcc",
            side_effect=ImportError("VCC not available"),
        ):
            result = generate_vcc_enhanced_summary(
                messages_to_summarize=basic_conversation,
                original_summary_fn=mock_fn,
            )

        assert result == "plain summary"
        mock_fn.assert_called_once()

        # The fallback input should NOT contain the skeleton header
        call_arg = mock_fn.call_args[0][0]
        assert "STRUCTURAL SKELETON" not in call_arg
        # But it should contain the serialized turns
        assert "[user]" in call_arg or "[assistant]" in call_arg


class TestMessagesToSerializedText:
    """_messages_to_serialized_text produces role-prefixed text."""

    def test_messages_to_serialized_text(self, basic_conversation):
        result = _messages_to_serialized_text(basic_conversation)

        # Should contain role markers
        assert "[system]" in result
        assert "[user]" in result
        assert "[assistant]" in result

        # Should contain message content
        assert "capital of France" in result
        assert "Paris" in result
        assert "Germany" in result
        assert "Berlin" in result

        # Messages should be separated by blank lines
        assert "\n\n" in result

    def test_messages_to_serialized_text_with_tool_calls(self, tool_heavy_session):
        result = _messages_to_serialized_text(tool_heavy_session)

        # Tool calls should appear as [tool_call] markers
        assert "[tool_call]" in result
        assert "Read" in result

    def test_messages_to_serialized_text_empty(self):
        result = _messages_to_serialized_text([])
        assert result == ""
