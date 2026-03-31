"""Tests for hermes_vcc.hooks — integration hooks for monkey-patching agents."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_vcc.config import VCCConfig
from hermes_vcc.hooks import (
    install_all,
    install_archive_hook,
    install_recovery_tool,
    VCC_RECOVERY_SCHEMA,
)


def _make_config(tmp_path):
    """Create a VCCConfig pointing at tmp_path for archives."""
    return VCCConfig(
        enabled=True,
        archive_dir=tmp_path / "vcc_archives",
        enhanced_summary=True,
        recovery_tool=True,
        retain_archives=10,
    )


def _make_agent(has_compress=True, has_tools=True):
    """Create a mock agent with the attributes hooks expect."""
    agent = MagicMock()
    agent.session_id = "test-session-123"

    if has_compress:
        agent._compress_context = MagicMock(return_value=([], "summary"))
        agent._compress_context._vcc_archive_wrapped = False
    else:
        # Simulate agent without _compress_context
        del agent._compress_context

    if has_tools:
        agent.tools = []
        agent.valid_tool_names = frozenset()
    else:
        del agent.tools

    return agent


class TestInstallArchiveHook:
    """install_archive_hook wraps _compress_context."""

    def test_install_archive_hook(self, tmp_path, vcc_py_path):
        config = _make_config(tmp_path)
        agent = _make_agent(has_compress=True)
        original_method = agent._compress_context

        result = install_archive_hook(agent, config)
        assert result is True

        # The method should be wrapped now
        assert hasattr(agent._compress_context, "_vcc_archive_wrapped")
        assert agent._compress_context._vcc_archive_wrapped is True

        # Calling the wrapper should call the original method
        messages = [{"role": "user", "content": "test"}]
        agent._compress_context(messages, "system prompt")
        original_method.assert_called_once_with(messages, "system prompt")

    def test_install_archive_hook_no_method(self, tmp_path):
        """Agent without _compress_context returns False."""
        config = _make_config(tmp_path)
        agent = _make_agent(has_compress=False)

        result = install_archive_hook(agent, config)
        assert result is False

    def test_install_archive_hook_double_patch(self, tmp_path, vcc_py_path):
        """Second call is idempotent."""
        config = _make_config(tmp_path)
        agent = _make_agent(has_compress=True)

        first = install_archive_hook(agent, config)
        assert first is True

        # Capture the wrapper after first install
        wrapper_after_first = agent._compress_context

        second = install_archive_hook(agent, config)
        assert second is True

        # The wrapper should be the same object (not double-wrapped)
        assert agent._compress_context is wrapper_after_first


class TestInstallRecoveryTool:
    """install_recovery_tool adds schema to agent.tools."""

    def test_install_recovery_tool(self, tmp_path):
        config = _make_config(tmp_path)
        agent = _make_agent(has_tools=True)

        result = install_recovery_tool(agent, config)
        assert result is True

        # Tool schema should be in the tools list
        assert len(agent.tools) == 1
        assert agent.tools[0]["function"]["name"] == "vcc_recover"

        # valid_tool_names should be updated
        assert "vcc_recover" in agent.valid_tool_names

        # Handler should be attached
        assert hasattr(agent, "_vcc_recover_handler")
        assert callable(agent._vcc_recover_handler)

    def test_install_recovery_tool_no_tools_attr(self, tmp_path):
        """Agent without tools attribute returns False."""
        config = _make_config(tmp_path)
        agent = _make_agent(has_tools=False)

        result = install_recovery_tool(agent, config)
        assert result is False

    def test_install_recovery_tool_idempotent(self, tmp_path):
        """Second install does not add duplicate schema."""
        config = _make_config(tmp_path)
        agent = _make_agent(has_tools=True)

        install_recovery_tool(agent, config)
        install_recovery_tool(agent, config)

        assert len(agent.tools) == 1


class TestInstallAll:
    """install_all installs both hooks and reports results."""

    def test_install_all(self, tmp_path, vcc_py_path):
        config = _make_config(tmp_path)
        agent = _make_agent(has_compress=True, has_tools=True)

        results = install_all(agent, config=config)

        assert results["archive_hook"] is True
        assert results["recovery_tool"] is True

    def test_install_all_disabled(self, tmp_path):
        """When VCC is disabled, both hooks report False."""
        config = _make_config(tmp_path)
        config.enabled = False
        agent = _make_agent()

        results = install_all(agent, config=config)

        assert results["archive_hook"] is False
        assert results["recovery_tool"] is False
