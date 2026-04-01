"""Tests for hermes_vcc.hooks — VCC installation on agent instances."""

from unittest.mock import MagicMock

from hermes_vcc.config import VCCConfig
from hermes_vcc.hooks import install, _install_archive, _install_summary


def _make_config(tmp_path):
    return VCCConfig(enabled=True, archive_dir=tmp_path / "vcc_archives", retain_archives=10)


def _make_agent(has_compress=True):
    agent = MagicMock()
    agent.session_id = "test-123"
    if has_compress:
        agent._compress_context = MagicMock(return_value=([], "sys"))
        agent._compress_context._vcc_wrapped = False
        agent.context_compressor = MagicMock()
        agent.context_compressor._generate_summary = MagicMock(return_value="llm summary")
        agent.context_compressor._generate_summary._vcc_wrapped = False
    else:
        del agent._compress_context
        del agent.context_compressor
    return agent


class TestInstallArchive:
    def test_installs(self, tmp_path, vcc_py_path):
        config = _make_config(tmp_path)
        agent = _make_agent()
        original = agent._compress_context
        assert _install_archive(agent, config) is True
        assert agent._compress_context._vcc_wrapped is True
        # Calling wrapper delegates to original
        agent._compress_context([{"role": "user", "content": "hi"}], "sys")
        original.assert_called_once()

    def test_no_method(self, tmp_path):
        config = _make_config(tmp_path)
        agent = _make_agent(has_compress=False)
        assert _install_archive(agent, config) is False

    def test_idempotent(self, tmp_path, vcc_py_path):
        config = _make_config(tmp_path)
        agent = _make_agent()
        _install_archive(agent, config)
        wrapper = agent._compress_context
        _install_archive(agent, config)
        assert agent._compress_context is wrapper  # not double-wrapped


class TestInstallSummary:
    def test_installs(self, tmp_path, vcc_py_path):
        config = _make_config(tmp_path)
        agent = _make_agent()
        assert _install_summary(agent, config) is True
        assert agent.context_compressor._generate_summary._vcc_wrapped is True

    def test_no_compressor(self, tmp_path):
        config = _make_config(tmp_path)
        agent = _make_agent(has_compress=False)
        assert _install_summary(agent, config) is False

    def test_idempotent(self, tmp_path, vcc_py_path):
        config = _make_config(tmp_path)
        agent = _make_agent()
        _install_summary(agent, config)
        fn = agent.context_compressor._generate_summary
        _install_summary(agent, config)
        assert agent.context_compressor._generate_summary is fn


class TestInstall:
    def test_both_hooks(self, tmp_path, vcc_py_path):
        config = _make_config(tmp_path)
        agent = _make_agent()
        results = install(agent, config)
        assert results["archive"] is True
        assert results["summary"] is True

    def test_disabled(self, tmp_path):
        config = _make_config(tmp_path)
        config.enabled = False
        agent = _make_agent()
        results = install(agent, config)
        assert results == {"archive": False, "summary": False}
