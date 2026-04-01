"""Install VCC into a running Hermes agent.

Two hooks:
1. Archive hook — before each compression, archive the full conversation
   as JSONL and compile VCC views (.txt, .min.txt).
2. Summary hook — replace the LLM summary with VCC's .min.txt.
   Deterministic, instant, free. Falls back to LLM if VCC fails.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from hermes_vcc.config import VCCConfig, load_config

logger = logging.getLogger(__name__)

_CYCLE_ATTR = "_vcc_compression_cycle"


def install(agent: Any, config: VCCConfig | None = None) -> dict[str, bool]:
    """Install VCC hooks on a Hermes agent.

    Args:
        agent: A Hermes AIAgent instance.
        config: Optional config. Loaded from config.yaml if None.

    Returns:
        {"archive": bool, "summary": bool} indicating success.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = VCCConfig()

    if not config.enabled:
        return {"archive": False, "summary": False}

    results = {
        "archive": _install_archive(agent, config),
        "summary": _install_summary(agent, config),
    }
    logger.info("VCC hooks: %s", results)
    return results


def _install_archive(agent: Any, config: VCCConfig) -> bool:
    """Patch _compress_context to archive before compression."""
    try:
        original = getattr(agent, "_compress_context", None)
        if original is None:
            return False
        if getattr(original, "_vcc_wrapped", False):
            return True  # already installed

        from hermes_vcc.archive import archive_before_compression, prune_archives
        from hermes_vcc.utils import ensure_dir

        archive_dir = ensure_dir(config.archive_dir)

        @functools.wraps(original)
        def wrapper(messages, system_message, *args, **kwargs):
            session_id = getattr(agent, "session_id", None) or "unknown"
            cycle = getattr(agent, _CYCLE_ATTR, 0) + 1
            setattr(agent, _CYCLE_ATTR, cycle)

            try:
                session_dir = archive_before_compression(
                    messages, session_id, archive_dir, cycle,
                )
                if config.retain_archives > 0:
                    prune_archives(session_dir, retain=config.retain_archives)
            except Exception as exc:
                logger.warning("VCC archive failed (non-fatal): %s", exc)

            return original(messages, system_message, *args, **kwargs)

        wrapper._vcc_wrapped = True  # type: ignore[attr-defined]
        agent._compress_context = wrapper
        return True

    except Exception as exc:
        logger.warning("VCC archive hook failed: %s", exc)
        return False


def _install_summary(agent: Any, config: VCCConfig) -> bool:
    """Replace compressor._generate_summary with VCC .min.txt.

    The .min.txt is the summary. No LLM call.
    Falls back to the original LLM summary if VCC compilation fails.
    """
    try:
        compressor = getattr(agent, "context_compressor", None)
        if compressor is None:
            return False

        original_gen = getattr(compressor, "_generate_summary", None)
        if original_gen is None:
            return False
        if getattr(original_gen, "_vcc_wrapped", False):
            return True

        from hermes_vcc.enhanced_summary import compile_to_brief

        def vcc_summary(turns, *args, **kwargs):
            brief = compile_to_brief(turns)
            if brief:
                logger.info("VCC summary: %d chars (no LLM call)", len(brief))
                return brief
            logger.info("VCC failed, falling back to LLM summary")
            return original_gen(turns, *args, **kwargs)

        vcc_summary._vcc_wrapped = True  # type: ignore[attr-defined]
        compressor._generate_summary = vcc_summary
        return True

    except Exception as exc:
        logger.warning("VCC summary hook failed: %s", exc)
        return False
