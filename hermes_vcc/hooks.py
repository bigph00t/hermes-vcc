"""Integration hooks for monkey-patching Hermes's AIAgent.

Provides non-invasive installation of VCC archiving and recovery into a
running agent instance.  All hooks are best-effort: failures are logged
and silently swallowed so the agent is never broken by VCC integration
issues.

Usage (typically in a plugin or startup script)::

    from hermes_vcc.hooks import install_all

    results = install_all(agent_instance)
    # results == {"archive_hook": True, "recovery_tool": True}
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from hermes_vcc.config import VCCConfig, load_config

logger = logging.getLogger(__name__)

# Counter used as a monotonic compression-cycle id per agent instance.
# Stored as an attribute on the wrapper so it survives across calls.
_CYCLE_ATTR = "_vcc_compression_cycle"


# ---------------------------------------------------------------------------
# Archive hook
# ---------------------------------------------------------------------------


def install_archive_hook(agent_instance: Any, config: VCCConfig) -> bool:
    """Monkey-patch ``_compress_context`` to archive before compression.

    The wrapper calls :func:`hermes_vcc.archive.archive_before_compression`
    with the current message list, then delegates to the original method.
    On any VCC error the original method is still called — archival never
    blocks compression.

    Args:
        agent_instance: A Hermes ``AIAgent`` (or compatible) instance.
        config: Resolved VCC configuration.

    Returns:
        *True* if the hook was installed, *False* if the target method was
        not found or an error occurred.
    """
    try:
        original_method = getattr(agent_instance, "_compress_context", None)
        if original_method is None:
            logger.info(
                "VCC archive hook: agent has no _compress_context — skipping"
            )
            return False

        # Avoid double-patching.
        if getattr(original_method, "_vcc_archive_wrapped", False):
            logger.debug("VCC archive hook already installed — skipping")
            return True

        from hermes_vcc.archive import archive_before_compression, prune_archives
        from hermes_vcc.utils import ensure_dir

        archive_dir = ensure_dir(config.archive_dir)

        @functools.wraps(original_method)
        def _vcc_compress_wrapper(
            messages: list,
            system_message: str,
            *args: Any,
            **kwargs: Any,
        ) -> tuple:
            # Determine session id (best-effort).
            session_id = getattr(agent_instance, "session_id", None) or "unknown"

            # Increment compression cycle counter.
            cycle = getattr(agent_instance, _CYCLE_ATTR, 0) + 1
            setattr(agent_instance, _CYCLE_ATTR, cycle)

            # Archive — swallowed on failure.
            try:
                session_dir = archive_before_compression(
                    messages=messages,
                    session_id=session_id,
                    archive_dir=archive_dir,
                    compression_cycle=cycle,
                )
                logger.info(
                    "VCC archived %d messages for session %s cycle %d",
                    len(messages),
                    session_id,
                    cycle,
                )

                # Prune old cycles if configured.
                if config.retain_archives > 0:
                    prune_archives(session_dir, retain=config.retain_archives)

            except Exception as exc:
                logger.warning(
                    "VCC archive hook failed (non-fatal): %s", exc
                )

            # Always call the original compressor.
            return original_method(messages, system_message, *args, **kwargs)

        _vcc_compress_wrapper._vcc_archive_wrapped = True  # type: ignore[attr-defined]
        agent_instance._compress_context = _vcc_compress_wrapper

        logger.info("VCC archive hook installed on %s", type(agent_instance).__name__)
        return True

    except Exception as exc:
        logger.warning("Failed to install VCC archive hook: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Summary mode hook
# ---------------------------------------------------------------------------


def install_summary_hook(agent_instance: Any, config: VCCConfig) -> bool:
    """Patch the compressor's summary generation based on config.summary_mode.

    Modes:
        ``"pure"``: Replace ``_generate_summary`` with VCC's
            ``compile_to_brief`` — deterministic, zero LLM calls, zero cost.
            Falls back to the original LLM summary if VCC fails.
        ``"hybrid"``: Feed VCC ``.min.txt`` skeleton + turns to the LLM.
        ``"llm"``: No summary patching (original LLM-only behavior).
            VCC still archives but doesn't touch the summary.

    Args:
        agent_instance: A Hermes ``AIAgent`` (or compatible) instance.
        config: Resolved VCC configuration.

    Returns:
        *True* if the hook was installed, *False* on error or ``"llm"`` mode.
    """
    if config.summary_mode == "llm":
        logger.debug("VCC summary_mode=llm — no summary hook needed")
        return False

    try:
        compressor = getattr(agent_instance, "context_compressor", None)
        if compressor is None:
            logger.info("VCC summary hook: no context_compressor found — skipping")
            return False

        original_gen = getattr(compressor, "_generate_summary", None)
        if original_gen is None:
            logger.info("VCC summary hook: compressor has no _generate_summary — skipping")
            return False

        if getattr(original_gen, "_vcc_summary_wrapped", False):
            logger.debug("VCC summary hook already installed — skipping")
            return True

        from hermes_vcc.enhanced_summary import (
            compile_to_brief,
            generate_pure_vcc_summary,
            generate_vcc_enhanced_summary,
            _messages_to_serialized_text,
        )

        if config.summary_mode == "pure":
            def _vcc_pure_summary(turns_to_summarize, *args, **kwargs):
                """Pure VCC summary: .min.txt only, no LLM call."""
                brief = compile_to_brief(turns_to_summarize)
                if brief:
                    logger.info(
                        "VCC pure summary: %d chars from .min.txt (no LLM call)",
                        len(brief),
                    )
                    return brief
                # VCC failed — fall back to original LLM summary
                logger.info("VCC compilation failed, falling back to LLM summary")
                return original_gen(turns_to_summarize, *args, **kwargs)

            _vcc_pure_summary._vcc_summary_wrapped = True  # type: ignore[attr-defined]
            compressor._generate_summary = _vcc_pure_summary
            logger.info("VCC summary hook installed (mode=pure, no LLM calls)")

        elif config.summary_mode == "hybrid":
            def _vcc_hybrid_summary(turns_to_summarize, *args, **kwargs):
                """Hybrid: VCC skeleton + LLM enrichment."""
                def _llm_fn(text):
                    # Reconstruct turns from text and call original
                    return original_gen(turns_to_summarize, *args, **kwargs)

                return generate_vcc_enhanced_summary(
                    turns_to_summarize, _llm_fn
                )

            _vcc_hybrid_summary._vcc_summary_wrapped = True  # type: ignore[attr-defined]
            compressor._generate_summary = _vcc_hybrid_summary
            logger.info("VCC summary hook installed (mode=hybrid, VCC + LLM)")

        return True

    except Exception as exc:
        logger.warning("Failed to install VCC summary hook: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Recovery tool
# ---------------------------------------------------------------------------

# OpenAI function-calling schema for the vcc_recover tool.
VCC_RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "vcc_recover",
        "description": (
            "Recover context from VCC conversation archives. "
            "Lists available archive cycles or retrieves the compiled "
            ".min.txt summary for a specific cycle. Use this after "
            "context compression to recall details that were lost."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read"],
                    "description": (
                        "'list' shows available archive cycles with metadata. "
                        "'read' returns the .min.txt content for a given cycle."
                    ),
                },
                "cycle_id": {
                    "type": "integer",
                    "description": (
                        "The compression cycle number to read. "
                        "Required when action is 'read'."
                    ),
                },
            },
            "required": ["action"],
        },
    },
}


def _handle_vcc_recover(
    agent_instance: Any,
    config: VCCConfig,
    action: str,
    cycle_id: int | None = None,
) -> str:
    """Execute the vcc_recover tool logic.

    Args:
        agent_instance: The agent, used to resolve the current session_id.
        config: VCC configuration (provides archive_dir).
        action: ``"list"`` or ``"read"``.
        cycle_id: Required when *action* is ``"read"``.

    Returns:
        A human-readable string suitable for returning as tool output.
    """
    import json
    from hermes_vcc.archive import get_archive_manifest

    session_id = getattr(agent_instance, "session_id", None) or "unknown"
    session_dir = config.archive_dir / session_id

    if not session_dir.is_dir():
        return f"No VCC archives found for session '{session_id}'."

    manifest = get_archive_manifest(session_dir)
    cycles = manifest.get("cycles", [])

    if action == "list":
        if not cycles:
            return f"No archived compression cycles for session '{session_id}'."

        lines = [f"VCC archives for session '{session_id}' ({len(cycles)} cycles):"]
        for c in cycles:
            cid = c.get("id", "?")
            ts = c.get("timestamp", "?")
            msg_count = c.get("message_count", "?")
            tokens = c.get("tokens_estimate", "?")
            lines.append(
                f"  cycle {cid}: {msg_count} messages, ~{tokens} tokens @ {ts}"
            )
        return "\n".join(lines)

    elif action == "read":
        if cycle_id is None:
            return "Error: cycle_id is required for action 'read'."

        # Try .min.txt first, then .txt
        for suffix in (".min.txt", ".txt"):
            path = session_dir / f"cycle_{cycle_id}{suffix}"
            if path.is_file():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    label = "min.txt" if suffix == ".min.txt" else "txt"
                    return (
                        f"VCC archive cycle {cycle_id} ({label}):\n\n{content}"
                    )

        # Fall back to raw JSONL if compiled views are missing.
        jsonl_path = session_dir / f"cycle_{cycle_id}.jsonl"
        if jsonl_path.is_file():
            content = jsonl_path.read_text(encoding="utf-8").strip()
            # Truncate very large JSONL to avoid blowing up context.
            if len(content) > 50_000:
                content = content[:50_000] + "\n... [truncated]"
            return f"VCC archive cycle {cycle_id} (raw JSONL):\n\n{content}"

        return f"No archive files found for cycle {cycle_id} in session '{session_id}'."

    else:
        return f"Unknown action '{action}'. Use 'list' or 'read'."


def install_recovery_tool(agent_instance: Any, config: VCCConfig) -> bool:
    """Add the ``vcc_recover`` tool to the agent's tool definitions.

    Looks for ``self.tools`` (list of OpenAI function-call schemas) and
    ``self.valid_tool_names`` (frozenset) on the agent.  If the tool is
    already present it is silently skipped.

    Args:
        agent_instance: A Hermes ``AIAgent`` (or compatible) instance.
        config: Resolved VCC configuration.

    Returns:
        *True* if the tool was added (or already present), *False* on error.
    """
    try:
        tools = getattr(agent_instance, "tools", None)
        if tools is None:
            logger.info(
                "VCC recovery tool: agent has no 'tools' attribute — skipping"
            )
            return False

        tool_name = VCC_RECOVERY_SCHEMA["function"]["name"]

        # Check if already registered.
        existing_names = {
            t.get("function", {}).get("name")
            for t in tools
            if isinstance(t, dict)
        }
        if tool_name in existing_names:
            logger.debug("VCC recovery tool already registered — skipping")
            return True

        # Append the schema.
        tools.append(VCC_RECOVERY_SCHEMA)

        # Update valid_tool_names if it exists.
        valid_names = getattr(agent_instance, "valid_tool_names", None)
        if isinstance(valid_names, frozenset):
            agent_instance.valid_tool_names = valid_names | frozenset({tool_name})

        # Register a handler so the agent can actually execute the tool.
        # Hermes dispatches tool calls through _execute_tool_call or similar;
        # we attach a callable the hook consumer can wire up.
        agent_instance._vcc_recover_handler = functools.partial(
            _handle_vcc_recover, agent_instance, config
        )

        logger.info(
            "VCC recovery tool '%s' registered on %s",
            tool_name,
            type(agent_instance).__name__,
        )
        return True

    except Exception as exc:
        logger.warning("Failed to install VCC recovery tool: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Aggregate installer
# ---------------------------------------------------------------------------


def install_all(
    agent_instance: Any,
    config: VCCConfig | None = None,
) -> dict[str, bool]:
    """Install all VCC hooks on *agent_instance*.

    Loads configuration from Hermes ``config.yaml`` if *config* is not
    provided.  Each hook is gated by its corresponding config flag
    (``enabled``, ``enhanced_summary``, ``recovery_tool``).

    Args:
        agent_instance: A Hermes ``AIAgent`` (or compatible) instance.
        config: Optional pre-loaded configuration.  When *None*, calls
            :func:`hermes_vcc.config.load_config`.

    Returns:
        A dict mapping hook names to their installation success status,
        e.g. ``{"archive_hook": True, "recovery_tool": False}``.
    """
    if config is None:
        try:
            config = load_config()
        except Exception as exc:
            logger.warning("Failed to load VCC config — using defaults: %s", exc)
            config = VCCConfig()

    results: dict[str, bool] = {}

    if not config.enabled:
        logger.info("VCC is disabled in config — skipping all hooks")
        results["archive_hook"] = False
        results["recovery_tool"] = False
        return results

    # Archive hook — always install when VCC is enabled.
    results["archive_hook"] = install_archive_hook(agent_instance, config)

    # Summary hook — gated by enhanced_summary flag and summary_mode.
    if config.enhanced_summary:
        results["summary_hook"] = install_summary_hook(agent_instance, config)
    else:
        logger.debug("VCC enhanced summary disabled in config — skipping")
        results["summary_hook"] = False

    # Recovery tool — gated by config flag.
    if config.recovery_tool:
        results["recovery_tool"] = install_recovery_tool(agent_instance, config)
    else:
        logger.debug("VCC recovery tool disabled in config — skipping")
        results["recovery_tool"] = False

    logger.info("VCC hook installation results: %s", results)
    return results
