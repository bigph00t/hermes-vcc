"""VCC-enhanced compressor: uses ``.min.txt`` as structural backbone for LLM summaries.

This module is intentionally decoupled from hermes-agent.  It provides
wrapper functions that can augment *any* summary generation callable with
VCC structural context, making the integration non-invasive and testable
in isolation.

All public functions swallow exceptions internally and fall back to the
original summary path so the compression pipeline is never broken by VCC
failures.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

from hermes_vcc.utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _messages_to_serialized_text(messages: list[dict[str, Any]]) -> str:
    """Produce a plain-text serialisation of messages for the LLM summary.

    This is the fallback representation when VCC is unavailable.  It mirrors
    the style most compressors expect: role-prefixed turns separated by
    blank lines.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""

        # Handle tool_calls on assistant messages.
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_lines: list[str] = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                tc_lines.append(f"  [tool_call] {name}({args[:200]})")
            if content:
                tc_lines.insert(0, content)
            content = "\n".join(tc_lines)

        if content:
            parts.append(f"[{role}]\n{content}")

    return "\n\n".join(parts)


def _write_temp_jsonl(messages: list[dict[str, Any]], work_dir: Path) -> Path:
    """Convert messages to VCC JSONL and write to a temp file in *work_dir*.

    Returns the path to the written JSONL file.
    """
    from hermes_vcc.adapter import convert_conversation, records_to_jsonl

    records = convert_conversation(messages)
    jsonl_text = records_to_jsonl(records)

    jsonl_path = work_dir / "enhanced_summary_input.jsonl"
    jsonl_path.write_text(jsonl_text, encoding="utf-8")
    return jsonl_path


def _read_min_txt(output_dir: Path, jsonl_stem: str) -> str | None:
    """Read the ``.min.txt`` file produced by VCC compile_pass.

    VCC names outputs based on the input stem, so we look for
    ``<stem>.min.txt`` in *output_dir*.

    Returns the content string, or *None* if the file does not exist or
    is empty.
    """
    min_path = output_dir / f"{jsonl_stem}.min.txt"
    if not min_path.is_file():
        return None
    content = min_path.read_text(encoding="utf-8").strip()
    return content or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_to_brief(messages: list[dict[str, Any]]) -> str | None:
    """Convert messages to VCC format, compile, and return ``.min.txt`` content.

    This is the low-level entry point for obtaining the structural skeleton
    of a conversation without involving any LLM call.

    Args:
        messages: Hermes OpenAI-format message list.

    Returns:
        The ``.min.txt`` content as a string, or *None* on any failure
        (missing VCC, compile error, empty output, etc.).
    """
    if not messages:
        return None

    try:
        vcc = None
        try:
            from hermes_vcc.utils import import_vcc
            vcc = import_vcc()
        except Exception as exc:
            logger.debug("VCC import failed in compile_to_brief: %s", exc)
            return None

        with tempfile.TemporaryDirectory(prefix="hermes_vcc_brief_") as tmpdir:
            work_dir = Path(tmpdir)
            jsonl_path = _write_temp_jsonl(messages, work_dir)

            vcc.compile_pass(
                str(jsonl_path),
                str(work_dir),
                truncate=128,
                truncate_user=256,
                quiet=True,
            )

            return _read_min_txt(work_dir, jsonl_path.stem)

    except Exception as exc:
        logger.warning("compile_to_brief failed: %s", exc)
        return None


def generate_vcc_enhanced_summary(
    messages_to_summarize: list[dict[str, Any]],
    original_summary_fn: Callable[[str], str],
    archive_dir: Path | None = None,
) -> str:
    """Generate a VCC-enhanced summary.

    Pipeline:
        1. Convert *messages_to_summarize* to VCC format.
        2. Write temporary JSONL and compile with VCC.
        3. Read the ``.min.txt`` as structural skeleton.
        4. Call *original_summary_fn* with enriched input that combines
           the skeleton and the serialized turns.
        5. Return the result.

    On any VCC failure the function falls back to calling
    *original_summary_fn* with just the serialized turns — it never
    raises.

    Args:
        messages_to_summarize: Hermes OpenAI-format messages to compress.
        original_summary_fn: Callable that takes a single string (the
            serialized conversation text) and returns a summary string.
        archive_dir: Optional directory for persisting the compiled VCC
            output.  When *None* a temporary directory is used and
            discarded after compilation.

    Returns:
        The summary string produced by *original_summary_fn*.
    """
    # Serialise turns as the fallback / baseline input.
    turns_text = _messages_to_serialized_text(messages_to_summarize)

    if not messages_to_summarize:
        return original_summary_fn(turns_text)

    skeleton: str | None = None

    try:
        from hermes_vcc.utils import import_vcc
        vcc = import_vcc()

        # Decide where to compile: persistent archive_dir or temp dir.
        if archive_dir is not None:
            work_dir = ensure_dir(archive_dir)
            jsonl_path = _write_temp_jsonl(messages_to_summarize, work_dir)

            vcc.compile_pass(
                str(jsonl_path),
                str(work_dir),
                truncate=128,
                truncate_user=256,
                quiet=True,
            )

            skeleton = _read_min_txt(work_dir, jsonl_path.stem)
        else:
            with tempfile.TemporaryDirectory(prefix="hermes_vcc_enh_") as tmpdir:
                work_dir = Path(tmpdir)
                jsonl_path = _write_temp_jsonl(messages_to_summarize, work_dir)

                vcc.compile_pass(
                    str(jsonl_path),
                    str(work_dir),
                    truncate=128,
                    truncate_user=256,
                    quiet=True,
                )

                skeleton = _read_min_txt(work_dir, jsonl_path.stem)

    except Exception as exc:
        logger.warning(
            "VCC compilation failed during enhanced summary — "
            "falling back to plain summary: %s",
            exc,
        )

    if skeleton:
        enriched_input = (
            "=== STRUCTURAL SKELETON (VCC .min.txt) ===\n"
            f"{skeleton}\n\n"
            "=== FULL CONVERSATION TURNS ===\n"
            f"{turns_text}"
        )
        try:
            return original_summary_fn(enriched_input)
        except Exception as exc:
            logger.warning(
                "original_summary_fn failed with enriched input — "
                "retrying with plain turns: %s",
                exc,
            )

    # Fallback: plain turns only.
    return original_summary_fn(turns_text)
