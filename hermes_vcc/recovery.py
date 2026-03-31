"""Post-compression recovery tool for Hermes VCC archives.

Provides a tool that the Hermes agent can invoke after context compression
to query archived VCC views and recover specific details that were compressed
out of the active context window.

Actions:
    list     - Show available archive cycles with metadata.
    overview - Return the .min.txt (brief mode) for a cycle.
    search   - Regex search over the .txt transcript with context lines.
    read     - Extract specific line ranges from the .txt transcript.

Usage as an agent tool:
    Register VCC_RECOVERY_SCHEMA in the tool list, then route calls
    through handle_vcc_recover().
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CONTENT = 50_000
"""Maximum characters returned in a single recovery response."""

CONTEXT_LINES = 3
"""Number of lines of context shown around each search match."""

# ---------------------------------------------------------------------------
# Tool schema — Hermes / OpenAI function-calling format
# ---------------------------------------------------------------------------

VCC_RECOVERY_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "vcc_recover",
        "description": (
            "Query archived VCC views to recover specific details that were "
            "compressed out of context. Use 'list' to see available archives, "
            "'overview' for a compact outline, 'search' for regex matches, or "
            "'read' for specific line ranges from the full transcript."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "overview", "search", "read"],
                    "description": "Action to perform.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "For 'search': regex pattern. "
                        "For 'read': line range like '45-80'. "
                        "Optional for other actions."
                    ),
                },
                "archive_id": {
                    "type": "string",
                    "description": (
                        "Specific archive cycle to query. "
                        "Defaults to most recent."
                    ),
                },
            },
            "required": ["action"],
        },
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_session_dir(
    archive_dir: Path, session_id: str | None
) -> Path | None:
    """Resolve the session directory inside the archive root.

    If *session_id* is provided, look for ``archive_dir / session_id``.
    Otherwise fall back to the most recently modified subdirectory
    (the "latest session" heuristic).

    Returns:
        Path to the session directory, or ``None`` if nothing exists.
    """
    if session_id:
        candidate = archive_dir / session_id
        if candidate.is_dir():
            return candidate
        logger.debug("Session dir %s does not exist", candidate)
        return None

    # Fall back: pick the subdirectory with the newest mtime.
    subdirs = [p for p in archive_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _load_manifest(session_dir: Path) -> list[dict] | None:
    """Load and return the cycles list from manifest.json.

    The manifest is expected to be a JSON object with a ``"cycles"`` key
    containing a list of cycle metadata dicts, each having at minimum
    ``"id"`` (int), ``"timestamp"`` (str), and ``"message_count"`` (int).

    Returns:
        The cycles list, or ``None`` if the manifest is missing or corrupt.
    """
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        cycles = data.get("cycles")
        if isinstance(cycles, list):
            return cycles
        # Legacy format: manifest is the list itself.
        if isinstance(data, list):
            return data
        logger.warning("Unexpected manifest structure in %s", manifest_path)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load manifest %s: %s", manifest_path, exc)
        return None


def _find_latest_cycle(session_dir: Path) -> int | None:
    """Find the highest cycle number from manifest or file scan.

    Checks the manifest first; falls back to scanning filenames for
    patterns like ``cycle_3.txt`` or ``cycle_3.min.txt``.

    Returns:
        The highest cycle number, or ``None`` if no cycles exist.
    """
    # Try manifest first.
    cycles = _load_manifest(session_dir)
    if cycles:
        ids = [c.get("id") for c in cycles if isinstance(c.get("id"), int)]
        if ids:
            return max(ids)

    # Fallback: scan for cycle_N.txt files.
    pattern = re.compile(r"cycle_(\d+)\.txt$")
    found: list[int] = []
    for p in session_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            found.append(int(m.group(1)))
    return max(found) if found else None


def _resolve_cycle_id(
    session_dir: Path, archive_id: str | None
) -> int | None:
    """Convert an archive_id string to a validated cycle number.

    If *archive_id* is ``None``, returns the latest cycle.
    Otherwise parses it as an integer and verifies the cycle exists.

    Returns:
        The cycle number, or ``None`` if resolution fails.
    """
    if archive_id is None:
        return _find_latest_cycle(session_dir)

    # Accept bare integers or "cycle_N" format.
    cleaned = archive_id.strip()
    if cleaned.lower().startswith("cycle_"):
        cleaned = cleaned[6:]
    try:
        cycle_num = int(cleaned)
    except ValueError:
        return None

    # Verify the cycle exists (at least one file).
    txt_path = session_dir / f"cycle_{cycle_num}.txt"
    min_path = session_dir / f"cycle_{cycle_num}.min.txt"
    if txt_path.exists() or min_path.exists():
        return cycle_num
    return None


def _cycle_txt_path(session_dir: Path, cycle: int) -> Path:
    """Path to the full transcript for a cycle."""
    return session_dir / f"cycle_{cycle}.txt"


def _cycle_min_path(session_dir: Path, cycle: int) -> Path:
    """Path to the brief/min transcript for a cycle."""
    return session_dir / f"cycle_{cycle}.min.txt"


def _cycle_jsonl_path(session_dir: Path, cycle: int) -> Path:
    """Path to the raw JSONL for a cycle."""
    return session_dir / f"cycle_{cycle}.jsonl"


def _cap(text: str, limit: int = MAX_CONTENT) -> str:
    """Truncate text to *limit* characters with a truncation notice."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... (truncated at {limit:,} chars)"


def _available_cycle_ids(session_dir: Path) -> list[int]:
    """Return sorted list of cycle IDs present in the session dir."""
    cycles = _load_manifest(session_dir)
    if cycles:
        ids = sorted(c.get("id") for c in cycles if isinstance(c.get("id"), int))
        if ids:
            return ids

    pattern = re.compile(r"cycle_(\d+)\.txt$")
    found = sorted(
        int(m.group(1))
        for p in session_dir.iterdir()
        if (m := pattern.match(p.name))
    )
    return found


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def _action_list(session_dir: Path) -> str:
    """List available archive cycles with metadata."""
    cycles = _load_manifest(session_dir)
    if cycles:
        lines = [f"VCC Archives ({len(cycles)} cycles):"]
        lines.append("")
        for c in cycles:
            cid = c.get("id", "?")
            ts = c.get("timestamp", "unknown")
            msg_count = c.get("message_count", "?")
            tokens = c.get("token_estimate")
            token_str = f", ~{tokens:,} tokens" if tokens else ""
            lines.append(f"  cycle_{cid}: {ts} ({msg_count} messages{token_str})")
        return "\n".join(lines)

    # Fallback: scan files.
    ids = _available_cycle_ids(session_dir)
    if not ids:
        return "No archive cycles found in this session."

    lines = [f"VCC Archives ({len(ids)} cycles, no manifest):"]
    lines.append("")
    for cid in ids:
        txt = _cycle_txt_path(session_dir, cid)
        size_str = ""
        if txt.exists():
            size_kb = txt.stat().st_size / 1024
            size_str = f" ({size_kb:.0f} KB)"
        lines.append(f"  cycle_{cid}{size_str}")
    return "\n".join(lines)


def _action_overview(session_dir: Path, cycle: int) -> str:
    """Return the .min.txt (brief mode) for the specified cycle."""
    min_path = _cycle_min_path(session_dir, cycle)
    if min_path.exists():
        content = min_path.read_text(encoding="utf-8")
        header = f"=== Overview: cycle_{cycle} (min.txt) ===\n\n"
        return _cap(header + content)

    # Fallback: if .min.txt is missing, try .txt with a note.
    txt_path = _cycle_txt_path(session_dir, cycle)
    if txt_path.exists():
        content = txt_path.read_text(encoding="utf-8")
        header = (
            f"=== Overview: cycle_{cycle} (full transcript, "
            f".min.txt not available) ===\n\n"
        )
        return _cap(header + content)

    return f"No transcript files found for cycle_{cycle}."


def _action_search(
    session_dir: Path, cycle: int, pattern_str: str
) -> str:
    """Regex search over the full transcript with context lines."""
    try:
        pattern = re.compile(pattern_str, re.IGNORECASE)
    except re.error as exc:
        return f"Invalid regex pattern: {exc}"

    txt_path = _cycle_txt_path(session_dir, cycle)
    if not txt_path.exists():
        return f"No transcript found for cycle_{cycle}."

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    total_lines = len(lines)

    # Collect match indices.
    match_indices: list[int] = []
    for i, line in enumerate(lines):
        if pattern.search(line):
            match_indices.append(i)

    if not match_indices:
        return (
            f"No matches for /{pattern_str}/ in cycle_{cycle} "
            f"({total_lines} lines searched)."
        )

    # Build output with context windows, merging overlapping ranges.
    output_parts: list[str] = [
        f"Search: /{pattern_str}/ in cycle_{cycle} "
        f"({len(match_indices)} matches in {total_lines} lines)",
        "",
    ]

    # Merge overlapping context windows into contiguous ranges.
    ranges: list[tuple[int, int]] = []
    for idx in match_indices:
        start = max(0, idx - CONTEXT_LINES)
        end = min(total_lines - 1, idx + CONTEXT_LINES)
        if ranges and start <= ranges[-1][1] + 1:
            # Extend the previous range.
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))

    for range_start, range_end in ranges:
        output_parts.append("---")
        for i in range(range_start, range_end + 1):
            # 1-indexed line numbers; highlight matching lines.
            line_num = i + 1
            prefix = ">>" if i in match_indices else "  "
            output_parts.append(f"{prefix} {line_num:>5}: {lines[i]}")

    result = "\n".join(output_parts)
    return _cap(result)


def _action_read(session_dir: Path, cycle: int, range_str: str) -> str:
    """Extract specific line ranges from the full transcript (1-indexed)."""
    # Parse range: "start-end" or just "start".
    range_str = range_str.strip()
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", range_str)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
    elif range_str.isdigit():
        start = end = int(range_str)
    else:
        return (
            f"Invalid line range: '{range_str}'. "
            f"Expected format: '45-80' or '45'."
        )

    if start < 1:
        return "Line numbers are 1-indexed. Start must be >= 1."
    if end < start:
        return f"Invalid range: end ({end}) < start ({start})."

    txt_path = _cycle_txt_path(session_dir, cycle)
    if not txt_path.exists():
        return f"No transcript found for cycle_{cycle}."

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    total_lines = len(lines)

    if start > total_lines:
        return (
            f"Start line {start} exceeds transcript length "
            f"({total_lines} lines)."
        )

    # Clamp end to actual file length.
    end = min(end, total_lines)

    header = (
        f"=== cycle_{cycle} lines {start}-{end} "
        f"(of {total_lines}) ===\n"
    )
    extracted = []
    for i in range(start - 1, end):  # Convert to 0-indexed.
        extracted.append(f"{i + 1:>5}: {lines[i]}")

    return _cap(header + "\n".join(extracted))


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

DEFAULT_ARCHIVE_DIR = Path.home() / ".hermes" / "vcc_archives"


def handle_vcc_recover(
    action: str,
    query: str | None = None,
    archive_id: str | None = None,
    archive_dir: Path | None = None,
    session_id: str | None = None,
) -> str:
    """Handle a vcc_recover tool invocation from the Hermes agent.

    Args:
        action: One of "list", "overview", "search", "read".
        query: For "search" -- regex pattern.  For "read" -- line range
            like "45-80".  Optional for "list" and "overview".
        archive_id: Specific archive cycle to target (e.g. "3" or
            "cycle_3").  Defaults to the most recent cycle.
        archive_dir: Root directory containing session archive folders.
            Defaults to ``~/.hermes/vcc_archives``.
        session_id: Subdirectory name for a specific session.  Defaults
            to the most recently modified session.

    Returns:
        Human-readable string with the requested data or an error
        message.  Always capped at MAX_CONTENT characters.
    """
    if archive_dir is None:
        archive_dir = DEFAULT_ARCHIVE_DIR

    if not archive_dir.is_dir():
        return (
            f"No VCC archive directory found at {archive_dir}. "
            f"Archives are created automatically during context compression."
        )

    # Resolve session directory.
    session_dir = _get_session_dir(archive_dir, session_id)
    if session_dir is None:
        if session_id:
            # List available sessions to help the caller.
            available = sorted(
                p.name for p in archive_dir.iterdir() if p.is_dir()
            )
            if available:
                return (
                    f"Session '{session_id}' not found. "
                    f"Available sessions: {', '.join(available)}"
                )
            return f"Session '{session_id}' not found and no sessions exist."
        return "No session directories found in the archive."

    # --- list ---
    if action == "list":
        return _action_list(session_dir)

    # For remaining actions, resolve cycle.
    cycle = _resolve_cycle_id(session_dir, archive_id)
    if cycle is None:
        available_ids = _available_cycle_ids(session_dir)
        if not available_ids:
            return "No archive cycles found in this session."
        if archive_id is not None:
            return (
                f"Archive '{archive_id}' not found. "
                f"Available cycles: {', '.join(f'cycle_{c}' for c in available_ids)}"
            )
        return "Could not determine the latest cycle."

    # --- overview ---
    if action == "overview":
        return _action_overview(session_dir, cycle)

    # --- search ---
    if action == "search":
        if not query:
            return "The 'search' action requires a 'query' parameter with a regex pattern."
        return _action_search(session_dir, cycle, query)

    # --- read ---
    if action == "read":
        if not query:
            return "The 'read' action requires a 'query' parameter with a line range (e.g. '45-80')."
        return _action_read(session_dir, cycle, query)

    return (
        f"Unknown action '{action}'. "
        f"Valid actions: list, overview, search, read."
    )
