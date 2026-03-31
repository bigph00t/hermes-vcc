"""Configuration for hermes-vcc.

Reads the ``compression.vcc`` section from Hermes's ``config.yaml`` and
exposes it as a typed :class:`VCCConfig` dataclass.  Missing keys fall
back to safe defaults so the module works out of the box without any
configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"


@dataclass
class VCCConfig:
    """Typed VCC configuration with safe defaults."""

    enabled: bool = True
    archive_dir: Path = field(
        default_factory=lambda: Path.home() / ".hermes" / "vcc_archives"
    )
    enhanced_summary: bool = True
    recovery_tool: bool = True
    retain_archives: int = 10

    def __post_init__(self) -> None:
        # Coerce string paths (e.g. from YAML) to Path objects.
        if isinstance(self.archive_dir, str):
            self.archive_dir = Path(self.archive_dir)


def _parse_section(raw: dict[str, Any]) -> VCCConfig:
    """Build a :class:`VCCConfig` from a raw dict, ignoring unknown keys."""
    kwargs: dict[str, Any] = {}

    if "enabled" in raw:
        kwargs["enabled"] = bool(raw["enabled"])

    if "archive_dir" in raw:
        kwargs["archive_dir"] = Path(str(raw["archive_dir"]))

    if "enhanced_summary" in raw:
        kwargs["enhanced_summary"] = bool(raw["enhanced_summary"])

    if "recovery_tool" in raw:
        kwargs["recovery_tool"] = bool(raw["recovery_tool"])

    if "retain_archives" in raw:
        try:
            kwargs["retain_archives"] = int(raw["retain_archives"])
        except (TypeError, ValueError):
            pass

    return VCCConfig(**kwargs)


def load_config(config_path: Path | None = None) -> VCCConfig:
    """Load VCC config from Hermes ``config.yaml`` under ``compression.vcc``.

    Falls back to defaults if the file is missing, unparseable, or the
    ``compression.vcc`` section does not exist.

    Args:
        config_path: Explicit path to Hermes config.yaml.  When *None*,
            uses ``~/.hermes/config.yaml``.

    Returns:
        A fully-populated :class:`VCCConfig` instance.
    """
    path = config_path or _DEFAULT_HERMES_CONFIG

    if not path.is_file():
        logger.debug("Hermes config not found at %s — using VCC defaults", path)
        return VCCConfig()

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "PyYAML not installed — cannot read %s, using VCC defaults", path
        )
        return VCCConfig()

    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s — using VCC defaults", path, exc)
        return VCCConfig()

    if not isinstance(data, dict):
        logger.debug("Hermes config at %s is not a mapping — using VCC defaults", path)
        return VCCConfig()

    compression = data.get("compression")
    if not isinstance(compression, dict):
        logger.debug("No 'compression' section in %s — using VCC defaults", path)
        return VCCConfig()

    vcc_section = compression.get("vcc")
    if not isinstance(vcc_section, dict):
        logger.debug(
            "No 'compression.vcc' section in %s — using VCC defaults", path
        )
        return VCCConfig()

    return _parse_section(vcc_section)
