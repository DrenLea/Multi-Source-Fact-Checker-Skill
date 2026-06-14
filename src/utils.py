"""
Utility module for the Multi-Source Fact Checker Skill.

Provides shared infrastructure: logging setup, YAML configuration loading,
and helper functions used across all modules.
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Path resolution – resolve relative to the package root (skill-verify-fact/)
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PACKAGE_ROOT


# ---------------------------------------------------------------------------
# Logging setup (loguru)
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """
    Configure loguru with a sensible default format.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )
    logger.info(f"Logging initialized at level {level}")


# ---------------------------------------------------------------------------
# Environment & config loading
# ---------------------------------------------------------------------------

def load_env(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to .env file. Defaults to <project_root>/.env .
    """
    if env_path is None:
        env_path = str(_PACKAGE_ROOT / ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path} — using existing env vars")


def load_yaml_config(filename: str, config_dir: Optional[str] = None) -> dict:
    """
    Load and parse a YAML configuration file.

    Args:
        filename: Name of the YAML file (e.g. 'weights.yaml').
        config_dir: Directory containing config files. Defaults to <root>/config/.

    Returns:
        Parsed configuration dictionary. Returns empty dict if file not found.
    """
    if config_dir is None:
        config_dir = str(_PACKAGE_ROOT / "config")
    filepath = os.path.join(config_dir, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            logger.info(f"Loaded config: {filepath}")
            return data or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found: {filepath}")
        return {}
    except yaml.YAMLError as exc:
        logger.error(f"Failed to parse YAML config {filepath}: {exc}")
        return {}


def resolve_env_vars(config: dict) -> dict:
    """
    Recursively resolve ${ENV_VAR} placeholders in a config dictionary
    using os.environ values.

    Args:
        config: A dictionary potentially containing '${...}' values.

    Returns:
        Dictionary with environment variable placeholders replaced.
    """
    import re

    # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
    _ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")

    def _resolve(value: Any) -> Any:
        if isinstance(value, str):
            def _replacer(match: re.Match) -> str:
                var_name = match.group(1)
                default = match.group(2)
                return os.environ.get(var_name, default if default is not None else match.group(0))

            return _ENV_PATTERN.sub(_replacer, value)
        elif isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve(item) for item in value]
        return value

    return _resolve(config)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# Auto-initialize logging with default level on import
_default_level = os.environ.get("LOG_LEVEL", "INFO")
setup_logging(level=_default_level)
