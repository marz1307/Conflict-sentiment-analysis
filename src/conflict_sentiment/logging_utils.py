"""Logging helpers used across the package."""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger once per process.

    Args:
        level: Logging level (default ``INFO``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Typically ``__name__`` of the caller.

    Returns:
        Configured :class:`logging.Logger`.
    """
    configure_root_logger()
    return logging.getLogger(name)
