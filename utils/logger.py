"""
Structured logger using Loguru with JSON output in production.
"""
from __future__ import annotations

import sys
from functools import lru_cache

from loguru import logger as _loguru_logger

from config import get_settings


@lru_cache(maxsize=None)
def get_logger(name: str):
    """Return a named loguru logger bound with module context."""
    settings = get_settings()
    _loguru_logger.remove()
    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[module]} | {message}"
        if settings.DEBUG
        else "{time} {level} {extra[module]} {message}"
    )
    _loguru_logger.add(
        sys.stderr,
        level="DEBUG" if settings.DEBUG else "INFO",
        format=fmt,
        colorize=settings.DEBUG,
    )
    return _loguru_logger.bind(module=name)
