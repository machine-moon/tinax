"""Isolated stream loggers that do not mutate the logging registry or root logger."""

import logging
from typing import TextIO


def make_stream_logger(
    name: str,
    *,
    level: int,
    stream: TextIO,
    format_string: str,
) -> logging.Logger:
    """Construct an unregistered non-propagating logger with one explicit stream handler.

    The logger is not added to the global registry, does not propagate to the root
    logger, and leaves ``logging.basicConfig`` and root handlers untouched.

    Args:
        name: Logger name. Must be non-empty.
        level: Logging level as an integer (booleans are rejected).
        stream: Writable text stream providing callable ``write`` and ``flush``.
        format_string: ``logging.Formatter`` format string, validated on creation.

    Returns:
        A configured ``logging.Logger`` with a single stream handler.

    Raises:
        TypeError: If ``name`` or ``format_string`` is not a string, ``level`` is not
            an integer (booleans are rejected), or ``stream`` lacks callable ``write``
            and ``flush`` methods.
        ValueError: If ``name`` is empty or ``format_string`` is not a valid format.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name:
        raise ValueError("name must not be empty")
    if not isinstance(level, int) or isinstance(level, bool):
        raise TypeError("level must be an integer and not a boolean")
    if not callable(getattr(stream, "write", None)) or not callable(getattr(stream, "flush", None)):
        raise TypeError("stream must provide callable write and flush methods")
    if not isinstance(format_string, str):
        raise TypeError("format_string must be a string")

    logger = logging.Logger(name, level=level)
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(format_string, validate=True))
    logger.addHandler(handler)
    return logger
