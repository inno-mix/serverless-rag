"""Structured single-line JSON logging.

Every log line is one JSON object, so CloudWatch Logs Insights can query fields
directly (PRD NFR-8). Handlers never call `print`; they obtain a
:class:`StructuredLogger` and pass context as keyword arguments::

    log = get_logger(__name__)
    log.info("ingest.completed", doc_id=doc_id, chunk_count=42, latency_ms=1180)

EMF custom metrics and the CloudWatch dashboard are layered on top of this in
T3.2; this module provides the log format those depend on.
"""

import json
import logging
import os
import sys
from typing import Any, TextIO

#: Log level, overridable per-Lambda through the environment.
DEFAULT_LOG_LEVEL = "INFO"


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize `record` to a one-line JSON string.

        Args:
            record: The record to render.

        Returns:
            A JSON object carrying the level, event name, and any context fields
            supplied by the caller.
        """
        payload: dict[str, Any] = {
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        payload.update(getattr(record, "context", {}))
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class StructuredLogger:
    """Thin wrapper that forces every log line through :class:`JsonFormatter`."""

    def __init__(self, name: str, *, stream: TextIO | None = None) -> None:
        """Create a logger writing single-line JSON.

        Args:
            name: Logger name, normally the calling module's `__name__`.
            stream: Destination stream. Defaults to stdout, which is what the
                Lambda runtime forwards to CloudWatch Logs; tests pass their own.
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper())
        # Own the output rather than inheriting the Lambda runtime's root handler,
        # which would emit a second, unstructured copy of every line.
        self._logger.handlers.clear()
        self._logger.propagate = False
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(JsonFormatter())
        self._logger.addHandler(handler)

    def info(self, event: str, **fields: Any) -> None:
        """Log an informational event with structured context."""
        self._logger.info(event, extra={"context": fields})

    def warning(self, event: str, **fields: Any) -> None:
        """Log a warning event with structured context."""
        self._logger.warning(event, extra={"context": fields})

    def error(self, event: str, *, exc_info: bool = False, **fields: Any) -> None:
        """Log an error event, optionally attaching the active exception."""
        self._logger.error(event, exc_info=exc_info, extra={"context": fields})


def get_logger(name: str, *, stream: TextIO | None = None) -> StructuredLogger:
    """Build a structured logger for `name`.

    Args:
        name: Logger name, normally the calling module's `__name__`.
        stream: Optional destination stream, used by tests.

    Returns:
        A logger that emits one JSON object per line.
    """
    return StructuredLogger(name, stream=stream)
