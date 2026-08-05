"""Structured single-line JSON logging.

Every log line is one JSON object, so CloudWatch Logs Insights can query fields
directly (PRD NFR-8). Handlers never call `print`; they obtain a
:class:`StructuredLogger` and pass context as keyword arguments::

    log = get_logger(__name__)
    log.info("ingest.completed", doc_id=doc_id, chunk_count=42, latency_ms=1180)

:func:`emit_metrics` writes CloudWatch Embedded Metric Format on the same
stream (T3.2). EMF is why token usage needs no `PutMetricData` call and no extra
IAM permission: CloudWatch extracts the metrics from the log line itself, so the
Lambda pays nothing beyond the log it was already writing.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from typing import Any, TextIO

#: Log level, overridable per-Lambda through the environment.
DEFAULT_LOG_LEVEL = "INFO"

#: CloudWatch namespace for this project's custom metrics (ARCHITECTURE.md §8).
METRIC_NAMESPACE = "ServerlessRag"


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

    def metrics(
        self,
        event: str,
        values: Mapping[str, float],
        *,
        dimensions: Mapping[str, str] | None = None,
        units: Mapping[str, str] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Publish custom metrics by writing one Embedded Metric Format line.

        CloudWatch extracts the metrics from the log event itself, so this needs
        no `PutMetricData` call and no extra IAM permission — the Lambda pays
        only for a log line it was already going to write (ARCHITECTURE.md §8).

        Args:
            event: Event name, for querying the line in Logs Insights.
            values: Metric name to value. Empty publishes nothing.
            dimensions: Dimension name to value, as a single dimension set.
            units: CloudWatch unit per metric, defaulting to `Count`.
            **fields: Extra context fields, logged but not published as metrics.

        Returns:
            The payload written, so callers and tests can assert on it.
        """
        if not values:
            return {}

        payload = build_metric_payload(values, dimensions=dimensions, units=units)
        self._logger.info(event, extra={"context": {**payload, **fields}})
        return payload


def build_metric_payload(
    metrics: Mapping[str, float],
    *,
    dimensions: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    namespace: str = METRIC_NAMESPACE,
) -> dict[str, Any]:
    """Build the Embedded Metric Format body for a set of metrics.

    Args:
        metrics: Metric name to value.
        dimensions: Dimension name to value, applied as a single dimension set.
            Keep the cardinality low: every distinct combination of dimension
            values becomes its own metric, and each one is billed.
        units: CloudWatch unit per metric, defaulting to `Count`.
        namespace: CloudWatch namespace to publish under.

    Returns:
        A payload whose `_aws` block tells CloudWatch which of the sibling
        fields to read as metrics.
    """
    dimensions = dimensions or {}
    units = units or {}
    return {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [list(dimensions)],
                    "Metrics": [
                        {"Name": name, "Unit": units.get(name, "Count")} for name in metrics
                    ],
                }
            ],
        },
        **dimensions,
        **dict(metrics),
    }


def get_logger(name: str, *, stream: TextIO | None = None) -> StructuredLogger:
    """Build a structured logger for `name`.

    Args:
        name: Logger name, normally the calling module's `__name__`.
        stream: Optional destination stream, used by tests.

    Returns:
        A logger that emits one JSON object per line.
    """
    return StructuredLogger(name, stream=stream)
