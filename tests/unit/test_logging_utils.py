"""Unit tests for structured JSON logging (CLAUDE.md conventions, PRD NFR-8)."""

import io
import json

import pytest

from shared.logging_utils import METRIC_NAMESPACE, get_logger


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


@pytest.fixture
def stream():
    return io.StringIO()


def test_each_call_emits_exactly_one_json_object(stream):
    log = get_logger("test.single", stream=stream)
    log.info("ingest.started")
    log.info("ingest.completed")

    assert len(_lines(stream)) == 2


def test_context_fields_are_top_level_keys(stream):
    """Insights queries filter on these directly, so they must not be nested."""
    log = get_logger("test.fields", stream=stream)
    log.info("query.completed", doc_id="d_1", latency_ms=1180, top_score=0.83, tokens_out=210)

    (line,) = _lines(stream)
    assert line["event"] == "query.completed"
    assert line["level"] == "INFO"
    assert line["doc_id"] == "d_1"
    assert line["latency_ms"] == 1180
    assert line["top_score"] == 0.83
    assert line["tokens_out"] == 210


def test_warning_and_error_levels_are_recorded(stream):
    log = get_logger("test.levels", stream=stream)
    log.warning("ingest.retrying", attempt=2)
    log.error("ingest.failed", reason="EMPTY_DOCUMENT")

    warning, error = _lines(stream)
    assert warning["level"] == "WARNING"
    assert error["level"] == "ERROR"
    assert error["reason"] == "EMPTY_DOCUMENT"


def test_exception_details_are_attached_on_request(stream):
    log = get_logger("test.exc", stream=stream)
    try:
        raise ValueError("boom")
    except ValueError:
        log.error("ingest.failed", exc_info=True, doc_id="d_1")

    (line,) = _lines(stream)
    assert "ValueError: boom" in line["error"]
    assert line["doc_id"] == "d_1"


def test_non_serializable_values_do_not_break_logging(stream):
    log = get_logger("test.serialize", stream=stream)
    log.info("ingest.completed", vector=object())

    (line,) = _lines(stream)
    assert "object" in line["vector"]


def test_unicode_is_preserved_rather_than_escaped(stream):
    log = get_logger("test.unicode", stream=stream)
    log.info("ingest.started", filename="rapport-financé.pdf")

    assert _lines(stream)[0]["filename"] == "rapport-financé.pdf"


def test_log_level_is_configurable_through_the_environment(monkeypatch, stream):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    log = get_logger("test.level.env", stream=stream)
    log.info("ignored.event")
    log.error("recorded.event")

    assert [line["event"] for line in _lines(stream)] == ["recorded.event"]


def test_handlers_are_not_duplicated_across_calls(stream):
    """A second get_logger for the same name must not double every line."""
    get_logger("test.dupe", stream=stream)
    log = get_logger("test.dupe", stream=stream)
    log.info("once")

    assert len(_lines(stream)) == 1


# --------------------------------------------------------------------------- #
# Embedded Metric Format (T3.2, ARCHITECTURE.md §8)
# --------------------------------------------------------------------------- #


def test_metrics_emit_one_embedded_metric_format_line(stream):
    log = get_logger("test.emf", stream=stream)
    log.metrics("query.answered", {"TokensIn": 1450, "TokensOut": 210})

    (line,) = _lines(stream)
    assert "_aws" in line
    assert line["event"] == "query.answered"


def test_metric_values_are_top_level_siblings_of_the_aws_block(stream):
    """CloudWatch reads the values from the same object, not a nested one."""
    log = get_logger("test.emf.values", stream=stream)
    log.metrics("query.answered", {"TokensIn": 1450, "TokensOut": 210})

    (line,) = _lines(stream)
    assert line["TokensIn"] == 1450
    assert line["TokensOut"] == 210


def test_declared_metrics_match_the_values_present(stream):
    log = get_logger("test.emf.declared", stream=stream)
    log.metrics("query.answered", {"TokensIn": 1, "TokensOut": 2})

    (line,) = _lines(stream)
    (directive,) = line["_aws"]["CloudWatchMetrics"]
    assert {metric["Name"] for metric in directive["Metrics"]} == {"TokensIn", "TokensOut"}
    assert directive["Namespace"] == METRIC_NAMESPACE


def test_dimensions_are_declared_and_present(stream):
    log = get_logger("test.emf.dimensions", stream=stream)
    log.metrics("query.answered", {"TokensIn": 1}, dimensions={"Service": "query"})

    (line,) = _lines(stream)
    (directive,) = line["_aws"]["CloudWatchMetrics"]
    assert directive["Dimensions"] == [["Service"]]
    assert line["Service"] == "query"


def test_units_default_to_count_and_can_be_overridden(stream):
    log = get_logger("test.emf.units", stream=stream)
    log.metrics(
        "query.answered",
        {"TokensIn": 1, "QueryLatency": 2870},
        units={"QueryLatency": "Milliseconds"},
    )

    (line,) = _lines(stream)
    (directive,) = line["_aws"]["CloudWatchMetrics"]
    units = {metric["Name"]: metric["Unit"] for metric in directive["Metrics"]}
    assert units == {"TokensIn": "Count", "QueryLatency": "Milliseconds"}


def test_extra_fields_are_logged_but_not_published_as_metrics(stream):
    log = get_logger("test.emf.fields", stream=stream)
    log.metrics("query.answered", {"TokensIn": 1}, citations=3)

    (line,) = _lines(stream)
    (directive,) = line["_aws"]["CloudWatchMetrics"]
    assert line["citations"] == 3
    assert "citations" not in {metric["Name"] for metric in directive["Metrics"]}


def test_no_metrics_writes_nothing(stream):
    log = get_logger("test.emf.empty", stream=stream)

    assert log.metrics("query.answered", {}) == {}
    assert _lines(stream) == []


def test_the_timestamp_is_in_milliseconds(stream):
    log = get_logger("test.emf.timestamp", stream=stream)
    log.metrics("query.answered", {"TokensIn": 1})

    (line,) = _lines(stream)
    # Seconds would be ~1.7e9; milliseconds ~1.7e12.
    assert line["_aws"]["Timestamp"] > 1_000_000_000_000
