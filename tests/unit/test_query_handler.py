"""Unit tests for the query pipeline (T2.2 acceptance criteria, TESTING.md §2).

Bedrock is stubbed on both sides — embedding and generation — so the retrieval
maths, the similarity floor, and the response contract are all exercised with
no AWS and no non-determinism. DynamoDB is moto-backed via `conftest.py`.

The single most important test here is
`test_below_floor_never_calls_the_generation_model`: ADR-003 is a safety
property, and a regression in it would silently turn the system into a
hallucination machine.
"""

import io
import json

import numpy as np
import pytest

from query import handler as query_handler
from query.handler import (
    MAX_QUESTION_CHARS,
    MAX_TOP_K,
    MIN_QUESTION_CHARS,
    NOT_FOUND_ANSWER,
    InvalidQuestionError,
    InvalidTopKError,
    answer_question,
    build_context,
    build_user_prompt,
    extract_citations,
    handler,
    parse_request,
)
from shared.embeddings import Generation, ModelUnavailableError
from shared.logging_utils import get_logger
from shared.storage import (
    ChunkRecord,
    DocumentStatus,
    StoredChunk,
    put_chunks,
    put_document_meta,
    update_status,
)
from shared.vector_search import ScoredChunk

FLOOR = 0.35
MAX_TOKENS = 1024


class _Embedder:
    """Returns a fixed query vector and records every call."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.array(vector, dtype=np.float32)
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[np.ndarray]:
        self.calls.append(texts)
        return [self.vector]


class _Generator:
    """Returns a queued completion and records every call."""

    def __init__(self, *completions: Generation) -> None:
        self._completions = list(completions)
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> Generation:
        self.calls.append(kwargs)
        if self._completions:
            return self._completions.pop(0)
        return Generation(text="unused", tokens_in=0, tokens_out=0)

    @property
    def called(self) -> bool:
        return bool(self.calls)


def _seed(table, doc_id: str, filename: str, *vectors: list[float]) -> None:
    """Write one document and its chunks, exactly as ingestion would."""
    put_document_meta(
        table,
        doc_id=doc_id,
        filename=filename,
        s3_key=f"uploads/{doc_id}/{filename}",
        uploaded_at=f"2026-08-04T10:0{len(filename) % 10}:00Z",
    )
    put_chunks(
        table,
        doc_id,
        [
            ChunkRecord(
                index=index,
                text=f"{filename} passage {index}",
                token_count=10,
                vector=np.array(vector, dtype=np.float32),
            )
            for index, vector in enumerate(vectors)
        ],
    )
    update_status(table, doc_id, DocumentStatus.READY, chunk_count=len(vectors))


def _scored(doc_id: str, chunk_id: str, text: str, score: float) -> ScoredChunk[StoredChunk]:
    return ScoredChunk(
        chunk=StoredChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=text,
            token_count=5,
            vector=np.array([1.0, 0.0], dtype=np.float32),
        ),
        score=score,
    )


# --------------------------------------------------------------------------- #
# Request validation (API_SPEC.md §4)
# --------------------------------------------------------------------------- #


def test_question_and_default_top_k_are_returned():
    assert parse_request({"question": "What was revenue?"}) == ("What was revenue?", 5)


def test_question_is_trimmed():
    assert parse_request({"question": "  spaced  "})[0] == "spaced"


@pytest.mark.parametrize("body", [{}, {"question": None}, {"question": 42}, {"question": ["a"]}])
def test_missing_or_non_string_question_is_rejected(body):
    with pytest.raises(InvalidQuestionError):
        parse_request(body)


@pytest.mark.parametrize("length", [0, 1, MIN_QUESTION_CHARS - 1, MAX_QUESTION_CHARS + 1])
def test_question_length_bounds_are_enforced(length):
    with pytest.raises(InvalidQuestionError):
        parse_request({"question": "a" * length})


@pytest.mark.parametrize("length", [MIN_QUESTION_CHARS, MAX_QUESTION_CHARS])
def test_question_length_boundaries_are_accepted(length):
    assert parse_request({"question": "a" * length})[0] == "a" * length


def test_whitespace_only_question_is_rejected():
    with pytest.raises(InvalidQuestionError):
        parse_request({"question": "        "})


@pytest.mark.parametrize("top_k", [1, 5, MAX_TOP_K])
def test_valid_top_k_is_accepted(top_k):
    assert parse_request({"question": "why?", "top_k": top_k})[1] == top_k


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1, 999])
def test_out_of_range_top_k_is_rejected(top_k):
    with pytest.raises(InvalidTopKError):
        parse_request({"question": "why?", "top_k": top_k})


@pytest.mark.parametrize("top_k", ["5", 5.0, [5], True])
def test_non_integer_top_k_is_rejected(top_k):
    """`True` is an int subclass, and must not be read as a retrieval depth."""
    with pytest.raises(InvalidTopKError):
        parse_request({"question": "why?", "top_k": top_k})


# --------------------------------------------------------------------------- #
# Prompt construction (ARCHITECTURE.md §5)
# --------------------------------------------------------------------------- #


def test_context_uses_the_documented_chunk_element():
    context = build_context([_scored("d_1", "chunk_00003", "the text", 0.9)], {"d_1": "report.pdf"})
    assert context == '<chunk id="chunk_00003" source="report.pdf">\nthe text\n</chunk>'


def test_context_separates_chunks_with_a_blank_line():
    context = build_context(
        [_scored("d_1", "chunk_00000", "a", 0.9), _scored("d_1", "chunk_00001", "b", 0.8)],
        {"d_1": "report.pdf"},
    )
    assert context.count("<chunk ") == 2
    assert "</chunk>\n\n<chunk " in context


def test_context_falls_back_to_the_doc_id_when_the_filename_is_unknown():
    context = build_context([_scored("d_9", "chunk_00000", "x", 0.9)], {})
    assert 'source="d_9"' in context


def test_user_prompt_carries_both_context_and_question():
    prompt = build_user_prompt("What was revenue?", '<chunk id="c">text</chunk>')
    assert "What was revenue?" in prompt
    assert '<chunk id="c">text</chunk>' in prompt


def test_system_prompt_states_the_three_grounding_rules():
    """ARCHITECTURE.md §5 requires context-only answering, citations, NOT_FOUND."""
    prompt = query_handler.SYSTEM_PROMPT
    assert "ONLY" in prompt
    assert "[chunk_" in prompt
    assert "NOT_FOUND" in prompt


# --------------------------------------------------------------------------- #
# Citation extraction
# --------------------------------------------------------------------------- #


def test_citations_follow_the_order_the_answer_refers_to_them():
    results = [
        _scored("d_1", "chunk_00000", "first", 0.9),
        _scored("d_1", "chunk_00001", "second", 0.8),
    ]
    citations = extract_citations(
        "Second point [chunk_00001], then first [chunk_00000].", results, {"d_1": "r.pdf"}
    )
    assert [c["chunk_id"] for c in citations] == ["chunk_00001", "chunk_00000"]


def test_repeated_markers_are_cited_once():
    results = [_scored("d_1", "chunk_00000", "text", 0.9)]
    citations = extract_citations("A [chunk_00000] and B [chunk_00000].", results, {"d_1": "r.pdf"})
    assert len(citations) == 1


def test_markers_for_chunks_that_were_not_retrieved_are_ignored():
    results = [_scored("d_1", "chunk_00000", "text", 0.9)]
    citations = extract_citations("Invented [chunk_99999].", results, {"d_1": "r.pdf"})
    assert [c["chunk_id"] for c in citations] == ["chunk_00000"]


def test_an_uncited_answer_falls_back_to_the_retrieved_chunks():
    """PRD G2: an answer must always show the evidence it was grounded in."""
    results = [_scored("d_1", "chunk_00000", "a", 0.9), _scored("d_1", "chunk_00001", "b", 0.8)]
    citations = extract_citations("No markers here.", results, {"d_1": "r.pdf"})
    assert len(citations) == 2


def test_a_marker_matching_two_documents_resolves_to_both():
    """Chunk ids are unique per document, so `doc_id` is the disambiguator."""
    results = [_scored("d_1", "chunk_00000", "a", 0.9), _scored("d_2", "chunk_00000", "b", 0.8)]
    citations = extract_citations("Both [chunk_00000].", results, {"d_1": "a.pdf", "d_2": "b.pdf"})
    assert [c["doc_id"] for c in citations] == ["d_1", "d_2"]


def test_citation_carries_every_documented_field():
    results = [_scored("d_1", "chunk_00007", "evidence text", 0.8312)]
    (citation,) = extract_citations("Claim [chunk_00007].", results, {"d_1": "q3.pdf"})
    assert citation == {
        "chunk_id": "chunk_00007",
        "doc_id": "d_1",
        "filename": "q3.pdf",
        "score": 0.8312,
        "text": "evidence text",
    }


# --------------------------------------------------------------------------- #
# The pipeline (T2.2 acceptance criteria)
# --------------------------------------------------------------------------- #


def test_happy_path_returns_an_answer_with_citations(table):
    _seed(table, "d_1", "q3-report.pdf", [1.0, 0.0], [0.0, 1.0])
    embed = _Embedder([1.0, 0.0])
    generate = _Generator(
        Generation(text="Revenue rose [chunk_00000].", tokens_in=1450, tokens_out=210)
    )

    body = answer_question(
        table=table,
        question="What was revenue?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=embed,
        generate=generate,
    )

    assert body["found"] is True
    assert body["answer"] == "Revenue rose [chunk_00000]."
    assert body["usage"] == {"tokens_in": 1450, "tokens_out": 210}
    assert isinstance(body["latency_ms"], int)

    (citation,) = body["citations"]
    assert citation["chunk_id"] == "chunk_00000"
    assert citation["doc_id"] == "d_1"
    assert citation["filename"] == "q3-report.pdf"
    assert citation["score"] == pytest.approx(1.0)


def test_generation_receives_the_grounding_prompt_and_the_retrieved_context(table):
    _seed(table, "d_1", "q3-report.pdf", [1.0, 0.0])
    generate = _Generator(Generation(text="Answer [chunk_00000].", tokens_in=1, tokens_out=1))

    answer_question(
        table=table,
        question="What was revenue?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([1.0, 0.0]),
        generate=generate,
    )

    (call,) = generate.calls
    assert call["system"] == query_handler.SYSTEM_PROMPT
    assert call["max_tokens"] == MAX_TOKENS
    assert '<chunk id="chunk_00000" source="q3-report.pdf">' in call["user"]
    assert "What was revenue?" in call["user"]


def test_the_question_is_embedded_once(table):
    _seed(table, "d_1", "q3-report.pdf", [1.0, 0.0])
    embed = _Embedder([1.0, 0.0])

    answer_question(
        table=table,
        question="What was revenue?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=embed,
        generate=_Generator(Generation(text="A [chunk_00000].", tokens_in=1, tokens_out=1)),
    )

    assert embed.calls == [["What was revenue?"]]


def test_top_k_limits_how_many_chunks_reach_the_prompt(table):
    _seed(table, "d_1", "many.pdf", *[[1.0, i / 100] for i in range(8)])
    generate = _Generator(Generation(text="A [chunk_00000].", tokens_in=1, tokens_out=1))

    answer_question(
        table=table,
        question="q?",
        k=3,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([1.0, 0.0]),
        generate=generate,
    )

    assert generate.calls[0]["user"].count("<chunk ") == 3


# --- ADR-003: the similarity floor ----------------------------------------- #


def test_below_floor_never_calls_the_generation_model(table):
    """ADR-003 / FR-2.4. The most important behavioral test in the suite.

    A weak retrieval must not reach Bedrock at all: answering from noise is
    exactly the hallucination the floor exists to prevent, and the skipped call
    is also the cost saving.
    """
    _seed(table, "d_1", "q3-report.pdf", [1.0, 0.0])
    generate = _Generator()

    body = answer_question(
        table=table,
        question="Who won the 1998 World Cup?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([3.0, 10.0]),  # cosine with [1, 0] is 0.287, under the floor
        generate=generate,
    )

    assert generate.called is False
    assert body["found"] is False
    assert body["answer"] == NOT_FOUND_ANSWER
    assert body["citations"] == []
    assert body["usage"] == {"tokens_in": 0, "tokens_out": 0}


def test_a_score_exactly_at_the_floor_is_answered(table):
    """The floor rejects strictly below itself, so equality still generates."""
    _seed(table, "d_1", "doc.pdf", [1.0, 0.0])
    generate = _Generator(Generation(text="A [chunk_00000].", tokens_in=1, tokens_out=1))

    body = answer_question(
        table=table,
        question="q?",
        k=5,
        min_similarity=1.0,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([1.0, 0.0]),
        generate=generate,
    )

    assert generate.called is True
    assert body["found"] is True


def test_empty_corpus_calls_neither_model(table):
    """API_SPEC.md §4: an empty corpus is a not-found, not an error."""
    embed = _Embedder([1.0, 0.0])
    generate = _Generator()

    body = answer_question(
        table=table,
        question="Anything at all?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=embed,
        generate=generate,
    )

    assert embed.calls == []
    assert generate.called is False
    assert body["found"] is False
    assert body["usage"] == {"tokens_in": 0, "tokens_out": 0}


def test_model_replying_not_found_maps_to_the_standard_response(table):
    _seed(table, "d_1", "q3-report.pdf", [1.0, 0.0])
    generate = _Generator(Generation(text="NOT_FOUND", tokens_in=900, tokens_out=4))

    body = answer_question(
        table=table,
        question="What is the capital of Mongolia?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([1.0, 0.0]),
        generate=generate,
    )

    assert body["found"] is False
    assert body["answer"] == NOT_FOUND_ANSWER
    assert body["citations"] == []
    # Generation did run here, so its real cost is reported.
    assert body["usage"] == {"tokens_in": 900, "tokens_out": 4}


@pytest.mark.parametrize("reply", ["NOT_FOUND", "  NOT_FOUND  ", "NOT_FOUND.", "not_found"])
def test_not_found_reply_is_recognized_despite_formatting(table, reply):
    _seed(table, "d_1", "doc.pdf", [1.0, 0.0])
    body = answer_question(
        table=table,
        question="q?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([1.0, 0.0]),
        generate=_Generator(Generation(text=reply, tokens_in=1, tokens_out=1)),
    )
    assert body["found"] is False


def test_chunks_from_several_documents_are_ranked_together(table):
    _seed(table, "d_1", "near.pdf", [1.0, 0.0])
    _seed(table, "d_2", "far.pdf", [0.0, 1.0])
    generate = _Generator(Generation(text="A [chunk_00000].", tokens_in=1, tokens_out=1))

    body = answer_question(
        table=table,
        question="q?",
        k=2,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([1.0, 0.0]),
        generate=generate,
    )

    assert generate.calls[0]["user"].index("near.pdf") < generate.calls[0]["user"].index("far.pdf")
    assert body["citations"][0]["filename"] == "near.pdf"


# --------------------------------------------------------------------------- #
# Handler adaptation (API_SPEC.md §4 and §6)
# --------------------------------------------------------------------------- #


@pytest.fixture
def api_env(monkeypatch):
    """Environment and cached table for handler-level tests."""
    monkeypatch.setenv("TABLE_NAME", "rag-store")
    monkeypatch.setenv("MIN_SIMILARITY", str(FLOOR))
    monkeypatch.setenv("MAX_TOKENS", str(MAX_TOKENS))
    monkeypatch.setenv("ALLOWED_ORIGIN", "https://example.cloudfront.net")
    monkeypatch.setattr(query_handler, "_table", object())


def _event(body: object) -> dict:
    return {"body": json.dumps(body) if body is not None else None}


def test_handler_returns_200_with_the_answer_body(api_env, monkeypatch):
    payload = {
        "answer": "Revenue rose [chunk_00000].",
        "found": True,
        "citations": [],
        "usage": {"tokens_in": 1, "tokens_out": 2},
        "latency_ms": 12,
    }
    monkeypatch.setattr(query_handler, "answer_question", lambda **_: payload)

    response = handler(_event({"question": "What was revenue?"}))

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == payload


def test_handler_sets_the_cors_and_content_type_headers(api_env, monkeypatch):
    monkeypatch.setattr(query_handler, "answer_question", lambda **_: {"found": False})
    headers = handler(_event({"question": "why?"}))["headers"]

    assert headers["Content-Type"] == "application/json"
    assert headers["Access-Control-Allow-Origin"] == "https://example.cloudfront.net"


def test_handler_passes_the_configured_floor_and_cap(api_env, monkeypatch):
    captured: dict = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return {"found": False}

    monkeypatch.setattr(query_handler, "answer_question", spy)
    handler(_event({"question": "why?", "top_k": 7}))

    assert captured["min_similarity"] == FLOOR
    assert captured["max_tokens"] == MAX_TOKENS
    assert captured["k"] == 7


def test_handler_rejects_a_bad_question_with_the_documented_envelope(api_env):
    response = handler(_event({"question": "a"}))

    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"]["code"] == "INVALID_QUESTION"


def test_handler_rejects_a_bad_top_k_with_the_documented_envelope(api_env):
    response = handler(_event({"question": "what is this?", "top_k": 50}))

    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"]["code"] == "INVALID_TOP_K"


@pytest.mark.parametrize("body", [None, "", "not json", "[1, 2]"])
def test_handler_treats_an_unusable_body_as_a_missing_question(api_env, body):
    """API_SPEC.md §6 defines no generic parse error, so this is INVALID_QUESTION."""
    response = handler({"body": body})

    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"]["code"] == "INVALID_QUESTION"


def test_handler_maps_bedrock_exhaustion_to_503(api_env, monkeypatch):
    def unavailable(**_):
        raise ModelUnavailableError("Bedrock unavailable after 3 attempts")

    monkeypatch.setattr(query_handler, "answer_question", unavailable)
    response = handler(_event({"question": "what is this?"}))

    assert response["statusCode"] == 503
    assert json.loads(response["body"])["error"]["code"] == "MODEL_UNAVAILABLE"


def test_handler_never_leaks_internal_detail(api_env, monkeypatch):
    """API_SPEC.md §6: details go to the logs, never to the caller."""

    def boom(**_):
        raise RuntimeError("table rag-store: credentials rejected for arn:aws:iam::123")

    monkeypatch.setattr(query_handler, "answer_question", boom)
    response = handler(_event({"question": "what is this?"}))
    body = json.loads(response["body"])

    assert response["statusCode"] == 500
    assert body["error"]["code"] == "INTERNAL"
    assert "arn:aws:iam" not in response["body"]
    assert "rag-store" not in response["body"]


def test_env_readers_use_the_configured_values(api_env):
    assert query_handler.min_similarity() == FLOOR
    assert query_handler.max_tokens() == MAX_TOKENS


# --------------------------------------------------------------------------- #
# Observability (T3.2)
# --------------------------------------------------------------------------- #


def test_an_answered_query_logs_the_documented_fields(table, monkeypatch):
    """T3.2 AC: one JSON line carrying latency_ms, top_score, and tokens_out."""
    stream = io.StringIO()
    monkeypatch.setattr(query_handler, "log", get_logger("test.query", stream=stream))
    _seed(table, "d_1", "q3-report.pdf", [1.0, 0.0])

    answer_question(
        table=table,
        question="What was revenue?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([1.0, 0.0]),
        generate=_Generator(Generation(text="A [chunk_00000].", tokens_in=1450, tokens_out=210)),
    )

    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line]
    (answered,) = [line for line in lines if line["event"] == "query.answered"]
    assert {"latency_ms", "top_score", "tokens_out"} <= answered.keys()
    assert answered["tokens_out"] == 210


def test_an_answered_query_publishes_token_metrics(table, monkeypatch):
    """Token counters reach CloudWatch as EMF, not a PutMetricData call (§8)."""
    stream = io.StringIO()
    monkeypatch.setattr(query_handler, "log", get_logger("test.query.emf", stream=stream))
    _seed(table, "d_1", "q3-report.pdf", [1.0, 0.0])

    answer_question(
        table=table,
        question="What was revenue?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([1.0, 0.0]),
        generate=_Generator(Generation(text="A [chunk_00000].", tokens_in=1450, tokens_out=210)),
    )

    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line]
    (emf,) = [line for line in lines if "_aws" in line]
    (directive,) = emf["_aws"]["CloudWatchMetrics"]

    assert {metric["Name"] for metric in directive["Metrics"]} == {
        "TokensIn",
        "TokensOut",
        "QueryLatency",
        "TopScore",
    }
    assert emf["Service"] == query_handler.SERVICE_NAME
    assert emf["TokensIn"] == 1450


def test_a_skipped_generation_publishes_no_token_metrics(table, monkeypatch):
    """Below the floor there is no model call, so there is nothing to meter."""
    stream = io.StringIO()
    monkeypatch.setattr(query_handler, "log", get_logger("test.query.floor", stream=stream))
    _seed(table, "d_1", "q3-report.pdf", [1.0, 0.0])

    answer_question(
        table=table,
        question="Who won the 1998 World Cup?",
        k=5,
        min_similarity=FLOOR,
        max_tokens=MAX_TOKENS,
        embed=_Embedder([3.0, 10.0]),
        generate=_Generator(),
    )

    assert "_aws" not in stream.getvalue()


def test_table_resource_is_cached(monkeypatch, aws):
    """Module-level caching is the intended warm-start pattern (CLAUDE.md)."""
    monkeypatch.setenv("TABLE_NAME", "rag-store")
    monkeypatch.setattr(query_handler, "_table", None)

    first, second = query_handler._get_table(), query_handler._get_table()
    assert first is second
