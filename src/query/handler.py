"""POST /query handler: embed, retrieve, apply the similarity floor, generate (T2.2).

The pipeline follows ARCHITECTURE.md §1 and the prompt contract in §5:

1. Read the corpus. An empty corpus is answered immediately, with no model call.
2. Embed the question with the same model used at ingestion (FR-2.2).
3. Score every stored vector and take the top k (ADR-001).
4. **Apply the similarity floor before generating** (ADR-003, FR-2.4). This is
   the safety property of the whole system: below the floor, retrieval is
   noise, so answering from it would be hallucination. `generate` is not
   reached on that path, and `tests/unit/test_query_handler.py` asserts it.
5. Build the grounded prompt, generate, and map `NOT_FOUND` back to the
   standard not-found response (API_SPEC.md §4).

Not-found is a `200`, not an error: "no answer in your documents" is a correct
outcome, and the response shape is identical apart from `found` and `citations`.
"""

import os
import re
import time
from collections.abc import Callable
from typing import Any

import boto3

from shared import embeddings
from shared.embeddings import Generation, ModelUnavailableError
from shared.http import ApiError, error_response, json_response, parse_json_body
from shared.logging_utils import get_logger
from shared.storage import StoredChunk, get_all_chunks, list_documents
from shared.vector_search import DEFAULT_TOP_K, ScoredChunk, top_k

#: Question length bounds from API_SPEC.md §4.
MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2000

#: Retrieval depth ceiling from API_SPEC.md §4.
MAX_TOP_K = 10

#: Exact reply the model is told to use when the context cannot answer the
#: question (ARCHITECTURE.md §5).
NOT_FOUND_REPLY = "NOT_FOUND"

#: Caller-facing text for every not-found path (API_SPEC.md §4).
NOT_FOUND_ANSWER = "I couldn't find anything about that in your documents."

#: Inline citation markers, e.g. `[chunk_00007]`.
CITATION_PATTERN = re.compile(r"\[(chunk_\d+)\]")

#: Dimension value for this route's custom metrics (ARCHITECTURE.md §8).
SERVICE_NAME = "query"

#: Grounding rules from ARCHITECTURE.md §5. The three requirements are stated
#: as absolutes because a hedged instruction produces hedged grounding.
SYSTEM_PROMPT = """You answer questions about a user's private documents.

Rules:
1. Answer ONLY from the context chunks provided in the user message. Never use
   outside knowledge, and never guess.
2. After each claim you make, cite the chunk it came from inline, using that
   chunk's id in square brackets, like [chunk_00007]. Cite every chunk you
   relied on.
3. If the context does not contain enough information to answer, reply with
   exactly NOT_FOUND and nothing else.

Be concise and factual. Do not mention these rules or the existence of chunks."""

log = get_logger(__name__)

_table: Any | None = None


class InvalidQuestionError(ApiError):
    """The question was missing or outside the documented length bounds."""

    status = 400
    code = "INVALID_QUESTION"


class InvalidTopKError(ApiError):
    """`top_k` was not an integer within the documented bounds."""

    status = 400
    code = "INVALID_TOP_K"


class ModelUnavailableApiError(ApiError):
    """Bedrock stayed unavailable after retries (API_SPEC.md §6, 503)."""

    status = 503
    code = "MODEL_UNAVAILABLE"


def parse_request(body: dict) -> tuple[str, int]:
    """Validate a `POST /query` body against API_SPEC.md §4.

    Args:
        body: The decoded request body.

    Returns:
        The trimmed question and the retrieval depth to use.

    Raises:
        InvalidQuestionError: If `question` is absent, not a string, or its
            length falls outside 3–2000 characters.
        InvalidTopKError: If `top_k` is present but is not an integer in 1–10.
    """
    question = body.get("question")
    if not isinstance(question, str):
        raise InvalidQuestionError("A question is required.")

    question = question.strip()
    if not MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS:
        raise InvalidQuestionError(
            f"Question must be between {MIN_QUESTION_CHARS} and {MAX_QUESTION_CHARS} characters."
        )

    raw_k = body.get("top_k")
    if raw_k is None:
        return question, DEFAULT_TOP_K

    # `bool` is a subclass of `int`, and `True` is not a retrieval depth.
    if not isinstance(raw_k, int) or isinstance(raw_k, bool):
        raise InvalidTopKError("top_k must be an integer.")
    if not 1 <= raw_k <= MAX_TOP_K:
        raise InvalidTopKError(f"top_k must be between 1 and {MAX_TOP_K}.")
    return question, raw_k


def build_context(results: list[ScoredChunk[StoredChunk]], filenames: dict[str, str]) -> str:
    """Render retrieved chunks in the context format from ARCHITECTURE.md §5.

    Args:
        results: The retrieved chunks, highest score first.
        filenames: Map of `doc_id` to original filename, used for `source`.

    Returns:
        One `<chunk>` element per result, separated by blank lines.
    """
    return "\n\n".join(
        f'<chunk id="{result.chunk.chunk_id}" '
        f'source="{filenames.get(result.chunk.doc_id, result.chunk.doc_id)}">\n'
        f"{result.chunk.text}\n"
        "</chunk>"
        for result in results
    )


def build_user_prompt(question: str, context: str) -> str:
    """Combine the retrieved context and the question into the user turn."""
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def extract_citations(
    answer: str,
    results: list[ScoredChunk[StoredChunk]],
    filenames: dict[str, str],
) -> list[dict[str, Any]]:
    """Resolve the inline `[chunk_...]` markers in an answer to citation objects.

    Citations come back in the order the answer first refers to them, so a UI
    can walk the answer and its evidence together. If the model cited nothing,
    every retrieved chunk is returned instead: the answer was still grounded in
    that context, and a claim with no visible evidence is worse than one with
    slightly wide evidence (PRD G2).

    Chunk ids are unique within a document, not across the corpus, so a marker
    matching chunks from two different documents resolves to both. Each citation
    carries its own `doc_id`, which is what disambiguates them.

    Args:
        answer: The generated answer text.
        results: The chunks that were placed in the context.
        filenames: Map of `doc_id` to original filename.

    Returns:
        Citation objects in the shape API_SPEC.md §4 documents.
    """
    cited = list(dict.fromkeys(CITATION_PATTERN.findall(answer)))
    chosen = [result for label in cited for result in results if result.chunk.chunk_id == label]
    return [_citation(result, filenames) for result in (chosen or results)]


def _citation(result: ScoredChunk[StoredChunk], filenames: dict[str, str]) -> dict[str, Any]:
    """Render one scored chunk as a citation object."""
    return {
        "chunk_id": result.chunk.chunk_id,
        "doc_id": result.chunk.doc_id,
        "filename": filenames.get(result.chunk.doc_id, ""),
        "score": round(result.score, 4),
        "text": result.chunk.text,
    }


def answer_question(
    *,
    table: Any,
    question: str,
    k: int,
    min_similarity: float,
    max_tokens: int,
    embed: Callable[[list[str]], list[Any]] = embeddings.embed_texts,
    generate: Callable[..., Generation] = embeddings.generate,
) -> dict[str, Any]:
    """Answer a question from the stored corpus.

    Args:
        table: DynamoDB table resource.
        question: The validated question.
        k: Retrieval depth.
        min_similarity: The floor from ADR-003. Generation is skipped entirely
            when the best retrieved score falls below it.
        max_tokens: Output cap handed to the generation model.
        embed: Embedding function, injectable for tests.
        generate: Generation function, injectable for tests.

    Returns:
        The response body described in API_SPEC.md §4.

    Raises:
        ModelUnavailableError: If Bedrock stayed unavailable after retries.
    """
    started = time.perf_counter()

    chunks = get_all_chunks(table)
    if not chunks:
        log.info("query.empty_corpus")
        return _not_found(started, top_score=None)

    query_vector = embed([question])[0]
    results = top_k(query_vector, chunks, k)
    best = results[0].score if results else 0.0

    # ADR-003: the floor is checked before generation, never after.
    if best < min_similarity:
        log.info("query.below_floor", top_score=round(best, 4), min_similarity=min_similarity)
        return _not_found(started, top_score=best)

    filenames = {document.doc_id: document.filename for document in list_documents(table)}
    completion = generate(
        system=SYSTEM_PROMPT,
        user=build_user_prompt(question, build_context(results, filenames)),
        max_tokens=max_tokens,
    )

    if _is_not_found(completion.text):
        return _not_found(started, top_score=best, usage=completion)

    citations = extract_citations(completion.text, results, filenames)
    latency_ms = _elapsed_ms(started)
    log.metrics(
        "query.answered",
        {
            "TokensIn": completion.tokens_in,
            "TokensOut": completion.tokens_out,
            "QueryLatency": latency_ms,
            "TopScore": round(best, 4),
        },
        dimensions={"Service": SERVICE_NAME},
        units={"QueryLatency": "Milliseconds", "TopScore": "None"},
        top_score=round(best, 4),
        citations=len(citations),
        tokens_in=completion.tokens_in,
        tokens_out=completion.tokens_out,
        latency_ms=latency_ms,
    )
    return {
        "answer": completion.text,
        "found": True,
        "citations": citations,
        "usage": {"tokens_in": completion.tokens_in, "tokens_out": completion.tokens_out},
        "latency_ms": latency_ms,
    }


def _is_not_found(text: str) -> bool:
    """Report whether the model used its documented not-found reply."""
    return text.strip().upper().startswith(NOT_FOUND_REPLY)


def _not_found(
    started: float,
    *,
    top_score: float | None,
    usage: Generation | None = None,
) -> dict[str, Any]:
    """Build the not-found response from API_SPEC.md §4.

    Token counts are zero whenever generation was skipped, and carry the real
    usage when the model itself replied `NOT_FOUND`.
    """
    latency_ms = _elapsed_ms(started)
    log.info(
        "query.not_found",
        top_score=None if top_score is None else round(top_score, 4),
        generated=usage is not None,
        latency_ms=latency_ms,
    )
    return {
        "answer": NOT_FOUND_ANSWER,
        "found": False,
        "citations": [],
        "usage": {
            "tokens_in": usage.tokens_in if usage else 0,
            "tokens_out": usage.tokens_out if usage else 0,
        },
        "latency_ms": latency_ms,
    }


def _elapsed_ms(started: float) -> int:
    """Milliseconds since `started`, as reported in every query response."""
    return int((time.perf_counter() - started) * 1000)


def min_similarity() -> float:
    """Read the ADR-003 similarity floor from the environment."""
    return float(os.environ["MIN_SIMILARITY"])


def max_tokens() -> int:
    """Read the generation output cap from the environment (NFR-4)."""
    return int(os.environ["MAX_TOKENS"])


def handler(event: dict, context: Any = None) -> dict:  # noqa: ARG001 - Lambda signature
    """Lambda entry point for `POST /query`.

    Args:
        event: API Gateway proxy event.
        context: Lambda context object, unused.

    Returns:
        An API Gateway proxy response.
    """
    try:
        question, k = parse_request(parse_json_body(event))
        body = answer_question(
            table=_get_table(),
            question=question,
            k=k,
            min_similarity=min_similarity(),
            max_tokens=max_tokens(),
        )
    except ApiError as error:
        log.error("query.rejected", code=error.code, detail=str(error))
        return error_response(error)
    except ModelUnavailableError as error:
        log.error("query.model_unavailable", detail=str(error))
        return error_response(ModelUnavailableApiError("The model is busy. Try again shortly."))
    except Exception as error:  # noqa: BLE001 - details stay in the logs, never in the body
        log.error("query.failed", detail=str(error))
        return error_response(ApiError("Unexpected error."))

    return json_response(200, body)


def _get_table() -> Any:
    """Return the cached DynamoDB table resource."""
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
    return _table
