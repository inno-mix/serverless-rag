"""Vectorized cosine similarity and stable top-k selection (T2.1).

This is the retrieval half of the RAG pipeline and the seam ADR-001 names as
the single place to swap if the corpus outgrows brute-force search. Everything
here is pure: it takes vectors in and returns scores out, with no AWS or
environment dependency, so it can be tested exhaustively (TESTING.md §2).

Scoring is done as one matrix multiply rather than a Python loop. Both sides are
L2-normalized first, which turns cosine similarity into a plain dot product, so
the whole corpus is scored in a single BLAS call. Zero-length vectors normalize
to zero rather than dividing by zero, so an empty or corrupt embedding scores
0.0 instead of poisoning the result with NaN.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

#: Default retrieval depth, matching `top_k` in API_SPEC.md §4.
DEFAULT_TOP_K = 5


class HasVector(Protocol):
    """Structural type for anything carrying an embedding.

    Declared as a protocol so this module never imports the storage layer: it
    scores `StoredChunk` in production and plain stubs in tests.
    """

    vector: np.ndarray


@dataclass(frozen=True)
class ScoredChunk[ChunkT: HasVector]:
    """A chunk paired with its cosine similarity to the query."""

    chunk: ChunkT
    score: float


class DimensionMismatchError(ValueError):
    """Corpus vectors disagree in length, or do not match the query.

    Raised rather than letting NumPy fail deep inside a broadcast, because the
    realistic cause is specific and worth naming: `embeddingModelId` was changed
    in `cdk.json` while chunks embedded by the previous model were still stored.
    Re-ingesting the affected documents is the fix.
    """


def cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Score every row of `matrix` against `query_vector`.

    Args:
        query_vector: The query embedding, shape `(d,)`.
        matrix: Corpus embeddings, shape `(n, d)`.

    Returns:
        Cosine similarities, shape `(n,)`, as float32. A zero-length vector on
        either side yields `0.0` for that pair rather than NaN.

    Raises:
        DimensionMismatchError: If the query length differs from the row length.
    """
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != query.size:
        width = matrix.shape[1] if matrix.ndim == 2 else "?"
        raise DimensionMismatchError(
            f"query has {query.size} dimensions but corpus vectors have {width}"
        )

    return _normalize_rows(matrix) @ _normalize(query)


def top_k[ChunkT: HasVector](
    query_vector: np.ndarray,
    chunks: Sequence[ChunkT],
    k: int = DEFAULT_TOP_K,
) -> list[ScoredChunk[ChunkT]]:
    """Return the `k` chunks most similar to the query, highest score first.

    Ties keep the order in which the chunks were supplied, so retrieval is
    deterministic for identical input — which matters because the caller feeds
    the result straight into a prompt.

    Args:
        query_vector: The embedded question.
        chunks: The corpus to search. Order is preserved for ties.
        k: How many chunks to return. Values above `len(chunks)` return
            everything; zero or negative returns nothing.

    Returns:
        At most `k` scored chunks, sorted by descending similarity.

    Raises:
        DimensionMismatchError: If the corpus vectors are ragged, or their
            length does not match the query.
    """
    if k <= 0 or not chunks:
        return []

    scores = cosine_scores(query_vector, _stack(chunks))
    ranking = np.argsort(-scores, kind="stable")[: min(k, len(chunks))]
    return [ScoredChunk(chunk=chunks[index], score=float(scores[index])) for index in ranking]


def _stack(chunks: Sequence[HasVector]) -> np.ndarray:
    """Assemble the corpus into one `(n, d)` float32 matrix."""
    vectors = [np.asarray(chunk.vector, dtype=np.float32).reshape(-1) for chunk in chunks]
    widths = {vector.size for vector in vectors}
    if len(widths) > 1:
        raise DimensionMismatchError(f"corpus vectors have mixed dimensions: {sorted(widths)}")
    return np.vstack(vectors)


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length, mapping the zero vector to itself."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros_like(vector)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, mapping zero rows to themselves."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
