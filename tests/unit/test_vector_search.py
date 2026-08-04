"""Unit tests for vector_search (T2.1 acceptance criteria, TESTING.md §2)."""

from dataclasses import dataclass

import numpy as np
import pytest

from shared.storage import StoredChunk
from shared.vector_search import (
    DEFAULT_TOP_K,
    DimensionMismatchError,
    ScoredChunk,
    cosine_scores,
    top_k,
)


@dataclass
class Chunk:
    """Minimal stand-in for a stored chunk, satisfying the HasVector protocol."""

    name: str
    vector: np.ndarray


def make(name: str, *values: float) -> Chunk:
    return Chunk(name=name, vector=np.array(values, dtype=np.float32))


def names(results: list[ScoredChunk[Chunk]]) -> list[str]:
    return [result.chunk.name for result in results]


# --- hand-computed geometry -------------------------------------------------


def test_identical_vector_scores_one():
    query = np.array([1.0, 0.0], dtype=np.float32)
    (result,) = top_k(query, [make("same", 1.0, 0.0)], k=1)
    assert result.score == pytest.approx(1.0)


def test_orthogonal_vector_scores_zero():
    query = np.array([1.0, 0.0], dtype=np.float32)
    (result,) = top_k(query, [make("perpendicular", 0.0, 1.0)], k=1)
    assert result.score == pytest.approx(0.0)


def test_opposite_vector_scores_minus_one():
    query = np.array([1.0, 0.0], dtype=np.float32)
    (result,) = top_k(query, [make("opposite", -1.0, 0.0)], k=1)
    assert result.score == pytest.approx(-1.0)


def test_forty_five_degree_vector_scores_one_over_root_two():
    query = np.array([1.0, 0.0], dtype=np.float32)
    (result,) = top_k(query, [make("diagonal", 1.0, 1.0)], k=1)
    assert result.score == pytest.approx(1 / np.sqrt(2), rel=1e-6)


def test_three_dimensional_score_is_hand_computed():
    """|q| = 3, |c| = 3, q·c = 8, so cosine = 8/9."""
    query = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    (result,) = top_k(query, [make("c", 2.0, 2.0, 1.0)], k=1)
    assert result.score == pytest.approx(8 / 9, rel=1e-6)


def test_magnitude_does_not_change_the_score():
    """Cosine measures direction only: a scaled vector scores identically."""
    query = np.array([3.0, 4.0], dtype=np.float32)
    unit, scaled = top_k(query, [make("unit", 4.0, 3.0), make("scaled", 400.0, 300.0)], k=2)
    assert unit.score == pytest.approx(scaled.score, rel=1e-6)
    assert unit.score == pytest.approx(0.96, rel=1e-6)


def test_ranking_follows_similarity():
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = [
        make("opposite", -1.0, 0.0),
        make("perpendicular", 0.0, 1.0),
        make("exact", 1.0, 0.0),
        make("diagonal", 1.0, 1.0),
    ]
    assert names(top_k(query, corpus, k=4)) == [
        "exact",
        "diagonal",
        "perpendicular",
        "opposite",
    ]


def test_scores_are_returned_in_descending_order():
    query = np.array([1.0, 0.25], dtype=np.float32)
    corpus = [make(str(i), 1.0, i / 10) for i in range(10)]
    scores = [result.score for result in top_k(query, corpus, k=10)]
    assert scores == sorted(scores, reverse=True)


# --- k handling -------------------------------------------------------------


def test_k_larger_than_corpus_returns_every_chunk():
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = [make("a", 1.0, 0.0), make("b", 0.0, 1.0)]
    assert len(top_k(query, corpus, k=99)) == 2


def test_k_zero_returns_nothing():
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert top_k(query, [make("a", 1.0, 0.0)], k=0) == []


def test_negative_k_returns_nothing():
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert top_k(query, [make("a", 1.0, 0.0)], k=-3) == []


def test_empty_corpus_returns_nothing():
    """The empty-corpus path in API_SPEC.md §4 must not touch NumPy."""
    assert top_k(np.array([1.0, 0.0], dtype=np.float32), [], k=5) == []


def test_default_k_matches_the_api_default():
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = [make(str(i), 1.0, i) for i in range(20)]
    assert len(top_k(query, corpus)) == DEFAULT_TOP_K == 5


# --- degenerate vectors -----------------------------------------------------


def test_zero_vector_in_corpus_scores_zero_without_dividing_by_zero():
    query = np.array([1.0, 0.0], dtype=np.float32)
    (result,) = top_k(query, [make("empty", 0.0, 0.0)], k=1)
    assert result.score == 0.0
    assert not np.isnan(result.score)


def test_zero_query_vector_scores_every_chunk_zero():
    query = np.zeros(2, dtype=np.float32)
    results = top_k(query, [make("a", 1.0, 0.0), make("b", 0.0, 1.0)], k=2)
    assert [result.score for result in results] == [0.0, 0.0]


def test_no_nan_leaks_into_scores():
    query = np.zeros(3, dtype=np.float32)
    scores = cosine_scores(query, np.zeros((4, 3), dtype=np.float32))
    assert not np.isnan(scores).any()


# --- determinism ------------------------------------------------------------


def test_ties_preserve_corpus_order():
    """Equal scores must not be reordered: retrieval feeds a prompt."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = [make(f"tie_{i}", 1.0, 0.0) for i in range(6)]
    assert names(top_k(query, corpus, k=6)) == [f"tie_{i}" for i in range(6)]


def test_ties_below_the_cut_do_not_disturb_the_winner():
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = [make("tie_a", 0.0, 1.0), make("winner", 1.0, 0.0), make("tie_b", 0.0, 1.0)]
    assert names(top_k(query, corpus, k=2)) == ["winner", "tie_a"]


def test_repeated_calls_return_identical_rankings():
    rng = np.random.default_rng(1234)
    query = rng.standard_normal(16).astype(np.float32)
    corpus = [Chunk(str(i), rng.standard_normal(16).astype(np.float32)) for i in range(50)]
    assert names(top_k(query, corpus, k=7)) == names(top_k(query, corpus, k=7))


# --- dimension guards -------------------------------------------------------


def test_ragged_corpus_raises_a_named_error():
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = [make("two", 1.0, 0.0), make("three", 1.0, 0.0, 0.0)]
    with pytest.raises(DimensionMismatchError, match="mixed dimensions"):
        top_k(query, corpus, k=2)


def test_query_of_the_wrong_width_raises_a_named_error():
    """The realistic cause is a changed embeddingModelId (ADR-005)."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(DimensionMismatchError, match="dimensions"):
        top_k(query, [make("two", 1.0, 0.0)], k=1)


def test_cosine_scores_rejects_a_one_dimensional_matrix():
    query = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(DimensionMismatchError):
        cosine_scores(query, np.array([1.0, 0.0], dtype=np.float32))


# --- interoperability with the storage layer --------------------------------


def test_scores_stored_chunks_from_the_storage_module():
    """The production corpus is `StoredChunk`; the protocol must accept it."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = [
        StoredChunk(
            doc_id="d_1",
            chunk_id="chunk_00000",
            text="near",
            token_count=3,
            vector=np.array([1.0, 0.0], dtype=np.float32),
        ),
        StoredChunk(
            doc_id="d_1",
            chunk_id="chunk_00001",
            text="far",
            token_count=3,
            vector=np.array([0.0, 1.0], dtype=np.float32),
        ),
    ]
    results = top_k(query, corpus, k=2)
    assert [result.chunk.chunk_id for result in results] == ["chunk_00000", "chunk_00001"]
    assert results[0].score == pytest.approx(1.0)
