"""Golden-set retrieval regression gate (T4.4).

Every other test proves the pipeline *runs*. This one proves it still finds the
right passage. That is the failure mode with no other guard: change the chunk
size, swap the embedding model, or adjust the similarity floor, and every unit
test stays green while answers quietly get worse.

The gate is deterministic and offline. Embeddings were produced once by the real
Bedrock model and committed as `vectors.npz` (see `record.py`) — fabricated
vectors would make the scores, and therefore the floor, meaningless. What the
gate actually exercises is the production chunker, `vector_search.top_k`, and
the ADR-003 floor against the real geometry of the real model.

It deliberately stops at retrieval. Generated wording is non-deterministic and
TESTING.md §7 excludes it; the model's own `NOT_FOUND` reply is what handles
questions that are topically close but unanswerable, and that is covered by the
integration smoke test instead.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from shared.chunking import chunk_text
from shared.vector_search import top_k

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parents[1]

#: Retrieval depth the gate scores at. Matches the API default (API_SPEC.md §4).
TOP_K = 5

#: Rank within which a question's evidence must always appear. Every answerable
#: question must clear this, so a regression anywhere in retrieval fails.
REQUIRED_RANK = 3

#: Floor on rank-1 accuracy. Measured at 10/12: the two misses put the evidence
#: at rank 2 within the *correct* document, which is the benign kind of miss.
#: Gated at the measured value, so any further slippage fails.
MIN_RANK_ONE = 10


@dataclass(frozen=True)
class EvalChunk:
    """A recorded corpus chunk, satisfying the `HasVector` protocol."""

    document: str
    text: str
    vector: np.ndarray


@pytest.fixture(scope="module")
def recording() -> dict:
    """Load the committed vectors, failing clearly if they were never recorded."""
    path = EVAL_ROOT / "vectors.npz"
    if not path.exists():
        pytest.fail(f"{path.name} is missing — run `python tests/eval/record.py`")
    return dict(np.load(path, allow_pickle=False))


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return json.loads((EVAL_ROOT / "golden_set.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def chunks(recording) -> list[EvalChunk]:
    return [
        EvalChunk(document=document, text=text, vector=vector)
        for document, text, vector in zip(
            recording["chunk_documents"].tolist(),
            recording["chunk_texts"].tolist(),
            recording["chunk_vectors"],
            strict=True,
        )
    ]


@pytest.fixture(scope="module")
def question_vectors(recording) -> dict[str, np.ndarray]:
    return dict(zip(recording["questions"].tolist(), recording["question_vectors"], strict=True))


@pytest.fixture(scope="module")
def configured_floor() -> float:
    """The `minSimilarity` the deployed system actually uses."""
    context = json.loads((REPO_ROOT / "cdk.json").read_text(encoding="utf-8"))["context"]
    return float(context["minSimilarity"])


def _retrieve(question: str, question_vectors, chunks, k: int = TOP_K):
    return top_k(question_vectors[question], chunks, k)


def _rank_of_evidence(results, case: dict) -> int | None:
    """1-based rank of the first chunk that both matches the document and holds the fact."""
    for rank, result in enumerate(results, start=1):
        if (
            result.chunk.document == case["document"]
            and case["evidence"].lower() in result.chunk.text.lower()
        ):
            return rank
    return None


def _answerable(cases: list[dict]) -> list[dict]:
    return [case for case in cases if case["answerable"]]


def _unanswerable(cases: list[dict]) -> list[dict]:
    return [case for case in cases if not case["answerable"]]


# --------------------------------------------------------------------------- #
# The recording must describe the code as it stands
# --------------------------------------------------------------------------- #


def test_the_recording_matches_the_current_chunker(recording):
    """A stale recording would make every result below it a comfortable fiction.

    The vectors were produced from specific chunk texts. If the chunker or the
    corpus has changed since, those texts no longer exist and the gate is
    scoring passages the system would never produce.
    """
    meta = json.loads(str(recording["meta"]))
    expected: list[str] = []
    for path in sorted((EVAL_ROOT / "corpus").glob("*.md")):
        expected.extend(
            chunk.text
            for chunk in chunk_text(
                path.read_text(encoding="utf-8"),
                target_tokens=meta["chunk_target_tokens"],
                overlap_tokens=meta["chunk_overlap_tokens"],
            )
        )

    assert recording["chunk_texts"].tolist() == expected, (
        "the corpus or the chunker changed since the vectors were recorded — "
        "re-run `python tests/eval/record.py`"
    )


def test_the_recording_used_the_configured_embedding_model(recording):
    """Scoring one model's geometry to validate another proves nothing."""
    meta = json.loads(str(recording["meta"]))
    context = json.loads((REPO_ROOT / "cdk.json").read_text(encoding="utf-8"))["context"]
    assert meta["embedding_model_id"] == context["embeddingModelId"], (
        "embeddingModelId changed — re-run `python tests/eval/record.py`"
    )


def test_every_golden_question_was_recorded(recording, cases):
    recorded = set(recording["questions"].tolist())
    missing = {case["question"] for case in cases} - recorded
    assert not missing, f"questions added without re-recording: {missing}"


def test_the_golden_set_is_large_enough_to_mean_something(cases):
    """T4.4 asks for ten pairs; negatives are extra and exercise the floor."""
    assert len(_answerable(cases)) >= 10
    assert len(_unanswerable(cases)) >= 3


# --------------------------------------------------------------------------- #
# Retrieval quality
# --------------------------------------------------------------------------- #


def test_every_answerable_question_retrieves_its_evidence(cases, chunks, question_vectors):
    """The hard guarantee: the passage holding the answer is always near the top.

    Reported in full rather than one assertion per case, because when this
    breaks the useful information is *which* questions moved and how far.
    """
    failures = []
    for case in _answerable(cases):
        results = _retrieve(case["question"], question_vectors, chunks)
        rank = _rank_of_evidence(results, case)
        if rank is None or rank > REQUIRED_RANK:
            failures.append(
                f"  {case['question']!r}\n"
                f"    wanted {case['document']} containing {case['evidence']!r} "
                f"within rank {REQUIRED_RANK}, got rank {rank}\n"
                f"    top hit: {results[0].chunk.document} @ {results[0].score:.3f}"
            )

    assert not failures, "retrieval regressed:\n" + "\n".join(failures)


def test_rank_one_accuracy_holds(cases, chunks, question_vectors):
    """Softer than the guarantee above: how often the best chunk is the right one."""
    answerable = _answerable(cases)
    hits = sum(
        _rank_of_evidence(_retrieve(case["question"], question_vectors, chunks), case) == 1
        for case in answerable
    )
    assert hits >= MIN_RANK_ONE, f"rank-1 accuracy fell to {hits}/{len(answerable)}"


def test_near_identical_documents_do_not_capture_each_other(cases, chunks, question_vectors):
    """A question about one document must not be answered from its near-twin.

    The corpus holds deliberate distractors — a Q2 report beside the Q3 one, a
    Widget Lite handbook beside the Pro — because that is the failure a small,
    topically uniform corpus hides completely.
    """
    for case in _answerable(cases):
        results = _retrieve(case["question"], question_vectors, chunks)
        assert results[0].chunk.document == case["document"], (
            f"{case['question']!r} was answered from {results[0].chunk.document}, "
            f"not {case['document']}"
        )


# --------------------------------------------------------------------------- #
# The similarity floor (ADR-003)
# --------------------------------------------------------------------------- #


def test_out_of_scope_questions_stay_below_the_floor(
    cases, chunks, question_vectors, configured_floor
):
    """Questions with no relationship to the corpus must never reach the model."""
    failures = []
    for case in _unanswerable(cases):
        best = _retrieve(case["question"], question_vectors, chunks, 1)[0]
        if best.score >= configured_floor:
            failures.append(f"  {case['question']!r} scored {best.score:.3f}")

    assert not failures, (
        f"these would have been sent to the model despite a floor of "
        f"{configured_floor}:\n" + "\n".join(failures)
    )


def test_answerable_questions_stay_above_the_floor(
    cases, chunks, question_vectors, configured_floor
):
    """The floor must not reject questions the corpus genuinely answers.

    This is the direction that fails quietly. A floor set too high produces
    "I couldn't find anything about that" for questions whose answer is sitting
    at rank 1, and nothing else in the suite would notice.
    """
    failures = []
    for case in _answerable(cases):
        best = _retrieve(case["question"], question_vectors, chunks, 1)[0]
        if best.score < configured_floor:
            failures.append(
                f"  {case['question']!r} scored {best.score:.3f} "
                f"(evidence was at rank {
                    _rank_of_evidence(_retrieve(case['question'], question_vectors, chunks), case)
                })"
            )

    assert not failures, (
        f"a floor of {configured_floor} rejects questions the corpus answers:\n"
        + "\n".join(failures)
    )


def test_the_floor_sits_between_the_two_populations(
    cases, chunks, question_vectors, configured_floor
):
    """The floor must land in the gap between the two populations.

    It is only meaningful if answerable and out-of-scope questions separate at
    all, and only correct if it sits between them. Reported with both margins,
    so a floor drifting toward either population is visible before it starts
    misclassifying.
    """
    worst_answerable = min(
        _retrieve(case["question"], question_vectors, chunks, 1)[0].score
        for case in _answerable(cases)
    )
    best_out_of_scope = max(
        _retrieve(case["question"], question_vectors, chunks, 1)[0].score
        for case in _unanswerable(cases)
    )

    assert best_out_of_scope < worst_answerable, (
        f"the populations overlap: out-of-scope reaches {best_out_of_scope:.3f} "
        f"while an answerable question scores {worst_answerable:.3f}. "
        "No floor can separate them."
    )
    assert best_out_of_scope < configured_floor < worst_answerable, (
        f"minSimilarity={configured_floor} is outside the usable band "
        f"({best_out_of_scope:.3f}, {worst_answerable:.3f})"
    )
