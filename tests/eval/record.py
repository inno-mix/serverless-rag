"""Record Bedrock embeddings for the golden-set eval corpus (T4.4).

The eval gate runs in CI with no AWS access (TESTING.md §1), but a retrieval
gate built on fake embeddings would measure nothing: the scores, and therefore
the similarity floor, only mean something if they come from the real model. So
the vectors are recorded once, here, and committed alongside the corpus.

Run this after changing the corpus, the golden set, the chunker, or
`embeddingModelId`:

    python tests/eval/record.py

It calls Bedrock — a few cents at most — and rewrites `vectors.npz`. The gate
detects a stale recording on its own and fails with a pointer back to this
script, so a forgotten re-record surfaces as a clear error rather than a
silently meaningless pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shared.chunking import chunk_text  # noqa: E402
from shared.embeddings import embed_texts  # noqa: E402

EVAL_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_ROOT / "corpus"
GOLDEN_SET = EVAL_ROOT / "golden_set.json"
VECTORS = EVAL_ROOT / "vectors.npz"

#: Must match `cdk.json` context. Recorded into the fixture so the gate can
#: report which model produced the vectors it is scoring.
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


def build_corpus() -> tuple[list[str], list[str]]:
    """Chunk every corpus document with the production chunker.

    Returns:
        Parallel lists of chunk text and the document each chunk came from,
        ordered by filename so the recording is reproducible.
    """
    texts: list[str] = []
    documents: list[str] = []

    for path in sorted(CORPUS_DIR.glob("*.md")):
        chunks = chunk_text(
            path.read_text(encoding="utf-8"),
            target_tokens=CHUNK_TARGET_TOKENS,
            overlap_tokens=CHUNK_OVERLAP_TOKENS,
        )
        texts.extend(chunk.text for chunk in chunks)
        documents.extend(path.name for _ in chunks)

    return texts, documents


def load_questions() -> list[str]:
    """Read every question in the golden set, in file order."""
    cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["cases"]
    return [case["question"] for case in cases]


def main() -> None:
    """Embed the corpus and the questions, and write the fixture."""
    chunk_texts, chunk_documents = build_corpus()
    questions = load_questions()

    print(f"corpus:    {len(chunk_texts)} chunks from {len(set(chunk_documents))} documents")
    print(f"questions: {len(questions)}")
    print(f"model:     {EMBEDDING_MODEL_ID}")

    chunk_vectors = embed_texts(chunk_texts, model_id=EMBEDDING_MODEL_ID)
    question_vectors = embed_texts(questions, model_id=EMBEDDING_MODEL_ID)

    np.savez_compressed(
        VECTORS,
        chunk_texts=np.array(chunk_texts),
        chunk_documents=np.array(chunk_documents),
        chunk_vectors=np.vstack(chunk_vectors),
        questions=np.array(questions),
        question_vectors=np.vstack(question_vectors),
        meta=np.array(
            json.dumps(
                {
                    "embedding_model_id": EMBEDDING_MODEL_ID,
                    "chunk_target_tokens": CHUNK_TARGET_TOKENS,
                    "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
                }
            )
        ),
    )
    print(f"wrote      {VECTORS.relative_to(EVAL_ROOT.parents[1])}")


if __name__ == "__main__":
    main()
