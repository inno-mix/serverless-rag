"""Unit tests for the paragraph-aware chunker (T1.2, TESTING.md §2)."""

import itertools
import re

import pytest

from shared.chunking import CHARS_PER_TOKEN, chunk_text, estimate_tokens

# Tolerance on the soft ceiling: the packer only closes a chunk once the *next*
# unit would overflow, so units never overshoot, but joining them reintroduces
# separator characters the per-unit estimate did not count.
TOLERANCE_TOKENS = 5

#: Every generated sentence is unique, so a shared prefix between consecutive
#: chunks can only come from the overlap mechanism and never from repeated text.
_SENTENCE_PATTERN = re.compile(r"^Finding number \d+ was reviewed and recorded\.$")


def _sentence(number: int) -> str:
    return f"Finding number {number} was reviewed and recorded."


def _document(paragraphs: int, sentences_each: int) -> str:
    counter = itertools.count()
    return "\n\n".join(
        " ".join(_sentence(next(counter)) for _ in range(sentences_each)) for _ in range(paragraphs)
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _pairs(chunks: list) -> list[tuple]:
    return list(zip(chunks[:-1], chunks[1:], strict=True))


def _overlap_length(previous: str, following: str) -> int:
    """Length of the longest suffix of `previous` that prefixes `following`."""
    for size in range(min(len(previous), len(following)), 0, -1):
        if previous.endswith(following[:size]):
            return size
    return 0


def _reconstruct(texts: list[str]) -> str:
    """Concatenate chunks, dropping the region each repeats from its predecessor."""
    rebuilt = texts[0]
    for previous, following in _pairs(texts):
        rebuilt += following[_overlap_length(previous, following) :]
    return rebuilt


# --------------------------------------------------------------------------- #
# estimate_tokens
# --------------------------------------------------------------------------- #


def test_estimate_tokens_is_zero_for_empty_text():
    assert estimate_tokens("") == 0


def test_estimate_tokens_uses_the_documented_heuristic():
    assert estimate_tokens("x" * (10 * CHARS_PER_TOKEN)) == 10


def test_estimate_tokens_never_returns_zero_for_non_empty_text():
    assert estimate_tokens("a") == 1


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "\t \n  \n"])
def test_blank_input_produces_no_chunks(text):
    assert chunk_text(text) == []


def test_single_short_paragraph_produces_exactly_one_chunk():
    paragraph = _document(paragraphs=1, sentences_each=2)
    chunks = chunk_text(paragraph, target_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].text == paragraph
    assert chunks[0].index == 0


def test_chunk_indexes_are_sequential():
    chunks = chunk_text(_document(6, 8), target_tokens=40, overlap_tokens=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))


# --------------------------------------------------------------------------- #
# Size and overlap
# --------------------------------------------------------------------------- #


def test_every_chunk_respects_the_target_size():
    chunks = chunk_text(_document(8, 6), target_tokens=60, overlap_tokens=15)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 60 + TOLERANCE_TOKENS


def test_consecutive_chunks_share_overlapping_text():
    chunks = chunk_text(_document(8, 6), target_tokens=60, overlap_tokens=20)
    assert len(chunks) > 1
    for previous, following in _pairs(chunks):
        assert _overlap_length(previous.text, following.text) > 0


def test_overlap_can_be_disabled():
    chunks = chunk_text(_document(4, 6), target_tokens=60, overlap_tokens=0)
    assert len(chunks) > 1
    for previous, following in _pairs(chunks):
        assert _overlap_length(previous.text, following.text) == 0


def test_reported_token_count_matches_the_chunk_text():
    for chunk in chunk_text(_document(5, 8), target_tokens=50, overlap_tokens=10):
        assert chunk.token_count == estimate_tokens(chunk.text)


# --------------------------------------------------------------------------- #
# Sentence safety
# --------------------------------------------------------------------------- #


def test_oversize_paragraph_is_split_on_sentence_boundaries():
    """A long single paragraph must break between sentences, never inside one."""
    chunks = chunk_text(_document(1, 30), target_tokens=40, overlap_tokens=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.endswith("."), chunk.text
        assert chunk.text[0].isupper(), chunk.text


def test_every_chunk_contains_only_whole_sentences():
    chunks = chunk_text(_document(4, 8), target_tokens=45, overlap_tokens=10)
    for chunk in chunks:
        for sentence in re.split(r"(?<=\.)\s+", chunk.text):
            assert _SENTENCE_PATTERN.match(sentence.strip()), sentence


def test_sentence_longer_than_the_target_is_split_on_word_boundaries():
    """An unsplittable sentence falls back to word boundaries rather than mid-word."""
    words = [f"word{i}" for i in range(200)]
    chunks = chunk_text(" ".join(words) + ".", target_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    emitted = " ".join(c.text for c in chunks).replace(".", "").split()
    assert set(emitted) == set(words)


# --------------------------------------------------------------------------- #
# Fidelity
# --------------------------------------------------------------------------- #


def test_unicode_survives_chunking_intact():
    document = (
        "Le chiffre d'affaires a augmenté de 12 %.\n\n"
        "日本語のテキストも保持されます。\n\n"
        "Это тоже работает."
    )
    chunks = chunk_text(document, target_tokens=500, overlap_tokens=50)
    assert _normalize(" ".join(c.text for c in chunks)) == _normalize(document)


def test_chunks_reconstruct_the_original_document():
    """Concatenating chunks minus their overlaps rebuilds the source text."""
    document = _document(6, 7)
    chunks = chunk_text(document, target_tokens=55, overlap_tokens=15)
    assert len(chunks) > 1
    assert _normalize(_reconstruct([c.text for c in chunks])) == _normalize(document)


def test_no_text_is_dropped_even_without_overlap():
    document = _document(5, 5)
    chunks = chunk_text(document, target_tokens=50, overlap_tokens=0)
    assert _normalize(" ".join(c.text for c in chunks)) == _normalize(document)


def test_paragraph_breaks_are_preserved_inside_a_chunk():
    document = "First paragraph here.\n\nSecond paragraph here."
    (chunk,) = chunk_text(document, target_tokens=500, overlap_tokens=50)
    assert "\n\n" in chunk.text


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("target", "overlap", "message"),
    [
        (0, 0, "target_tokens must be positive"),
        (-1, 0, "target_tokens must be positive"),
        (100, -1, "overlap_tokens must not be negative"),
        (100, 100, "overlap_tokens must be smaller"),
        (100, 500, "overlap_tokens must be smaller"),
    ],
)
def test_invalid_arguments_are_rejected(target, overlap, message):
    with pytest.raises(ValueError, match=message):
        chunk_text("some text", target_tokens=target, overlap_tokens=overlap)
