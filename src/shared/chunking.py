"""Paragraph-aware text chunking with sentence-safe splitting and overlap (T1.2).

Chunking strategy, per PRD FR-1.4:

1. Split the document on blank lines into paragraphs.
2. A paragraph that fits within the target size stays whole.
3. An oversize paragraph is split on sentence boundaries; an oversize *sentence*
   is split on word boundaries as a last resort.
4. The resulting units are packed greedily into chunks up to the target size.
5. Each chunk after the first is seeded with whole trailing units from its
   predecessor, giving the configured overlap without ever starting or ending a
   chunk mid-sentence.

Token counts are estimated as ``len(text) // 4``. This heuristic avoids shipping a
tokenizer into the Lambda bundle; it runs low on English prose and further off on
code or non-Latin scripts, so `target_tokens` is an approximate ceiling rather
than a guarantee.
"""

import re
from dataclasses import dataclass

#: Average characters per token used by :func:`estimate_tokens`.
CHARS_PER_TOKEN = 4

#: Blank line (optionally carrying whitespace) separating two paragraphs.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")

#: Sentence-ending punctuation, optionally followed by a closing quote or bracket
#: and anchored to whitespace or end of text. Matched forwards rather than with a
#: lookbehind, which Python's `re` only supports at fixed width.
_SENTENCE_END = re.compile(r"[.!?][\"'\)\]]?(?=\s|$)")


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document, ready to embed.

    Attributes:
        index: Zero-based position of this chunk within the document.
        text: The chunk's text, including any overlap carried from the previous
            chunk.
        token_count: Estimated token count for `text`.
    """

    index: int
    text: str
    token_count: int


@dataclass(frozen=True)
class _Unit:
    """An indivisible piece of text: a whole paragraph, sentence, or word run."""

    text: str
    tokens: int
    starts_paragraph: bool


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in `text`.

    Args:
        text: The text to measure.

    Returns:
        The estimated token count, using the documented four-characters-per-token
        heuristic. Non-empty text always counts as at least one token.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_text(text: str, target_tokens: int = 500, overlap_tokens: int = 50) -> list[Chunk]:
    """Split a document into overlapping, sentence-aligned chunks.

    Args:
        text: The full document text.
        target_tokens: Soft ceiling on the estimated size of each chunk.
        overlap_tokens: Approximate number of tokens each chunk repeats from the
            end of the previous one, rounded down to whole sentences.

    Returns:
        The chunks in document order. Empty or whitespace-only input yields an
        empty list.

    Raises:
        ValueError: If `target_tokens` is not positive, or if `overlap_tokens` is
            negative or not smaller than `target_tokens`.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must not be negative")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    units = _split_into_units(text, target_tokens)
    if not units:
        return []

    chunks: list[Chunk] = []
    for index, group in enumerate(_pack(units, target_tokens, overlap_tokens)):
        joined = _join(group)
        chunks.append(Chunk(index=index, text=joined, token_count=estimate_tokens(joined)))
    return chunks


def _split_into_units(text: str, target_tokens: int) -> list[_Unit]:
    """Break `text` into the smallest pieces that never need splitting again."""
    units: list[_Unit] = []
    for paragraph in _PARAGRAPH_BREAK.split(text.strip()):
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        first = True
        for piece in _split_paragraph(cleaned, target_tokens):
            units.append(_Unit(text=piece, tokens=estimate_tokens(piece), starts_paragraph=first))
            first = False
    return units


def _split_paragraph(paragraph: str, target_tokens: int) -> list[str]:
    """Split one paragraph down to pieces that each fit within `target_tokens`."""
    if estimate_tokens(paragraph) <= target_tokens:
        return [paragraph]

    pieces: list[str] = []
    for sentence in _split_sentences(paragraph):
        if estimate_tokens(sentence) <= target_tokens:
            pieces.append(sentence)
        else:
            pieces.extend(_split_on_words(sentence, target_tokens))
    return pieces


def _split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph after each sentence terminator, keeping the terminator."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        sentence = paragraph[start : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()

    trailing = paragraph[start:].strip()
    if trailing:
        sentences.append(trailing)
    return sentences


def _split_on_words(sentence: str, target_tokens: int) -> list[str]:
    """Last-resort split of a single oversize sentence on whitespace."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in sentence.split():
        word_tokens = estimate_tokens(word)
        if current and current_tokens + word_tokens > target_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _pack(units: list[_Unit], target_tokens: int, overlap_tokens: int) -> list[list[_Unit]]:
    """Group units into chunks, seeding each chunk with overlap from the last."""
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tokens = 0

    for unit in units:
        if current and current_tokens + unit.tokens > target_tokens:
            groups.append(current)
            current = _overlap_seed(current, overlap_tokens, target_tokens - unit.tokens)
            current_tokens = sum(u.tokens for u in current)
        current.append(unit)
        current_tokens += unit.tokens

    if current:
        groups.append(current)
    return groups


def _overlap_seed(previous: list[_Unit], overlap_tokens: int, budget: int) -> list[_Unit]:
    """Take whole trailing units from `previous` as the next chunk's overlap.

    Units are taken from the end while they fit inside both `overlap_tokens` and
    `budget` — the room left once the incoming unit is accounted for. Taking only
    whole units is what keeps chunk boundaries off mid-sentence positions.
    """
    seed: list[_Unit] = []
    seed_tokens = 0
    for unit in reversed(previous):
        if seed_tokens + unit.tokens > overlap_tokens:
            break
        seed.insert(0, unit)
        seed_tokens += unit.tokens

    while seed and seed_tokens > budget:
        seed_tokens -= seed.pop(0).tokens
    return seed


def _join(units: list[_Unit]) -> str:
    """Reassemble units into text, restoring paragraph and sentence spacing."""
    parts: list[str] = []
    for position, unit in enumerate(units):
        if position == 0:
            parts.append(unit.text)
        elif unit.starts_paragraph:
            parts.append("\n\n" + unit.text)
        else:
            parts.append(" " + unit.text)
    return "".join(parts)
