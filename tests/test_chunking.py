import re

import pytest

from src.chunking import chunk_document, validate_chunk_settings
from src.schemas import PageRecord


class WhitespaceTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return result


class TrailingWhitespaceTokenizer(WhitespaceTokenizer):
    """Model token offsets can include the whitespace after punctuation."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (m.start(), min(len(text), m.end() + 1)) for m in matches
            ]
        return result


def _pages(text):
    return [PageRecord("d1", "Doc", "doc.pdf", 1, text,
                       {"authors": [], "year": None, "subject": None}, "checksum", False)]


@pytest.mark.parametrize("size,overlap", [(128, 24), (256, 48), (512, 96)])
def test_allowed_settings_offsets_and_token_limits(size, overlap):
    text = " ".join(f"token{i}." for i in range(620))
    chunks = chunk_document(_pages(text), size, overlap, tokenizer=WhitespaceTokenizer())
    assert chunks
    assert all(c.text and c.token_count <= size for c in chunks)
    assert all(text[c.char_start:c.char_end] == c.text for c in chunks)
    assert all(c.chunk_size == size and c.overlap == overlap for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_invalid_chunk_pair_rejected():
    with pytest.raises(ValueError, match="allowed pairs"):
        validate_chunk_settings(128, 23)


def test_sentence_boundaries_cannot_create_tiny_sliding_chunks():
    # Decimal/table punctuation can be misread as sentence boundaries. Even
    # when a candidate boundary is very early in a window, chunking must make
    # bounded progress instead of emitting one-token-shifted duplicates.
    text = " ".join(f"metric{i} 0.123 result{i}." for i in range(900))
    size, overlap = 128, 24
    chunks = chunk_document(_pages(text), size, overlap, tokenizer=WhitespaceTokenizer())

    assert len(chunks) <= 2 * (900 * 3 // (size // 2))
    assert all(c.token_count >= size // 2 for c in chunks[:-1])
    assert all(text[c.char_start:c.char_end] == c.text for c in chunks)


def test_boundary_inside_token_span_still_makes_bounded_progress():
    # The first sentence boundary is just before the first token's reported
    # end offset. Without the minimum advance, the chunker emits a one-token
    # chunk before processing the rest of the document.
    text = "a. " + " ".join(f"word{i}" for i in range(300))
    size, overlap = 128, 24
    chunks = chunk_document(_pages(text), size, overlap, tokenizer=TrailingWhitespaceTokenizer())

    assert len(chunks) < 10
    assert all(c.token_count >= size // 2 for c in chunks[:-1])
