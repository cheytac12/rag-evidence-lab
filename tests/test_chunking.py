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
