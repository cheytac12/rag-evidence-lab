"""Reproducible tokenizer-based chunking with source character offsets."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .schemas import ChunkRecord, PageRecord, read_jsonl, validate_page_record, write_jsonl


CHUNK_OVERLAPS = {128: 24, 256: 48, 512: 96}
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOKENIZER: Any = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers is required for token counts. Install requirements.txt first.") from exc
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
        except Exception as exc:
            raise RuntimeError(f"Could not load Hugging Face tokenizer {TOKENIZER_MODEL}: {exc}") from exc
    return _TOKENIZER


def validate_chunk_settings(size: int, overlap: int) -> None:
    if size not in CHUNK_OVERLAPS or CHUNK_OVERLAPS[size] != overlap:
        allowed = ", ".join(f"({s},{o})" for s, o in CHUNK_OVERLAPS.items())
        raise ValueError(f"Unsupported chunk size/overlap ({size},{overlap}); allowed pairs: {allowed}")


def token_count(text: str, tokenizer: Any | None = None) -> int:
    tokenizer = tokenizer or get_tokenizer()
    tokens = tokenizer(text, add_special_tokens=False)
    ids = tokens["input_ids"] if isinstance(tokens, dict) else tokens.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def _token_offsets(text: str, tokenizer: Any) -> list[tuple[int, int]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else getattr(encoded, "offset_mapping", None)
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    if offsets is None:
        raise RuntimeError("A fast Hugging Face tokenizer with offset mappings is required for reproducible chunk provenance.")
    return [(int(start), int(end)) for start, end in offsets if int(end) > int(start)]


def _boundary_ends(text: str) -> list[int]:
    boundaries = {m.end() for m in re.finditer(r"\n\s*\n", text)}
    boundaries.update(m.end() for m in re.finditer(r"[.!?](?:[\"'”’)]*)?(?=\s|$)", text))
    return sorted(boundaries)


def chunk_document(pages: list[PageRecord], size: int, overlap: int, tokenizer: Any | None = None) -> list[ChunkRecord]:
    validate_chunk_settings(size, overlap)
    if not pages:
        return []
    tokenizer = tokenizer or get_tokenizer()
    pages = sorted(pages, key=lambda page: page.page_number)
    doc_id = pages[0].doc_id
    if any(page.doc_id != doc_id for page in pages):
        raise ValueError("chunk_document expects pages from exactly one document")
    pieces: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for index, page in enumerate(pages):
        if index:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(page.text)
        cursor += len(page.text)
        page_ranges.append((start, cursor, page.page_number))
    text = "".join(pieces)
    offsets = _token_offsets(text, tokenizer)
    if not offsets:
        return []
    boundaries = _boundary_ends(text)
    chunks: list[ChunkRecord] = []
    token_start = 0
    while token_start < len(offsets):
        token_limit = min(token_start + size, len(offsets))
        char_start = offsets[token_start][0]
        hard_end = offsets[token_limit - 1][1]
        candidates = [point for point in boundaries if char_start < point <= hard_end]
        if candidates:
            char_end = candidates[-1]
            token_end = token_limit
            while token_end > token_start + 1 and offsets[token_end - 1][1] > char_end:
                token_end -= 1
            if token_end <= token_start:
                token_end = token_limit
                char_end = hard_end
        else:
            token_end = token_limit
            char_end = hard_end
        raw_start, raw_end = char_start, char_end
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace():
            raw_end -= 1
        body = text[raw_start:raw_end]
        count = token_count(body, tokenizer)
        if body and count > size:
            raise ValueError(f"Chunk token limit exceeded ({count} > {size}) at offsets {raw_start}:{raw_end}")
        if body:
            touched = [page_number for pstart, pend, page_number in page_ranges if pend > raw_start and pstart < raw_end]
            first = min(touched) if touched else pages[0].page_number
            last = max(touched) if touched else first
            checksum = pages[0].source_sha256
            stable = f"{checksum}|{size}|{overlap}|{raw_start}|{raw_end}"
            chunks.append(ChunkRecord(
                chunk_id="chunk_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24],
                doc_id=doc_id, title=pages[0].title, source_file=pages[0].source_file,
                page_start=first, page_end=last, char_start=raw_start, char_end=raw_end,
                text=body, token_count=count, source_sha256=checksum,
                is_demo=pages[0].is_demo, chunk_size=size, overlap=overlap,
            ))
        if token_end >= len(offsets):
            break
        token_start = max(token_start + 1, token_end - overlap)
    for chunk in chunks:
        if not chunk.text.strip() or chunk.token_count > size:
            raise ValueError(f"Invalid chunk produced: {chunk.chunk_id}")
    return chunks


def chunk_pages(pages: list[PageRecord], size: int, overlap: int, tokenizer: Any | None = None) -> list[ChunkRecord]:
    grouped: dict[str, list[PageRecord]] = {}
    for page in pages:
        grouped.setdefault(page.doc_id, []).append(page)
    result = []
    for doc_id in sorted(grouped):
        result.extend(chunk_document(grouped[doc_id], size, overlap, tokenizer=tokenizer))
    return result


def chunk_jsonl(input_path: str | Path, output_path: str | Path, size: int, overlap: int, tokenizer: Any | None = None) -> list[ChunkRecord]:
    pages = read_jsonl(input_path, validate_page_record)
    chunks = chunk_pages(pages, size, overlap, tokenizer=tokenizer)
    write_jsonl(output_path, chunks)
    return chunks
