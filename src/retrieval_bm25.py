"""BM25 retrieval using rank-bm25 with deterministic score tie handling."""
from __future__ import annotations

import re
from typing import Any, Iterable

from .schemas import ChunkRecord, RetrievedPassage


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def retrieve_bm25(query: str, chunks: Iterable[ChunkRecord], top_k: int = 5) -> list[RetrievedPassage]:
    docs = sorted(list(chunks), key=lambda chunk: chunk.chunk_id)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not docs:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("rank-bm25 is required for BM25 retrieval. Install requirements.txt first.") from exc
    index = BM25Okapi([tokenize(chunk.text) for chunk in docs])
    scores = index.get_scores(tokenize(query))
    order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), docs[i].chunk_id))[:top_k]
    return [RetrievedPassage(
        rank=rank, chunk_id=docs[i].chunk_id, doc_id=docs[i].doc_id, title=docs[i].title,
        source_file=docs[i].source_file, page_start=docs[i].page_start, page_end=docs[i].page_end,
        text=docs[i].text, raw_score=float(scores[i]), method="bm25",
        char_start=docs[i].char_start, char_end=docs[i].char_end,
    ) for rank, i in enumerate(order, start=1)]
