"""Reciprocal Rank Fusion over BM25 and dense ranked lists."""
from __future__ import annotations

from typing import Any, Iterable

from .embeddings import DEFAULT_EMBEDDING, retrieve_dense
from .retrieval_bm25 import retrieve_bm25
from .schemas import ChunkRecord, RetrievedPassage

RRF_K = 60


def reciprocal_rank_fusion(*ranked_lists: list[RetrievedPassage], top_k: int = 5) -> list[RetrievedPassage]:
    aggregate: dict[str, float] = {}
    lookup: dict[str, RetrievedPassage] = {}
    for results in ranked_lists:
        for passage in results:
            aggregate[passage.chunk_id] = aggregate.get(passage.chunk_id, 0.0) + 1.0 / (RRF_K + passage.rank)
            lookup[passage.chunk_id] = passage
    ids = sorted(aggregate, key=lambda cid: (-aggregate[cid], cid))[:top_k]
    return [RetrievedPassage(
        rank=rank, chunk_id=cid, doc_id=lookup[cid].doc_id, title=lookup[cid].title,
        source_file=lookup[cid].source_file, page_start=lookup[cid].page_start,
        page_end=lookup[cid].page_end, text=lookup[cid].text, raw_score=aggregate[cid],
        method="hybrid", char_start=lookup[cid].char_start, char_end=lookup[cid].char_end,
    ) for rank, cid in enumerate(ids, start=1)]


def retrieve_hybrid(
    query: str,
    chunks: Iterable[ChunkRecord],
    top_k: int = 5,
    model_id: str = DEFAULT_EMBEDDING,
    *,
    candidate_k: int | None = None,
    cache_dir: str = "results/embedding_cache",
    embedding_model: Any | None = None,
) -> list[RetrievedPassage]:
    docs = list(chunks)
    k = candidate_k or max(top_k, 20)
    bm25 = retrieve_bm25(query, docs, top_k=k)
    dense = retrieve_dense(query, docs, top_k=k, model_id=model_id, cache_dir=cache_dir, model=embedding_model)
    return reciprocal_rank_fusion(bm25, dense, top_k=top_k)
