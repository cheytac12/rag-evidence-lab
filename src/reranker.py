"""Fixed CrossEncoder reranker with stable tie handling."""
from __future__ import annotations

from typing import Any, Iterable

from .schemas import ChunkRecord, RetrievedPassage

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_passages(
    query: str,
    candidates: Iterable[RetrievedPassage],
    top_k: int = 5,
    *,
    model: Any | None = None,
) -> list[RetrievedPassage]:
    passages = list(candidates)
    if not passages:
        return []
    passages = sorted(passages, key=lambda p: p.rank)[:20]
    if model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for reranking. Install requirements.txt first.") from exc
        try:
            model = CrossEncoder(RERANKER_MODEL)
        except Exception as exc:
            raise RuntimeError(f"Could not load reranker model {RERANKER_MODEL}: {exc}") from exc
    scores = model.predict([(query, passage.text) for passage in passages])
    decorated = [(passage, float(score)) for passage, score in zip(passages, scores)]
    decorated.sort(key=lambda pair: (-pair[1], pair[0].rank, pair[0].chunk_id))
    return [RetrievedPassage(
        rank=rank, chunk_id=p.chunk_id, doc_id=p.doc_id, title=p.title, source_file=p.source_file,
        page_start=p.page_start, page_end=p.page_end, text=p.text, raw_score=p.raw_score,
        method=f"{p.method}+rerank", reranker_score=score,
        char_start=p.char_start, char_end=p.char_end,
    ) for rank, (p, score) in enumerate(decorated[:top_k], start=1)]
