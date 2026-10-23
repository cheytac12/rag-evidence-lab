import numpy as np

from src.embeddings import retrieve_dense
from src.reranker import rerank_passages
from src.retrieval_bm25 import retrieve_bm25
from src.retrieval_hybrid import reciprocal_rank_fusion, retrieve_hybrid
from src.schemas import ChunkRecord, RetrievedPassage


def chunk(cid, text):
    return ChunkRecord(cid, "d1", "Doc", "doc.pdf", 1, 1, 0, len(text), text,
                       len(text.split()), "checksum", False, 128, 24)


class FakeEmbedding:
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        out = []
        for text in texts:
            lowered = text.lower()
            out.append([1.0, 0.0] if "apple" in lowered else [0.0, 1.0] if "pear" in lowered else [1.0, 0.0])
        return np.asarray(out, dtype=np.float32)


class FakeCrossEncoder:
    def predict(self, pairs):
        return [1.0 if "pear" in text.lower() else 0.5 for _, text in pairs]


def test_bm25_ties_are_chunk_id_ordered():
    docs = [chunk("z", "same text"), chunk("a", "same text")]
    results = retrieve_bm25("no matching query", docs, top_k=2)
    assert [p.chunk_id for p in results] == ["a", "z"]


def test_dense_cosine_and_cache_are_deterministic(tmp_path):
    docs = [chunk("b", "pear detail"), chunk("a", "apple detail")]
    model = FakeEmbedding()
    results = retrieve_dense("apple", docs, 2, cache_dir=tmp_path, model=model)
    assert [p.chunk_id for p in results] == ["a", "b"]
    again = retrieve_dense("apple", docs, 2, cache_dir=tmp_path, model=model)
    assert [p.chunk_id for p in again] == ["a", "b"]


def test_hybrid_uses_rrf_and_tie_chunk_id(tmp_path):
    a = RetrievedPassage(1, "b", "d1", "Doc", "doc", 1, 1, "b", 0.9, "bm25")
    b = RetrievedPassage(1, "a", "d1", "Doc", "doc", 1, 1, "a", 0.2, "dense")
    fused = reciprocal_rank_fusion([a], [b], top_k=2)
    assert [p.chunk_id for p in fused] == ["a", "b"]
    docs = [chunk("b", "pear"), chunk("a", "apple")]
    hybrid = retrieve_hybrid("apple", docs, top_k=2, candidate_k=2, cache_dir=str(tmp_path), embedding_model=FakeEmbedding())
    assert len(hybrid) == 2
    assert all(p.method == "hybrid" for p in hybrid)


def test_reranker_orders_score_then_original_rank_then_chunk_id():
    passages = [RetrievedPassage(2, "b", "d", "T", "f", 1, 1, "pear", 0.2, "bm25"),
                RetrievedPassage(1, "a", "d", "T", "f", 1, 1, "pear", 0.1, "bm25")]
    reranked = rerank_passages("q", passages, top_k=2, model=FakeCrossEncoder())
    assert [p.chunk_id for p in reranked] == ["a", "b"]
    assert reranked[0].reranker_score == 1.0
