"""Dense retrieval embeddings and deterministic NumPy cosine ranking."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schemas import ChunkRecord, RetrievedPassage

DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
ALLOWED_EMBEDDINGS = {DEFAULT_EMBEDDING, "BAAI/bge-small-en-v1.5"}


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _cache_key(chunks: list[ChunkRecord], model_id: str) -> str:
    material = {
        "checksums": sorted({chunk.source_sha256 for chunk in chunks}),
        "settings": sorted({(chunk.chunk_size, chunk.overlap) for chunk in chunks}),
        "model": model_id,
        "chunks": [(chunk.chunk_id, chunk.text) for chunk in chunks],
    }
    return hashlib.sha256(repr(material).encode("utf-8")).hexdigest()


def _load_model(model_id: str):
    if model_id not in ALLOWED_EMBEDDINGS:
        raise ValueError(f"Unsupported embedding model: {model_id}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for dense retrieval. Install requirements.txt first.") from exc
    try:
        return SentenceTransformer(model_id)
    except Exception as exc:
        raise RuntimeError(f"Could not load embedding model {model_id}: {exc}") from exc


def retrieve_dense(
    query: str,
    chunks: Iterable[ChunkRecord],
    top_k: int = 5,
    model_id: str = DEFAULT_EMBEDDING,
    cache_dir: str | Path = "results/embedding_cache",
    *,
    model: Any | None = None,
) -> list[RetrievedPassage]:
    docs = sorted(list(chunks), key=lambda chunk: chunk.chunk_id)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not docs:
        return []
    key = _cache_key(docs, model_id)
    cache_path = Path(cache_dir) / f"{key}.npz"
    vectors: np.ndarray | None = None
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                if str(cache["cache_key"].item()) == key and list(cache["chunk_ids"].astype(str)) == [c.chunk_id for c in docs]:
                    vectors = cache["vectors"].astype(np.float32)
        except Exception:
            vectors = None
    model = model or _load_model(model_id)
    if vectors is None:
        raw = model.encode([chunk.text for chunk in docs], convert_to_numpy=True, show_progress_bar=False)
        vectors = normalize_rows(np.asarray(raw))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, cache_key=np.asarray(key), chunk_ids=np.asarray([c.chunk_id for c in docs]), vectors=vectors)
    query_vec = np.asarray(model.encode([query], convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    query_vec = normalize_rows(query_vec)[0]
    scores = vectors @ query_vec
    order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), docs[i].chunk_id))[:top_k]
    return [RetrievedPassage(
        rank=rank, chunk_id=docs[i].chunk_id, doc_id=docs[i].doc_id, title=docs[i].title,
        source_file=docs[i].source_file, page_start=docs[i].page_start, page_end=docs[i].page_end,
        text=docs[i].text, raw_score=float(scores[i]), method="dense",
        char_start=docs[i].char_start, char_end=docs[i].char_end,
    ) for rank, i in enumerate(order, start=1)]
