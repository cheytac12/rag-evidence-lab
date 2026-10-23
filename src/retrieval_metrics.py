"""Span-overlap based retrieval metrics for annotated evidence units."""
from __future__ import annotations

import math
from typing import Iterable

from .schemas import EvidenceSpan, RetrievedPassage


def passage_covers(passage: RetrievedPassage, evidence: EvidenceSpan) -> bool:
    """Use provenance overlap; lexical similarity to the annotation quote is irrelevant."""
    return (
        passage.doc_id == evidence.doc_id
        and passage.page_start <= evidence.page_end
        and passage.page_end >= evidence.page_start
        and passage.char_start < evidence.char_end
        and passage.char_end > evidence.char_start
    )


def retrieval_metrics(passages: Iterable[RetrievedPassage], evidence: list[EvidenceSpan], k: int = 5, *, answerable: bool = True) -> dict[str, float | int | None]:
    ranked = sorted(list(passages), key=lambda p: p.rank)[:k]
    if not answerable:
        return {"recall": None, "hit": None, "precision": None, "mrr": None, "ndcg": None, "complete_evidence_recall": None}
    if not evidence:
        raise ValueError("Answerable retrieval metrics require at least one evidence unit")
    unit_hits = [any(passage_covers(p, span) for p in ranked) for span in evidence]
    relevant = [any(passage_covers(p, span) for span in evidence) for p in ranked]
    recall = sum(unit_hits) / len(evidence)
    hit = float(any(unit_hits))
    denom = min(k, len(ranked))
    precision = sum(relevant) / denom if denom else 0.0
    first = next((idx for idx, is_relevant in enumerate(relevant, start=1) if is_relevant), None)
    mrr = 1 / first if first else 0.0
    dcg = sum((1.0 / math.log2(index + 1)) for index, is_relevant in enumerate(relevant, start=1) if is_relevant)
    ideal_n = min(len(evidence), denom)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_n + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "recall": recall,
        "hit": hit,
        "precision": precision,
        "mrr": mrr,
        "ndcg": ndcg,
        "complete_evidence_recall": float(all(unit_hits)) if len(evidence) > 1 else None,
    }


def unanswerable_metrics(answer: str | None, *, explicit_abstention: bool) -> dict[str, float]:
    answered = bool((answer or "").strip()) and not explicit_abstention
    return {"false_positive_answer": float(answered), "abstention": float(not answered)}
