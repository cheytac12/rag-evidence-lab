from src.retrieval_metrics import retrieval_metrics, unanswerable_metrics
from src.schemas import EvidenceSpan, RetrievedPassage


def passage(rank, start, end, chunk_id):
    return RetrievedPassage(rank, chunk_id, "d", "Doc", "doc", 1, 1, "evidence", 1.0, "bm25", char_start=start, char_end=end)


def test_retrieval_metric_formulas_and_multihop_completeness():
    evidence = [EvidenceSpan("d", 1, 1, 10, 20, "first span"), EvidenceSpan("d", 1, 1, 30, 40, "second span")]
    values = retrieval_metrics([passage(1, 10, 20, "c1"), passage(2, 30, 40, "c2"), passage(3, 50, 60, "c3")], evidence, 3)
    assert values["recall"] == 1.0
    assert values["hit"] == 1.0
    assert abs(values["precision"] - 2/3) < 1e-9
    assert values["mrr"] == 1.0
    assert values["ndcg"] == 1.0
    assert values["complete_evidence_recall"] == 1.0


def test_missing_evidence_and_unanswerable_edge_cases():
    evidence = [EvidenceSpan("d", 1, 1, 10, 20, "first"), EvidenceSpan("d", 1, 1, 30, 40, "second")]
    values = retrieval_metrics([passage(1, 10, 20, "c1")], evidence, 5)
    assert values["recall"] == 0.5
    assert values["complete_evidence_recall"] == 0.0
    nulls = retrieval_metrics([], [], 5, answerable=False)
    assert all(value is None for value in nulls.values())
    assert unanswerable_metrics("An answer", explicit_abstention=False) == {"false_positive_answer": 1.0, "abstention": 0.0}
    assert unanswerable_metrics("", explicit_abstention=True) == {"false_positive_answer": 0.0, "abstention": 1.0}
