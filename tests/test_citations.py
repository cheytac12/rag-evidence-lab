import pytest

from src.answer_metrics import claim_metric_summary, validate_citation_ids, validate_generated_payload
from src.schemas import RetrievedPassage


def test_citation_presence_is_not_citation_support():
    passages = [RetrievedPassage(1, "p1", "d", "Doc", "f", 1, 1, "Text", 1.0, "bm25")]
    cited = [{"claim_text": "Claim", "passage_id": "p1"}, {"claim_text": "Claim", "passage_id": "unknown"}]
    checked = validate_citation_ids(cited, passages)
    assert checked[0]["citation_present"] is True
    assert checked[1]["citation_present"] is False
    claims = [{"claim_text": "Claim", "citation_ids": ["p1"]}]
    metrics = claim_metric_summary(claims, [{"label": "UNSUPPORTED"}],
                                   [{"claim_text": "Claim", "citation_present": True, "citation_supporting": False}])
    assert metrics["citation_present_rate"] == 1.0
    assert metrics["citation_correctness"] == 0.0
    assert metrics["faithfulness"] == 0.0


def test_generated_payload_rejects_unknown_structure():
    with pytest.raises(ValueError, match="exactly answer"):
        validate_generated_payload({"answer": "x"}, [])


def test_no_claims_produce_unavailable_ratios_not_zero():
    summary = claim_metric_summary([], [], [])
    assert summary["faithfulness"] is None
    assert summary["unsupported_claim_rate"] is None
    assert summary["unsupported_answer"] is None
    assert summary["citation_present_rate"] is None
    assert summary["citation_correctness"] is None
