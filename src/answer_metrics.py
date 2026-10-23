"""Claim-level answer metrics and citation ID/support separation."""
from __future__ import annotations

from typing import Any, Iterable

from .schemas import RetrievedPassage

CLAIM_LABELS = {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}


def validate_citation_ids(citations: list[dict[str, Any]], passages: Iterable[RetrievedPassage]) -> list[dict[str, Any]]:
    valid_ids = {p.chunk_id for p in passages}
    return [{**citation, "citation_present": citation.get("passage_id") in valid_ids,
             "citation_valid": citation.get("passage_id") in valid_ids} for citation in citations]


def validate_generated_payload(payload: Any, passages: Iterable[RetrievedPassage]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"answer", "claims", "citations", "uncertainties"}:
        raise ValueError("Generation output must contain exactly answer, claims, citations, and uncertainties")
    if not isinstance(payload["answer"], str) or not isinstance(payload["claims"], list) or not isinstance(payload["citations"], list) or not isinstance(payload["uncertainties"], list):
        raise ValueError("Generation output has incorrect field types")
    valid_ids = {p.chunk_id for p in passages}
    checked_claims = []
    for index, claim in enumerate(payload["claims"], 1):
        if not isinstance(claim, dict) or set(claim) != {"claim_text", "citation_ids"}:
            raise ValueError(f"claims[{index}] must contain claim_text and citation_ids")
        if not isinstance(claim["claim_text"], str) or not isinstance(claim["citation_ids"], list) or any(not isinstance(x, str) for x in claim["citation_ids"]):
            raise ValueError(f"claims[{index}] has invalid field types")
        checked_claims.append({**claim, "citation_id_validity": {pid: pid in valid_ids for pid in claim["citation_ids"]}})
    for index, citation in enumerate(payload["citations"], 1):
        if not isinstance(citation, dict) or set(citation) != {"claim_text", "passage_id"}:
            raise ValueError(f"citations[{index}] must contain claim_text and passage_id")
        if not all(isinstance(citation[key], str) for key in ("claim_text", "passage_id")):
            raise ValueError(f"citations[{index}] has invalid field types")
    expected_links = {(claim["claim_text"], passage_id) for claim in checked_claims for passage_id in claim["citation_ids"]}
    returned_links = {(citation["claim_text"], citation["passage_id"]) for citation in payload["citations"]}
    if expected_links != returned_links:
        raise ValueError("claims.citation_ids and citations must contain the same claim/passage links")
    if any(not isinstance(x, str) for x in payload["uncertainties"]):
        raise ValueError("uncertainties must contain only strings")
    return {**payload, "claims": checked_claims, "citations": validate_citation_ids(payload["citations"], passages)}


def validate_claim_evaluations(claims: list[dict[str, Any]], evaluations: list[dict[str, Any]], passage_ids: set[str]) -> list[dict[str, Any]]:
    if len(claims) != len(evaluations):
        raise ValueError("Evaluator must return exactly one evaluation per factual claim")
    output = []
    for idx, (claim, evaluation) in enumerate(zip(claims, evaluations), 1):
        if not isinstance(evaluation, dict) or set(evaluation) != {"claim_text", "supporting_passage_ids", "label", "explanation"}:
            raise ValueError(f"evaluation[{idx}] has an invalid schema")
        if evaluation["claim_text"] != claim["claim_text"]:
            raise ValueError(f"evaluation[{idx}] claim text does not match claim order")
        if evaluation["label"] not in CLAIM_LABELS:
            raise ValueError(f"evaluation[{idx}] label must be one of {sorted(CLAIM_LABELS)}")
        if not isinstance(evaluation["supporting_passage_ids"], list) or any(pid not in passage_ids for pid in evaluation["supporting_passage_ids"]):
            raise ValueError(f"evaluation[{idx}] includes passage IDs that were not in the supplied context")
        if not isinstance(evaluation["explanation"], str):
            raise ValueError(f"evaluation[{idx}] explanation must be a string")
        output.append(evaluation)
    return output


def claim_metric_summary(claims: list[dict[str, Any]], evaluations: list[dict[str, Any]], citations: list[dict[str, Any]]) -> dict[str, float | int | None]:
    count = len(claims)
    if not count:
        faithfulness = unsupported_rate = None
    else:
        labels = [item["label"] for item in evaluations]
        faithfulness = labels.count("SUPPORTED") / count
        unsupported_rate = labels.count("UNSUPPORTED") / count
    citations_by_claim: dict[str, list[dict[str, Any]]] = {}
    for citation in citations:
        citations_by_claim.setdefault(citation["claim_text"], []).append(citation)
    claims_with_citations = [claim for claim in claims if citations_by_claim.get(claim["claim_text"])]
    cited_claim_count = sum(any(item.get("citation_present") for item in citations_by_claim[claim["claim_text"]]) for claim in claims)
    correct_claim_count = sum(any(item.get("citation_supporting") is True for item in citations_by_claim[claim["claim_text"]]) for claim in claims_with_citations)
    citation_correctness = (correct_claim_count / len(claims_with_citations)) if claims_with_citations else None
    return {
        "factual_claim_count": count,
        "faithfulness": faithfulness,
        "unsupported_claim_rate": unsupported_rate,
        "unsupported_answer": float(any(item["label"] in {"UNSUPPORTED", "CONTRADICTED"} for item in evaluations)) if evaluations else None,
        "citation_present_rate": cited_claim_count / count if count else None,
        "citation_correctness": citation_correctness,
    }


def correctness_score(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1, 2, 3}:
        raise ValueError("Correctness must be an integer on the 0-3 rubric")
    return value
