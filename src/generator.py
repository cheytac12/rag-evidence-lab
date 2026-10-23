"""Credential-gated OpenAI Responses API generation and claim evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .answer_metrics import validate_claim_evaluations, validate_generated_payload
from .schemas import RetrievedPassage

PROMPT_VERSION = "rag-evidence-v1"
EVALUATOR_PROMPT_VERSION = "claim-review-v1"
MAX_OUTPUT_TOKENS = 1200
MAX_CONTEXT_TOKENS = 3000
TRANSMISSION_NOTICE = "The question and retrieved passages will be sent to your configured OpenAI model to generate an answer."
SYSTEM_PROMPT = """Answer using only the supplied evidence passages. If evidence is insufficient, set answer to exactly: The indexed documents do not provide enough evidence to answer this question. Split the answer's factual content into atomic claims. Distinguish explicit statements from inference. Cite only supplied passage IDs. Keep claims[].citation_ids and citations[] consistent, with the same claim/passage links. Do not invent titles, authors, page numbers, or other source metadata. Return only the requested JSON object."""
EVALUATOR_PROMPT = """Evaluate each supplied atomic factual claim only against the supplied passages. Use SUPPORTED when evidence establishes it, CONTRADICTED when a passage directly conflicts, and UNSUPPORTED when the passages do not establish it (this does not mean globally false). Return one short evidence-based explanation and only supplied passage IDs. Do not treat a citation's presence as proof of support."""
CORRECTNESS_PROMPT = """Score the answer against the reference using this fixed 0-3 rubric: 0 incorrect, irrelevant, or no answer when enough evidence exists; 1 partly correct but misses a required fact or contains a major error; 2 mostly correct with a minor omission and no major factual error; 3 correct and sufficiently complete. Return only the integer score and a concise explanation. Do not reward unsupported extra claims."""


class MissingOpenAIConfiguration(RuntimeError):
    pass


def load_local_env(path: str | Path = ".env") -> None:
    """Load non-overriding KEY=VALUE lines from a local .env without logging values."""
    target = Path(path)
    if not target.exists():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key in {"OPENAI_API_KEY", "OPENAI_MODEL"} and key not in os.environ:
            os.environ[key] = value


def openai_configured(env_file: str | Path = ".env") -> bool:
    load_local_env(env_file)
    return bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"))


def missing_configuration_message() -> str:
    load_local_env()
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_MODEL") if not os.environ.get(name)]
    return "Missing OpenAI configuration: set " + " and ".join(missing) + " in environment or .env. Generation and LLM claim evaluation are disabled; local ingestion and retrieval remain available."


def _client():
    if not openai_configured():
        raise MissingOpenAIConfiguration(missing_configuration_message())
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The official openai Python SDK is required. Install requirements.txt first.") from exc
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), os.environ["OPENAI_MODEL"]


def _passage_context(passages: Iterable[RetrievedPassage]) -> list[dict[str, str]]:
    return [{"passage_id": p.chunk_id, "text": p.text} for p in passages]


def generate_answer(question: str, passages: list[RetrievedPassage], *, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if not passages:
        return {"answer": "The indexed documents do not provide enough evidence to answer this question.", "claims": [], "citations": [], "uncertainties": ["No supporting passages were retrieved."], "model": None, "prompt_version": PROMPT_VERSION, "max_output_tokens": max_output_tokens, "temperature": 0}
    client, model_name = _client()
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "claims": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"claim_text": {"type": "string"}, "citation_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["claim_text", "citation_ids"]}},
            "citations": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"claim_text": {"type": "string"}, "passage_id": {"type": "string"}}, "required": ["claim_text", "passage_id"]}},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        }, "required": ["answer", "claims", "citations", "uncertainties"],
    }
    # Only the explicit question and selected passages enter the user payload.
    user_payload = json.dumps({"question": question, "passages": _passage_context(passages)}, ensure_ascii=False)
    response = client.responses.create(
        model=model_name,
        input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_payload}],
        temperature=0,
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_schema", "name": "rag_answer", "strict": True, "schema": schema}},
    )
    try:
        payload = json.loads(response.output_text)
    except Exception as exc:
        raise ValueError("OpenAI returned malformed structured answer JSON; the result was not repaired.") from exc
    validated = validate_generated_payload(payload, passages)
    usage = getattr(response, "usage", None)
    return {**validated, "model": model_name, "prompt_version": PROMPT_VERSION,
            "max_output_tokens": max_output_tokens, "temperature": 0,
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None), "provider_cost": None}


def evaluate_claims(question: str, answer: str, claims: list[dict[str, Any]], passages: list[RetrievedPassage]) -> list[dict[str, Any]]:
    if not claims:
        return []
    client, model_name = _client()
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"evaluations": {"type": "array", "items": {"type": "object", "additionalProperties": False,
              "properties": {"claim_text": {"type": "string"}, "supporting_passage_ids": {"type": "array", "items": {"type": "string"}}, "label": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]}, "explanation": {"type": "string"}},
              "required": ["claim_text", "supporting_passage_ids", "label", "explanation"]}}}, "required": ["evaluations"]}
    user_payload = json.dumps({"question": question, "answer": answer, "claims": claims,
                               "passages": _passage_context(passages)}, ensure_ascii=False)
    response = client.responses.create(
        model=model_name,
        input=[{"role": "system", "content": EVALUATOR_PROMPT}, {"role": "user", "content": user_payload}],
        temperature=0, max_output_tokens=MAX_OUTPUT_TOKENS,
        text={"format": {"type": "json_schema", "name": "claim_evaluations", "strict": True, "schema": schema}},
    )
    try:
        payload = json.loads(response.output_text)
    except Exception as exc:
        raise ValueError("OpenAI returned malformed claim evaluation JSON; the result was not repaired.") from exc
    evaluations = validate_claim_evaluations(claims, payload["evaluations"], {p.chunk_id for p in passages})
    for item in evaluations:
        item["model"] = model_name
        item["prompt_version"] = EVALUATOR_PROMPT_VERSION
        item["human_verified"] = False
    return evaluations


def evaluate_correctness(question: str, answer: str, reference_answer: str, required_facts: list[str] | None = None) -> dict[str, Any]:
    """Apply the specified 0-3 rubric; result is automated, never human-verified."""
    client, model_name = _client()
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"score": {"type": "integer", "enum": [0, 1, 2, 3]}, "explanation": {"type": "string"}},
              "required": ["score", "explanation"]}
    user_payload = json.dumps({"question": question, "answer": answer,
                               "reference_answer": reference_answer,
                               "required_facts": required_facts or []}, ensure_ascii=False)
    response = client.responses.create(
        model=model_name,
        input=[{"role": "system", "content": CORRECTNESS_PROMPT}, {"role": "user", "content": user_payload}],
        temperature=0, max_output_tokens=300,
        text={"format": {"type": "json_schema", "name": "answer_correctness", "strict": True, "schema": schema}},
    )
    try:
        result = json.loads(response.output_text)
    except Exception as exc:
        raise ValueError("OpenAI returned malformed correctness JSON; the result was not repaired.") from exc
    if not isinstance(result, dict) or set(result) != {"score", "explanation"} or result["score"] not in (0, 1, 2, 3) or not isinstance(result["explanation"], str):
        raise ValueError("OpenAI correctness output does not match the fixed 0-3 rubric schema")
    return {**result, "model": model_name, "prompt_version": "answer-correctness-v1", "human_verified": False}
