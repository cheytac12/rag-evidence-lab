"""Credential-gated OpenAI and Google Gemini generation and evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .answer_metrics import validate_claim_evaluations, validate_generated_payload
from .schemas import RetrievedPassage

PROMPT_VERSION = "rag-evidence-v1"
EVALUATOR_PROMPT_VERSION = "claim-review-v1"
MAX_OUTPUT_TOKENS = 1200
MAX_CONTEXT_TOKENS = 3000
TRANSMISSION_NOTICE = "The question and retrieved passages will be sent to your configured model provider to generate an answer."
SYSTEM_PROMPT = """Answer using only the supplied evidence passages. If evidence is insufficient, set answer to exactly: The indexed documents do not provide enough evidence to answer this question. Split the answer's factual content into atomic claims. Distinguish explicit statements from inference. Cite only supplied passage IDs. Keep claims[].citation_ids and citations[] consistent, with the same claim/passage links. Do not invent titles, authors, page numbers, or other source metadata. Return only the requested JSON object."""
EVALUATOR_PROMPT = """Evaluate each supplied atomic factual claim only against the supplied passages. Use SUPPORTED when evidence establishes it, CONTRADICTED when a passage directly conflicts, and UNSUPPORTED when the passages do not establish it (this does not mean globally false). Return one short evidence-based explanation and only supplied passage IDs. Do not treat a citation's presence as proof of support."""
CORRECTNESS_PROMPT = """Score the answer against the reference using this fixed 0-3 rubric: 0 incorrect, irrelevant, or no answer when enough evidence exists; 1 partly correct but misses a required fact or contains a major error; 2 mostly correct with a minor omission and no major factual error; 3 correct and sufficiently complete. Return only the integer score and a concise explanation. Do not reward unsupported extra claims."""


class MissingLLMConfiguration(RuntimeError):
    pass


MissingOpenAIConfiguration = MissingLLMConfiguration


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
        if key in {"LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL"} and key not in os.environ:
            os.environ[key] = value


def configured_provider(env_file: str | Path = ".env") -> str:
    load_local_env(env_file)
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider
    # Keep existing OpenAI-only .env files working, while recognizing a Gemini
    # key when it is the only configured provider.
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "gemini"
    return "openai"


def configured_model(env_file: str | Path = ".env") -> str | None:
    load_local_env(env_file)
    provider = configured_provider(env_file)
    if provider == "gemini":
        return os.environ.get("GEMINI_MODEL") or None
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL") or None
    return None


def llm_configured(env_file: str | Path = ".env") -> bool:
    load_local_env(env_file)
    provider = configured_provider(env_file)
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_MODEL"))
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"))
    return False


def openai_configured(env_file: str | Path = ".env") -> bool:
    """Backward-compatible name; true when the selected LLM provider is configured."""
    return llm_configured(env_file)


def missing_configuration_message() -> str:
    load_local_env()
    provider = configured_provider()
    if provider == "gemini":
        required = ("GEMINI_API_KEY", "GEMINI_MODEL")
        label = "Google Gemini"
    elif provider == "openai":
        required = ("OPENAI_API_KEY", "OPENAI_MODEL")
        label = "OpenAI"
    else:
        return f"Unsupported LLM_PROVIDER '{provider}'. Set LLM_PROVIDER to 'gemini' or 'openai'. Generation and LLM evaluation are disabled; local ingestion and retrieval remain available."
    missing = [name for name in required if not os.environ.get(name)]
    if not missing:
        return f"{label} configuration is ready."
    return "Missing " + label + " configuration: set " + " and ".join(missing) + " in environment or .env. Generation and LLM claim evaluation are disabled; local ingestion and retrieval remain available."


def _selected_credentials() -> tuple[str, str, str]:
    if not llm_configured():
        raise MissingLLMConfiguration(missing_configuration_message())
    provider = configured_provider()
    if provider == "gemini":
        return provider, os.environ["GEMINI_API_KEY"], os.environ["GEMINI_MODEL"]
    return provider, os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"]


def _call_openai(api_key: str, model_name: str, system_prompt: str, user_payload: str,
                 schema: dict[str, Any], max_output_tokens: int) -> tuple[str, int | None, int | None]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The official openai Python SDK is required. Install requirements.txt first.") from exc
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_name,
        input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
        temperature=0,
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_schema", "name": "rag_result", "strict": True, "schema": schema}},
    )
    usage = getattr(response, "usage", None)
    return response.output_text, getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Keep the JSON Schema subset supported by Gemini structured output."""
    supported = {"type", "properties", "items", "required", "enum", "description", "format", "minimum", "maximum", "minItems", "maxItems"}
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in supported:
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {name: _gemini_schema(child) for name, child in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = _gemini_schema(value)
        else:
            result[key] = value
    return result


def _call_gemini(api_key: str, model_name: str, system_prompt: str, user_payload: str,
                 schema: dict[str, Any], max_output_tokens: int) -> tuple[str, int | None, int | None]:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model_name, safe='-_.')}:generateContent"
    request_body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_payload}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": _gemini_schema(schema),
        },
    }
    request = Request(endpoint, data=json.dumps(request_body).encode("utf-8"),
                      headers={"Content-Type": "application/json", "x-goog-api-key": api_key}, method="POST")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:500].replace(api_key, "[REDACTED]")
            if exc.code == 503 and attempt < 2:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = min(max(float(retry_after), 1.0), 8.0) if retry_after else 2 ** attempt
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(delay)
                continue
            if exc.code == 503:
                raise RuntimeError(f"Gemini is temporarily overloaded (HTTP 503) after 3 attempts. Please retry shortly. Details: {details}") from exc
            raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach the Gemini API: {exc.reason}") from exc
    candidates = result.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    output_text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not output_text:
        reason = candidates[0].get("finishReason", "no candidate returned") if candidates else "no candidate returned"
        raise ValueError(f"Gemini returned no structured text (finish reason: {reason}).")
    usage = result.get("usageMetadata", {})
    return output_text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount")


def _call_model(system_prompt: str, user_payload: str, schema: dict[str, Any], max_output_tokens: int):
    provider, api_key, model_name = _selected_credentials()
    if provider == "gemini":
        output, input_tokens, output_tokens = _call_gemini(api_key, model_name, system_prompt, user_payload, schema, max_output_tokens)
    else:
        output, input_tokens, output_tokens = _call_openai(api_key, model_name, system_prompt, user_payload, schema, max_output_tokens)
    return output, input_tokens, output_tokens, model_name, provider


def _passage_context(passages: Iterable[RetrievedPassage]) -> list[dict[str, str]]:
    return [{"passage_id": p.chunk_id, "text": p.text} for p in passages]


def generate_answer(question: str, passages: list[RetrievedPassage], *, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if not passages:
        return {"answer": "The indexed documents do not provide enough evidence to answer this question.", "claims": [], "citations": [], "uncertainties": ["No supporting passages were retrieved."], "model": None, "prompt_version": PROMPT_VERSION, "max_output_tokens": max_output_tokens, "temperature": 0}
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
    output_text, input_tokens, output_tokens, model_name, provider = _call_model(
        SYSTEM_PROMPT, user_payload, schema, max_output_tokens)
    try:
        payload = json.loads(output_text)
    except Exception as exc:
        raise ValueError(f"{provider} returned malformed structured answer JSON; the result was not repaired.") from exc
    validated = validate_generated_payload(payload, passages)
    return {**validated, "provider": provider, "model": model_name, "prompt_version": PROMPT_VERSION,
            "max_output_tokens": max_output_tokens, "temperature": 0,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "provider_cost": None}


def evaluate_claims(question: str, answer: str, claims: list[dict[str, Any]], passages: list[RetrievedPassage]) -> list[dict[str, Any]]:
    if not claims:
        return []
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"evaluations": {"type": "array", "items": {"type": "object", "additionalProperties": False,
              "properties": {"claim_text": {"type": "string"}, "supporting_passage_ids": {"type": "array", "items": {"type": "string"}}, "label": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]}, "explanation": {"type": "string"}},
              "required": ["claim_text", "supporting_passage_ids", "label", "explanation"]}}}, "required": ["evaluations"]}
    user_payload = json.dumps({"question": question, "answer": answer, "claims": claims,
                               "passages": _passage_context(passages)}, ensure_ascii=False)
    output_text, _, _, model_name, provider = _call_model(
        EVALUATOR_PROMPT, user_payload, schema, MAX_OUTPUT_TOKENS)
    try:
        payload = json.loads(output_text)
    except Exception as exc:
        raise ValueError(f"{provider} returned malformed claim evaluation JSON; the result was not repaired.") from exc
    evaluations = validate_claim_evaluations(claims, payload["evaluations"], {p.chunk_id for p in passages})
    for item in evaluations:
        item["model"] = model_name
        item["provider"] = provider
        item["prompt_version"] = EVALUATOR_PROMPT_VERSION
        item["human_verified"] = False
    return evaluations


def evaluate_correctness(question: str, answer: str, reference_answer: str, required_facts: list[str] | None = None) -> dict[str, Any]:
    """Apply the specified 0-3 rubric; result is automated, never human-verified."""
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"score": {"type": "integer", "enum": [0, 1, 2, 3]}, "explanation": {"type": "string"}},
              "required": ["score", "explanation"]}
    user_payload = json.dumps({"question": question, "answer": answer,
                               "reference_answer": reference_answer,
                               "required_facts": required_facts or []}, ensure_ascii=False)
    output_text, _, _, model_name, provider = _call_model(
        CORRECTNESS_PROMPT, user_payload, schema, 300)
    try:
        result = json.loads(output_text)
    except Exception as exc:
        raise ValueError(f"{provider} returned malformed correctness JSON; the result was not repaired.") from exc
    if not isinstance(result, dict) or set(result) != {"score", "explanation"} or result["score"] not in (0, 1, 2, 3) or not isinstance(result["explanation"], str):
        raise ValueError(f"{provider} correctness output does not match the fixed 0-3 rubric schema")
    return {**result, "provider": provider, "model": model_name, "prompt_version": "answer-correctness-v1", "human_verified": False}
