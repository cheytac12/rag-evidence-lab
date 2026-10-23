"""Thin reusable entry point for automated claim evidence review."""
from .generator import evaluate_claims


def detect_unsupported_claims(question: str, answer: str, claims: list[dict], passages: list):
    """Return automated model labels; callers must keep human review separate."""
    return evaluate_claims(question, answer, claims, passages)
