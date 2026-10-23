"""Typed records and strict JSONL validation for the project data contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, TypeVar


QUESTION_TYPES = {"factual_lookup", "technical_explanation", "comparison", "multi_hop", "unanswerable"}
T = TypeVar("T")


@dataclass(frozen=True)
class EvidenceSpan:
    doc_id: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    quote: str


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    title: str
    source_file: str
    page_number: int
    text: str
    metadata: dict[str, Any]
    source_sha256: str
    is_demo: bool = False


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    source_file: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    text: str
    token_count: int
    source_sha256: str
    is_demo: bool = False
    chunk_size: int | None = None
    overlap: int | None = None


@dataclass(frozen=True)
class BenchmarkQuestion:
    question_id: str
    question: str
    answerable: bool
    reference_answer: str | None
    question_type: str
    required_facts: list[str]
    supporting_evidence: list[EvidenceSpan]
    annotation_verified: bool
    is_demo: bool = False


@dataclass(frozen=True)
class RetrievedPassage:
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    raw_score: float
    method: str
    reranker_score: float | None = None
    char_start: int = 0
    char_end: int = 0


def _required(record: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"{context}: missing required field(s): {', '.join(missing)}")


def _string(value: Any, name: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be {'a string' if empty else 'a non-empty string'}")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _boolean(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def validate_page_record(obj: Any) -> PageRecord:
    ctx = "page record"
    if not isinstance(obj, dict):
        raise ValueError(f"{ctx} must be a JSON object")
    _required(obj, ("doc_id", "title", "source_file", "page_number", "text", "metadata", "source_sha256", "is_demo"), ctx)
    for key in ("doc_id", "title", "source_file", "source_sha256"):
        _string(obj[key], key)
    _integer(obj["page_number"], "page_number", 1)
    _string(obj["text"], "text", empty=True)
    if not isinstance(obj["metadata"], dict):
        raise ValueError("metadata must be an object")
    for key in ("authors", "year", "subject"):
        if key not in obj["metadata"]:
            raise ValueError(f"metadata.{key} is required; use null or an empty value when unknown")
    if not (obj["metadata"]["authors"] is None or isinstance(obj["metadata"]["authors"], list)):
        raise ValueError("metadata.authors must be a list or null")
    if not (obj["metadata"]["year"] is None or isinstance(obj["metadata"]["year"], (int, str))):
        raise ValueError("metadata.year must be a year value or null")
    if not (obj["metadata"]["subject"] is None or isinstance(obj["metadata"]["subject"], str)):
        raise ValueError("metadata.subject must be a string or null")
    _boolean(obj["is_demo"], "is_demo")
    return PageRecord(**{key: obj[key] for key in ("doc_id", "title", "source_file", "page_number", "text", "metadata", "source_sha256", "is_demo")})


def validate_chunk_record(obj: Any) -> ChunkRecord:
    ctx = "chunk record"
    if not isinstance(obj, dict):
        raise ValueError(f"{ctx} must be a JSON object")
    keys = ("chunk_id", "doc_id", "title", "source_file", "page_start", "page_end", "char_start", "char_end", "text", "token_count", "source_sha256", "is_demo")
    _required(obj, keys, ctx)
    for key in ("chunk_id", "doc_id", "title", "source_file", "source_sha256"):
        _string(obj[key], key)
    for key, minimum in (("page_start", 1), ("page_end", 1), ("char_start", 0), ("char_end", 0), ("token_count", 1)):
        _integer(obj[key], key, minimum)
    if obj["page_end"] < obj["page_start"] or obj["char_end"] < obj["char_start"]:
        raise ValueError("page_end/page_start or char_end/char_start range is reversed")
    _string(obj["text"], "text")
    _boolean(obj["is_demo"], "is_demo")
    for optional in ("chunk_size", "overlap"):
        if optional in obj and obj[optional] is not None:
            _integer(obj[optional], optional, 0)
    return ChunkRecord(**{key: obj[key] for key in ChunkRecord.__dataclass_fields__ if key in obj})


def validate_question_record(obj: Any) -> BenchmarkQuestion:
    ctx = "benchmark question"
    if not isinstance(obj, dict):
        raise ValueError(f"{ctx} must be a JSON object")
    keys = ("question_id", "question", "answerable", "reference_answer", "question_type", "required_facts", "supporting_evidence", "annotation_verified", "is_demo")
    _required(obj, keys, ctx)
    _string(obj["question_id"], "question_id")
    _string(obj["question"], "question")
    _boolean(obj["answerable"], "answerable")
    _boolean(obj["annotation_verified"], "annotation_verified")
    _boolean(obj["is_demo"], "is_demo")
    if obj["question_type"] not in QUESTION_TYPES:
        raise ValueError(f"question_type must be one of {sorted(QUESTION_TYPES)}")
    if not (obj["reference_answer"] is None or isinstance(obj["reference_answer"], str)):
        raise ValueError("reference_answer must be a string or null")
    if not isinstance(obj["required_facts"], list) or any(not isinstance(item, str) for item in obj["required_facts"]):
        raise ValueError("required_facts must be a list of strings")
    if not isinstance(obj["supporting_evidence"], list):
        raise ValueError("supporting_evidence must be a list")
    evidence: list[EvidenceSpan] = []
    for index, item in enumerate(obj["supporting_evidence"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"supporting_evidence[{index}] must be an object")
        _required(item, ("doc_id", "page_start", "page_end", "char_start", "char_end", "quote"), f"supporting_evidence[{index}]")
        _string(item["doc_id"], f"supporting_evidence[{index}].doc_id")
        _string(item["quote"], f"supporting_evidence[{index}].quote")
        for key, minimum in (("page_start", 1), ("page_end", 1), ("char_start", 0), ("char_end", 1)):
            _integer(item[key], f"supporting_evidence[{index}].{key}", minimum)
        if item["page_end"] < item["page_start"] or item["char_end"] < item["char_start"]:
            raise ValueError(f"supporting_evidence[{index}] has reversed offsets/pages")
        evidence.append(EvidenceSpan(**item))
    if obj["answerable"]:
        if not isinstance(obj["reference_answer"], str) or not obj["reference_answer"].strip():
            raise ValueError("answerable questions require a non-empty reference_answer")
        if not evidence:
            raise ValueError("answerable questions require at least one supporting_evidence span")
    elif obj["reference_answer"] is not None or evidence:
        raise ValueError("unanswerable questions require reference_answer=null and empty supporting_evidence")
    return BenchmarkQuestion(
        question_id=obj["question_id"], question=obj["question"], answerable=obj["answerable"],
        reference_answer=obj["reference_answer"], question_type=obj["question_type"],
        required_facts=obj["required_facts"], supporting_evidence=evidence,
        annotation_verified=obj["annotation_verified"], is_demo=obj["is_demo"],
    )


def read_jsonl(path: str | Path, validator: Any) -> list[Any]:
    """Read every record and surface the source path and line on any error."""
    source = Path(path)
    records = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{source}:{line_number}: blank JSONL records are not allowed")
                try:
                    decoded = json.loads(line)
                    records.append(validator(decoded))
                except Exception as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read {source}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            obj = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(target)


def validate_evidence_against_pages(question: BenchmarkQuestion, pages: Iterable[PageRecord]) -> None:
    """Require every annotated quote to occur exactly at its source offsets."""
    by_doc: dict[str, list[PageRecord]] = {}
    for page in pages:
        by_doc.setdefault(page.doc_id, []).append(page)
    full_text: dict[str, str] = {}
    for doc_id, doc_pages in by_doc.items():
        ordered = sorted(doc_pages, key=lambda p: p.page_number)
        full_text[doc_id] = "\n\n".join(page.text for page in ordered)
    for idx, span in enumerate(question.supporting_evidence, 1):
        source = full_text.get(span.doc_id)
        if source is None:
            raise ValueError(f"{question.question_id}: evidence span {idx} references unknown doc_id {span.doc_id}")
        if span.char_end > len(source) or source[span.char_start:span.char_end] != span.quote:
            raise ValueError(f"{question.question_id}: evidence span {idx} quote does not match source text at the supplied offsets")
