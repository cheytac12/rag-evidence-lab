import json

import pytest

from src.schemas import validate_chunk_record, validate_page_record, validate_question_record, read_jsonl, validate_evidence_against_pages


def page_record():
    return {"doc_id": "d1", "title": "Note", "source_file": "note.pdf", "page_number": 1,
            "text": "Source sentence.", "metadata": {"authors": [], "year": None, "subject": None},
            "source_sha256": "abc", "is_demo": False}


def test_jsonl_validates_and_reports_path_and_line(tmp_path):
    path = tmp_path / "pages.jsonl"
    path.write_text(json.dumps(page_record()) + "\n", encoding="utf-8")
    assert len(read_jsonl(path, validate_page_record)) == 1
    path.write_text(json.dumps(page_record()) + "\n{\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"pages.jsonl:2"):
        read_jsonl(path, validate_page_record)


def test_required_fields_and_page_numbers_are_strict():
    record = page_record()
    del record["source_sha256"]
    with pytest.raises(ValueError, match="source_sha256"):
        validate_page_record(record)
    record = page_record()
    record["page_number"] = 0
    with pytest.raises(ValueError, match="page_number"):
        validate_page_record(record)


def test_chunk_and_question_contracts():
    chunk = {"chunk_id": "c1", "doc_id": "d1", "title": "Note", "source_file": "note.pdf",
             "page_start": 1, "page_end": 1, "char_start": 0, "char_end": 4,
             "text": "Text", "token_count": 1, "source_sha256": "abc", "is_demo": False}
    assert validate_chunk_record(chunk).chunk_id == "c1"
    question = {"question_id": "q1", "question": "What?", "answerable": True,
                "reference_answer": "A.", "question_type": "factual_lookup", "required_facts": ["A"],
                "supporting_evidence": [{"doc_id": "d1", "page_start": 1, "page_end": 1,
                    "char_start": 0, "char_end": 16, "quote": "Source sentence."}],
                "annotation_verified": False, "is_demo": False}
    parsed = validate_question_record(question)
    validate_evidence_against_pages(parsed, [validate_page_record(page_record())])
    question["supporting_evidence"][0]["quote"] = "incorrect"
    parsed = validate_question_record(question)
    with pytest.raises(ValueError, match="does not match source text"):
        validate_evidence_against_pages(parsed, [validate_page_record(page_record())])


def test_unanswerable_record_requires_null_and_empty_evidence():
    record = {"question_id": "q2", "question": "Absent?", "answerable": False,
              "reference_answer": "No", "question_type": "unanswerable", "required_facts": [],
              "supporting_evidence": [], "annotation_verified": False, "is_demo": False}
    with pytest.raises(ValueError, match="reference_answer=null"):
        validate_question_record(record)
