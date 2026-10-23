from pathlib import Path

import fitz

from src.document_loader import ingest_pdfs
from src.schemas import read_jsonl, validate_page_record


def _make_pdf(path: Path):
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "First page: 12 items. Keep punctuation!")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Second page has source text.")
    doc.save(path)
    doc.close()


def test_page_provenance_and_stable_ingestion(tmp_path):
    pdf = tmp_path / "source.pdf"
    _make_pdf(pdf)
    out1, out2 = tmp_path / "pages1.jsonl", tmp_path / "pages2.jsonl"
    first = ingest_pdfs(tmp_path, out1)
    second = ingest_pdfs(tmp_path, out2)
    pages = read_jsonl(out1, validate_page_record)
    assert first.page_count == second.page_count == 2
    assert [p.page_number for p in pages] == [1, 2]
    assert all(p.source_file == "source.pdf" for p in pages)
    assert pages[0].doc_id == pages[1].doc_id
    assert len(pages[0].source_sha256) == 64
    assert "12 items." in pages[0].text
    assert out1.read_bytes() == out2.read_bytes()


def test_duplicate_pdf_checksum_is_not_indexed_twice(tmp_path):
    _make_pdf(tmp_path / "a.pdf")
    (tmp_path / "b.pdf").write_bytes((tmp_path / "a.pdf").read_bytes())
    output = tmp_path / "pages.jsonl"
    report = ingest_pdfs(tmp_path, output)
    pages = read_jsonl(output, validate_page_record)
    assert len(report.duplicate_files) == 1
    assert len(pages) == 2
