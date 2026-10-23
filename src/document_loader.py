"""Deterministic, local-only PDF ingestion with page-level provenance."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import re
from pathlib import Path
from typing import Any

from .schemas import PageRecord, write_jsonl


@dataclass
class IngestionReport:
    processed_files: list[str]
    failed_files: list[dict[str, str]]
    empty_text_pages: list[dict[str, Any]]
    image_only_pages: list[dict[str, Any]]
    duplicate_files: list[dict[str, str]]
    page_count: int


def normalize_page_text(text: str) -> str:
    """Normalize layout whitespace while preserving punctuation, symbols and units."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if out and not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdfs(input_path: str | Path, output_path: str | Path, *, is_demo: bool = False) -> IngestionReport:
    """Extract PDFs in stable filename order. PDFs are never sent to a service."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF ingestion. Install requirements.txt first.") from exc

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"PDF input path does not exist: {source}")
    if source.is_file() and source.suffix.lower() != ".pdf":
        raise ValueError(f"Input file is not a PDF: {source.name}")
    files = [source] if source.is_file() and source.suffix.lower() == ".pdf" else sorted(
        (p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: (p.name.casefold(), p.name),
    )
    if source.is_dir() and not files:
        raise ValueError(f"No PDF files found in {source}")

    pages: list[PageRecord] = []
    report = IngestionReport([], [], [], [], [], 0)
    seen: dict[str, str] = {}
    for path in files:
        try:
            checksum = sha256_file(path)
        except OSError as exc:
            report.failed_files.append({"file": path.name, "reason": str(exc)})
            continue
        if checksum in seen:
            report.duplicate_files.append({"file": path.name, "duplicate_of": seen[checksum], "sha256": checksum})
            continue
        seen[checksum] = path.name
        try:
            doc = fitz.open(path)
            if getattr(doc, "is_pdf", True) is False:
                raise ValueError("File is not a readable PDF")
            metadata = doc.metadata or {}
            raw_title = metadata.get("title")
            title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else path.stem
            authors_raw = metadata.get("author")
            authors = [authors_raw.strip()] if isinstance(authors_raw, str) and authors_raw.strip() else []
            for page_idx, page in enumerate(doc, start=1):
                text = normalize_page_text(page.get_text("text", sort=True))
                if not text:
                    image_count = len(page.get_images(full=True))
                    record = {"file": path.name, "page": page_idx, "image_count": image_count}
                    report.empty_text_pages.append(record)
                    if image_count:
                        report.image_only_pages.append(record)
                pages.append(PageRecord(
                    doc_id=f"doc_{checksum[:20]}", title=title, source_file=path.name,
                    page_number=page_idx, text=text,
                    metadata={"authors": authors, "year": None, "subject": None},
                    source_sha256=checksum, is_demo=is_demo,
                ))
                report.page_count += 1
            doc.close()
            report.processed_files.append(path.name)
        except Exception as exc:
            # Remove any page rows already emitted for this unreadable file.
            pages = [page for page in pages if page.source_file != path.name]
            report.failed_files.append({"file": path.name, "reason": f"{type(exc).__name__}: {exc}"})
    write_jsonl(output_path, pages)
    return report


def report_dict(report: IngestionReport) -> dict[str, Any]:
    return asdict(report)
