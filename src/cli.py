"""Command-line interface for ingestion, chunking and controlled experiments."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from .chunking import CHUNK_OVERLAPS, chunk_jsonl, validate_chunk_settings
from .document_loader import ingest_pdfs, report_dict
from .experiment_runner import (benchmark_counts, freeze_test_configuration,
                                load_benchmark, load_matrix, require_formal_benchmark,
                                run_experiments)
from .schemas import read_jsonl, validate_page_record, validate_question_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="RAG Evidence Lab project commands")
    subs = parser.add_subparsers(dest="command", required=True)
    ingest = subs.add_parser("ingest", help="Extract PDF pages locally with provenance")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--demo", action="store_true", help="Label extracted pages as demo-only")
    chunk = subs.add_parser("chunk", help="Create tokenizer-counted chunks")
    chunk.add_argument("--input", required=True)
    chunk.add_argument("--output", default="data/processed/chunks.jsonl")
    chunk.add_argument("--size", type=int, required=True)
    chunk.add_argument("--overlap", type=int, required=True)
    validate = subs.add_parser("validate-benchmark", help="Validate benchmark records and source spans")
    validate.add_argument("--split", choices=("dev", "test"), required=True)
    validate.add_argument("--demo", action="store_true", help="Validate the isolated synthetic demo benchmark")
    validate.add_argument("--benchmark", help="Override benchmark JSONL path")
    validate.add_argument("--pages", help="Override extracted pages JSONL path")
    run = subs.add_parser("run-experiments", help="Run benchmark questions through the experiment matrix")
    run.add_argument("--matrix", required=True)
    run.add_argument("--split", choices=("dev", "test"), required=True)
    run.add_argument("--overwrite", action="store_true", help="Explicitly replace existing runs with identical fingerprints")
    run.add_argument("--demo", action="store_true", help="Run only the isolated synthetic demo set")
    run.add_argument("--pages", help="Override page JSONL path")
    run.add_argument("--benchmark", help="Override benchmark JSONL path")
    run.add_argument("--database", default="results/experiments.sqlite3")
    freeze = subs.add_parser("freeze-test-config", help="Freeze four test configurations selected on dev data")
    freeze.add_argument("--matrix", required=True)
    freeze.add_argument("--output", default="results/frozen_test_config.yaml")
    freeze.add_argument("--database", default="results/experiments.sqlite3")
    return parser


def _validate_command(args) -> int:
    if args.demo:
        pages_path = args.pages or "data/demo/demo_pages.jsonl"
        benchmark_path = args.benchmark or "data/demo/demo_questions.jsonl"
    else:
        pages_path = args.pages or "data/processed/pages.jsonl"
        benchmark_path = args.benchmark or ("data/benchmark_dev.jsonl" if args.split == "dev" else "data/benchmark_test.jsonl")
    pages = read_jsonl(pages_path, validate_page_record)
    questions, report = load_benchmark(benchmark_path, pages)
    report["split"] = "demo" if args.demo else args.split
    if not report["valid"]:
        print(json.dumps(report, indent=2))
        return 1
    if args.demo:
        if not questions or not all(q.is_demo for q in questions) or not all(p.is_demo for p in pages):
            report["valid"] = False
            report["issues"].append("Demo validation requires all pages and questions to have is_demo=true.")
            print(json.dumps(report, indent=2))
            return 1
    else:
        try:
            require_formal_benchmark(questions, args.split)
        except ValueError as exc:
            report["valid"] = False
            report["issues"].append(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ingest":
            report = ingest_pdfs(args.input, args.output, is_demo=args.demo)
            print(json.dumps(report_dict(report), indent=2, ensure_ascii=False))
            if report.image_only_pages:
                print("OCR is not implemented. Pages with images and no extracted text are listed above.", file=sys.stderr)
            return 1 if report.failed_files else 0
        if args.command == "chunk":
            validate_chunk_settings(args.size, args.overlap)
            chunks = chunk_jsonl(args.input, args.output, args.size, args.overlap)
            print(json.dumps({"output": args.output, "chunk_count": len(chunks), "size": args.size,
                              "overlap": args.overlap, "demo_count": sum(c.is_demo for c in chunks),
                              "real_count": sum(not c.is_demo for c in chunks)}, indent=2))
            return 0
        if args.command == "validate-benchmark":
            return _validate_command(args)
        if args.command == "freeze-test-config":
            result = freeze_test_configuration(args.matrix, args.output, args.database)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "run-experiments":
            pages_path = args.pages or ("data/demo/demo_pages.jsonl" if args.demo else "data/processed/pages.jsonl")
            benchmark_path = args.benchmark
            results = run_experiments(args.matrix, args.split, pages_path=pages_path,
                benchmark_path=benchmark_path, db_path=args.database, overwrite=args.overwrite, demo=args.demo)
            print(json.dumps(results, indent=2, ensure_ascii=False))
            return 1 if any(item.get("status") in {"failed", "blocked"} for item in results) else 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for secret_name in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
            secret = os.environ.get(secret_name)
            if secret:
                message = message.replace(secret, "[REDACTED]")
        print(message, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
