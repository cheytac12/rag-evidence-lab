"""Controlled experiment expansion, split validation, freezing and execution."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any
import uuid

from .answer_metrics import claim_metric_summary
from .chunking import CHUNK_OVERLAPS, chunk_pages
from .document_loader import sha256_file
from .embeddings import DEFAULT_EMBEDDING
from .generator import (EVALUATOR_PROMPT_VERSION, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, PROMPT_VERSION,
                        configured_model, configured_provider, evaluate_claims, evaluate_correctness,
                        generate_answer, llm_configured, missing_configuration_message)
from .retrieval_bm25 import retrieve_bm25
from .retrieval_hybrid import retrieve_hybrid
from .retrieval_metrics import retrieval_metrics, unanswerable_metrics
from .reranker import RERANKER_MODEL, rerank_passages
from .schemas import (BenchmarkQuestion, ChunkRecord, EvidenceSpan, PageRecord,
                      RetrievedPassage, read_jsonl, validate_chunk_record,
                      validate_evidence_against_pages, validate_page_record,
                      validate_question_record)
from .storage import ExperimentStore

ALLOWED_RETRIEVERS = {"bm25", "dense", "hybrid"}
TIE_ORDER = {"hybrid": 0, "dense": 1, "bm25": 2}


def _yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for experiment configuration. Install requirements.txt first.") from exc
    return yaml


def load_matrix(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        matrix = _yaml().safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot load experiment matrix {target}: {exc}") from exc
    if not isinstance(matrix, dict):
        raise ValueError("Experiment matrix must be a YAML object")
    validate_matrix(matrix)
    return matrix


def validate_matrix(matrix: dict[str, Any]) -> None:
    allowed = {(128, 24), (256, 48), (512, 96)}
    for item in matrix.get("chunk_settings", []):
        pair = (item.get("size"), item.get("overlap")) if isinstance(item, dict) else None
        if pair not in allowed:
            raise ValueError(f"Invalid chunk size/overlap in matrix: {pair}")
    if set(matrix.get("allowed_top_k", [])) != {3, 5, 10}:
        raise ValueError("Matrix must support exactly top_k values 3, 5, and 10")
    stages = matrix.get("stages")
    if not isinstance(stages, dict) or not {"E1", "E2", "E3", "E4", "E5"}.issubset(stages):
        raise ValueError("Matrix must define all stages E1 through E5")
    for stage, definition in stages.items():
        configs = definition.get("configurations", []) if isinstance(definition, dict) else []
        for config in configs:
            if "chunk_size" in config and config["chunk_size"] not in {128, 256, 512}:
                raise ValueError(f"{stage}/{config.get('id', config.get('role'))}: chunk_size must be 128, 256, or 512")
            if "top_k" in config and config["top_k"] not in {3, 5, 10}:
                raise ValueError(f"{stage}/{config.get('id', config.get('role'))}: top_k must be 3, 5, or 10")
            if "retriever" in config and config["retriever"] not in ALLOWED_RETRIEVERS | {"selected"}:
                raise ValueError(f"{stage}: unsupported retriever {config['retriever']}")


def file_sha256(path: str | Path) -> str:
    return sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def benchmark_counts(questions: list[BenchmarkQuestion], pages: list[PageRecord], split: str, path: str | Path) -> dict[str, Any]:
    for question in questions:
        validate_evidence_against_pages(question, pages)
    by_type = {name: sum(q.question_type == name for q in questions) for name in sorted({q.question_type for q in questions} | {"factual_lookup", "technical_explanation", "comparison", "multi_hop", "unanswerable"})}
    report = {
        "split": split,
        "source_file": str(path),
        "question_count": len(questions),
        "by_question_type": by_type,
        "answerability": {"answerable": sum(q.answerable for q in questions), "unanswerable": sum(not q.answerable for q in questions)},
        "verification": {"verified": sum(q.annotation_verified for q in questions), "unverified": sum(not q.annotation_verified for q in questions)},
        "data_label": {"demo": sum(q.is_demo for q in questions), "real": sum(not q.is_demo for q in questions)},
        "valid": True,
        "issues": [],
    }
    if any(question.is_demo for question in questions) and any(not question.is_demo for question in questions):
        report["valid"] = False
        report["issues"].append("Demo and real benchmark records may not be mixed in one split.")
    if any(page.is_demo for page in pages) and any(not page.is_demo for page in pages):
        report["valid"] = False
        report["issues"].append("Demo and real corpus pages may not be mixed in one benchmark.")
    if questions and pages and any(question.is_demo != pages[0].is_demo for question in questions):
        report["valid"] = False
        report["issues"].append("Benchmark and corpus data labels do not match.")
    duplicate_ids = sorted(question_id for question_id, count in Counter(q.question_id for q in questions).items() if count > 1)
    if duplicate_ids:
        report["valid"] = False
        report["issues"].append("Duplicate question_id values: " + ", ".join(duplicate_ids))
    return report


def load_benchmark(path: str | Path, pages: list[PageRecord]) -> tuple[list[BenchmarkQuestion], dict[str, Any]]:
    questions = read_jsonl(path, validate_question_record)
    report = benchmark_counts(questions, pages, Path(path).stem, path)
    return questions, report


def require_formal_benchmark(questions: list[BenchmarkQuestion], split: str) -> None:
    target = {"dev": 20, "test": 60}.get(split)
    if target is None:
        raise ValueError("Formal split must be 'dev' or 'test'")
    if len(questions) != target:
        raise ValueError(f"Formal {split} split is incomplete: expected {target} questions, found {len(questions)}")
    if any(q.is_demo for q in questions):
        raise ValueError("Demo questions cannot be included in formal runs")
    if any(not q.annotation_verified for q in questions):
        count = sum(not q.annotation_verified for q in questions)
        raise ValueError(f"Formal {split} split has {count} unverified annotation(s); verify every record before running")


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.mean(values) if values else None


def select_best(rows: list[dict[str, Any]], metric: str = "mrr", *, tie_order: dict[str, int] | None = None) -> dict[str, Any]:
    rows = [row for row in rows if _mean_metric(row.get("question_metrics", []), metric) is not None]
    if not rows:
        raise ValueError(f"No measured development {metric} values are available for configuration selection")
    tie_order = tie_order or {}
    ranked = sorted(rows, key=lambda row: (
        -(_mean_metric(row.get("question_metrics", []), metric) if _mean_metric(row.get("question_metrics", []), metric) is not None else -1),
        tie_order.get(row.get("settings", {}).get("retriever"), 99),
        str(row.get("configuration_id", "")),
    ))
    return ranked[0]


def _dev_result_groups(store: ExperimentStore, stage: str, data_label: str = "real") -> list[dict[str, Any]]:
    output = []
    for run in store.runs(data_label=data_label, split="dev"):
        if run["status"] not in {"complete", "blocked"}:
            continue
        settings = json.loads(run["settings_json"])
        if settings.get("stage") != stage:
            continue
        rows = store.question_results(run["run_id"])
        output.append({"run_id": run["run_id"], "configuration_id": run["configuration_id"], "settings": settings,
                       "question_metrics": [item["result"].get("retrieval_metrics", {}) for item in rows]})
    return output


def _stage_configurations(matrix: dict[str, Any], stage: str, store: ExperimentStore | None = None, data_label: str = "real") -> list[dict[str, Any]]:
    stages = matrix["stages"]
    if stage == "E1":
        return [{**config, "stage": stage} for config in stages[stage]["configurations"]]
    if stage == "E2":
        best_e1 = select_best(_dev_result_groups(store, "E1", data_label))
        size = best_e1["settings"]["chunk_size"]
        overlap = CHUNK_OVERLAPS[size]
        return [{**config, "stage": stage, "chunk_size": size, "overlap": overlap} for config in stages[stage]["configurations"]]
    if stage == "E3":
        best_e1 = select_best(_dev_result_groups(store, "E1", data_label))
        best_e2 = select_best(_dev_result_groups(store, "E2", data_label))
        size = best_e1["settings"]["chunk_size"]
        model = best_e2["settings"].get("embedding_model", DEFAULT_EMBEDDING)
        return [{"id": f"E3-{retriever}-k{k}", "stage": stage, "chunk_size": size,
                 "overlap": CHUNK_OVERLAPS[size], "embedding_model": model,
                 "retriever": retriever, "top_k": k, "reranker": False}
                for retriever in ("bm25", "dense", "hybrid") for k in (3, 5, 10)]
    if stage == "E4":
        e3_rows = _dev_result_groups(store, "E3", data_label)
        # Rank retriever methods by macro MRR; ties use the fixed prescribed order.
        method_rows = []
        for method in ("bm25", "dense", "hybrid"):
            matching = [row for row in e3_rows if row["settings"].get("retriever") == method
                        and _mean_metric(row.get("question_metrics", []), "mrr") is not None]
            if matching:
                # The strongest top-k variant of each method represents that method.
                method_rows.append(select_best(matching))
        strongest: dict[str, dict[str, Any]] = {}
        for row in method_rows:
            strongest[row["settings"]["retriever"]] = row
        selected = sorted(strongest.values(), key=lambda row: (
            -(_mean_metric(row["question_metrics"], "mrr") or 0), TIE_ORDER[row["settings"]["retriever"]]))[:2]
        configs = []
        for row in selected:
            base = row["settings"]
            for enabled in (False, True):
                configs.append({**base, "stage": stage,
                                "id": f"E4-{base['retriever']}-rerank-{str(enabled).lower()}",
                                "top_k": 5, "candidate_k": 20 if enabled else 5,
                                "reranker": enabled, "final_passages": 5})
        return configs
    if stage == "E5":
        return [{**config, "stage": stage} for config in stages[stage]["configurations"]]
    raise ValueError(f"Unknown experiment stage {stage}")


def _resolve_e5(matrix: dict[str, Any], store: ExperimentStore) -> list[dict[str, Any]]:
    e1 = select_best(_dev_result_groups(store, "E1"))
    e2 = select_best(_dev_result_groups(store, "E2"))
    e3rows = _dev_result_groups(store, "E3")
    best_dense = select_best([r for r in e3rows if r["settings"].get("retriever") == "dense"])
    best_hybrid = select_best([r for r in e3rows if r["settings"].get("retriever") == "hybrid"])
    e4rows = _dev_result_groups(store, "E4")
    reranked = [r for r in e4rows if r["settings"].get("reranker")]
    best_reranked = select_best(reranked)
    base_size = e1["settings"]["chunk_size"]
    baseline = {"id": "E5-bm25-baseline", "role": "baseline", "chunk_size": base_size,
                "overlap": CHUNK_OVERLAPS[base_size], "retriever": "bm25", "top_k": 5,
                "reranker": False, "embedding_model": None, "stage": "E5"}
    output = [baseline]
    for role, source in (("best_dense_dev", best_dense), ("best_hybrid_dev", best_hybrid), ("best_reranked_dev", best_reranked)):
        settings = source["settings"]
        output.append({"id": f"E5-{role}", "role": role, "chunk_size": settings["chunk_size"],
                       "overlap": settings["overlap"], "retriever": settings["retriever"],
                       "top_k": 5, "reranker": settings.get("reranker", False),
                       "embedding_model": settings.get("embedding_model", e2["settings"].get("embedding_model")),
                       "candidate_k": 20 if settings.get("reranker") else 5, "stage": "E5"})
    return output


def freeze_test_configuration(matrix_path: str | Path, output_path: str | Path = "results/frozen_test_config.yaml",
                              db_path: str | Path = "results/experiments.sqlite3") -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    store = ExperimentStore(db_path)
    for stage in ("E1", "E2", "E3", "E4"):
        rows = _dev_result_groups(store, stage)
        if not rows or all(not row["question_metrics"] for row in rows):
            raise ValueError(f"Cannot freeze test settings: stage {stage} has no measured development retrieval results")
    selected = _resolve_e5(matrix, store)
    frozen = {"matrix_sha256": file_sha256(matrix_path), "created_at": datetime.now(timezone.utc).isoformat(),
              "selection_split": "dev", "selection_rule": "Macro MRR; ties for E4 retriever ordering use hybrid, dense, BM25.",
              "generation": {"provider": configured_provider() if llm_configured() else None,
                             "model": configured_model() if llm_configured() else None,
                             "prompt_version": PROMPT_VERSION,
                             "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
                             "temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS,
                             "max_context_tokens": MAX_CONTEXT_TOKENS},
              "configurations": selected}
    yaml_text = _yaml().safe_dump(frozen, sort_keys=True, allow_unicode=True)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write the exact bytes used by the digest (without Windows newline translation).
    target.write_bytes(yaml_text.encode("utf-8"))
    digest = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_bytes((digest + "  " + target.name + "\n").encode("utf-8"))
    return {**frozen, "snapshot_sha256": digest, "snapshot_path": str(target)}


def load_frozen_configuration(path: str | Path, matrix_path: str | Path) -> tuple[dict[str, Any], str]:
    target = Path(path)
    try:
        raw = target.read_bytes()
        snapshot = _yaml().safe_load(raw.decode("utf-8"))
        expected = target.with_suffix(target.suffix + ".sha256").read_text(encoding="utf-8").split()[0]
    except Exception as exc:
        raise ValueError(f"Frozen test configuration or checksum is missing/invalid: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError("Frozen test configuration checksum mismatch; snapshot may have changed")
    matrix_hash = file_sha256(matrix_path)
    if snapshot.get("matrix_sha256") != matrix_hash:
        raise ValueError("Experiment matrix differs from the matrix used for the frozen test configuration")
    configs = snapshot.get("configurations")
    if not isinstance(configs, list) or len(configs) != 4:
        raise ValueError("Frozen held-out configuration must contain exactly four settings")
    for config in configs:
        validate_configuration(config)
    generation = snapshot.get("generation")
    expected_generation = {"prompt_version": PROMPT_VERSION, "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
                           "temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS,
                           "max_context_tokens": MAX_CONTEXT_TOKENS}
    if not isinstance(generation, dict) or any(generation.get(key) != value for key, value in expected_generation.items()):
        raise ValueError("Frozen generation settings do not match the current fixed prompt/decoding/context settings")
    return snapshot, actual


def validate_configuration(config: dict[str, Any]) -> None:
    if config.get("retriever") not in ALLOWED_RETRIEVERS:
        raise ValueError(f"Unsupported retriever: {config.get('retriever')}")
    if config.get("top_k") not in (3, 5, 10):
        raise ValueError("top_k must be one of 3, 5, or 10")
    size, overlap = config.get("chunk_size"), config.get("overlap")
    if (size, overlap) not in {(128, 24), (256, 48), (512, 96)}:
        raise ValueError(f"Invalid chunk size/overlap pair: {(size, overlap)}")
    if config.get("reranker") and config.get("top_k") != 5:
        raise ValueError("Reranked comparisons must return exactly five passages")
    embedding = config.get("embedding_model")
    if config.get("retriever") in {"dense", "hybrid"} and not embedding:
        raise ValueError("Dense and hybrid retrieval require an embedding_model")


def _retrieve(query: str, chunks: list[ChunkRecord], config: dict[str, Any], cache_dir: str,
              retrieve_k: int | None = None) -> list[RetrievedPassage]:
    method = config["retriever"]
    k = retrieve_k or (config.get("candidate_k", 20) if config.get("reranker") else config["top_k"])
    embedding = config.get("embedding_model", DEFAULT_EMBEDDING)
    if method == "bm25":
        return retrieve_bm25(query, chunks, top_k=k)
    if method == "dense":
        from .embeddings import retrieve_dense
        return retrieve_dense(query, chunks, top_k=k, model_id=embedding, cache_dir=cache_dir)
    return retrieve_hybrid(query, chunks, top_k=k, model_id=embedding, candidate_k=k, cache_dir=cache_dir)


def _configuration_id(config: dict[str, Any]) -> str:
    return config.get("id") or "cfg-" + canonical_sha256(config)[:12]


def aggregate_question_results(rows: list[dict[str, Any]], *, data_label: str, split: str | None = None) -> dict[str, Any]:
    """Aggregate only the requested label/split; demo and real rows cannot mix."""
    if data_label not in {"demo", "real"}:
        raise ValueError("Aggregate requires an explicit data_label of demo or real")
    selected = [row for row in rows if row.get("data_label") == data_label and (split is None or row.get("split_name", row.get("split")) == split)]
    metrics: dict[str, list[float]] = {"recall_at_5": [], "hit_at_5": [], "precision_at_5": [], "mrr": [], "ndcg": [],
                                      "faithfulness": [], "correctness": [], "unsupported_answer_rate": [], "median_total_latency": [],
                                      "citation_correctness": [], "false_positive_answer_rate": [], "abstention_rate": []}
    for row in selected:
        result = row.get("result", row)
        retrieval = result.get("retrieval_metrics", {})
        metrics_at_5 = result.get("retrieval_metrics_at_5", retrieval)
        for target, source in (("recall_at_5", "recall"), ("hit_at_5", "hit"), ("precision_at_5", "precision"), ("mrr", "mrr"), ("ndcg", "ndcg")):
            value = metrics_at_5.get(source) if target.endswith("at_5") else retrieval.get(source)
            if value is not None:
                metrics[target].append(float(value))
        value = result.get("faithfulness")
        if value is not None: metrics["faithfulness"].append(float(value))
        correctness = result.get("correctness_automated")
        if isinstance(correctness, dict): correctness = correctness.get("score")
        if correctness is not None: metrics["correctness"].append(float(correctness))
        for target, source in (("unsupported_answer_rate", "unsupported_answer_rate"), ("citation_correctness", "citation_correctness_rate")):
            value = result.get(source)
            if value is not None: metrics[target].append(float(value))
        latency = result.get("latency", {}).get("total_seconds")
        if latency is not None: metrics["median_total_latency"].append(float(latency))
        unanswerable = result.get("unanswerable_metrics", {})
        for target, source in (("false_positive_answer_rate", "false_positive_answer"), ("abstention_rate", "abstention")):
            value = unanswerable.get(source)
            if value is not None: metrics[target].append(float(value))
    reduced = {}
    for key, values in metrics.items():
        if not values:
            reduced[key] = {"value": None, "scored_question_count": 0}
        else:
            reduced[key] = {"value": statistics.median(values) if key == "median_total_latency" else statistics.mean(values),
                            "scored_question_count": len(values)}
    return {"data_label": data_label, "split": split, "question_count": len(selected), "metrics": reduced}


def _run_one_configuration(config: dict[str, Any], questions: list[BenchmarkQuestion], page_groups: dict[str, list[PageRecord]],
                           chunks_by_size: dict[int, list[ChunkRecord]], *, split: str, data_label: str,
                           corpus_hash: str, benchmark_hash: str, store: ExperimentStore,
                           cache_dir: str, overwrite: bool = False, require_generation: bool = False,
                           frozen_hash: str | None = None) -> dict[str, Any]:
    validate_configuration(config)
    cid = _configuration_id(config)
    fingerprint = canonical_sha256([split, data_label, cid, corpus_hash, benchmark_hash])[:24]
    run_id = "run-" + fingerprint
    existing = store.run(run_id)
    if existing and existing["status"] == "complete" and not overwrite:
        raise ValueError(f"Configuration {cid} already completed with these data fingerprints; rerun with --overwrite")
    if existing and overwrite:
        store.delete_run(run_id)
        existing = None
    if existing is None:
        status = "running"
        error = None
        stored_settings = {**config, "temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS,
                           "max_context_tokens": MAX_CONTEXT_TOKENS,
                           "prompt_version": PROMPT_VERSION,
                           "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION}
        store.create_run(run_id=run_id, split=split, data_label=data_label, corpus_sha256=corpus_hash,
                         benchmark_sha256=benchmark_hash, configuration_id=cid, settings=stored_settings,
                         embedding_model=config.get("embedding_model"),
                         reranker_model=RERANKER_MODEL if config.get("reranker") else None,
                         generator_model=configured_model() if require_generation and llm_configured() else None,
                         prompt_version=PROMPT_VERSION if require_generation else None, status=status)
    else:
        store.set_running(run_id)
    frozen_model = config.get("frozen_generator_model")
    frozen_provider = config.get("frozen_generator_provider", "openai")
    current_model = configured_model() if llm_configured() else None
    current_provider = configured_provider() if llm_configured() else None
    if require_generation and current_model and (current_model != frozen_model or current_provider != frozen_provider):
        reason = "Configured LLM provider or model differs from the frozen test configuration. Freeze again using development-only selection."
        store.finish_run(run_id, "blocked", reason)
        return {"run_id": run_id, "configuration_id": cid, "status": "blocked", "reason": reason}
    generation_available = llm_configured() if require_generation else False
    defer_e4_generation_block = require_generation and split == "dev" and config.get("stage") == "E4" and not generation_available
    if require_generation and not generation_available and not defer_e4_generation_block:
        reason = missing_configuration_message()
        store.finish_run(run_id, "blocked", reason)
        return {"run_id": run_id, "configuration_id": cid, "status": "blocked", "reason": reason}
    chunks = chunks_by_size.get(config["chunk_size"], [])
    if not chunks:
        reason = f"No chunks available for chunk size {config['chunk_size']}"
        store.finish_run(run_id, "blocked", reason)
        return {"run_id": run_id, "configuration_id": cid, "status": "blocked", "reason": reason}
    done_ids = {item["question_id"] for item in store.question_results(run_id)
                if not require_generation or item["result"].get("answer") is not None}
    try:
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for question in questions:
            if question.question_id in done_ids:
                continue
            t0 = time.perf_counter()
            retrieval_start = time.perf_counter()
            search_k = 20 if config.get("reranker") else max(config["top_k"], 5)
            candidates = _retrieve(question.question, chunks, config, cache_dir, retrieve_k=search_k)
            retrieval_latency = time.perf_counter() - retrieval_start
            rerank_latency = 0.0
            if config.get("reranker"):
                t = time.perf_counter()
                passages = rerank_passages(question.question, candidates, top_k=5)
                rerank_latency = time.perf_counter() - t
            else:
                passages = candidates[:config["top_k"]]
            if split == "test" and len(passages) > 5:
                passages = passages[:5]
            retrieval = retrieval_metrics(passages, question.supporting_evidence, 5 if split == "test" else config["top_k"], answerable=question.answerable)
            metrics_at_5 = retrieval_metrics(passages if config.get("reranker") or split == "test" else candidates[:5],
                                             question.supporting_evidence, 5, answerable=question.answerable)
            payload: dict[str, Any] = {
                "question_id": question.question_id,
                "question": question.question,
                "answerable": question.answerable,
                "reference_answer": question.reference_answer,
                "question_type": question.question_type,
                "required_evidence": [asdict(span) for span in question.supporting_evidence],
                "retrieved_passages": [asdict(passage) for passage in passages],
                "retrieval_metrics": retrieval,
                "retrieval_metrics_at_5": metrics_at_5,
                "latency": {"retrieval_seconds": retrieval_latency, "reranking_seconds": rerank_latency,
                            "generation_seconds": None, "total_seconds": None},
                "context_token_count": None,
                "output_token_count": None,
                "provider_cost": None,
                "answer": None,
                "claims": [], "claim_evaluations": [], "citations": [],
                "correctness_automated": None, "correctness_manual": None,
                "faithfulness": None, "unsupported_claim_rate": None,
                "unsupported_answer_rate": None, "citation_correctness_rate": None,
            }
            payload["context_token_count"] = sum(chunk_by_id[p.chunk_id].token_count for p in passages if p.chunk_id in chunk_by_id)
            payload["context_token_budget"] = MAX_CONTEXT_TOKENS
            if require_generation and payload["context_token_count"] > MAX_CONTEXT_TOKENS:
                raise ValueError(f"Selected context exceeds the frozen {MAX_CONTEXT_TOKENS}-token maximum")
            if require_generation:
                if not generation_available:
                    payload["generation_blocked_reason"] = missing_configuration_message()
                    payload["latency"]["total_seconds"] = time.perf_counter() - t0
                    store.add_question_result(run_id, question.question_id, payload)
                    continue
                gen_start = time.perf_counter()
                generated = generate_answer(question.question, passages)
                evaluations = evaluate_claims(question.question, generated["answer"], generated["claims"], passages)
                passages_by_id = {p.chunk_id: p for p in passages}
                eval_by_claim = {item["claim_text"]: item for item in evaluations}
                citations = []
                for citation in generated["citations"]:
                    evaluated = eval_by_claim.get(citation["claim_text"], {})
                    supported_ids = set(evaluated.get("supporting_passage_ids", []))
                    citations.append({**citation, "citation_supporting": bool(citation["citation_present"] and citation["passage_id"] in supported_ids and evaluated.get("label") == "SUPPORTED")})
                correctness = None
                if question.reference_answer is not None:
                    correctness = evaluate_correctness(question.question, generated["answer"], question.reference_answer, question.required_facts)
                cm = claim_metric_summary(generated["claims"], evaluations, citations)
                generation_latency = time.perf_counter() - gen_start
                payload.update({"answer": generated["answer"], "claims": generated["claims"],
                                "claim_evaluations": evaluations, "citations": citations,
                                "uncertainties": generated["uncertainties"],
                                "citation_presence": [citation["citation_present"] for citation in citations],
                                "citation_correctness_rate": cm["citation_correctness"],
                                "faithfulness": cm["faithfulness"],
                                "unsupported_claim_rate": cm["unsupported_claim_rate"],
                                "unsupported_answer_rate": cm["unsupported_answer"],
                                "correctness_automated": correctness,
                                "generator_model": generated["model"], "prompt_version": generated["prompt_version"],
                                "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
                                "generation_provider": generated["provider"],
                                "temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS,
                                "output_token_count": generated["output_tokens"],
                                "provider_cost": generated["provider_cost"]})
                if not question.answerable:
                    abstained = "The indexed documents do not provide enough evidence to answer this question.".casefold() in generated["answer"].casefold()
                    payload["unanswerable_metrics"] = unanswerable_metrics(generated["answer"], explicit_abstention=abstained)
                payload["latency"]["generation_seconds"] = generation_latency
            payload["latency"]["total_seconds"] = time.perf_counter() - t0
            store.add_question_result(run_id, question.question_id, payload)
        if defer_e4_generation_block:
            reason = missing_configuration_message() + " Retrieval metrics were recorded; answer metrics are unavailable."
            store.finish_run(run_id, "blocked", reason)
            return {"run_id": run_id, "configuration_id": cid, "status": "blocked", "reason": reason}
        store.finish_run(run_id, "complete")
        return {"run_id": run_id, "configuration_id": cid, "status": "complete", "question_count": len(questions)}
    except Exception as exc:
        # Error strings are sanitized so credentials can never be persisted accidentally.
        error = f"{type(exc).__name__}: {exc}"
        for secret_name in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
            error = error.replace(os.environ.get(secret_name, "\0"), "[REDACTED]")
        status = "blocked" if "required" in error.lower() or "could not load" in error.lower() or "blocked" in error.lower() else "failed"
        store.finish_run(run_id, status, error[:1000])
        return {"run_id": run_id, "configuration_id": cid, "status": status, "reason": error[:1000]}


def run_experiments(matrix_path: str | Path, split: str, *, pages_path: str | Path = "data/processed/pages.jsonl",
                    benchmark_path: str | Path | None = None, db_path: str | Path = "results/experiments.sqlite3",
                    cache_dir: str = "results/embedding_cache", overwrite: bool = False,
                    demo: bool = False, frozen_path: str | Path = "results/frozen_test_config.yaml") -> list[dict[str, Any]]:
    matrix = load_matrix(matrix_path)
    pages = read_jsonl(pages_path, validate_page_record)
    if not pages:
        raise ValueError(f"No pages found in {pages_path}; ingest PDFs first")
    if demo:
        if split != "dev":
            raise ValueError("Demo records may only be used in the development/demo split")
        if str(pages_path) == "data/processed/pages.jsonl":
            pages_path = "data/demo/demo_pages.jsonl"
        benchmark_path = benchmark_path or "data/demo/demo_questions.jsonl"
        data_label = "demo"
    else:
        benchmark_path = benchmark_path or ("data/benchmark_dev.jsonl" if split == "dev" else "data/benchmark_test.jsonl")
        data_label = "real"
    questions, report = load_benchmark(benchmark_path, pages)
    if not report["valid"]:
        raise ValueError("Benchmark validation failed: " + "; ".join(report["issues"]))
    if not demo:
        require_formal_benchmark(questions, split)
        if any(page.is_demo for page in pages):
            raise ValueError("Real experiment cannot use demo corpus pages")
    elif any(not page.is_demo for page in pages) or any(not q.is_demo for q in questions):
        raise ValueError("Demo run requires every page and question to be labeled is_demo=true")
    corpus_hash = file_sha256(pages_path)
    benchmark_hash = file_sha256(benchmark_path)
    store = ExperimentStore(db_path)
    if split == "test":
        snapshot, frozen_hash = load_frozen_configuration(frozen_path, matrix_path)
        if not llm_configured():
            # Create one blocked run per frozen configuration with the concrete dependency reason.
            configs = snapshot["configurations"]
        else:
            configs = snapshot["configurations"]
        configs = [{**config, "frozen_snapshot_sha256": frozen_hash,
                    "frozen_generator_model": snapshot["generation"].get("model"),
                    "frozen_generator_provider": snapshot["generation"].get("provider", "openai")} for config in configs]
        generation = True
    elif split == "dev":
        # E1 -> E2 -> E3 -> E4 selection is sequential and based only on this dev split.
        configs = _stage_configurations(matrix, "E1", data_label=data_label)
        generation = False
        chunks_by_size: dict[int, list[ChunkRecord]] = {}
        page_groups: dict[str, list[PageRecord]] = {}
        for page in pages:
            page_groups.setdefault(page.doc_id, []).append(page)
        results: list[dict[str, Any]] = []
        for stage in ("E1", "E2", "E3", "E4"):
            configs = _stage_configurations(matrix, stage, store if stage != "E1" else None, data_label=data_label)
            if stage == "E4":
                generation = True
            for config in configs:
                size = config["chunk_size"]
                if size not in chunks_by_size:
                    chunks_by_size[size] = chunk_pages(pages, size, CHUNK_OVERLAPS[size])
                results.append(_run_one_configuration(config, questions, page_groups, chunks_by_size,
                    split=split, data_label=data_label, corpus_hash=corpus_hash, benchmark_hash=benchmark_hash,
                    store=store, cache_dir=cache_dir, overwrite=overwrite, require_generation=generation))
        return results
    else:
        raise ValueError("split must be dev or test")

    page_groups = {}
    for page in pages:
        page_groups.setdefault(page.doc_id, []).append(page)
    chunks_by_size = {}
    results = []
    for config in configs:
        size = config["chunk_size"]
        try:
            if size not in chunks_by_size:
                chunks_by_size[size] = chunk_pages(pages, size, CHUNK_OVERLAPS[size])
            results.append(_run_one_configuration(config, questions, page_groups, chunks_by_size,
                split=split, data_label=data_label, corpus_hash=corpus_hash, benchmark_hash=benchmark_hash,
                store=store, cache_dir=cache_dir, overwrite=overwrite, require_generation=generation,
                frozen_hash=config.get("frozen_snapshot_sha256")))
        except Exception as exc:
            results.append({"configuration_id": _configuration_id(config), "status": "blocked", "reason": str(exc)})
    return results
