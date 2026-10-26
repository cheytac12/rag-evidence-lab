import hashlib

import pytest
import yaml

from src.experiment_runner import aggregate_question_results, benchmark_counts, file_sha256, load_frozen_configuration
from src.generator import PROMPT_VERSION
from src.schemas import BenchmarkQuestion, PageRecord
from src.storage import ExperimentStore


def _frozen(path, matrix_path):
    configs = [
        {"id": "base", "stage": "E5", "chunk_size": 128, "overlap": 24, "retriever": "bm25", "top_k": 5, "reranker": False},
        {"id": "dense", "stage": "E5", "chunk_size": 128, "overlap": 24, "retriever": "dense", "top_k": 5, "reranker": False, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
        {"id": "hybrid", "stage": "E5", "chunk_size": 128, "overlap": 24, "retriever": "hybrid", "top_k": 5, "reranker": False, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
        {"id": "reranked", "stage": "E5", "chunk_size": 128, "overlap": 24, "retriever": "hybrid", "top_k": 5, "reranker": True, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
    ]
    body = yaml.safe_dump({"matrix_sha256": file_sha256(matrix_path),
                           "generation": {"model": None, "prompt_version": PROMPT_VERSION,
                                          "evaluator_prompt_version": "claim-review-v1", "temperature": 0,
                                          "max_output_tokens": 1200, "max_context_tokens": 3000},
                           "configurations": configs})
    path.write_bytes(body.encode("utf-8"))
    path.with_suffix(path.suffix + ".sha256").write_bytes(hashlib.sha256(body.encode()).hexdigest().encode("utf-8"))


def test_test_freeze_checksum_enforcement(tmp_path):
    matrix = "experiments/experiment_matrix.yaml"
    frozen = tmp_path / "frozen.yaml"
    _frozen(frozen, matrix)
    loaded, digest = load_frozen_configuration(frozen, matrix)
    assert len(loaded["configurations"]) == 4
    assert len(digest) == 64
    frozen.write_text(frozen.read_text() + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frozen_configuration(frozen, matrix)


def test_demo_results_never_enter_real_aggregate():
    rows = [
        {"data_label": "real", "split_name": "dev", "retrieval_metrics": {"recall": 0.5}},
        {"data_label": "demo", "split_name": "dev", "retrieval_metrics": {"recall": 1.0}},
    ]
    real = aggregate_question_results(rows, data_label="real", split="dev")
    demo = aggregate_question_results(rows, data_label="demo", split="dev")
    assert real["question_count"] == 1
    assert real["metrics"]["recall_at_5"]["value"] == 0.5
    assert demo["metrics"]["recall_at_5"]["value"] == 1.0
    with pytest.raises(ValueError, match="explicit data_label"):
        aggregate_question_results(rows, data_label="all")


def test_sqlite_persists_demo_label_and_results(tmp_path):
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    store.create_run(run_id="r1", split="dev", data_label="demo", corpus_sha256="c", benchmark_sha256="b",
                     configuration_id="cfg", settings={"retriever": "bm25"})
    store.add_question_result("r1", "q1", {"retrieval_metrics": {"recall": 1.0}})
    assert store.runs(data_label="real") == []
    assert store.question_results("r1")[0]["result"]["retrieval_metrics"]["recall"] == 1.0


def test_benchmark_rejects_duplicate_question_ids_and_mixed_demo_labels():
    page = PageRecord("d1", "Note", "note.pdf", 1, "", {"authors": [], "year": None, "subject": None}, "abc", True)
    question = BenchmarkQuestion("q1", "What is present?", False, None, "unanswerable", [], [], True, False)
    report = benchmark_counts([question, question], [page], "demo", "questions.jsonl")
    assert not report["valid"]
    assert any("Duplicate question_id" in issue for issue in report["issues"])
    assert any("data labels do not match" in issue for issue in report["issues"])
