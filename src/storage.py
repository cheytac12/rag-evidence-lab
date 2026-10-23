"""SQLite persistence for reproducible run metadata and per-question results."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


class ExperimentStore:
    def __init__(self, path: str | Path = "results/experiments.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self):
        with self._connect() as conn:
            conn.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              split_name TEXT NOT NULL,
              data_label TEXT NOT NULL CHECK(data_label IN ('demo','real')),
              corpus_sha256 TEXT NOT NULL,
              benchmark_sha256 TEXT NOT NULL,
              configuration_id TEXT NOT NULL,
              settings_json TEXT NOT NULL,
              embedding_model TEXT,
              reranker_model TEXT,
              generator_model TEXT,
              prompt_version TEXT,
              status TEXT NOT NULL CHECK(status IN ('running','complete','failed','blocked')),
              error_message TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_runs_settings ON runs(split_name, data_label, configuration_id);
            CREATE TABLE IF NOT EXISTS question_results (
              run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
              question_id TEXT NOT NULL,
              result_json TEXT NOT NULL,
              PRIMARY KEY(run_id, question_id)
            );
            CREATE TABLE IF NOT EXISTS reviews (
              run_id TEXT NOT NULL,
              question_id TEXT NOT NULL,
              reviewer TEXT NOT NULL,
              reviewed_at TEXT NOT NULL,
              note TEXT,
              PRIMARY KEY(run_id, question_id),
              FOREIGN KEY(run_id, question_id) REFERENCES question_results(run_id, question_id) ON DELETE CASCADE
            );
            """)

    def create_run(self, *, run_id: str, split: str, data_label: str, corpus_sha256: str,
                   benchmark_sha256: str, configuration_id: str, settings: dict[str, Any],
                   embedding_model: str | None = None, reranker_model: str | None = None,
                   generator_model: str | None = None, prompt_version: str | None = None,
                   status: str = "running", error_message: str | None = None) -> None:
        if data_label not in {"demo", "real"}:
            raise ValueError("data_label must be demo or real")
        with self._connect() as conn:
            conn.execute("""INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, datetime.now(timezone.utc).isoformat(), split, data_label, corpus_sha256,
                 benchmark_sha256, configuration_id, json.dumps(settings, sort_keys=True), embedding_model,
                 reranker_model, generator_model, prompt_version, status, error_message))

    def add_question_result(self, run_id: str, question_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO question_results(run_id,question_id,result_json) VALUES (?,?,?)",
                         (run_id, question_id, json.dumps(payload, ensure_ascii=False, sort_keys=True)))

    def finish_run(self, run_id: str, status: str, error_message: str | None = None) -> None:
        if status not in {"complete", "failed", "blocked"}:
            raise ValueError("final run status must be complete, failed, or blocked")
        with self._connect() as conn:
            conn.execute("UPDATE runs SET status=?, error_message=? WHERE run_id=?", (status, error_message, run_id))

    def set_running(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE runs SET status='running', error_message=NULL WHERE run_id=?", (run_id,))

    def delete_run(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))

    def run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def runs(self, *, data_label: str | None = None, split: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        for column, value in (("data_label", data_label), ("split_name", split), ("status", status)):
            if value is not None and value != "All":
                where.append(f"{column}=?")
                args.append(value)
        sql = "SELECT * FROM runs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, args)]

    def question_results(self, run_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT r.*, q.question_id, q.result_json FROM runs r JOIN question_results q USING(run_id)"
        args: tuple[Any, ...] = ()
        if run_id:
            sql += " WHERE r.run_id=?"
            args = (run_id,)
        sql += " ORDER BY r.created_at, q.question_id"
        with self._connect() as conn:
            output = []
            for row in conn.execute(sql, args):
                item = dict(row)
                item["settings"] = json.loads(item.pop("settings_json"))
                item["result"] = json.loads(item.pop("result_json"))
                output.append(item)
            return output

    def completed_same_fingerprints(self, split: str, data_label: str, configuration_id: str,
                                    corpus_sha256: str, benchmark_sha256: str) -> list[dict[str, Any]]:
        return [r for r in self.runs(data_label=data_label, split=split, status="complete")
                if r["configuration_id"] == configuration_id and r["corpus_sha256"] == corpus_sha256
                and r["benchmark_sha256"] == benchmark_sha256]

    def mark_reviewed(self, run_id: str, question_id: str, reviewer: str, note: str | None = None) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer label cannot be empty")
        with self._connect() as conn:
            conn.execute("""INSERT OR REPLACE INTO reviews(run_id,question_id,reviewer,reviewed_at,note)
                            VALUES(?,?,?,?,?)""", (run_id, question_id, reviewer.strip(), datetime.now(timezone.utc).isoformat(), note))

    def review(self, run_id: str, question_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reviews WHERE run_id=? AND question_id=?", (run_id, question_id)).fetchone()
            return dict(row) if row else None

    def export_json(self, runs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        selected_ids = {row["run_id"] for row in runs} if runs is not None else None
        chosen = [row for row in self.runs() if selected_ids is None or row["run_id"] in selected_ids]
        output_runs = []
        for run in chosen:
            run["settings"] = json.loads(run.pop("settings_json"))
            run["questions"] = [item["result"] for item in self.question_results(run["run_id"])]
            output_runs.append(run)
        return {"runs": output_runs}
