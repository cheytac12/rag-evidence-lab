# Data, privacy, and result files

RAG Evidence Lab works with any user-selected PDF subject. The project repository is a software and documentation baseline; it should not become a copy of a user's private corpus or result history.

## Local file locations

| Path | Purpose | Git behavior |
|---|---|---|
| `data/pdfs/` | User-provided source PDFs | Ignored; `.gitkeep` only is shareable |
| `data/processed/pages.jsonl` | Locally extracted page text and provenance | Ignored |
| `data/processed/chunks.jsonl` | Locally derived chunk text and offsets | Ignored |
| `data/benchmark_dev.jsonl` | Real development questions and evidence annotations | Ignored |
| `data/benchmark_test.jsonl` | Real held-out questions and evidence annotations | Ignored |
| `data/demo/` | Generic synthetic example fixture | Included; records are labeled `is_demo: true` |
| `results/experiments.sqlite3` | Runs, per-question outputs, evaluation fields, human review timestamps | Ignored |
| `results/embedding_cache/` | Local dense-vector cache | Ignored |
| `results/frozen_test_config.yaml` and checksum | Frozen test-stage configuration snapshot | Ignored |
| `.env` | Optional local OpenAI key and model setting | Ignored |

The root `.gitignore` keeps local content out of Git by default. This includes the documents themselves and derived page/chunk text, which can reproduce source content. Do not force-add these files to a GitHub repository unless you have deliberately reviewed and intend to publish them.

## What gets sent to the configured API

The code does not send PDFs or the full corpus during ingestion and retrieval. If a user configures the OpenAI Responses API:

- Answer generation sends the question and selected retrieved passages.
- Claim evaluation sends the question, generated answer, extracted claims, and selected passages.
- Correctness evaluation sends the question, answer, reference answer, and required facts.

These calls can consume paid API usage. The API is separate from ChatGPT billing. Keep credentials out of the repository and do not configure the API unless you accept the account's terms and possible charges. Local ingestion and retrieval work without an API key.

## Benchmark records and review

Real benchmark JSONL files are created for a specific user corpus and must remain separate from the synthetic demo. Each answerable question stores a reference answer, required facts, source document ID, page range, character offsets, exact quote, a verification flag, and `is_demo: false`. Unanswerable items store no reference answer or evidence span. The validator checks exact quote text at the recorded offsets.

The expected formal split sizes are 20 development questions and 60 test questions, with all annotations reviewed before formal runs. More detail and a record checklist are in [`../reports/annotation_guide.md`](../reports/annotation_guide.md).

## Local experiment records

The SQLite store contains configuration metadata and one result record per question, including retrieved passages, model answer/claims when available, citations, metric values, latency, token counts, source fingerprints, and review status. Automated evaluation fields are not human review. The UI can export:

- per-question results CSV;
- aggregate configuration metrics CSV;
- full run records and metadata JSON;
- frozen test configuration YAML;
- benchmark validation report JSON.

These exports can contain questions, answers, passages, or user source-derived text. Review them before sharing, even if generated automatically.

## Repository publishing checklist

Before publishing or pushing a local repository:

1. Confirm the destination repository visibility and owner.
2. Inspect `git status` and staged file names; verify `.env`, PDFs, extracted JSONL, real benchmark files, result databases, caches, and frozen snapshots are absent.
3. Check that demo-only data is labeled and that no demo metrics are presented as research results.
4. Review exports separately before attaching or publishing them.

Git ignore rules prevent accidental adds, but they cannot remove data that was already committed or explicitly force-added.
