# RAG Evidence Lab

**A domain-agnostic application and reproducible experiment framework for evaluating evidence retrieval in Retrieval-Augmented Generation (RAG).** It accepts PDF collections supplied by the user, shows which passages were retrieved, and compares controlled retrieval configurations against a source-grounded benchmark.

> **Research status:** the repository intentionally contains no real user corpus, verified benchmark, or formal real-corpus results. A small generic synthetic fixture is included for a labeled `DEMO` mode only. Demo behavior and scores are excluded from real evaluation and conclusions.

## What this project investigates

The research question is: **How do chunk size, embedding model, retrieval method, number of retrieved passages, and reranking affect evidence retrieval, answer quality, and runtime for a supplied PDF collection?**

The project is designed to measure corpus-specific behavior. It does not assume a subject domain or claim that one configuration is best for every collection.

## What is compared

The experiment matrix in [`experiments/experiment_matrix.yaml`](experiments/experiment_matrix.yaml) defines five stages. Each stage changes a controlled set of factors; development results select settings for later stages.

| Stage | Comparison | Main control / purpose |
|---|---|---|
| E1 | Three chunk sizes (128, 256, 512 tokens) and BM25 vs. dense retrieval | Compare chunking and lexical/semantic retrieval at `top_k=5`. Overlaps are 24, 48, and 96 tokens. |
| E2 | Two embedding models with dense and hybrid retrieval | Use the E1 development-selected chunk setting; compare MiniLM and BGE embeddings. |
| E3 | BM25, dense, and hybrid retrieval at `top_k=3, 5, 10` | Use the development-selected chunk and embedding settings. |
| E4 | Reranking off vs. on for the two strongest base methods | Select base methods using development MRR; the reranker orders up to 20 candidates and returns five. |
| E5 | Four selected configurations on held-out questions | Compare a BM25 baseline, best dense, best hybrid, and best reranked method after freezing the configuration. |

The generation model and prompt are held constant within a run so retrieval choices can be compared. See [`docs/experiments.md`](docs/experiments.md) for selection rules, metrics, and the held-out protocol.

## Where to see results

The web app navigation separates interactive questions from benchmark results:

| App page | What it shows |
|---|---|
| **Overview** | Corpus and benchmark readiness, completed runs, and a short summary of recent results. |
| **Corpus** | Add PDFs, inspect extracted documents, and preview chunks. Processing happens locally. |
| **Ask** | Choose BM25, Dense, or Hybrid (and optional reranking) for a question; inspect the retrieved source passages and citations. A generated answer requires an explicitly configured model API. |
| **Experiments** | Filter saved benchmark runs; compare metrics in tables, charts, and side-by-side configuration cards. This page displays stored runs; the CLI starts formal experiment runs. |
| **Claims** | Review automated supported, unsupported, and contradicted claim labels alongside their evidence; record a separate human review. |
| **Retrieval debugger** | Compare retrieved passages to annotated gold evidence and inspect missed evidence. |
| **Exports** | Download per-question and aggregate CSV, full result/metadata JSON, frozen configuration YAML, and benchmark validation JSON. |

The three retrievers can be tried one at a time on the **Ask** page. To compare their measured quality, first create a verified benchmark and run the experiment matrix; free-form questions alone do not produce benchmark metrics.

## Quick start

Use Python 3.11. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m streamlit run app.py
```

If `py` is unavailable, create the virtual environment with the path to an installed Python 3.11 interpreter. Open the local URL printed by Streamlit.

The first use of the tokenizer or a Sentence Transformers model may download model files from Hugging Face. This is a local model download; it is separate from hosted model API usage.

## Use your own PDFs

1. In the sidebar, select **REAL**.
2. Open **Corpus**, upload one or more PDFs, and choose **Process PDFs**.
3. Open **Ask**, choose a retrieval method, enter a question, and submit it.
4. Inspect the returned passages, page ranges, and citations. If answer generation is configured, inspect the generated answer and its claim-level evidence as well.

PDF extraction and retrieval run locally. The app keeps page provenance and source checksums. Scanned image-only pages are reported for review; OCR is not implemented.

### API and cost behavior

An API key is optional. Ingestion, indexing, retrieval, evidence display, and deterministic retrieval metrics run without a hosted model API. For a Google AI Studio key, create a key in AI Studio and add these entries to your local `.env` file:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash
```

The key creation page does not ask you to choose a model; the app selects the model by sending its model ID with each request. The values above do not need quotation marks. Google’s free tier has model access and rate limits, and its free tier terms say submitted content may be used to improve Google products. Check the current [Gemini API pricing and tier details](https://ai.google.dev/gemini-api/docs/pricing) before sending anything you consider sensitive.

OpenAI remains available as an alternative: set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. OpenAI API usage may cost money and is billed separately from ChatGPT. The app sends the question and selected retrieved passages for answer generation, the question/answer/claims/passages for claim review, and reference-answer material for correctness evaluation. The PDF files and full corpus are not sent by these code paths. With no configured credentials, hosted generation and LLM-based evaluation are disabled.

When generation runs, the question and selected retrieved passages are sent to the configured model. Claim review sends the question, answer, claims, and selected passages; correctness evaluation sends the question, answer, reference answer, and required facts. The PDF files and full corpus are not sent by these code paths. With no credentials, generation and LLM-based evaluation are disabled; no paid API is called.

Copy [`.env.example`](.env.example) to `.env` only if enabling an API. Keep the key private; never paste it into chat or commit it. `.env` is ignored by Git.

## Formal evaluation workflow

Formal conclusions require a corpus-specific benchmark. The repository includes a detailed [annotation guide](reports/annotation_guide.md); real PDFs, real questions, and annotations are deliberately excluded from the repository.

1. Add and process your PDFs in **Corpus**. The CLI alternative is `python -m src.cli ingest --input data/pdfs --output data/processed/pages.jsonl`.
2. Create the real development and test JSONL files using the annotation guide. Target sizes are **20 verified development questions** and **60 verified held-out test questions**. Every answerable question must point to exact evidence spans in the supplied source text; test annotations must all be verified.
3. Validate development and test data:

   ```powershell
   python -m src.cli validate-benchmark --split dev
   python -m src.cli validate-benchmark --split test
   ```

4. Run development experiments, select settings from development metrics, and freeze the test configuration:

   ```powershell
   python -m src.cli run-experiments --matrix experiments/experiment_matrix.yaml --split dev
   python -m src.cli freeze-test-config --matrix experiments/experiment_matrix.yaml
   ```

5. Run the held-out test only after the configuration is frozen:

   ```powershell
   python -m src.cli run-experiments --matrix experiments/experiment_matrix.yaml --split test
   ```

Runs are saved locally to `results/experiments.sqlite3`. The **Experiments**, **Claims**, **Retrieval debugger**, and **Exports** pages read those stored records. Do not tune settings based on held-out test results. Missing annotations, source files, or required local model files block the affected evaluation; unavailable metrics remain null rather than being estimated.

## Data and reproducibility

- A clean repository checkout has no real user corpus or formal benchmark.
- `data/demo/` contains generic synthetic fixtures flagged `is_demo: true`; they stay isolated from real records and are not evidence of research performance.
- User PDFs, extracted page/chunk JSONL, real benchmark files, local vector caches, SQLite results, frozen test snapshots, and `.env` files are excluded by [`.gitignore`](.gitignore).
- PDF SHA-256 checksums, stable chunk IDs, document/page/character offsets, configuration IDs, benchmark split, and model/prompt settings are retained in local records.
- The test configuration snapshot stores a checksum of its settings and the source matrix so accidental changes are detected.
- Results are specific to the corpus, annotations, models, and settings used. Automated LLM judgments are not human verification.

See [`docs/architecture.md`](docs/architecture.md), [`docs/experiments.md`](docs/experiments.md), and [`docs/data-and-results.md`](docs/data-and-results.md) for implementation and research details.

## Repository map

```text
app.py                         Streamlit entry point
src/                           PDF processing, chunking, retrieval, generation, metrics, storage, UI
experiments/experiment_matrix.yaml
                               Versioned E1–E5 experimental plan
reports/annotation_guide.md   Benchmark construction and verification instructions
reports/final_report.md       Academic report; findings stay pending until real runs exist
reports/presentation_outline.md
                               Presentation structure and demo cautions
data/demo/                     Generic synthetic, isolated demo fixture
data/pdfs/                     Local user PDFs; ignored by Git
data/processed/                Local extracted pages/chunks; ignored by Git
data/benchmark_*.jsonl         Local real benchmark files; ignored by Git
results/                       Local database, cache, and frozen configuration; ignored by Git
tests/                         Unit and integrity tests
```

## Current findings and limitations

No formal real-corpus experiment has been run for the repository snapshot, so there are no measured findings or recommended winning configuration to report. The report's Results and Failure Analysis sections are explicitly pending. The synthetic demo exists to exercise application paths only.

Other limitations include no OCR, no automatic real-benchmark authoring, and reliance on manual evidence annotation. Dense retrieval, chunking, and reranking require local model files. LLM evaluation is provider-generated and should be reviewed by a human before it is used as a research conclusion.

## References

The academic context and full bibliography are in [`reports/final_report.md`](reports/final_report.md). The implementation is an independent comparison framework; it does not claim to reproduce RAGAS, Self-RAG, or the cited retrieval systems.
