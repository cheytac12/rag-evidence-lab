# Experimental design and metrics

This document explains what the project compares, how configuration selection works, and how to interpret the displayed results. The versioned source of truth is [`../experiments/experiment_matrix.yaml`](../experiments/experiment_matrix.yaml).

## Research question

For a user-provided PDF collection and a fixed benchmark, how do chunk size, embedding model, retrieval method, retrieved-passage count, and reranking affect evidence retrieval quality, answer quality, faithfulness, citations, latency, and provider usage?

This is a within-project comparison, not a claim that one method is universally best. A conclusion applies only to the selected corpus, annotations, models, prompt, and settings.

## Factors and stages

| Stage | Settings that change | Settings held fixed or selected |
|---|---|---|
| E1 — chunking and base retrieval | Chunk size 128/256/512; BM25 or dense | `top_k=5`, MiniLM dense model, reranker off |
| E2 — embedding choice | MiniLM or BGE; dense or hybrid | Best E1 chunk setting from development; `top_k=5`; reranker off |
| E3 — retrieval depth and method | BM25, dense, hybrid; `top_k=3/5/10` | Best chunk setting and embedding choice selected from development; reranker off |
| E4 — reranking | Reranker off/on for the strongest two base methods | Choose base methods by development MRR; reranked path scores up to 20 candidates and returns five |
| E5 — held-out check | Evaluation split changes from development to test | Four configurations selected/frozen from development; same generator, prompt, temperature, and token limits |

E1 has six explicit configurations. E2 has four explicit configurations. E3 expands to nine retriever/top-k combinations. E4 compares reranking for two selected base methods. E5 evaluates four frozen roles: BM25 baseline, best dense, best hybrid, and best reranked configuration.

## Benchmark and split rules

- The formal target is 20 verified development questions and 60 verified held-out test questions.
- Each answerable question includes a reference answer, required facts, and one or more exact source evidence spans. Each unanswerable question has no reference answer or supporting evidence.
- Evidence quotes and offsets are checked against extracted source text. Multi-hop questions should include every evidence unit required for a complete answer.
- Demo records are flagged `is_demo: true`, are kept separate from real records, and cannot satisfy formal benchmark validation.
- All formal test annotations must be verified before the run. The E5 snapshot and source-matrix SHA-256 checksums protect the held-out configuration from accidental changes.
- Select settings using development results only. Do not use test scores to choose chunking, embeddings, retriever, `top_k`, or reranking.

See [`../reports/annotation_guide.md`](../reports/annotation_guide.md) for exact annotation instructions and JSONL requirements.

## Metrics

### Retrieval metrics

Relevance is based on overlap between a retrieved passage's document/page/character span and the annotated gold evidence spans.

| Metric | Interpretation |
|---|---|
| Recall@k | Fraction of annotated evidence units overlapped by the top `k` passages. |
| Hit@k | Whether at least one annotated evidence unit appears in the top `k`. |
| Precision@k | Fraction of returned top-`k` passages that overlap any annotated evidence unit. |
| MRR | Reciprocal rank of the first passage that overlaps annotated evidence. |
| nDCG@k | Binary-relevance ranking quality with higher ranks weighted more. |
| Complete evidence recall | Whether every annotated evidence unit is retrieved; particularly useful for multi-hop items. |

Unanswerable items are measured separately by whether the pipeline abstains or produces an answer. They do not receive retrieval recall values.

### Answer and citation metrics

- **Correctness** is an automated 0–3 score against a reference answer and required facts. The rubric is stored in `src/generator.py`; it is not a human score.
- **Faithfulness** is the fraction of extracted factual claims labeled `SUPPORTED` by the configured model.
- **Unsupported claim rate** is the fraction labeled `UNSUPPORTED`. A separate `CONTRADICTED` label means retrieved evidence directly conflicts with the claim.
- **Unsupported-answer rate** marks an answer when any of its claims is unsupported or contradicted.
- **Citation presence/validity** checks whether claims point to IDs in the retrieved context. A citation being present does not prove that it supports the claim; claim evidence review is separate.
- **Abstention and false-positive answer behavior** are measured for unanswerable items.

Automated labels are not independent ground truth. Review them in **Claims** and use **Mark reviewed** only after a person inspects the evidence.

### Operational measurements

The run records retrieval, reranking, generation, and total latency where applicable; context and output token counts when returned by the provider; and provider cost only when available. Unavailable values remain null. Do not interpret null as zero cost or zero latency.

## Running and viewing experiments

The CLI populates the local SQLite result store. The web UI visualizes those stored records:

```powershell
python -m src.cli validate-benchmark --split dev
python -m src.cli run-experiments --matrix experiments/experiment_matrix.yaml --split dev
python -m src.cli freeze-test-config --matrix experiments/experiment_matrix.yaml
python -m src.cli validate-benchmark --split test
python -m src.cli run-experiments --matrix experiments/experiment_matrix.yaml --split test
python -m streamlit run app.py
```

Open **Experiments** to filter runs by split, data label, method, chunk size, `top_k`, and reranking; inspect summary tables and metric charts; and compare up to four configurations side by side. Open **Retrieval debugger** to inspect per-question gold spans and missed evidence. Open **Claims** to examine claim-level model labels and record human review. **Exports** downloads per-question and aggregate CSV, full result JSON, frozen YAML, and a benchmark validation report.

If OpenAI credentials are absent, retrieval and its evidence-span metrics remain available. Answer generation and model-judged metrics will be unavailable or blocked; do not replace them with fabricated or estimated values. An API model may incur cost when configured.

## Interpreting results

1. Compare configurations on the same split and same question set.
2. Check `question_count` and the number of scored values beside each aggregate; missing values mean the metric was not available for those records.
3. Look at retrieval and answer metrics together. Higher retrieval recall does not automatically imply more correct or more faithful answers.
4. Check latency and token counts when comparing quality; additional context or reranking can change runtime.
5. Inspect failures at the question level before making a conclusion.
6. Report the corpus checksum set, benchmark version, split, matrix/frozen checksum, model IDs, prompt version, and sample count with any result.

## Current result status

The repository does not include a real corpus, formal benchmark, or real experiment database. Therefore no real-corpus score, p-value, ranking, or winning configuration is currently available. `reports/final_report.md` leaves Results and Failure Analysis pending. The synthetic DEMO is for software walkthroughs only and is excluded from the research conclusions.
