# Abstract

This project provides a domain-agnostic Retrieval-Augmented Generation (RAG) application and a controlled experiment platform for comparing chunking and retrieval choices. The repository snapshot intentionally excludes user-provided PDFs and derived corpus files. It does not include a verified benchmark or formal experiment results, so all research findings remain pending. This report separates implemented design from measured findings.

# Introduction

RAG systems retrieve passages from an external document collection and condition a language model on those passages. The system is intended to help users trace answers back to the documents they provide. Retrieval can still miss evidence, and generated text can exceed what retrieved passages support, so the system records passages, citations, claim assessments, and operational measurements.

# Problem Statement

The project studies how chunk size, embedding model, retrieval method, number of passages, and reranking affect retrieval quality, answer correctness, evidence faithfulness, latency, and computational cost for a selected PDF corpus. No claim about a universally best configuration is in scope.

# Objectives

1. Extract and index user-selected PDFs with page and checksum provenance.
2. Answer document questions using BM25, dense, or hybrid retrieval and show the passages used.
3. Compare controlled retrieval configurations on common development and held-out questions.
4. Measure retrieval, answer, citation, latency, token, and cost metrics without replacing missing values with estimates.
5. Keep synthetic demo behavior separate from real corpus and benchmark results.

# Research Questions and Hypotheses

The research question is: How do chunk size, embedding model, retrieval method, number of retrieved passages, and reranking affect retrieval quality, answer correctness, faithfulness, latency, and computational cost?

Planned hypotheses are that dense or hybrid retrieval may improve evidence recall for some question types, reranking may change passage ordering, and larger contexts may increase latency and token usage. These are testable expectations only; they are not findings and may be rejected by measured results.

# Related Work

Lewis et al. introduced a retrieval-augmented generation architecture that combines a parametric generator with retrieved non-parametric memory. Karpukhin et al. studied dense passage retrieval for open-domain question answering. Self-RAG explores retrieval decisions and model self-critique, while RAGAS presents an automated evaluation framework for RAG pipelines. This project uses these works as context; its planned BM25, dense, fusion, reranking, and claim-review comparisons are not reproductions of those systems.

# Dataset and Annotation

The project accepts user-selected PDFs without imposing a subject domain. The repository does not include user PDFs, derived page/chunk text, or verified benchmark annotations; local user documents remain excluded by Git ignore rules. The generic synthetic fixture is labeled `DEMO`, is not a research dataset, and is excluded from real aggregates. The target benchmark has 20 development and 60 held-out test questions; all formal test annotations must be verified against exact source spans.

# System Architecture

PyMuPDF extracts normalized page text locally. A Hugging Face tokenizer produces token counts and source-offset chunks. BM25 uses `rank-bm25`; dense retrieval encodes chunks with Sentence Transformers and computes cosine similarity in NumPy; hybrid retrieval uses Reciprocal Rank Fusion with constant 60. An optional Sentence Transformers CrossEncoder reranks up to 20 candidates. Streamlit calls reusable modules in `src/`. SQLite stores run metadata and per-question results. The optional hosted answer generator and LLM evaluator support Google Gemini and OpenAI; no real-corpus result is claimed by this repository snapshot.

# Experimental Methodology

The planned experiment matrix contains E1 chunk-size comparisons, E2 embedding-model comparisons, E3 retrieval-method and top-k comparisons, E4 reranking comparisons, and E5 four frozen held-out configurations. Development data selects settings; held-out test settings are checksummed before test execution. Every configuration uses the same questions within its split. Provider model, prompt version, temperature, output limit, timings, token counts, and available cost are stored. Missing source files, model files, annotations, or credentials block the affected run rather than producing estimates.

# Metrics

Retrieval measures include Recall@k, Hit@k, Precision@k, MRR, binary nDCG@k, and complete evidence recall for multi-hop questions. Answer measures include a 0–3 correctness rubric, claim faithfulness, unsupported-claim rate, unsupported-answer rate, citation presence, citation correctness, and unanswerable abstention/false-positive behavior. Operational measures include retrieval, reranking, generation, and total latency; context/output token counts; and provider-reported cost when available. Aggregates display their scored-question count and preserve unavailable values as null.

# Results

Pending experiment run

# Failure Analysis

Pending experiment run. The retrieval debugger will identify missed annotated spans using document, page, and character-span overlap. Claim review will distinguish unsupported claims from contradicted claims. No failure measurements are available yet.

# Limitations and Threats to Validity

The application does not include OCR; scanned pages with no extractable text are reported for review. The repository includes no verified benchmark or formal experiment measurements, so no empirical conclusion can be drawn from this snapshot. Automated LLM claim and correctness judgments are not human-verified, may depend on the configured model, and require sending the relevant question and retrieved passages to that configured provider. Findings from any eventual run will be corpus-, benchmark-, model-, and setting-specific. The synthetic fixture checks software behavior only.

# Conclusion

The repository establishes a reproducible workflow for comparing document retrieval and answer evidence. Its research conclusions remain pending until user documents, reviewed annotations, and actual experiment runs are available.

# References

1. Patrick Lewis et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” NeurIPS 2020. https://arxiv.org/abs/2005.11401
2. Vladimir Karpukhin et al. “Dense Passage Retrieval for Open-Domain Question Answering.” EMNLP 2020. https://arxiv.org/abs/2004.04906
3. Akari Asai et al. “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.” ICLR 2024. https://arxiv.org/abs/2310.11511
4. Shahul Es et al. “RAGAS: Automated Evaluation of Retrieval Augmented Generation.” 2023. https://arxiv.org/abs/2309.15217
