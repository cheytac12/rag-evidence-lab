# Benchmark annotation guide

This guide defines how to build a real, corpus-specific benchmark from the documents the user supplies. The application accepts any document subject. Never copy the synthetic demo question into a real benchmark without independently validating it against a real source.

## Target and split

The project target is 80 questions: 20 development questions and 60 held-out test questions, distributed across 25–40 real source documents when available. Include factual lookup, technical explanation, comparison, multi-hop, and unanswerable questions. Aim for 15–20% unanswerable items. Treat these as targets, not completed data: files remain empty until reviewed annotations exist.

Draft and revise questions using development documents. Keep the test question file inaccessible to configuration selection. Record the split outside the question text and do not use held-out questions to tune chunking, embeddings, retrieval, top-k, or reranking.

## Selecting a question

1. Write a clear, self-contained question that a reader can answer from the supplied corpus, or a deliberately unanswerable question for which the corpus has been checked.
2. Assign exactly one allowed type: `factual_lookup`, `technical_explanation`, `comparison`, `multi_hop`, or `unanswerable`.
3. Keep the wording distinct from a direct copy of the evidence sentence. Avoid vague pronouns, hidden assumptions, and questions requiring outside knowledge.
4. Add concise `required_facts` for the essential pieces a complete answer must contain. For an unanswerable item, use an empty list.

## Writing a reference answer

For answerable questions, write a concise paraphrase that covers the necessary facts and does not add claims beyond the corpus. Use `reference_answer: null` only for an unanswerable question. A reference answer is an annotation target, not a generated model answer.

## Marking exact evidence spans

For each answerable question, identify one or more exact source text spans that establish the answer. Copy each `quote` exactly as it appears in the extracted, whitespace-normalized source text. Record the true `doc_id`, inclusive one-based `page_start` and `page_end`, and zero-based `char_start` and `char_end` offsets into the normalized concatenated document text. The quote must equal the source substring at those offsets. Evidence spans refer to the source document and pages, never to chunk IDs.

For a multi-hop question, annotate every evidence unit needed to answer it. Each independently necessary span should be recorded, including spans from different pages or documents when needed. The system counts complete evidence recall only when every annotated evidence unit overlaps retrieved text.

## Confirming unanswerability

An unanswerable label means that the supplied corpus does not contain enough information to answer the question. Search the relevant documents and likely synonyms, inspect neighboring sections and tables where possible, and record why the corpus does not establish the answer in the annotation review notes. A retriever failing to return evidence is not proof that a question is unanswerable. Unanswerable records must have `answerable: false`, `reference_answer: null`, and an empty `supporting_evidence` array.

## Verification

1. Validate every quote and offset against the extracted pages.
2. Have a second reviewer independently check answerability, required facts, reference answer, evidence span boundaries, and question type.
3. Resolve disagreements before setting `annotation_verified: true`.
4. Do not mark synthetic demo records as real or include them in a formal split.
5. Validate the complete dev/test files with the CLI before a formal run. A held-out test run requires all 60 annotations verified and a frozen configuration selected from development results.

Preserve the source PDF filename and checksum in page/chunk provenance. Unknown authors, years, or subjects remain null/empty; do not infer bibliographic metadata.
