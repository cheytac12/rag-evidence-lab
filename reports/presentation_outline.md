1. **Title and project team placeholder** — An Empirical Evaluation of Retrieval Strategies for Reducing Hallucinations in Retrieval-Augmented Generation Systems; team names and institution to be filled in.
2. **Problem and motivation** — Ground answers in user-selected documents and make evidence gaps visible.
3. **Research question and objectives** — Compare chunk size, embeddings, retrieval methods, top-k, and reranking across quality and operational measures.
4. **RAG and system architecture** — Local PDF extraction, source-aware chunks, BM25/dense/hybrid retrieval, optional reranking, cited generation, SQLite results.
5. **Dataset and benchmark construction** — Any user-provided PDF domain; 20 development and 60 held-out questions are targets; annotations require exact spans and review.
6. **Controlled experiment design** — E1–E5 matrix, development-only tuning, frozen test configuration, same split questions across configurations.
7. **Metrics and implementation** — Retrieval recall/ranking, claim faithfulness, correctness, citation behavior, latency, token counts, and available cost.
8. **Measured results** — Pending formal experiment run. The repository contains no verified real questions or measured findings; DEMO results are software checks only.
9. **Failure analysis and limitations** — Pending real runs; planned analysis uses missed source-span overlap and claim-level evidence review. OCR is not included.
10. **Conclusion and live demo sequence** — Research conclusions pending. Demo: select the clearly labeled synthetic fixture, retrieve passages, inspect citations, and open experiment/debugger pages; do not present demo values as findings.
