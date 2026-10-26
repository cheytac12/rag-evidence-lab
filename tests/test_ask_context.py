from src.schemas import ChunkRecord, RetrievedPassage
from src.ui_components import (
    _add_opening_page_context,
    _corpus_fingerprint,
    _is_document_overview_question,
    _needs_opening_page_context,
    _retrieve_ask_candidates,
)


def _chunk(chunk_id, page, text, doc_id="d1"):
    return ChunkRecord(
        chunk_id, doc_id, "Document", "doc.pdf", page, page, 0, len(text), text,
        len(text.split()), "checksum", False, 256, 48,
    )


def test_document_overview_question_detection_is_domain_neutral():
    assert _is_document_overview_question("What problem does this research paper address?")
    assert _is_document_overview_question("Summarize this document")
    assert not _is_document_overview_question("What learning rate is used in this paper?")


def test_overview_context_adds_opening_match_and_keeps_unique_ranks():
    query = "What problem does this research paper address?"
    opening = _chunk("intro", 1, "This paper addresses the problem of sequential computation.")
    later = _chunk("reference", 9, "A bibliography entry about another model.")
    dense = RetrievedPassage(
        1, "reference", "d1", "Document", "doc.pdf", 9, 9,
        later.text, 0.16, "dense", char_start=0, char_end=len(later.text),
    )

    merged, used = _add_opening_page_context(query, [opening, later], [dense], top_k=5, method_label="Hybrid")

    assert used
    assert [passage.chunk_id for passage in merged] == ["intro", "reference"]
    assert [passage.rank for passage in merged] == [1, 2]
    assert merged[0].method == "hybrid+opening-page-context"


def test_overview_context_is_method_agnostic_for_single_document_questions():
    question = "What problem does this research paper address?"
    assert _needs_opening_page_context(question, selected_document_count=1)
    assert not _needs_opening_page_context(question, selected_document_count=2)
    assert not _needs_opening_page_context("What learning rate is used in this paper?", 1)


def test_overview_context_keeps_opening_evidence_and_regular_retrieval(monkeypatch):
    chunks = [
        _chunk("meta", 1, "This paper has the listed authors and affiliations."),
        _chunk("abstract", 1, "This paper addresses an important problem and introduces a solution."),
        _chunk("problem", 2, "The fundamental constraint is a problem for this research."),
        _chunk("later", 4, "A later passage retrieved by the selected method."),
    ]
    selected = []

    def fake_bm25(question, candidates, top_k):
        selected.append(top_k)
        return [
            RetrievedPassage(i, chunk.chunk_id, chunk.doc_id, chunk.title, chunk.source_file,
                             chunk.page_start, chunk.page_end, chunk.text, 1.0, "bm25")
            for i, chunk in enumerate(candidates, start=1)
        ][:top_k]

    monkeypatch.setattr("src.ui_components.retrieve_bm25", fake_bm25)
    later = RetrievedPassage(1, "later", "d1", "Document", "doc.pdf", 4, 4,
                             chunks[3].text, 0.8, "hybrid")
    merged, used = _add_opening_page_context(
        "What problem does this research paper address?", chunks, [later], top_k=5, method_label="BM25"
    )

    assert used
    assert selected == [3]
    assert [passage.chunk_id for passage in merged] == ["meta", "abstract", "problem", "later"]


def test_overview_context_is_skipped_for_specific_queries_or_multiple_documents():
    query = "What learning rate is stated in this paper?"
    first = _chunk("intro", 1, "Learning rate details are summarized.")
    second_doc = _chunk("other", 1, "Another document has an introduction.", doc_id="d2")
    dense = RetrievedPassage(1, "other", "d2", "Document", "doc.pdf", 1, 1, "text", 0.5, "dense")

    results, used = _add_opening_page_context(query, [first, second_doc], [dense], top_k=5)

    assert not used
    assert results == [dense]


def test_overview_context_is_skipped_when_corpus_has_no_opening_pages():
    later = _chunk("body", 3, "Only a later section is indexed.")
    result = RetrievedPassage(1, "body", "d1", "Document", "doc.pdf", 3, 3,
                              later.text, 0.5, "hybrid")

    merged, used = _add_opening_page_context(
        "What problem does this research paper address?", [later], [result], top_k=5
    )

    assert not used
    assert merged == [result]


def test_corpus_fingerprint_is_order_independent_and_changes_with_index():
    first = _chunk("intro", 1, "Opening passage.")
    second = _chunk("body", 3, "Body passage.")
    replacement = _chunk("intro-v2", 1, "Rebuilt opening passage.")

    assert _corpus_fingerprint([first, second]) == _corpus_fingerprint([second, first])
    assert _corpus_fingerprint([first, second]) != _corpus_fingerprint([replacement, second])


def test_hybrid_ask_uses_wide_candidate_pool_before_fusion(monkeypatch):
    captured = {}

    def fake_hybrid(question, chunks, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.ui_components.retrieve_hybrid", fake_hybrid)
    _retrieve_ask_candidates("question", [], "Hybrid", top_k=5, use_reranker=False)

    assert captured["top_k"] == 5
    assert captured["candidate_k"] == 20
