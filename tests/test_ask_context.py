from src.schemas import ChunkRecord, RetrievedPassage
from src.ui_components import (
    _add_opening_page_context,
    _corpus_fingerprint,
    _is_document_overview_question,
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


def test_dense_overview_context_adds_opening_match_and_keeps_unique_ranks():
    query = "What problem does this research paper address?"
    opening = _chunk("intro", 1, "This paper addresses the problem of sequential computation.")
    later = _chunk("reference", 9, "A bibliography entry about another model.")
    dense = RetrievedPassage(
        1, "reference", "d1", "Document", "doc.pdf", 9, 9,
        later.text, 0.16, "dense", char_start=0, char_end=len(later.text),
    )

    merged, used = _add_opening_page_context(query, [opening, later], [dense], top_k=5)

    assert used
    assert [passage.chunk_id for passage in merged] == ["intro", "reference"]
    assert [passage.rank for passage in merged] == [1, 2]
    assert merged[0].method == "dense+opening-page-context"


def test_overview_context_is_skipped_for_specific_queries_or_multiple_documents():
    query = "What learning rate is stated in this paper?"
    first = _chunk("intro", 1, "Learning rate details are summarized.")
    second_doc = _chunk("other", 1, "Another document has an introduction.", doc_id="d2")
    dense = RetrievedPassage(1, "other", "d2", "Document", "doc.pdf", 1, 1, "text", 0.5, "dense")

    results, used = _add_opening_page_context(query, [first, second_doc], [dense], top_k=5)

    assert not used
    assert results == [dense]


def test_corpus_fingerprint_is_order_independent_and_changes_with_index():
    first = _chunk("intro", 1, "Opening passage.")
    second = _chunk("body", 3, "Body passage.")
    replacement = _chunk("intro-v2", 1, "Rebuilt opening passage.")

    assert _corpus_fingerprint([first, second]) == _corpus_fingerprint([second, first])
    assert _corpus_fingerprint([first, second]) != _corpus_fingerprint([replacement, second])
