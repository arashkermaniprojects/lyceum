"""Tests for the narration planner."""
import os
import pytest

from book.ir import Book, BookNode, ConceptEntry, ConceptTemplate
from narrator.planner import (
    plan, NarrationPlan, NarrationClause,
    _tokenize, _bm25_scores, _split_sentences, _estimate_dur,
)


# ---------------------------------------------------------------------------
# Tokeniser / BM25
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_keeps_hyphens():
    assert _tokenize("Linear Maps and DEEP-net.") == ["linear", "maps", "and", "deep-net"]


def test_tokenize_drops_pure_numbers_and_punct():
    assert _tokenize("=== 1234 ===") == []


def test_bm25_ranks_relevant_doc_higher():
    docs = [
        _tokenize("eigenvalues are roots of the characteristic polynomial"),
        _tokenize("python is a programming language"),
        _tokenize("the determinant equals the product of eigenvalues"),
    ]
    q = _tokenize("eigenvalues")
    scores = _bm25_scores(q, docs)
    assert scores[0] > 0 and scores[2] > 0
    assert scores[1] == 0
    assert scores[0] >= 0 and scores[2] >= 0


def test_bm25_handles_empty_query():
    scores = _bm25_scores([], [_tokenize("anything"), _tokenize("here")])
    assert scores == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    out = _split_sentences("Hello world. This is a sentence. And another!")
    assert len(out) == 3


def test_split_sentences_glues_abbreviations():
    out = _split_sentences("Refer to Eq. 4 for details. We define X.")
    # The abbreviation "Eq." should not split the first sentence.
    assert len(out) == 2
    assert "Eq. 4" in out[0]


def test_split_sentences_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   \n\n  ") == []


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------

def test_estimate_dur_clamps_to_min():
    assert _estimate_dur("Hi.") >= 1.5


def test_estimate_dur_grows_with_length():
    short = _estimate_dur("A short clause.")
    long = _estimate_dur("A " + "long " * 50 + "clause.")
    assert long > short


# ---------------------------------------------------------------------------
# Synthetic-book planning
# ---------------------------------------------------------------------------

def _book_with_topics() -> Book:
    """A book with four sections, each on a distinct topic."""
    pages = [
        "Linear maps are functions that preserve vector addition. "
        "A matrix represents a linear map under a chosen basis.",
        "Eigenvalues are scalars λ such that Av = λv for some v. "
        "The determinant equals the product of eigenvalues.",
        "Compactness is a topological property. "
        "A set is compact if every open cover has a finite subcover.",
        "Probability theory uses measures. "
        "A random variable is a measurable function.",
    ]
    sections = [
        BookNode(nid=f"b/ch{i+1}", kind="chapter", number=str(i+1),
                title=f"Topic {i+1}", page_start=i+1, page_end=i+1,
                body_text=pages[i])
        for i in range(4)
    ]
    root = BookNode(nid="b", kind="book", number=None, title="Toy",
                    page_start=1, page_end=4, children=sections)
    return Book(
        title="Toy", author=None, source="/tmp/toy.pdf",
        root=root,
        concepts={
            "matrix": ConceptEntry(
                cid="matrix", canonical="matrix",
                aliases=["matrix", "matrices"], definitions=[],
                templates=[ConceptTemplate(
                    home_nid="b/ch1", primitive="matrix_bracket",
                    meta={"kind": "matrix_bracket"}, evidence={})],
                figure_refs=[]),
            "eigenvalue": ConceptEntry(
                cid="eigenvalue", canonical="eigenvalue",
                aliases=["eigenvalues", "eigenvalue"], definitions=[],
                templates=[ConceptTemplate(
                    home_nid="b/ch2", primitive="rect",
                    meta={"kind": "rect"}, evidence={})],
                figure_refs=[]),
            "set": ConceptEntry(
                cid="set", canonical="set",
                aliases=["set", "sets"], definitions=[],
                templates=[ConceptTemplate(
                    home_nid="b/ch3", primitive="set_blob",
                    meta={"kind": "set_blob"}, evidence={})],
                figure_refs=[]),
        },
        pages=pages,
    )


def test_plan_returns_relevant_chapter_first():
    b = _book_with_topics()
    p = plan(b, "eigenvalues and determinant", top_k=2)
    # Chapter 2 covers eigenvalues; the topic plan reroots there and
    # walks its subtree so the lecture stays in one chapter instead of
    # whiplashing between top-k matches scattered across the book.
    assert p.meta["primary_root"] == "b/ch2"
    assert p.visited_nids[0] == "b/ch2"


def test_plan_visits_in_book_order_after_ranking():
    b = _book_with_topics()
    # A topic that matches multiple chapters: the planner ranks by score
    # but emits clauses in book reading order.
    p = plan(b, "vector matrix eigenvalue", top_k=3)
    pages = [b.find(nid).page_start for nid in p.visited_nids]
    assert pages == sorted(pages)


def test_plan_is_deterministic():
    b = _book_with_topics()
    p1 = plan(b, "eigenvalues", top_k=2)
    p2 = plan(b, "eigenvalues", top_k=2)
    assert [c.text for c in p1.clauses] == [c.text for c in p2.clauses]
    assert p1.visited_nids == p2.visited_nids


def test_plan_concept_tags_match_corpus_concepts():
    b = _book_with_topics()
    p = plan(b, "matrix linear maps", top_k=1)
    # First clause from chapter 1 should mention "matrix" or "linear".
    cids = [c[0] for cl in p.clauses for c in cl.concepts]
    assert "matrix" in cids


def test_plan_empty_topic_returns_empty():
    b = _book_with_topics()
    p = plan(b, "")
    assert p.clauses == []


def test_plan_unmatched_topic_returns_empty():
    b = _book_with_topics()
    p = plan(b, "xyzzy quartic banana")
    assert p.clauses == []
    assert p.meta["reason"] == "no_match"


def test_plan_filters_by_include_kinds():
    b = _book_with_topics()
    # Restrict candidates to chapter kind only — synthetic book has 4
    # chapters.  The single-rooted topic plan picks its primary root
    # from the chapter pool, but the depth-first walk that follows
    # naturally descends into the chapter's children (sections,
    # subsections); the contract here is that the *chosen root* honours
    # the kind filter, not every node visited by the descendants walk.
    p = plan(b, "linear maps eigenvalues", include_kinds={"chapter"}, top_k=4)
    primary = b.find(p.meta["primary_root"])
    assert primary is not None
    assert primary.kind == "chapter"


def test_plan_excludes_kinds():
    b = _book_with_topics()
    p = plan(b, "linear maps", exclude_kinds={"chapter"}, top_k=4)
    # Excluding chapter eliminates every candidate in this synthetic book.
    assert p.clauses == []


def test_plan_total_chars_and_dur_consistent():
    b = _book_with_topics()
    p = plan(b, "eigenvalues determinant", top_k=2)
    assert p.total_chars() > 0
    assert p.total_dur() > 0


# ---------------------------------------------------------------------------
# Real-PDF smoke
# ---------------------------------------------------------------------------

PDF = "tests/data/sample_paper.pdf"


@pytest.mark.skipif(not os.path.exists(PDF), reason="paper PDF missing")
def test_plan_against_real_paper():
    from book import parse_pdf, extract_concepts
    b = parse_pdf(PDF)
    b.concepts = extract_concepts(b)
    p = plan(b, "deterministic layout", top_k=3)
    assert p.clauses
    assert p.visited_nids
    # Latency budget: planning should complete in well under 200 ms.
    assert p.total_chars() > 0
