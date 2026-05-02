"""Pin the concept-graph extractor + qa.set_concept_graph plumbing.
"""
from __future__ import annotations

import pytest

from book.ir import Book, BookNode
from book.concept_graph import (
    extract_concept_graph,
    prerequisites,
    _is_concept_phrase,
)


def _book_with_text(*passages, concepts=None) -> Book:
    """passages = [(nid, body_text)]"""
    root = BookNode(nid="b", kind="book", number=None, title="t",
                    page_start=1, page_end=999)
    for nid, text in passages:
        root.children.append(BookNode(
            nid=nid, kind="section", number="",
            title="t", page_start=1, page_end=2, body_text=text,
        ))
    return Book(title="t", author=None, source="", root=root,
                concepts=concepts or {}, pages=[], figures=[], cross_refs=[])


# ---------------------------------------------------------------------------
# is_concept_phrase
# ---------------------------------------------------------------------------

def test_concept_phrase_rejects_pronouns():
    assert not _is_concept_phrase("we")
    assert not _is_concept_phrase("we will")
    assert not _is_concept_phrase("which")
    assert not _is_concept_phrase("here we")


def test_concept_phrase_rejects_conjunctions_and_articles():
    assert not _is_concept_phrase("and then")
    assert not _is_concept_phrase("the matrix")  # article start
    assert not _is_concept_phrase("a generalization")


def test_concept_phrase_accepts_real_concept_names():
    assert _is_concept_phrase("ridge regression")
    assert _is_concept_phrase("kernel matrix")
    assert _is_concept_phrase("support vector machine")


# ---------------------------------------------------------------------------
# Extraction restricted to known terms
# ---------------------------------------------------------------------------

def test_extracts_simple_built_on_relation():
    """The classic case — two known concepts joined by a clear pattern.
    Restrict known terms via concept index so the extraction is
    high-precision."""
    concepts = {
        "ridge regression": {"canonical": "ridge regression",
                              "aliases": [], "definitions": [],
                              "templates": [], "figure_refs": [], "cid": "ridge"},
        "linear regression": {"canonical": "linear regression",
                               "aliases": [], "definitions": [],
                               "templates": [], "figure_refs": [], "cid": "linear"},
    }
    book = _book_with_text(
        ("b/x", "ridge regression is built on linear regression by adding a penalty term."),
        concepts=concepts,
    )
    g = extract_concept_graph(book)
    assert "ridge regression" in g
    prereqs = [p for p, _ in g["ridge regression"]]
    assert "linear regression" in prereqs


def test_drops_unrecognised_heads():
    """Heads / prereqs that aren't in the known-terms set never form edges."""
    concepts = {"matrix": {"canonical": "matrix", "aliases": [],
                            "definitions": [], "templates": [],
                            "figure_refs": [], "cid": "matrix"}}
    book = _book_with_text(
        ("b/x", "we use a complicated method to derive the matrix."),
        concepts=concepts,
    )
    g = extract_concept_graph(book)
    # 'we' is not a known term, so no edge from "we use" sentences.
    assert all(head != "we" for head in g)


def test_prerequisites_returns_empty_when_concept_missing():
    g = {"ridge regression": [("linear regression", 2)]}
    assert prerequisites(g, "lasso") == []
    assert prerequisites(g, "ridge regression") == [("linear regression", 2)]


def test_prerequisites_substring_fallback():
    g = {"linear regression ridge regression": [("least squares", 1)]}
    out = prerequisites(g, "ridge regression")
    # Substring fallback finds the entry containing the query term.
    assert out == [("least squares", 1)]


# ---------------------------------------------------------------------------
# qa module integration
# ---------------------------------------------------------------------------

def test_qa_set_concept_graph_round_trip():
    from narrator import qa as qa_mod
    g = {"ridge regression": [("linear regression", 3)]}
    qa_mod.set_concept_graph(g)
    try:
        assert qa_mod.get_concept_graph() == g
    finally:
        qa_mod.set_concept_graph({})


def test_qa_dependencies_includes_concept_clause_when_graph_set():
    """Concept-graph entries flow into the dependencies plan as an
    'at the concept level, this builds on …' clause."""
    from narrator import qa as qa_mod
    book = _book_with_text(
        ("b/ridge",
         "Ridge regression adds a penalty.  See Section 3.2 for "
         "details and Equation 3.42 for the closed form."),
        concepts={"ridge regression": {"canonical": "ridge regression",
                                        "aliases": [], "definitions": [],
                                        "templates": [], "figure_refs": [],
                                        "cid": "ridge"}},
    )
    # Add a fake chapter title so dependencies has a focus
    book.root.children[0].number = "3"
    book.root.children[0].kind = "chapter"
    book.root.children[0].title = "ridge regression"

    qa_mod.set_concept_graph({
        "ridge regression": [("linear regression", 4),
                              ("regularization", 3)],
    })
    try:
        plan = qa_mod.dependencies(book, "b/ridge")
        text = " ".join(c.text for c in plan.clauses)
        assert ("concept level" in text.lower()
                or "linear regression" in text.lower())
    finally:
        qa_mod.set_concept_graph({})
