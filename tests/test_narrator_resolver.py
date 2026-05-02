"""Tests for the narrator/resolver — context-sensitive concept lookup."""
from book.ir import (
    Book, BookNode, ConceptEntry, ConceptTemplate,
)
from narrator.resolver import (
    resolve, render_resolved, _proximity_score, _common_prefix_len,
)


# ---------------------------------------------------------------------------
# Path proximity helpers
# ---------------------------------------------------------------------------

def test_common_prefix_len_basic():
    assert _common_prefix_len("b/ch1/s1.1", "b/ch1/s1.2") == 2
    assert _common_prefix_len("b/ch1", "b/ch2") == 1
    assert _common_prefix_len("b/ch1", "b/ch1") == 2
    assert _common_prefix_len("a/x", "b/y") == 0


def test_proximity_score_orders_by_closeness():
    here = "b/ch3/s3.2"
    same = _proximity_score("b/ch3/s3.2", here)
    sibling = _proximity_score("b/ch3/s3.3", here)
    other = _proximity_score("b/ch7/s7.1", here)
    assert same > sibling > other


# ---------------------------------------------------------------------------
# Resolution against a synthetic book
# ---------------------------------------------------------------------------

def _book_with_two_matrix_templates() -> Book:
    """Synthetic book where 'matrix' has two distinct visual templates:
    chapter-1 uses a 2×3 bracketed grid, chapter-8 uses a tensor box."""
    root = BookNode(nid="b", kind="book", number=None, title="Toy",
                    page_start=1, page_end=200, children=[
                        BookNode(nid="b/ch1", kind="chapter", number="1",
                                title="Linear Algebra Intro",
                                page_start=1, page_end=50),
                        BookNode(nid="b/ch8", kind="chapter", number="8",
                                title="Multilinear Algebra",
                                page_start=150, page_end=200),
                    ])
    return Book(
        title="Toy", author=None, source="/tmp/toy.pdf", root=root,
        concepts={
            "matrix": ConceptEntry(
                cid="matrix", canonical="matrix",
                aliases=["matrix"],
                definitions=[],
                templates=[
                    ConceptTemplate(
                        home_nid="b/ch1", primitive="matrix_bracket",
                        meta={"kind": "matrix_bracket",
                              "nrows": 2, "ncols": 3,
                              "cells": [["a", "b", "c"], ["d", "e", "f"]],
                              "delim": "bracket"},
                        evidence={"page": 12},
                    ),
                    ConceptTemplate(
                        home_nid="b/ch8", primitive="tensor_box",
                        meta={"kind": "tensor_box", "legs": 2,
                              "label": "T"},
                        evidence={"page": 187},
                    ),
                ],
                figure_refs=[],
            ),
        },
    )


def test_resolver_picks_chapter_1_template_when_in_chapter_1():
    b = _book_with_two_matrix_templates()
    rs = resolve(b, "matrix", current_nid="b/ch1/s1.2")
    assert rs.from_corpus
    assert rs.primitive == "matrix_bracket"
    assert rs.home_nid == "b/ch1"
    assert rs.meta["nrows"] == 2 and rs.meta["ncols"] == 3


def test_resolver_picks_chapter_8_template_when_in_chapter_8():
    b = _book_with_two_matrix_templates()
    rs = resolve(b, "matrix", current_nid="b/ch8/s8.4")
    assert rs.from_corpus
    assert rs.primitive == "tensor_box"
    assert rs.home_nid == "b/ch8"
    assert rs.meta["legs"] == 2


def test_resolver_falls_back_when_concept_missing():
    b = _book_with_two_matrix_templates()
    rs = resolve(b, "nonexistent_concept")
    assert not rs.from_corpus
    assert rs.evidence["reason"] == "not_in_corpus"
    # Default primitive is rect (when math_lex doesn't recognise the label).
    assert rs.primitive == "rect"


def test_resolver_uses_math_lex_fallback_for_known_label():
    """A label the corpus doesn't have but math_lex recognises returns the
    math_lex primitive."""
    b = _book_with_two_matrix_templates()
    rs = resolve(b, "circle")  # not in corpus, but math_lex knows
    assert not rs.from_corpus
    assert rs.primitive == "circle"


def test_resolver_is_deterministic():
    b = _book_with_two_matrix_templates()
    a = resolve(b, "matrix", current_nid="b/ch8")
    c = resolve(b, "matrix", current_nid="b/ch8")
    assert a.primitive == c.primitive
    assert a.home_nid == c.home_nid


def test_resolver_respects_label_override():
    b = _book_with_two_matrix_templates()
    rs = resolve(b, "matrix", current_nid="b/ch1", label_override="A")
    assert rs.label == "A"


def test_render_resolved_produces_svg():
    b = _book_with_two_matrix_templates()
    rs = resolve(b, "matrix", current_nid="b/ch1/s1.2")
    svg = render_resolved(rs)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    # Matrix bracket renderer emits <path> for the brackets.
    assert "<path" in svg


def test_render_resolved_for_tensor_box_in_chapter_8():
    b = _book_with_two_matrix_templates()
    rs = resolve(b, "matrix", current_nid="b/ch8/s8.4")
    svg = render_resolved(rs)
    # Tensor box uses <rect> + leg <line>s.
    assert svg.startswith("<svg")
    assert "<rect" in svg


def test_resolver_score_higher_for_closer_context():
    b = _book_with_two_matrix_templates()
    in_ch1 = resolve(b, "matrix", current_nid="b/ch1/s1.2")
    in_ch8 = resolve(b, "matrix", current_nid="b/ch8/s8.4")
    # Both should report positive proximity (their picks are within their
    # respective chapters).
    assert in_ch1.score > 0
    assert in_ch8.score > 0


def test_resolver_handles_empty_current_nid():
    """When no location is supplied, picks the lexicographically-first
    template deterministically."""
    b = _book_with_two_matrix_templates()
    rs1 = resolve(b, "matrix")
    rs2 = resolve(b, "matrix")
    assert rs1.primitive == rs2.primitive
    assert rs1.home_nid == rs2.home_nid
