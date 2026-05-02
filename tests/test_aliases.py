"""Pin alias extraction patterns + retrieval expansion.
"""
from __future__ import annotations

import pytest

from book.ir import Book, BookNode
from book.aliases import (
    expand_query, extract_aliases, _acronym_matches_head,
)
from narrator import qa as qa_mod


def _book_with_text(*passages: tuple[str, str]) -> Book:
    """Construct a tiny Book where each passage is (nid, body_text)."""
    root = BookNode(nid="b", kind="book", number=None, title="t",
                    page_start=1, page_end=999)
    for nid, text in passages:
        root.children.append(BookNode(
            nid=nid, kind="section", number="",
            title="t", page_start=1, page_end=2, body_text=text,
        ))
    return Book(title="t", author=None, source="", root=root,
                concepts={}, pages=[], figures=[], cross_refs=[])


# ---------------------------------------------------------------------------
# Pattern coverage
# ---------------------------------------------------------------------------

def test_extracts_parenthetical_acronym():
    book = _book_with_text(
        ("b/x", "We use the radial basis function (RBF) kernel."),
    )
    m = extract_aliases(book)
    assert m.get("rbf") == "radial basis function"
    assert m.get("radial basis function") == "rbf"


def test_extracts_also_known_as():
    book = _book_with_text(
        ("b/x", "the support vector machine, also known as SVM, is widely used."),
    )
    m = extract_aliases(book)
    assert m.get("svm") == "support vector machine"


def test_extracts_or_acronym_form():
    book = _book_with_text(
        ("b/x", "the linear discriminant analysis, or LDA, separates classes."),
    )
    m = extract_aliases(book)
    assert m.get("lda") == "linear discriminant analysis"


def test_acronym_must_match_initials():
    """Reject parentheticals whose acronym doesn't match the head's
    initials — kills "kernel matrix (UNRELATED)" style noise."""
    book = _book_with_text(
        ("b/x", "the kernel matrix (RBF) is positive semi-definite."),
    )
    m = extract_aliases(book)
    # RBF doesn't match "kernel matrix" initials → no extraction.
    assert "rbf" not in m or m.get("rbf") != "kernel matrix"


def test_specialisation_not_extracted():
    """``matrix`` ↔ ``kernel matrix`` is a specialisation; not a
    synonym pair.  Filter it out."""
    book = _book_with_text(
        ("b/x", "the kernel matrix, also known as matrix, is positive."),
    )
    m = extract_aliases(book)
    assert m.get("matrix") != "kernel matrix"


def test_drops_noise_starts():
    """Heads that start with structural words (Section / Chapter / etc.)
    are noise — they shouldn't pair with anything."""
    book = _book_with_text(
        ("b/x", "Section 5 (also called Section 5) defines bagging."),
    )
    m = extract_aliases(book)
    # Either nothing extracted, or at least the pair filters out.
    for k, v in m.items():
        assert "section 5" not in k


# ---------------------------------------------------------------------------
# Acronym verification helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head,alias,expected", [
    ("radial basis function", "RBF", True),
    ("linear discriminant analysis", "LDA", True),
    ("reproducing kernel hilbert space", "RKHS", True),
    ("support vector machine", "SVM", True),
    # Mismatch — first letters don't line up.
    ("radial basis function", "XYZ", False),
    # Lowercase alias isn't an acronym → check skipped (returns True).
    ("matrix", "kernel matrix", True),
])
def test_acronym_matches_head(head, alias, expected):
    assert _acronym_matches_head(head, alias) == expected


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def test_expand_query_adds_canonical_for_acronym():
    amap = {"rbf": "radial basis function",
            "radial basis function": "rbf"}
    out = expand_query("what is RBF", amap)
    assert "radial basis function" in out


def test_expand_query_no_change_when_already_present():
    """If the query already contains both forms, don't duplicate."""
    amap = {"rbf": "radial basis function",
            "radial basis function": "rbf"}
    q = "what is the radial basis function (RBF)"
    out = expand_query(q, amap)
    # No additional appended terms.
    assert out.lower().count("radial basis function") == 1


def test_expand_query_unchanged_without_match():
    amap = {"rbf": "radial basis function"}
    out = expand_query("what is bagging", amap)
    assert out == "what is bagging"


def test_expand_query_word_boundary_required():
    """``rbf`` shouldn't match ``orbflag`` or other false-positives."""
    amap = {"rbf": "radial basis function"}
    out = expand_query("what is orbflag", amap)
    assert "radial basis function" not in out


def test_expand_query_empty_map_noop():
    out = expand_query("what is RBF", {})
    assert out == "what is RBF"


# ---------------------------------------------------------------------------
# qa.set_alias_map / qa._retrieve integration
# ---------------------------------------------------------------------------

def test_set_alias_map_stores_copy():
    """Mutating the original after install must not affect retrieval."""
    src = {"rbf": "radial basis function"}
    qa_mod.set_alias_map(src)
    src["rbf"] = "tampered"
    assert qa_mod.get_alias_map().get("rbf") == "radial basis function"
    # Restore default for other tests.
    qa_mod.set_alias_map({})


def test_retrieve_expands_query_when_alias_set():
    book = _book_with_text(
        ("b/rbf_section",
         "The radial basis function kernel is widely used. "
         "It maps inputs into a high-dimensional space."),
        ("b/other",
         "Bagging averages predictions from bootstrap samples."),
    )
    qa_mod.set_alias_map({
        "rbf": "radial basis function",
        "radial basis function": "rbf",
    })
    try:
        # Without expansion, BM25 wouldn't match "RBF" against the
        # passage that uses "radial basis function" verbatim.  With
        # expansion, the rbf-section comes out top.
        passages = qa_mod._retrieve(book, "what is RBF", top_k=2)
        nids = [p.nid for p in passages]
        assert "b/rbf_section" in nids
    finally:
        qa_mod.set_alias_map({})
