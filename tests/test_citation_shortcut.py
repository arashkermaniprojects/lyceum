"""Pin the citation-aware retrieval shortcut: explicit references
to a section/chapter/theorem/equation by number short-circuit
BM25/dense and return that node directly.
"""
from __future__ import annotations

import pytest

from book.ir import Book, BookNode
from narrator.qa import _citation_shortcut, _CITATION_SHORTCUT_RE


def _book() -> Book:
    root = BookNode(nid="b", kind="book", number=None, title="t",
                    page_start=1, page_end=999)
    ch3 = BookNode(nid="b/ch3", kind="chapter", number="3",
                   title="Linear Methods", page_start=40, page_end=80,
                   body_text="Chapter 3 covers ordinary least squares.")
    s3_4 = BookNode(nid="b/ch3/s3_4", kind="section", number="3.4",
                    title="Shrinkage Methods", page_start=60, page_end=70,
                    body_text="Ridge regression shrinks the coefficients.")
    ch3.children.append(s3_4)
    ch5 = BookNode(nid="b/ch5", kind="chapter", number="5",
                   title="Basis Expansions", page_start=100, page_end=200,
                   body_text="Chapter 5 introduces basis expansions.")
    s5_8 = BookNode(nid="b/ch5/s5_8", kind="section", number="5.8",
                    title="RKHS", page_start=170, page_end=190,
                    body_text="Reproducing Kernel Hilbert Spaces.")
    ch5.children.append(s5_8)
    thm = BookNode(nid="b/ch3/thm3_2", kind="theorem", number="3.2",
                   title="Gauss-Markov", page_start=50, page_end=51,
                   body_text="Among unbiased linear estimators, OLS has the smallest variance.")
    ch3.children.append(thm)
    root.children.extend([ch3, ch5])
    return Book(title="t", author=None, source="", root=root,
                concepts={}, pages=[], figures=[], cross_refs=[])


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,group,expected", [
    ("explain section 5.8", "sec", "5.8"),
    ("Section 3.4 please", "sec", "3.4"),
    ("Chapter 5", "ch", "5"),
    ("§3.4", "sec_alt", "3.4"),
    ("§ 5.8.1", "sec_alt", "5.8.1"),
    ("what does theorem 3.2 say", "thm", "3.2"),
    ("Figure 5.14", "fig", "5.14"),
    ("Equation 3.42", "eq", "3.42"),
])
def test_citation_regex_extracts(query, group, expected):
    m = _CITATION_SHORTCUT_RE.search(query)
    assert m is not None, f"no match in {query!r}"
    assert (m.group(group) or "") == expected


def test_citation_regex_doesnt_match_random_numbers():
    """Plain numbers without a kind keyword shouldn't trigger."""
    for q in ["I have 5 apples", "the algorithm runs in O(n^2) time"]:
        assert _CITATION_SHORTCUT_RE.search(q) is None


# ---------------------------------------------------------------------------
# Shortcut returns the right node
# ---------------------------------------------------------------------------

def test_section_shortcut_finds_node():
    book = _book()
    out = _citation_shortcut(book, "tell me about section 5.8")
    assert len(out) == 1
    assert out[0].nid == "b/ch5/s5_8"


def test_chapter_shortcut_finds_node():
    book = _book()
    out = _citation_shortcut(book, "explain Chapter 3")
    assert len(out) == 1
    assert out[0].nid == "b/ch3"


def test_section_alt_shortcut():
    book = _book()
    out = _citation_shortcut(book, "§5.8")
    assert len(out) == 1
    assert out[0].nid == "b/ch5/s5_8"


def test_theorem_shortcut_finds_node():
    book = _book()
    out = _citation_shortcut(book, "what does Theorem 3.2 say")
    assert len(out) == 1
    assert out[0].nid == "b/ch3/thm3_2"


def test_shortcut_misses_unknown_number():
    book = _book()
    assert _citation_shortcut(book, "Section 99.99") == []


def test_shortcut_misses_plain_query():
    book = _book()
    assert _citation_shortcut(book, "what is bagging") == []
