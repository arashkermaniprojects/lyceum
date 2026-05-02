"""Pin chapter_graph aggregation, the citation_graph endpoint
shape, and the dependencies planner + router intent.
"""
from __future__ import annotations

import json

import pytest

from book.ir import Book, BookNode, CrossRef
from book import xref


def _book_with_xrefs() -> Book:
    """Three-chapter fixture with hand-rolled cross-refs."""
    root = BookNode(nid="b", kind="book", number=None,
                    title="ESL", page_start=1, page_end=999)
    ch3 = BookNode(nid="b/ch3", kind="chapter", number="3",
                   title="Linear Methods for Regression",
                   page_start=40, page_end=80)
    ch5 = BookNode(nid="b/ch5", kind="chapter", number="5",
                   title="Basis Expansions and Regularization",
                   page_start=100, page_end=200)
    s5_8 = BookNode(nid="b/ch5/s5_8", kind="section", number="5.8",
                    title="Reproducing Kernel Hilbert Spaces",
                    page_start=170, page_end=190)
    ch5.children.append(s5_8)
    ch12 = BookNode(nid="b/ch12", kind="chapter", number="12",
                    title="Support Vector Machines",
                    page_start=400, page_end=440)
    s12_3 = BookNode(nid="b/ch12/s12_3", kind="section", number="12.3",
                     title="SVMs and Kernels",
                     page_start=420, page_end=430)
    ch12.children.append(s12_3)
    root.children.extend([ch3, ch5, ch12])
    crs = [
        # ch3 → ch5 (twice)
        CrossRef(from_nid="b/ch3", to_nid="b/ch5",
                 label="Chapter 5", char_offset=10),
        CrossRef(from_nid="b/ch3", to_nid="b/ch5",
                 label="Chapter 5", char_offset=20),
        # ch12/s12_3 → ch5/s5_8 (chapter 12 cites chapter 5)
        CrossRef(from_nid="b/ch12/s12_3", to_nid="b/ch5/s5_8",
                 label="Section 5.8", char_offset=30),
        # ch12 → ch3
        CrossRef(from_nid="b/ch12", to_nid="b/ch3",
                 label="Chapter 3", char_offset=40),
        # Intra-chapter ref (ch5 cites its own subsection) — should NOT
        # become a chapter-graph edge.
        CrossRef(from_nid="b/ch5", to_nid="b/ch5/s5_8",
                 label="Section 5.8", char_offset=50),
        # Self-loop (chapter cites itself directly) — should drop.
        CrossRef(from_nid="b/ch3", to_nid="b/ch3",
                 label="Chapter 3", char_offset=60),
    ]
    return Book(title="ESL", author=None, source="", root=root,
                concepts={}, pages=[], figures=[], cross_refs=crs)


# ---------------------------------------------------------------------------
# chapter_graph aggregation
# ---------------------------------------------------------------------------

def test_chapter_graph_node_count():
    book = _book_with_xrefs()
    g = xref.chapter_graph(book)
    nids = {n["nid"] for n in g["nodes"]}
    assert nids == {"b/ch3", "b/ch5", "b/ch12"}


def test_chapter_graph_drops_intra_and_self():
    book = _book_with_xrefs()
    g = xref.chapter_graph(book)
    # Edges are only between distinct chapters.
    for e in g["edges"]:
        assert e["src"] != e["dst"]
    # ch5 → ch5 must not appear (intra-chapter ref was dropped).
    pairs = {(e["src"], e["dst"]) for e in g["edges"]}
    assert ("b/ch5", "b/ch5") not in pairs
    assert ("b/ch3", "b/ch3") not in pairs


def test_chapter_graph_aggregates_counts():
    book = _book_with_xrefs()
    g = xref.chapter_graph(book)
    pair_to_count = {(e["src"], e["dst"]): e["count"] for e in g["edges"]}
    # ch3 → ch5 has 2 edges in fixture.
    assert pair_to_count[("b/ch3", "b/ch5")] == 2
    # ch12 → ch3 has 1.
    assert pair_to_count[("b/ch12", "b/ch3")] == 1
    # ch12 → ch5 (via s12_3 → s5_8) has 1.
    assert pair_to_count[("b/ch12", "b/ch5")] == 1


def test_chapter_graph_node_counts():
    book = _book_with_xrefs()
    g = xref.chapter_graph(book)
    by_nid = {n["nid"]: n for n in g["nodes"]}
    # ch5: in_count = ch3→ch5 (2) + ch12→ch5 (1) = 3; out_count = 0.
    assert by_nid["b/ch5"]["in_count"] == 3
    assert by_nid["b/ch5"]["out_count"] == 0
    # ch3: in_count = 1 (from ch12); out_count = 2 (to ch5).
    assert by_nid["b/ch3"]["in_count"] == 1
    assert by_nid["b/ch3"]["out_count"] == 2


def test_chapter_graph_sorted_by_chapter_number():
    book = _book_with_xrefs()
    g = xref.chapter_graph(book)
    nums = [int(n["number"]) for n in g["nodes"]]
    assert nums == sorted(nums)


# ---------------------------------------------------------------------------
# Dependencies planner
# ---------------------------------------------------------------------------

def test_dependencies_lists_external_targets():
    """Ch.12's prerequisites should surface ch5 (via ch5/s5_8 cite)
    and ch3 — but NOT ch12 itself or its sub-sections."""
    from narrator.qa import dependencies
    book = _book_with_xrefs()
    plan = dependencies(book, "b/ch12")
    text = " ".join(c.text for c in plan.clauses)
    # External chapter prerequisites surface.
    assert "Linear Methods for Regression" in text or "§3" in text
    # Internal section names stay out.
    assert "SVMs and Kernels" not in text
    assert plan.meta["focus_nid"] == "b/ch12"
    assert plan.meta["n_prereqs"] >= 1


def test_dependencies_handles_chapter_with_no_outgoing():
    """A chapter that cites nothing external should still produce
    a sensible plan."""
    from narrator.qa import dependencies
    book = _book_with_xrefs()
    plan = dependencies(book, "b/ch5")
    text = " ".join(c.text for c in plan.clauses).lower()
    # ch5 has no outgoing in the fixture — should explain that.
    assert ("stands on its own" in text or "no outgoing" in text
            or "n_prereqs" in plan.meta)


def test_dependencies_handles_unknown_focus():
    from narrator.qa import dependencies
    book = _book_with_xrefs()
    plan = dependencies(book, "b/nope")
    text = " ".join(c.text for c in plan.clauses).lower()
    assert "ask me" in text or "studying" in text


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("utterance", [
    "what are the prerequisites",
    "what do I need to know first",
    "what should I know before",
    "what does this depend on",
    "background for kernels",
    "before we learn",
    "what comes before",
])
def test_router_classifies_dependencies(utterance):
    from narrator import router
    r = router.route(
        utterance,
        last_focus_nid="b/ch12",
        last_focus_topic="support vector machines",
    )
    assert r.intent == router.INTENT_DEPENDENCIES
    assert r.target_nid == "b/ch12"
