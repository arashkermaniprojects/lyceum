"""Pin ``narrator.qa.book_overview`` — three depth modes,
chapter-aware narration, no front-matter leakage.
"""
from __future__ import annotations

import pytest

from book.ir import Book, BookNode
from narrator.qa import book_overview


def _book(*, with_author: bool = True) -> Book:
    root = BookNode(nid="b", kind="book", number=None,
                    title="The Elements of Statistical Learning",
                    page_start=1, page_end=999)
    for num, title in [
        ("1", "Introduction"),
        ("2", "Overview of Supervised Learning"),
        ("3", "Linear Methods for Regression"),
        ("4", "Linear Methods for Classification"),
        ("5", "Basis Expansions and Regularization"),
        ("6", "Kernel Smoothing Methods"),
    ]:
        root.children.append(
            BookNode(nid=f"b/ch{num}", kind="chapter", number=num,
                     title=title, page_start=0, page_end=0)
        )
    return Book(
        title=root.title,
        author="Hastie, Tibshirani, Friedman" if with_author else None,
        source="", root=root, concepts={}, pages=[], figures=[],
    )


# ---------------------------------------------------------------------------
# Depth modes
# ---------------------------------------------------------------------------

def test_short_overview_two_clauses():
    plan = book_overview(_book(), depth="short")
    assert plan.meta["mode"] == "book_overview"
    assert plan.meta["depth"] == "short"
    assert len(plan.clauses) == 2
    assert "Hastie" in plan.clauses[0].text
    assert "21" not in plan.clauses[1].text  # only 6 in fixture
    assert "6 chapters" in plan.clauses[1].text


def test_default_overview_groups_chapters():
    plan = book_overview(_book())
    assert plan.meta["mode"] == "book_overview"
    # First clause names the book; subsequent group chapter titles.
    assert "Elements of Statistical Learning" in plan.clauses[0].text
    later = " ".join(c.text for c in plan.clauses[1:])
    assert "Introduction" in later
    assert "Kernel Smoothing Methods" in later
    # Closes with an invitation.
    assert "Ask me about" in plan.clauses[-1].text


def test_deep_overview_one_clause_per_chapter():
    plan = book_overview(_book(), depth="deep")
    chapter_clauses = [c for c in plan.clauses if c.text.startswith("Chapter")]
    assert len(chapter_clauses) == 6
    assert any("Linear Methods for Regression" in c.text for c in chapter_clauses)
    # Visited nids include every chapter's nid.
    assert "b/ch5" in plan.visited_nids


def test_overview_handles_missing_author():
    plan = book_overview(_book(with_author=False))
    assert "by " not in plan.clauses[0].text
    assert "The Elements of Statistical Learning" in plan.clauses[0].text


def test_overview_topic_marker():
    """The plan is marked with a synthetic topic so the orchestrator
    knows this isn't a regular Q&A run."""
    plan = book_overview(_book())
    assert plan.topic == "<book-overview>"
