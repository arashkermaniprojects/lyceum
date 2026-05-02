"""Pin the per-session card caps + tightened garbled detector.

User reported §5.8.1 emitting 5+ formula cards (overlapping) and 2
duplicate canonical figures plus an Equation 5.47 reference whose
body ``i/γi < ∞,`` slipped past the multi-line garbled detector.
"""
from __future__ import annotations

from unittest.mock import patch

from book.ir import Book, BookNode
from chalkboard import Chalkboard
from narrator import NarrationClause, NarrationPlan
from narrator.tts import NullTTS
from serve.orchestrator import (
    CANONICAL_FIGURES_MAX,
    FORMULA_CARDS_MAX,
    Orchestrator,
    _looks_garbled_equation,
)


def test_garbled_detector_catches_single_line_partial_inequality():
    """``i/γi < ∞,`` is half an inequality with a trailing comma —
    clearly a fragment of a bigger equation."""
    assert _looks_garbled_equation("i/γi < ∞,")
    assert _looks_garbled_equation("γi < ∞,")
    # Whitespace doesn't save it.
    assert _looks_garbled_equation("  i / γi < ∞,  ")


def test_garbled_detector_keeps_complete_inequalities():
    """Complete one-line equations / inequalities still pass."""
    samples = [
        "y = X beta + epsilon",
        "MSE = (1/n) sum_i (y_i - hat y_i)^2",
        "P(A | B) > 0",
        "x = a + b",
    ]
    for s in samples:
        assert not _looks_garbled_equation(s), f"falsely flagged: {s!r}"


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------

def _orch_with_synthetic_ops() -> Orchestrator:
    root = BookNode(nid="b", kind="book", number=None, title="t",
                    page_start=1, page_end=10)
    book = Book(title="t", author=None, source="", root=root,
                concepts={}, pages=[], figures=[])
    plan = NarrationPlan(
        topic="<full-book>", book_title="t",
        clauses=[NarrationClause(text="x", home_nid="b",
                                  concepts=[], suggested_dur=1.0)],
        visited_nids=["b"], meta={"mode": "full"},
    )
    return Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())


def test_formula_cards_capped():
    orch = _orch_with_synthetic_ops()
    ops = [
        {"primitive": "formula_card", "nid": f"f{i}",
         "kind": "add", "label": f"f{i}",
         "svg": "<rect/>", "t": 0.0, "w": 100, "h": 50}
        for i in range(FORMULA_CARDS_MAX + 4)
    ]
    kept = orch._enforce_caps(ops)
    n_formula = sum(1 for o in kept if o.get("primitive") == "formula_card")
    assert n_formula == FORMULA_CARDS_MAX


def test_canonical_figures_capped():
    orch = _orch_with_synthetic_ops()
    ops = [
        {"primitive": "canonical_figure", "nid": f"c{i}",
         "kind": "add", "label": f"c{i}",
         "svg": "<rect/>", "t": 0.0, "w": 200, "h": 100}
        for i in range(CANONICAL_FIGURES_MAX + 3)
    ]
    kept = orch._enforce_caps(ops)
    n_canon = sum(1 for o in kept if o.get("primitive") == "canonical_figure")
    assert n_canon == CANONICAL_FIGURES_MAX


def test_other_primitives_pass_through_uncapped():
    """Passage banners, book figures, reference cards — unaffected."""
    orch = _orch_with_synthetic_ops()
    ops = [
        {"primitive": "passage_card",   "nid": "p1", "kind": "add",
         "svg": "<rect/>", "t": 0.0, "w": 100, "h": 50},
        {"primitive": "book_figure",    "nid": "f1", "kind": "add",
         "svg": "<rect/>", "t": 0.0, "w": 200, "h": 100},
        {"primitive": "reference_card", "nid": "r1", "kind": "add",
         "svg": "<rect/>", "t": 0.0, "w": 200, "h": 80},
    ] * 4  # 12 ops total, none should drop.
    kept = orch._enforce_caps(ops)
    assert len(kept) == len(ops)


def test_caps_persist_across_clauses():
    """Once the formula cap is hit on clause 0, clause 1's formula
    cards are also dropped."""
    orch = _orch_with_synthetic_ops()
    first_batch = [
        {"primitive": "formula_card", "nid": f"a{i}",
         "kind": "add", "label": f"a{i}",
         "svg": "<rect/>", "t": 0.0, "w": 100, "h": 50}
        for i in range(FORMULA_CARDS_MAX)
    ]
    second_batch = [
        {"primitive": "formula_card", "nid": "b1", "kind": "add",
         "label": "b1", "svg": "<rect/>", "t": 0.0, "w": 100, "h": 50},
    ]
    orch._enforce_caps(first_batch)
    kept = orch._enforce_caps(second_batch)
    assert kept == []
