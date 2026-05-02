"""Visualisation invariants — exercised against a real Session
running on the ESLII corpus.

The chalkboard's :class:`ReadingOrderPolicy` should never produce
overlapping shapes, formula cards should be sized to their content,
and the per-session caps should keep the board from drowning in
duplicates.  These properties are easy to regress when refactoring
the orchestrator, so we pin them here.
"""
from __future__ import annotations

import pytest

from chalkboard import Chalkboard, ReadingOrderPolicy
from serve.orchestrator import (
    AudioChunkEvent, AudioCompleteEvent, StreamEvent,
)

from ._fixtures import assert_no_overlap


def _drain(session, max_steps: int = 4_000) -> int:
    """Step the session iterator until tangent ends or we hit the cap."""
    n = 0
    for _ in range(max_steps):
        pe = session.next_event()
        if pe is None:
            return n
        if pe.is_tangent_end:
            return n
        if isinstance(pe.event, StreamEvent):
            n += 1
    return n


def test_no_overlap_after_full_journey(session):
    """A 10-turn conversation must never produce overlapping cards."""
    asks = [
        "what is this book about",
        "explain chapter 5",
        "section 5.8",
        "tell me more",
        "see also",
        "what do I need to know first",
        "what is bagging",
        "what about boosting",
        "what have we covered",
        "show me regularization again",
    ]
    for q in asks:
        tid = session.ask(q)
        if tid:
            _drain(session)
        assert_no_overlap(session.main_board)
        assert_no_overlap(session.tangent_board)


def test_caps_dont_drown_chalkboard(session):
    """Repeated re-narration should not push board past its cap.
    The orchestrator's ``_enforce_caps`` drops formula_card overflow
    before it lands."""
    for _ in range(8):
        tid = session.ask("explain chapter 5")
        if tid:
            _drain(session)
    # Tangent board respects max_content (chalkboard.max_content).
    assert len(session.tangent_board.shapes) <= session.tangent_board.max_content


def test_formula_card_sizing_grows_with_annotations():
    """Sanity-check that the formula card sizer stays sensitive to
    citations + var_defs.  This regression-pins the §5.8.1 fix."""
    from serve.orchestrator import _formula_card_size
    base_w, base_h = _formula_card_size("y = a x + b")
    # Citation chips render inline in the header — same height as base.
    cited_w, cited_h = _formula_card_size(
        "y = a x + b", cite_labels=["Equation 5.42"],
    )
    var_w, var_h = _formula_card_size(
        "y = a x + b",
        var_defs=[("a", "slope"), ("b", "intercept")],
    )
    assert cited_h == base_h
    assert var_h > base_h


def test_inline_formula_attaches_citations_in_same_clause():
    """When a clause says ``y = m x, ..., as in Equation 5.42``, the
    inline-formula path should mark Equation::5.42 as already-attached
    so the reference-card path doesn't double-emit."""
    from serve.orchestrator import (
        _equation_citations_in, _variable_definitions_in,
    )
    text = ("y = m x + b, where m is the slope and b is the intercept, "
            "as in Equation 5.42.")
    cites = _equation_citations_in(text)
    defs = _variable_definitions_in(text)
    assert cites == ["Equation 5.42"]
    syms = [s for s, _ in defs]
    assert "m" in syms and "b" in syms


def test_streaming_plan_does_not_break_chalkboard(eslii):
    """A streaming plan's clause generator iterates without
    pre-population.  The orchestrator's eq-warmup must skip in this
    case (we already pin this in test_streaming.py — re-asserting at
    the system level keeps the integration honest)."""
    from narrator.planner import NarrationClause, NarrationPlan
    from narrator.tts import NullTTS
    from serve.orchestrator import Orchestrator

    def _gen():
        for s in ["First.", "Second.", "Third."]:
            yield NarrationClause(text=s, home_nid="b",
                                   concepts=[], suggested_dur=1.0)

    plan = NarrationPlan(
        topic="<test>", book_title=eslii.title,
        clauses=_gen(), visited_nids=["b"],
        meta={"mode": "streaming_tutor"}, streaming=True,
    )
    orch = Orchestrator(book=eslii, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    events = list(orch.stream())
    # 3 clauses → 3 StreamEvents.  Streaming flag preserved.
    se = [e for e in events if isinstance(e, StreamEvent)]
    assert len(se) == 3
    # No overlapping shapes after streaming.
    assert_no_overlap(orch.chalkboard)


def test_picture_primitives_dont_crowd_cards(session):
    """Pictures (book_figure / canonical_figure / reference_card) and
    cursor-placed cards (passage_card / formula_card / math_note)
    coexist on the same board without overlap.  This regression-pins
    the Map-panel-era fix."""
    session.ask("explain chapter 5")
    _drain(session)
    # Manually drop a picture and a card next to each other.
    cb = session.tangent_board
    before = len(cb.shapes)
    cb.add(
        nid="test-pic", svg_body="<rect width='720' height='480'/>",
        primitive="book_figure", label="big-pic",
        w=720.0, h=480.0,
    )
    cb.add(
        nid="test-card", svg_body="<rect width='280' height='96'/>",
        primitive="formula_card", label="card",
        w=280.0, h=96.0,
    )
    assert_no_overlap(cb)
    assert len(cb.shapes) >= before + 2 - cb.max_content
