"""Pin Session.dialogue_history + smart routing in Session.ask.

Each turn the user takes flows through ``narrator.router``; the
session updates dialogue_history and refreshes its focus pointers
so the next "tell me more" lands in context.  Control intents
(pause/stop) short-circuit and never produce a tangent plan.
"""
from __future__ import annotations

from book.ir import Book, BookNode
from chalkboard import Chalkboard
from narrator import NarrationClause, NarrationPlan
from narrator import router as router_mod
from narrator.tts import NullTTS

from serve.orchestrator import Orchestrator
from serve.session import Session, build_session


def _book() -> Book:
    root = BookNode(nid="b", kind="book", number=None,
                    title="ESL", page_start=1, page_end=999)
    ch5 = BookNode(nid="b/ch5", kind="chapter", number="5",
                   title="Basis Expansions and Regularization",
                   page_start=100, page_end=200)
    ch3 = BookNode(nid="b/ch3", kind="chapter", number="3",
                   title="Linear Methods for Regression",
                   page_start=40, page_end=80)
    root.children.extend([ch3, ch5])
    return Book(title="ESL", author=None, source="", root=root,
                concepts={}, pages=[], figures=[])


def _session() -> Session:
    book = _book()
    plan = NarrationPlan(
        topic="<test>", book_title=book.title,
        clauses=[NarrationClause(text="t", home_nid="b",
                                  concepts=[], suggested_dur=1.0)],
        visited_nids=["b"], meta={"mode": "full"},
    )
    return build_session(
        plan_id="test-plan",
        book=book, plan=plan,
        tts_factory=lambda: NullTTS(),
    )


# ---------------------------------------------------------------------------
# Dialogue history grows
# ---------------------------------------------------------------------------

def test_chapter_overview_records_focus():
    s = _session()
    tangent = s.ask("explain chapter 5")
    assert tangent != ""
    assert len(s.dialogue_history) == 1
    turn = s.dialogue_history[0]
    assert turn.intent == router_mod.INTENT_CHAPTER_OVERVIEW
    assert turn.focus_nid == "b/ch5"
    # Focus pointers updated for follow-ups.
    assert s._last_focus_nid == "b/ch5"
    assert s._last_focus_topic == "chapter 5"


def test_follow_up_uses_last_focus():
    s = _session()
    s.ask("explain chapter 5")
    s.ask("tell me more")
    assert len(s.dialogue_history) == 2
    follow = s.dialogue_history[1]
    assert follow.intent == router_mod.INTENT_FOLLOW_UP
    assert follow.focus_topic == "chapter 5"
    assert follow.focus_nid == "b/ch5"


def test_control_pause_short_circuits():
    s = _session()
    tangent = s.ask("pause")
    assert tangent == ""           # control intent: no tangent
    assert s._paused is True
    assert s.dialogue_history[-1].intent == router_mod.INTENT_CONTROL


def test_control_stop_cancels_session():
    s = _session()
    s.ask("stop")
    assert s.is_active() is False


def test_book_overview_routes_to_book_overview_planner():
    s = _session()
    s.ask("what is this book about")
    turn = s.dialogue_history[-1]
    assert turn.intent == router_mod.INTENT_BOOK_OVERVIEW


def test_history_capped():
    """Beyond _history_cap turns, oldest is dropped."""
    s = _session()
    # Use a tiny cap so the test runs cheaply.
    s._history_cap = 3
    for q in ["chapter 5", "chapter 3", "what is bagging",
              "what is boosting", "chapter 5"]:
        s.ask(q)
    assert len(s.dialogue_history) == 3


def test_focus_does_not_move_on_control_or_followup():
    """Pause / follow-up don't advance the focus pointer — only a
    fresh content turn does."""
    s = _session()
    s.ask("explain chapter 5")
    s.ask("pause")
    s.ask("tell me more")
    assert s._last_focus_nid == "b/ch5"
    assert s._last_focus_topic == "chapter 5"
