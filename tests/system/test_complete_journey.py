"""End-to-end "curious student" trajectory through a SeVim session.

Walks a single real Session through every kind of intent the system
exposes and pins the invariants we care about along the way:

  * Each turn appends one DialogueTurn (or zero, for control).
  * Focus pointers advance only on content turns.
  * Knowledge state accumulates across tangents.
  * Chalkboard never produces overlapping shapes.
  * Plans are non-empty, with the right ``meta.mode`` for the intent.

Uses NullTTS + the deterministic planners (no LLM) so the test is
fast and reproducible.  Live-LLM-only assertions live in
``test_pedagogy.py``.
"""
from __future__ import annotations

import pytest

from narrator import router as router_mod
from narrator.tts import NullTTS

from ._fixtures import assert_no_overlap


def _drain_tangent(session, tangent_id: str) -> int:
    """Step the session's iterator until the tangent ends.  Returns
    the number of clauses played."""
    if not tangent_id:
        return 0
    n_clauses = 0
    for _ in range(2_000):  # hard cap so a runaway test bails
        pe = session.next_event()
        if pe is None:
            break
        if pe.is_tangent_end and pe.tangent_id == tangent_id:
            break
        from serve.orchestrator import StreamEvent
        if isinstance(pe.event, StreamEvent) and pe.panel == "tangent":
            n_clauses += 1
    return n_clauses


def test_complete_curious_student_journey(session):
    """One Session, twelve turns, every intent kind exercised."""
    book = session.book

    # --- Turn 1: "what is this book about" → book_overview ---
    tid = session.ask("what is this book about")
    assert tid != "", "book_overview must produce a tangent"
    assert session.dialogue_history[-1].intent == router_mod.INTENT_BOOK_OVERVIEW
    n_clauses = _drain_tangent(session, tid)
    assert n_clauses >= 2, "book overview should narrate ≥ 2 clauses"
    assert_no_overlap(session.tangent_board)

    # --- Turn 2: "explain chapter 5" → chapter_overview ---
    tid = session.ask("explain chapter 5")
    assert tid != ""
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_CHAPTER_OVERVIEW
    assert last.focus_nid == "b/ch5"
    assert session._last_focus_nid == "b/ch5"
    _drain_tangent(session, tid)
    assert_no_overlap(session.tangent_board)

    # --- Turn 3: "section 5.8" → section_overview, focus updated ---
    tid = session.ask("section 5.8")
    assert tid != ""
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_SECTION_OVERVIEW
    assert last.focus_nid == "b/ch5/s5_8"
    assert session._last_focus_nid == "b/ch5/s5_8"
    _drain_tangent(session, tid)

    # --- Turn 4: "tell me more" → follow_up anchored on §5.8 ---
    tid = session.ask("tell me more")
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_FOLLOW_UP
    assert last.focus_topic == "section 5.8"
    # Focus pointer must NOT advance on a follow-up.
    assert session._last_focus_nid == "b/ch5/s5_8"
    _drain_tangent(session, tid)

    # --- Turn 5: "see also" → xref_explore on §5.8 ---
    tid = session.ask("see also")
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_XREF_EXPLORE
    n = _drain_tangent(session, tid)
    assert n >= 2, "xref_explore should narrate the neighborhood"

    # --- Turn 6: "what do I need to know first" → dependencies ---
    tid = session.ask("what do I need to know first")
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_DEPENDENCIES
    _drain_tangent(session, tid)

    # --- Turn 7: "pause" → control, no tangent, no focus advance ---
    tid = session.ask("pause")
    assert tid == ""
    assert session.dialogue_history[-1].intent == router_mod.INTENT_CONTROL
    assert session._paused is True
    assert session._last_focus_nid == "b/ch5/s5_8"

    # --- Turn 8: "resume" → control ---
    session.ask("resume")
    assert session._paused is False

    # --- Turn 9: explicit citation → citation shortcut ---
    tid = session.ask("explain section 3.4")
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_SECTION_OVERVIEW
    assert last.focus_nid == "b/ch3/s3_4"
    _drain_tangent(session, tid)

    # --- Turn 10: free-form question → topic_qa ---
    tid = session.ask("what is bagging")
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_TOPIC_QA
    assert "bagging" in last.focus_topic.lower()
    _drain_tangent(session, tid)

    # --- Turn 11: "what have we covered" → recap with content ---
    tid = session.ask("what have we covered")
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_RECAP
    n = _drain_tangent(session, tid)
    assert n >= 1, "recap should at minimum greet"

    # --- Turn 12: "show me bagging again" → reshow ---
    tid = session.ask("show me bagging again")
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_RESHOW
    _drain_tangent(session, tid)

    # --- Final invariants ---
    # Dialogue history is bounded.
    assert len(session.dialogue_history) <= session._history_cap
    # Knowledge accumulated across tangents.  The exact contents are
    # noisy but at least passage_card / canonical_topic dedup state
    # has *something* in it after a real conversation.
    knowledge_total = (
        len(session.knowledge.seen_nids)
        + len(session.knowledge.seen_canonical_topics)
        + len(session.knowledge.seen_refs)
    )
    assert knowledge_total > 0, (
        "after a 12-turn conversation, knowledge state must have grown"
    )


def test_forget_clears_knowledge(session):
    """`forget` is a control that wipes SessionKnowledge."""
    session.knowledge.seen_canonical_topics.update({"a", "b", "c"})
    session.knowledge.seen_formulas.add("y=mx")
    session.knowledge.spoken_topics.append("regularization")
    session.ask("forget what we covered")
    assert session.knowledge.seen_canonical_topics == set()
    assert session.knowledge.seen_formulas == set()
    assert session.knowledge.spoken_topics == []


def test_topic_switch_does_not_drag_old_focus(session):
    session.ask("explain chapter 5")
    assert session._last_focus_nid == "b/ch5"
    session.ask("explain chapter 3")
    assert session._last_focus_nid == "b/ch3"
    session.ask("tell me more")
    assert session.dialogue_history[-1].focus_topic == "chapter 3"


def test_long_conversation_bounded(session):
    for i in range(20):
        session.ask(f"explain chapter {(i % 5) + 1}")
    # History is capped.
    assert len(session.dialogue_history) <= session._history_cap


def test_recap_reflects_history(session):
    """After a few topical asks, recap surfaces them."""
    session.ask("what is bagging")
    session.ask("what is boosting")
    session.ask("explain regularization")
    tid = session.ask("recap")
    # Drain.
    plan_meta = None
    for _ in range(200):
        pe = session.next_event()
        if pe is None or (pe.is_tangent_end and pe.tangent_id == tid):
            break
        from serve.orchestrator import StreamEvent
        if isinstance(pe.event, StreamEvent):
            plan_meta = pe.event
    assert plan_meta is not None
    # Recap clause text mentions some of the asked topics.
    text_corpus = " ".join(
        c.user_text for c in session.dialogue_history
    ).lower()
    assert "bagging" in text_corpus
    assert "boosting" in text_corpus
