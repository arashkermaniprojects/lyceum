"""Pin session persistence: snapshot round-trip, save/load to disk,
auto-save callback firing on each turn, resumable sessions.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from book.ir import Book, BookNode
from chalkboard import Chalkboard
from chalkboard.state import ChalkShape
from narrator import NarrationClause, NarrationPlan
from narrator import router as router_mod
from narrator.tts import NullTTS

from serve import persistence
from serve.session import (
    DialogueTurn, Session, SessionKnowledge, build_session,
)


@pytest.fixture
def tmp_session_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("SEVIM_SESSION_DIR", td)
        yield td


def _book() -> Book:
    root = BookNode(nid="b", kind="book", number=None,
                    title="ESL", page_start=1, page_end=999)
    ch5 = BookNode(nid="b/ch5", kind="chapter", number="5",
                   title="Basis Expansions and Regularization",
                   page_start=100, page_end=200)
    root.children.append(ch5)
    return Book(title="ESL", author=None, source="", root=root,
                concepts={}, pages=[], figures=[], cross_refs=[])


def _session() -> Session:
    book = _book()
    plan = NarrationPlan(
        topic="<test>", book_title=book.title,
        clauses=[NarrationClause(text="t", home_nid="b",
                                  concepts=[], suggested_dur=1.0)],
        visited_nids=["b"], meta={"mode": "full"},
    )
    return build_session(
        plan_id="plan-test-1",
        book=book, plan=plan,
        tts_factory=lambda: NullTTS(),
    )


# ---------------------------------------------------------------------------
# Persistence module — save / load / list
# ---------------------------------------------------------------------------

def test_session_dir_creates(tmp_session_dir):
    d = persistence.session_dir()
    assert os.path.isdir(d)
    assert d == tmp_session_dir


def test_save_and_load_round_trip(tmp_session_dir):
    snap = {
        "plan_id": "abc-123",
        "dialogue_history": [
            {"user_text": "hi", "intent": "topic_qa"},
        ],
        "knowledge": {"seen_canonical_topics": ["regularization"]},
    }
    assert persistence.save_session_dict("abc-123", snap) is True
    loaded = persistence.load_session_dict("abc-123")
    assert loaded is not None
    assert loaded["plan_id"] == "abc-123"
    assert loaded["dialogue_history"][0]["user_text"] == "hi"
    assert "updated_at" in loaded


def test_load_returns_none_when_missing(tmp_session_dir):
    assert persistence.load_session_dict("does-not-exist") is None


def test_unsafe_plan_id_rejected(tmp_session_dir):
    """Path-traversal must not write outside the session dir."""
    bad = "../etc/passwd"
    assert persistence.save_session_dict(bad, {"x": 1}) is False
    assert persistence.load_session_dict(bad) is None


def test_list_session_summaries_orders_newest_first(tmp_session_dir):
    persistence.save_session_dict("old", {
        "dialogue_history": [{"user_text": "old"}],
        "last_focus_topic": "Topic A",
    })
    # Ensure timestamps differ — sleep would slow tests; just back-date.
    import json, os, time
    old_path = os.path.join(tmp_session_dir, "old.json")
    with open(old_path) as f: snap = json.load(f)
    snap["updated_at"] = time.time() - 3600  # one hour ago
    with open(old_path, "w") as f: json.dump(snap, f)

    persistence.save_session_dict("new", {
        "dialogue_history": [{"user_text": "new1"}, {"user_text": "new2"}],
        "last_focus_topic": "Topic B",
    })
    summaries = persistence.list_session_summaries()
    assert summaries[0]["plan_id"] == "new"
    assert summaries[0]["dialogue_len"] == 2
    assert summaries[1]["plan_id"] == "old"


def test_delete_session_removes_file(tmp_session_dir):
    persistence.save_session_dict("ephemeral", {"x": 1})
    assert os.path.exists(os.path.join(tmp_session_dir, "ephemeral.json"))
    assert persistence.delete_session("ephemeral") is True
    assert not os.path.exists(os.path.join(tmp_session_dir, "ephemeral.json"))
    # Double-delete is a no-op.
    assert persistence.delete_session("ephemeral") is False


# ---------------------------------------------------------------------------
# Session snapshot / restore
# ---------------------------------------------------------------------------

def test_snapshot_dict_captures_state():
    sess = _session()
    sess.dialogue_history.append(DialogueTurn(
        user_text="explain bagging",
        intent=router_mod.INTENT_TOPIC_QA,
        focus_topic="bagging", focus_nid="b/ch8",
    ))
    sess.knowledge.seen_canonical_topics.add("regularization")
    sess.knowledge.seen_refs.add("Equation::5.42")
    sess.knowledge.spoken_topics.append("bagging")
    sess._last_focus_topic = "bagging"
    sess._last_focus_nid = "b/ch8"
    # Add a chalkboard shape so the snapshot has something visual.
    sess.main_board.add(
        nid="card-1", svg_body="<rect/>",
        primitive="formula_card", label="y = m x",
        w=120, h=80, meta={"fragment": "y = m x"},
    )

    snap = sess.snapshot_dict()
    assert snap["plan_id"] == "plan-test-1"
    assert snap["last_focus_topic"] == "bagging"
    assert "regularization" in snap["knowledge"]["seen_canonical_topics"]
    assert "Equation::5.42" in snap["knowledge"]["seen_refs"]
    assert len(snap["dialogue_history"]) == 1
    shapes = snap["chalkboard"]["shapes"]
    assert len(shapes) == 1
    assert shapes[0]["nid"] == "card-1"
    assert shapes[0]["primitive"] == "formula_card"


def test_restore_from_dict_rebuilds_state():
    """A fresh session can adopt another's snapshot wholesale."""
    sess1 = _session()
    sess1.dialogue_history.append(DialogueTurn(
        user_text="explain regularization",
        intent=router_mod.INTENT_TOPIC_QA,
        focus_topic="regularization",
    ))
    sess1.knowledge.seen_canonical_topics.add("ridge")
    sess1.main_board.add(
        nid="c", svg_body="<rect/>",
        primitive="formula_card", label="L",
        w=100, h=60, meta={"x": 1},
    )
    snap = sess1.snapshot_dict()

    # Build a fresh resumable session and restore.
    book = _book()
    sess2 = Session(plan_id="plan-test-1", book=book, main_orch=None)
    sess2.restore_from_dict(snap)

    assert sess2.dialogue_history[0].user_text == "explain regularization"
    assert "ridge" in sess2.knowledge.seen_canonical_topics
    assert len(sess2.main_board.shapes) == 1
    assert sess2.main_board.shapes[0].nid == "c"


def test_resumable_session_has_no_main_iter():
    """A session with main_orch=None must still be usable for tangents."""
    book = _book()
    sess = Session(plan_id="plan-resumed", book=book, main_orch=None)
    assert sess._main_iter is None
    assert sess.is_active() is True
    # next_event returns None gracefully (no main events).
    assert sess.next_event() is None


# ---------------------------------------------------------------------------
# Auto-save callback fires on every turn
# ---------------------------------------------------------------------------

def test_save_callback_fires_on_turn(tmp_session_dir):
    sess = _session()
    calls = []
    sess._save_callback = lambda s: calls.append(s.plan_id)

    sess.ask("explain chapter 5")
    sess.ask("pause")
    sess.ask("tell me more")
    assert len(calls) >= 3
    assert all(c == "plan-test-1" for c in calls)


def test_save_callback_failure_does_not_crash_turn(tmp_session_dir):
    sess = _session()
    def bad(_): raise RuntimeError("disk full")
    sess._save_callback = bad
    # Must not raise.
    sess.ask("explain chapter 5")
    assert sess.dialogue_history[-1].user_text == "explain chapter 5"


# ---------------------------------------------------------------------------
# Summary shape — frontend's sessions panel relies on these fields
# ---------------------------------------------------------------------------

def test_session_summary_fields_stable(tmp_session_dir):
    """The fields the frontend's sessions sidebar reads must keep
    stable names: plan_id, updated_at, dialogue_len, last_focus_topic,
    book_title."""
    persistence.save_session_dict("alpha", {
        "dialogue_history": [
            {"user_text": "what is bagging", "intent": "topic_qa"},
            {"user_text": "tell me more", "intent": "follow_up"},
        ],
        "last_focus_topic": "bagging",
        "book_title": "ESL",
    })
    summaries = persistence.list_session_summaries()
    assert len(summaries) == 1
    s = summaries[0]
    for key in ("plan_id", "updated_at", "dialogue_len",
                "last_focus_topic", "book_title"):
        assert key in s, f"missing {key!r}"
    assert s["dialogue_len"] == 2
    assert s["last_focus_topic"] == "bagging"
    assert s["book_title"] == "ESL"


def test_summary_handles_missing_optional_fields(tmp_session_dir):
    """Old / partial snapshots without last_focus_topic must not
    break the listing."""
    persistence.save_session_dict("partial", {
        "dialogue_history": [],
    })
    summaries = persistence.list_session_summaries()
    s = next(x for x in summaries if x["plan_id"] == "partial")
    assert s["dialogue_len"] == 0
    assert s["last_focus_topic"] == ""
    assert s["book_title"] == ""


def test_summary_skips_corrupt_files(tmp_session_dir):
    """Junk JSON files in the session dir must be skipped, not crash."""
    bad_path = os.path.join(tmp_session_dir, "broken.json")
    with open(bad_path, "w") as f:
        f.write("{ not valid json")
    persistence.save_session_dict("good", {"dialogue_history": []})
    summaries = persistence.list_session_summaries()
    assert any(s["plan_id"] == "good" for s in summaries)
    assert not any(s["plan_id"] == "broken" for s in summaries)


def test_summary_limit(tmp_session_dir):
    for i in range(8):
        persistence.save_session_dict(f"s{i}", {
            "dialogue_history": [{"user_text": f"q{i}"}],
        })
    summaries = persistence.list_session_summaries(limit=3)
    assert len(summaries) == 3


# ---------------------------------------------------------------------------
# Book-tagged sessions
# ---------------------------------------------------------------------------

def test_snapshot_includes_book_name():
    """A session bound to a known book should record its name in the
    snapshot so cross-book persistence stays sane."""
    sess = _session()
    snap = sess.snapshot_dict()
    # Without a server registry, book_name is "" — but the field
    # MUST be present so resume code can branch on it.
    assert "book_name" in snap


def test_summary_surfaces_book_name(tmp_session_dir):
    persistence.save_session_dict("alpha", {
        "dialogue_history": [],
        "book_name": "ESLII",
    })
    summaries = persistence.list_session_summaries()
    assert summaries[0]["book_name"] == "ESLII"
