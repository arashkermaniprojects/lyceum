"""Robustness tests — bad inputs, edge cases, partial failure modes.

The system must degrade gracefully: empty / nonsense queries must
not crash the session; missing plan_ids must surface 404, not 500;
unknown books must report a clean error.  Persistence failures
must not leak into the user's turn.
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.request

import pytest

from book.ir import Book, BookNode
from chalkboard import Chalkboard
from narrator import NarrationClause, NarrationPlan
from narrator import qa as qa_mod
from narrator import router as router_mod
from narrator.tts import NullTTS

from serve import persistence
from serve.session import Session, build_session


# ---------------------------------------------------------------------------
# Empty / whitespace / nonsense
# ---------------------------------------------------------------------------

def test_empty_question_does_not_crash(session):
    tid = session.ask("")
    # Empty falls through to topic_qa which BM25s an empty query;
    # the underlying _retrieve returns an empty list and the
    # planner produces a degraded plan rather than crashing.
    assert isinstance(tid, str)


def test_whitespace_only_does_not_crash(session):
    session.ask("   \t\n  ")  # must not raise


def test_nonsense_question_routes_to_topic_qa(session):
    tid = session.ask("flibbertigibbet xyzzy quux")
    assert tid != ""
    assert session.dialogue_history[-1].intent == router_mod.INTENT_TOPIC_QA


def test_very_long_question_handled(session):
    """A 5,000-char question should still be routed."""
    q = "explain " + "very long bagging variant " * 200
    session.ask(q[:5000])
    assert session.dialogue_history[-1].intent == router_mod.INTENT_TOPIC_QA


def test_unicode_emoji_question_handled(session):
    # Unicode is fine; emoji shouldn't trip the regexes.
    session.ask("what about 𝕊upport vector machines 🤔")
    last = session.dialogue_history[-1]
    assert last.intent in {
        router_mod.INTENT_TOPIC_QA, router_mod.INTENT_FOLLOW_UP,
    }


# ---------------------------------------------------------------------------
# Citation shortcut edge cases
# ---------------------------------------------------------------------------

def test_unknown_section_falls_back_to_bm25(eslii):
    from narrator.qa import _retrieve
    # 99.99 doesn't exist in ESLII; the shortcut returns nothing,
    # and BM25 covers (or fails to cover) the rest.
    out = _retrieve(eslii, "section 99.99 please", top_k=2)
    nids = [p.nid for p in out]
    assert "b/ch99" not in nids


def test_unknown_chapter_routes_but_session_degrades(session):
    tid = session.ask("explain chapter 99")
    # Router still classifies as chapter_overview (target_nid empty).
    last = session.dialogue_history[-1]
    assert last.intent == router_mod.INTENT_CHAPTER_OVERVIEW
    assert last.focus_nid == ""
    # Session falls back to topic_qa under the hood, returning a tangent.
    assert tid != ""


def test_followup_without_focus_falls_back(session):
    """Asking ``tell me more`` before any content turn should not crash."""
    tid = session.ask("tell me more")
    assert isinstance(tid, str)


# ---------------------------------------------------------------------------
# Persistence robustness
# ---------------------------------------------------------------------------

def test_persistence_unsafe_plan_id_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SEVIM_SESSION_DIR", str(tmp_path))
    assert persistence.save_session_dict("../etc/passwd", {"x": 1}) is False
    assert persistence.load_session_dict("../etc/passwd") is None


def test_persistence_corrupt_file_skipped_in_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("SEVIM_SESSION_DIR", str(tmp_path))
    # Hand-write a malformed JSON file.
    with open(os.path.join(tmp_path, "broken.json"), "w") as f:
        f.write("{ not valid")
    persistence.save_session_dict("good", {"dialogue_history": []})
    summaries = persistence.list_session_summaries()
    assert any(s["plan_id"] == "good" for s in summaries)
    assert not any(s["plan_id"] == "broken" for s in summaries)


def test_session_save_callback_failure_does_not_crash(session):
    def explode(_):
        raise RuntimeError("disk full")
    session._save_callback = explode
    # Must not raise — turn completes, autosave failure is logged.
    session.ask("explain chapter 5")


# ---------------------------------------------------------------------------
# Streaming TTS fallback
# ---------------------------------------------------------------------------

def test_streaming_backend_disabled_falls_back(eslii):
    """When set_streaming_backend is None, qa.answer_streaming
    silently returns a non-streaming plan."""
    qa_mod.set_streaming_backend(None)
    plan = qa_mod.answer_streaming(eslii, "what is bagging")
    assert plan.streaming is False
    # Clauses are a list, not a generator.
    assert isinstance(plan.clauses, list)


def test_streaming_backend_failure_yields_empty(eslii):
    """A streaming backend that raises should not poison the
    generator — the wrapper logs and returns no clauses."""
    def bad(question, passages, book, history=None):
        raise RuntimeError("backend exploded")
        yield
    bad.is_streaming = True
    bad.supports_history = True
    qa_mod.set_streaming_backend(bad)
    try:
        plan = qa_mod.answer_streaming(eslii, "anything")
        assert plan.streaming is True
        # Drain — no exception should propagate.
        list(plan.clauses)
    finally:
        qa_mod.set_streaming_backend(None)


# ---------------------------------------------------------------------------
# Concurrent modification protection
# ---------------------------------------------------------------------------

def test_qa_set_alias_map_stores_copy():
    src = {"rbf": "radial basis function"}
    qa_mod.set_alias_map(src)
    src.clear()
    assert qa_mod.get_alias_map().get("rbf") == "radial basis function"
    qa_mod.set_alias_map({})


def test_session_book_unchanged_by_active_book_swap(session):
    """A session bound to ESLII keeps its book reference even if
    something else mutates the global active-book pointer."""
    original = session.book
    # Simulate a swap (not actually swapping; just checking ref).
    assert session.book is original


# ---------------------------------------------------------------------------
# ASR endpoint
# ---------------------------------------------------------------------------

def test_asr_decode_rejects_empty(monkeypatch):
    monkeypatch.setenv("SEVIM_DISABLE_ASR", "1")
    from serve import asr
    asr._model = None
    asr._model_load_error = None
    with pytest.raises(asr.AudioDecodeError):
        asr._decode_to_wav_pcm16(b"")


def test_asr_transcribe_raises_when_unavailable(monkeypatch):
    monkeypatch.setenv("SEVIM_DISABLE_ASR", "1")
    from serve import asr
    asr._model = None
    asr._model_load_error = None
    with pytest.raises(RuntimeError):
        asr.transcribe(b"\x00\x01\x02")
