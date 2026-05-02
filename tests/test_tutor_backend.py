"""Pin the new tutor backend's prompt-shape, history threading, and
qa.answer integration.  We don't hit a real LLM in these tests —
the backend is replaced by a fake that records its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from book.ir import Book, BookNode
from narrator import qa as qa_mod


@pytest.fixture(autouse=True)
def _isolate_qa_state():
    """Snapshot/restore the qa module's installed backends + caches
    so a server fixture in another test file doesn't leak global
    state into these tests."""
    saved_synth = qa_mod._synth_backend
    saved_intro = qa_mod._intro_backend
    saved_stream = qa_mod._streaming_backend
    saved_alias = qa_mod._alias_map
    saved_concept = qa_mod._concept_graph
    qa_mod._intro_backend = None
    qa_mod._streaming_backend = None
    qa_mod._alias_map = {}
    qa_mod._concept_graph = {}
    yield
    qa_mod._synth_backend = saved_synth
    qa_mod._intro_backend = saved_intro
    qa_mod._streaming_backend = saved_stream
    qa_mod._alias_map = saved_alias
    qa_mod._concept_graph = saved_concept
from narrator.qa import (
    RetrievedPassage,
    _format_history_for_prompt,
    _TUTOR_PROMPT_SYSTEM,
)


def _book() -> Book:
    root = BookNode(nid="b", kind="book", number=None, title="ESL",
                    page_start=1, page_end=999)
    ch5 = BookNode(nid="b/ch5", kind="chapter", number="5",
                   title="Basis Expansions and Regularization",
                   page_start=100, page_end=200,
                   body_text="Basis expansion ridge lambda regularization "
                             "smoothing splines reproducing kernel.")
    root.children.append(ch5)
    return Book(title="ESL", author=None, source="", root=root,
                concepts={}, pages=[], figures=[], cross_refs=[])


@dataclass
class _StubTurn:
    user_text: str = ""
    intent: str = ""
    focus_topic: str = ""
    focus_nid: str = ""


# ---------------------------------------------------------------------------
# History formatter
# ---------------------------------------------------------------------------

def test_format_history_skips_control_and_recap():
    history = [
        _StubTurn(user_text="what is bagging", intent="topic_qa",
                  focus_topic="what is bagging"),
        _StubTurn(user_text="pause", intent="control"),
        _StubTurn(user_text="explain section 5.8", intent="section_overview",
                  focus_topic="section 5.8"),
        _StubTurn(user_text="recap", intent="recap"),
    ]
    blob = _format_history_for_prompt(history)
    assert "bagging" in blob
    assert "5.8" in blob
    assert "pause" not in blob
    assert "recap" not in blob


def test_format_history_caps_at_max_turns():
    history = [
        _StubTurn(user_text=f"q{i}", intent="topic_qa")
        for i in range(20)
    ]
    blob = _format_history_for_prompt(history, max_turns=3)
    # Only the last 3 should remain.
    assert "q17" in blob and "q18" in blob and "q19" in blob
    assert "q0" not in blob and "q5" not in blob


def test_format_history_empty_returns_empty():
    assert _format_history_for_prompt(None) == ""
    assert _format_history_for_prompt([]) == ""


# ---------------------------------------------------------------------------
# qa.answer routes history into history-aware backends
# ---------------------------------------------------------------------------

def test_qa_answer_threads_history_to_supports_backend():
    """When the backend has supports_history=True, qa.answer passes
    the history list as the 4th positional arg."""
    book = _book()
    captured = {}

    def fake(question, passages, book, history=None):
        captured["question"] = question
        captured["history"] = history
        captured["n_passages"] = len(passages)
        return ["Bagging averages predictions across bootstrap samples."]
    fake.supports_history = True

    history = [_StubTurn(user_text="what is bagging", intent="topic_qa")]
    plan = qa_mod.answer(
        book, "explain bagging in detail",
        backend=fake, history=history,
    )
    assert captured["question"] == "explain bagging in detail"
    assert captured["history"] == history
    assert plan.clauses[0].text.startswith("Bagging averages")


def test_qa_answer_omits_history_for_legacy_backend():
    """A backend without supports_history should be called the
    legacy 3-arg way — history must not leak in."""
    book = _book()
    captured = {}

    def legacy(question, passages, book):
        captured["called"] = True
        return ["legacy answer."]
    # No supports_history attribute.

    plan = qa_mod.answer(
        book, "hi", backend=legacy,
        history=[_StubTurn(user_text="prior", intent="topic_qa")],
    )
    assert captured["called"] is True


# ---------------------------------------------------------------------------
# Tutor prompt content
# ---------------------------------------------------------------------------

def test_tutor_prompt_explicitly_describes_length_adaptivity():
    """The prompt sets a concise default with an opt-in for depth.
    Default = 1–2 sentences; depth keywords trigger longer answers."""
    p = _TUTOR_PROMPT_SYSTEM.lower()
    assert "concise" in p or "default length" in p
    assert "depth" in p or "detail" in p
    assert "follow-up" in p or "follow up" in p


def test_tutor_prompt_allows_latex_delimiters():
    p = _TUTOR_PROMPT_SYSTEM
    assert "\\(" in p or "LaTeX" in p


def test_tutor_prompt_grounds_in_passages():
    p = _TUTOR_PROMPT_SYSTEM.lower()
    assert "passage" in p or "ground" in p
    assert "equation" in p


def test_tutor_prompt_requires_verbalised_math():
    """The narrator's voice goes through TTS — symbol names like 'sigma'
    or 'partial' make for terrible audio.  The prompt must tell the
    model to spell formulas out in English alongside the LaTeX."""
    p = _TUTOR_PROMPT_SYSTEM.lower()
    # The rule itself.
    assert "verbalize" in p or "verbalise" in p or "in words" in p, (
        "Prompt must instruct the model to verbalize math notation."
    )
    # Concrete worked examples — without these the model defaults to
    # reading symbols literally ("sigma i equals 1 N x i").
    assert "sum" in p and "integral" in p
    # Greek letters are explicitly handled.
    assert "alpha" in p and "beta" in p
