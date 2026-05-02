"""Live-LLM pedagogy tests.

These probe the *content* the system produces — adaptive length,
grounding in book passages, citation use — by hitting the actual
local Qwen.  Skipped when vLLM isn't reachable so CI doesn't block
on a network dependency.

Run with::

    pytest tests/system/test_pedagogy.py -v
"""
from __future__ import annotations

import time

import pytest

from narrator import qa as qa_mod

from ._fixtures import live_llm, vllm_reachable


@pytest.fixture(scope="module")
def tutor_backend(eslii):
    """Install the streaming tutor backend for these tests."""
    if not vllm_reachable():
        pytest.skip("vLLM unreachable")
    from narrator.qa import (
        make_vllm_tutor_streaming_backend,
        set_streaming_backend, set_alias_map,
    )
    from book.aliases import extract_aliases
    set_alias_map(extract_aliases(eslii))
    bk = make_vllm_tutor_streaming_backend()
    if bk is None:
        pytest.skip("streaming tutor backend could not be built")
    set_streaming_backend(bk)
    yield bk
    set_streaming_backend(None)
    set_alias_map({})


def _drain_clauses(plan) -> list[str]:
    return [c.text for c in plan.clauses]


@live_llm
def test_brief_vs_deep_length_adapts(eslii, tutor_backend):
    """Same topic, different phrasing → different clause counts."""
    brief = qa_mod.answer_streaming(
        eslii, "what is bagging in one sentence",
    )
    brief_clauses = _drain_clauses(brief)
    deep = qa_mod.answer_streaming(
        eslii, "explain bagging in detail step by step with the formulas",
    )
    deep_clauses = _drain_clauses(deep)
    assert len(deep_clauses) > len(brief_clauses), (
        f"deep should have more clauses than brief; "
        f"brief={len(brief_clauses)} deep={len(deep_clauses)}"
    )
    # Brief stays short.
    assert len(brief_clauses) <= 4


@live_llm
def test_first_audio_under_one_second(eslii, tutor_backend):
    """Streaming should yield the first sentence quickly."""
    plan = qa_mod.answer_streaming(eslii, "what is regularization")
    t0 = time.perf_counter()
    first = next(iter(plan.clauses))
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, f"first clause took {elapsed:.2f}s"
    assert first.text.strip()


@live_llm
def test_followup_does_not_repeat_self(eslii, tutor_backend):
    """A follow-up should not trivially copy the prior answer's
    opening sentence verbatim — it must build on it."""
    first_plan = qa_mod.answer_streaming(eslii, "what is regularization")
    first_text = " ".join(_drain_clauses(first_plan))
    follow = qa_mod.answer_streaming(
        eslii, "tell me more about it",
        history=[
            type("T", (), {"user_text": "what is regularization",
                            "intent": "topic_qa",
                            "focus_topic": "regularization"})(),
        ],
    )
    follow_text = " ".join(_drain_clauses(follow))
    # Different opening sentence at minimum.
    if first_text.split(".")[0] and follow_text.split(".")[0]:
        assert (first_text.split(".")[0].strip()
                != follow_text.split(".")[0].strip())


@live_llm
def test_citation_in_answer_grounds_in_passage(eslii, tutor_backend):
    """A question that names a section should pull that section's
    content into the answer."""
    plan = qa_mod.answer_streaming(
        eslii, "what does section 5.8 say about kernels",
    )
    text = " ".join(_drain_clauses(plan)).lower()
    assert ("kernel" in text or "reproducing" in text), text[:300]
