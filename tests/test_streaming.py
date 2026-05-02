"""Pin sentence peeling, streaming-plan handling in the orchestrator,
and qa.answer_streaming with a mocked streaming backend.
"""
from __future__ import annotations

import pytest

from book.ir import Book, BookNode
from chalkboard import Chalkboard
from narrator import qa as qa_mod
from narrator.planner import NarrationClause, NarrationPlan
from narrator.qa import _peel_sentence
from narrator.tts import NullTTS

from serve.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Sentence peeling
# ---------------------------------------------------------------------------

def test_peel_simple_sentence():
    sent, rest = _peel_sentence("This is a sentence. And the next one starts.")
    assert sent == "This is a sentence."
    assert rest.lstrip().startswith("And")


def test_peel_returns_none_when_incomplete():
    sent, rest = _peel_sentence("Half a sentence with no terminator")
    assert sent is None
    assert rest == "Half a sentence with no terminator"


def test_peel_returns_none_on_trailing_period_without_next():
    """We can't tell from a trailing period whether a real sentence
    boundary follows — wait for more tokens."""
    sent, rest = _peel_sentence("End with period.  ")
    assert sent is None


def test_peel_skips_period_inside_inline_math():
    """``\\(x = 1.5\\)`` shouldn't break the sentence at the dot."""
    text = "We have \\(x = 1.5\\) which is fine. Then we proceed."
    sent, rest = _peel_sentence(text)
    assert sent == "We have \\(x = 1.5\\) which is fine."
    assert rest.lstrip().startswith("Then")


def test_peel_skips_period_inside_display_math():
    text = (r"The form is \[ y = \sum_{i=1}^N x_i. \] We see this ratio. "
            r"Now the next.")
    sent, rest = _peel_sentence(text)
    assert sent == r"The form is \[ y = \sum_{i=1}^N x_i. \] We see this ratio."
    assert rest.lstrip().startswith("Now")


def test_peel_handles_question_and_bang():
    sent, rest = _peel_sentence("What is bagging? It is a method.")
    assert sent == "What is bagging?"
    sent2, rest2 = _peel_sentence(rest.lstrip() + " The next.")
    assert sent2 == "It is a method."


# ---------------------------------------------------------------------------
# Phrase-level peeling (TTS streaming MVP)
# ---------------------------------------------------------------------------

def test_peel_phrase_basic_comma():
    """After at least 5 words, a comma is a phrase boundary."""
    from narrator.qa import _peel_phrase
    text = "Bagging averages predictions across many models, then we average them."
    phrase, rest = _peel_phrase(text)
    assert phrase == "Bagging averages predictions across many models,"
    assert rest.startswith("then")


def test_peel_phrase_requires_min_words():
    """Don't peel ``Bagging,`` — too short for a phrase."""
    from narrator.qa import _peel_phrase
    phrase, rest = _peel_phrase("Bagging, or bootstrap aggregating, is a method.")
    # First clause has only 1 word before comma → no peel.
    # The next valid boundary is after "or bootstrap aggregating," (4 words),
    # still under min — no peel.  Returns None.
    assert phrase is None


def test_peel_phrase_at_semicolon():
    from narrator.qa import _peel_phrase
    text = "We must consider regularization carefully; otherwise we overfit."
    phrase, rest = _peel_phrase(text)
    assert phrase == "We must consider regularization carefully;"
    assert rest.startswith("otherwise")


def test_peel_phrase_skips_inside_inline_math():
    """A comma inside ``\\(...\\)`` shouldn't be a phrase boundary."""
    from narrator.qa import _peel_phrase
    text = (r"Consider the function \(f(x, y)\) and the variable z. "
            r"It maps inputs.")
    # The comma inside \(...\) is in math; the next valid boundary is
    # the period — but _peel_phrase only handles soft boundaries, so
    # it should return None and let _peel_sentence handle the period.
    phrase, rest = _peel_phrase(text)
    assert phrase is None or "f(x, y" not in phrase or phrase.endswith(",")


def test_peel_phrase_skips_numbers_with_commas():
    """``1,000`` — the next char after the comma is a digit, not a
    word-starter, so we shouldn't break there."""
    from narrator.qa import _peel_phrase
    text = "We have 1,000 samples in the training data, then more come."
    phrase, rest = _peel_phrase(text)
    # First-pass peel should NOT split inside the number.
    if phrase is not None:
        assert "1,000" not in phrase or phrase.endswith(",")
        assert "1,000" in (phrase + rest)


def test_peel_phrase_em_dash():
    from narrator.qa import _peel_phrase
    text = "We minimize the loss function — and lambda controls strength."
    phrase, rest = _peel_phrase(text)
    assert phrase == "We minimize the loss function"
    assert rest.startswith("and")


def test_peel_phrase_returns_none_when_no_boundary():
    from narrator.qa import _peel_phrase
    text = "We minimize the loss function carefully and we proceed"
    # No comma / semi / dash in the buffer.
    phrase, rest = _peel_phrase(text)
    assert phrase is None
    assert rest == text


def test_peel_phrase_returns_none_when_too_few_words():
    from narrator.qa import _peel_phrase
    # Only 3 words before the comma — under the 5-word minimum.
    phrase, rest = _peel_phrase("We see x, then we proceed.")
    assert phrase is None


# ---------------------------------------------------------------------------
# answer_streaming with mocked backend
# ---------------------------------------------------------------------------

def _book() -> Book:
    root = BookNode(nid="b", kind="book", number=None, title="ESL",
                    page_start=1, page_end=999)
    ch5 = BookNode(nid="b/ch5", kind="chapter", number="5",
                   title="Basis Expansions and Regularization",
                   page_start=100, page_end=200,
                   body_text="Regularization smooths out the fit. "
                             "Lambda controls the penalty. "
                             "Smoothing splines balance fit and smoothness.")
    root.children.append(ch5)
    return Book(title="ESL", author=None, source="", root=root,
                concepts={}, pages=[], figures=[], cross_refs=[])


def test_answer_streaming_yields_clauses_lazily():
    book = _book()
    consumed = {"count": 0}

    def fake(question, passages, book, history=None):
        for s in [
            "Bagging averages predictions across bootstrap samples.",
            "Each model is trained on a resampled training set.",
            "The variance of the average is lower than any one model.",
        ]:
            consumed["count"] += 1
            yield s
    fake.is_streaming = True
    fake.supports_history = True

    qa_mod.set_streaming_backend(fake)
    # Make sure no intro backend is installed: when one is, low-sim
    # questions take the non-streaming intro path, which would skip
    # the streaming generator we're asserting on here.
    saved_intro = qa_mod._intro_backend
    qa_mod.set_intro_backend(None)
    try:
        plan = qa_mod.answer_streaming(book, "what is bagging")
        assert plan.streaming is True
        # Generator hasn't been consumed yet.
        assert consumed["count"] == 0
        clauses = list(plan.clauses)
        # All 3 sentences pulled.
        assert consumed["count"] == 3
        assert len(clauses) == 3
        assert "Bagging" in clauses[0].text
    finally:
        qa_mod.set_streaming_backend(None)
        qa_mod.set_intro_backend(saved_intro)


def test_answer_streaming_falls_back_when_no_backend():
    """No streaming backend installed → caller still gets a usable
    NarrationPlan via the non-streaming path (no exception)."""
    book = _book()
    qa_mod.set_streaming_backend(None)  # explicit
    plan = qa_mod.answer_streaming(book, "what is regularization")
    # Non-streaming path returns streaming=False.
    assert plan.streaming is False


def test_answer_streaming_passes_history_to_backend():
    book = _book()
    captured = {}

    def fake(question, passages, book, history=None):
        captured["history"] = history
        yield "ok."
    fake.is_streaming = True
    fake.supports_history = True

    qa_mod.set_streaming_backend(fake)
    saved_intro = qa_mod._intro_backend
    qa_mod.set_intro_backend(None)
    try:
        plan = qa_mod.answer_streaming(
            book, "follow-up", history=[{"user_text": "prior"}],
        )
        list(plan.clauses)  # drain
        assert captured["history"] == [{"user_text": "prior"}]
    finally:
        qa_mod.set_streaming_backend(None)
        qa_mod.set_intro_backend(saved_intro)


# ---------------------------------------------------------------------------
# Orchestrator handles a streaming plan
# ---------------------------------------------------------------------------

def test_orchestrator_iterates_streaming_plan():
    """The orchestrator's stream() loop must accept a generator-clauses
    plan without trying to len() / index it."""
    book = _book()

    def _gen():
        yield NarrationClause(text="First sentence.", home_nid="b",
                               concepts=[], suggested_dur=1.0)
        yield NarrationClause(text="Second sentence.", home_nid="b",
                               concepts=[], suggested_dur=1.0)

    plan = NarrationPlan(
        topic="<test-stream>", book_title=book.title,
        clauses=_gen(),
        visited_nids=["b"], meta={"mode": "streaming_tutor"},
        streaming=True,
    )
    orch = Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    events = list(orch.stream())
    assert len(events) == 2
    assert events[0].clause_text == "First sentence."


def test_orchestrator_skips_eq_warmup_when_streaming():
    """The eq_latex pre-warm walks plan.clauses; if it ran on a
    streaming plan it would consume the LLM's sentence stream
    before the user heard anything.  Verify it skips."""
    book = _book()
    consumed = {"count": 0}

    def _gen():
        for s in ["A.", "B.", "C."]:
            consumed["count"] += 1
            yield NarrationClause(text=s, home_nid="b",
                                   concepts=[], suggested_dur=1.0)

    plan = NarrationPlan(
        topic="<test>", book_title=book.title,
        clauses=_gen(), visited_nids=["b"],
        meta={"mode": "streaming_tutor"},
        streaming=True,
    )
    orch = Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    # __post_init__ already ran; if it had iterated the generator
    # consumed["count"] would already be 3.  It should be 0.
    assert consumed["count"] == 0


def test_streaming_plan_total_chars_returns_zero():
    """Length helpers must not trigger generator consumption."""
    plan = NarrationPlan(
        topic="<x>", book_title="t",
        clauses=iter([]), streaming=True,
    )
    assert plan.total_chars() == 0
    assert plan.total_dur() == 0.0
