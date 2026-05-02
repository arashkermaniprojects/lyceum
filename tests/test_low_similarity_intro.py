"""Tests for low-similarity routing of qa.answer().

When the question doesn't match anything in the book, the orchestrator
should route to the LLM intro backend instead of splicing weak
passage extracts.  These tests cover:

  * the threshold helper (`is_low_similarity`)
  * dispatch logic in `answer()` when an intro backend is installed
  * graceful fallback when no intro backend is registered
  * meta fields exposed for observability
"""
from __future__ import annotations

from book.ir import Book, BookNode, ConceptEntry, ConceptTemplate
from narrator import answer
from narrator.qa import (
    DENSE_THRESHOLD,
    RetrievedPassage,
    is_low_similarity,
    set_intro_backend,
)


# ---------------------------------------------------------------------------
# Toy book — three short chapters; deliberately small so retrieval is
# cheap and the BM25 / cosine scoring is easy to reason about.
# ---------------------------------------------------------------------------

def _toy_book() -> Book:
    """Three short chapters; titles carry the chapter topic so BM25 can
    discriminate query→chapter well enough for the threshold tests."""
    chapters = [
        ("Linear maps",
         "A linear map is a function preserving vector addition. "
         "A matrix represents a linear map between vector spaces."),
        ("Eigenvalues",
         "Eigenvalues are scalars where A v equals lambda v. "
         "The determinant is the product of eigenvalues. "
         "Eigenvalues describe how a linear map stretches its eigenvectors."),
        ("Topology",
         "A topological space is a set with a topology. "
         "Compactness means every cover has a finite subcover."),
    ]
    sections = [
        BookNode(nid=f"b/ch{i + 1}", kind="chapter", number=str(i + 1),
                 title=title, page_start=i + 1, page_end=i + 1,
                 body_text=body)
        for i, (title, body) in enumerate(chapters)
    ]
    root = BookNode(nid="b", kind="book", number=None, title="Toy",
                    page_start=1, page_end=3, children=sections)
    return Book(
        title="Toy", author=None, source="/tmp/toy.pdf",
        root=root, concepts={}, pages=[c[1] for c in chapters],
    )


# ---------------------------------------------------------------------------
# is_low_similarity
# ---------------------------------------------------------------------------

def test_low_similarity_when_no_passages():
    assert is_low_similarity([]) is True


def test_high_dense_similarity_is_not_low():
    p = RetrievedPassage(nid="b/ch1", text="x", score=1.0,
                         bm25_score=0.5, dense_score=0.9)
    assert is_low_similarity([p]) is False


def test_low_dense_similarity_is_low():
    p = RetrievedPassage(nid="b/ch1", text="x", score=1.0,
                         bm25_score=0.5, dense_score=0.05)
    assert is_low_similarity([p]) is True


def test_dense_threshold_used_when_any_passage_has_dense():
    """If any passage has a non-zero dense_score, the dense threshold
    governs — we don't fall back to BM25 just because some passages
    happen to lack embeddings."""
    p_dense = RetrievedPassage(nid="b/ch1", text="x", score=1.0,
                               bm25_score=10.0, dense_score=0.10)
    p_sparse = RetrievedPassage(nid="b/ch2", text="y", score=0.5,
                                bm25_score=10.0, dense_score=0.0)
    assert is_low_similarity([p_dense, p_sparse]) is True


def test_bm25_floor_used_when_no_dense_signal():
    """When every passage has dense_score == 0, fall back to BM25."""
    weak = RetrievedPassage(nid="b/ch1", text="x", score=0.1,
                            bm25_score=0.1, dense_score=0.0)
    assert is_low_similarity([weak]) is True
    strong = RetrievedPassage(nid="b/ch1", text="x", score=5.0,
                              bm25_score=5.0, dense_score=0.0)
    assert is_low_similarity([strong]) is False


def test_threshold_overrides_respected():
    p = RetrievedPassage(nid="b/ch1", text="x", score=1.0,
                         bm25_score=0.5, dense_score=0.50)
    # default threshold (0.40) → high enough.
    assert is_low_similarity([p]) is False
    # caller bumps threshold above the score → now low.
    assert is_low_similarity([p], dense_threshold=0.80) is True


# ---------------------------------------------------------------------------
# answer() routing
# ---------------------------------------------------------------------------

def _intro_stub_factory():
    """Return (backend, calls) — a stub intro backend that records each
    call's arguments so tests can assert dispatch happened."""
    calls: list[tuple[str, int]] = []

    def _stub(question: str, passages, book: Book):
        calls.append((question, len(passages)))
        return [
            f"Introduction: the topic '{question}' is being explained.",
            "This intro came from the LLM, not from the book.",
        ]
    _stub.__name__ = "intro_stub"
    return _stub, calls


def test_low_similarity_routes_to_intro_backend():
    """Off-topic question → intro backend is invoked, plan reflects it."""
    book = _toy_book()
    stub, calls = _intro_stub_factory()
    plan = answer(
        book, "What is a Hopf algebra?",
        intro_backend=stub,
    )
    assert calls, "intro backend should have been called"
    assert plan.meta["intro_source"] == "llm"
    assert plan.meta["low_similarity"] is True
    # First clause came from the stub, not the retrieval template.
    assert any("LLM" in c.text or "Introduction" in c.text
               for c in plan.clauses)


def test_high_similarity_does_not_invoke_intro():
    """On-topic question → intro backend is not called.

    Uses tokens that appear verbatim in the eigenvalue chapter (the
    BM25 tokenizer is non-stemming) so the title-boost path lights up
    and the BM25 score crosses the default low-similarity floor."""
    book = _toy_book()
    stub, calls = _intro_stub_factory()
    plan = answer(
        book, "explain eigenvalues and the determinant",
        intro_backend=stub,
    )
    assert plan.meta["intro_source"] == "book"
    assert plan.meta["low_similarity"] is False
    assert calls == []


def test_low_similarity_falls_through_when_intro_backend_returns_none():
    """If the intro backend declines (e.g. vLLM down), behaviour
    must match the pre-feature retrieval-only template — no crash."""
    book = _toy_book()

    def _declining_intro(*_args, **_kwargs):
        return None
    _declining_intro.__name__ = "declining_intro"

    plan = answer(
        book, "What is a Hopf algebra?",
        intro_backend=_declining_intro,
    )
    assert plan.meta["low_similarity"] is True
    # Falls back to retrieval-only / closed-book; intro not used.
    assert plan.meta["intro_source"] == "book"
    assert plan.clauses, "should still produce some narration"


def test_module_level_intro_backend_install_and_clear():
    """`set_intro_backend` swaps the active intro backend."""
    book = _toy_book()
    stub, calls = _intro_stub_factory()
    try:
        set_intro_backend(stub)
        plan = answer(book, "What is a Hopf algebra?")
        assert calls
        assert plan.meta["intro_source"] == "llm"
    finally:
        set_intro_backend(None)
    # After clearing, the same off-topic question routes through the
    # default closed-book path (no LLM).
    plan2 = answer(book, "What is a Hopf algebra?")
    assert plan2.meta["intro_source"] == "book"


def test_dense_threshold_default_is_sane():
    """Sanity check: the default cosine threshold is in (0, 1)."""
    assert 0.0 < DENSE_THRESHOLD < 1.0


def test_plan_meta_carries_score_telemetry():
    """`top_dense_score` / `top_bm25_score` are exposed for diagnostics."""
    book = _toy_book()
    plan = answer(book, "what is an eigenvalue")
    assert "top_dense_score" in plan.meta
    assert "top_bm25_score" in plan.meta
    assert plan.meta["top_bm25_score"] >= 0.0
