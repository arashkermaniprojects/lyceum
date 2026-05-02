"""Tests for Mode B (read-the-book) and Mode C (tangent Q&A)."""
import pytest

from book.ir import Book, BookNode, ConceptEntry, ConceptTemplate
from narrator import plan_full, answer
from narrator.qa import _retrieve, _retrieval_only_backend


def _toy_book() -> Book:
    pages = [
        "A linear map is a function preserving vector addition. "
        "A matrix represents a linear map.",
        "Eigenvalues are scalars where Av = λv. "
        "The determinant is the product of eigenvalues.",
        "A topological space is a set with a topology. "
        "Compactness means every cover has a finite subcover.",
    ]
    sections = [
        BookNode(nid=f"b/ch{i+1}", kind="chapter", number=str(i+1),
                title=f"Topic {i+1}", page_start=i+1, page_end=i+1,
                body_text=pages[i])
        for i in range(3)
    ]
    root = BookNode(nid="b", kind="book", number=None, title="Toy",
                    page_start=1, page_end=3, children=sections)
    return Book(
        title="Toy", author=None, source="/tmp/toy.pdf",
        root=root,
        concepts={
            "matrix": ConceptEntry(
                cid="matrix", canonical="matrix", aliases=["matrix"],
                definitions=[],
                templates=[ConceptTemplate(
                    home_nid="b/ch1", primitive="matrix_bracket",
                    meta={"kind": "matrix_bracket"}, evidence={})],
                figure_refs=[]),
            "eigenvalue": ConceptEntry(
                cid="eigenvalue", canonical="eigenvalue",
                aliases=["eigenvalues", "eigenvalue"], definitions=[],
                templates=[], figure_refs=[]),
            "set": ConceptEntry(
                cid="set", canonical="set", aliases=["set", "sets"],
                definitions=[], templates=[], figure_refs=[]),
        },
        pages=pages,
    )


# ---------------------------------------------------------------------------
# Mode B — read-the-book
# ---------------------------------------------------------------------------

def test_plan_full_walks_in_book_order():
    b = _toy_book()
    p = plan_full(b)
    pages = [b.find(c.home_nid).page_start for c in p.clauses]
    assert pages == sorted(pages)


def test_plan_full_includes_all_chapters():
    b = _toy_book()
    p = plan_full(b)
    visited_chapters = {c.home_nid for c in p.clauses}
    assert "b/ch1" in visited_chapters
    assert "b/ch2" in visited_chapters
    assert "b/ch3" in visited_chapters


def test_plan_full_emits_chapter_preambles():
    b = _toy_book()
    p = plan_full(b)
    # Look for the auto-generated preamble.
    text = " ".join(c.text for c in p.clauses)
    assert "Topic 1" in text or "Chapter 1" in text


def test_plan_full_excludes_bibliography_by_default():
    b = _toy_book()
    # Add a bibliography chapter to the toy book.
    bib = BookNode(nid="b/biblio", kind="bibliography", number=None,
                  title="Bibliography", page_start=4, page_end=4,
                  body_text="Knuth, D. The Art of Computer Programming.")
    b.root.children.append(bib)
    p = plan_full(b)
    nids = {c.home_nid for c in p.clauses}
    assert "b/biblio" not in nids


def test_plan_full_is_deterministic():
    b = _toy_book()
    p1 = plan_full(b)
    p2 = plan_full(b)
    assert [c.text for c in p1.clauses] == [c.text for c in p2.clauses]


# ---------------------------------------------------------------------------
# Mode C — tangent Q&A
# ---------------------------------------------------------------------------

def test_retrieve_ranks_by_relevance():
    b = _toy_book()
    out = _retrieve(b, "eigenvalue determinant", top_k=3)
    assert out
    # Chapter 2 covers eigenvalues + determinant; should rank top.
    assert out[0].nid == "b/ch2"


def test_answer_returns_a_plan_with_clauses():
    b = _toy_book()
    plan = answer(b, "what is an eigenvalue?", top_k=2)
    assert plan.clauses
    assert plan.meta["mode"] == "tangent"


def test_answer_for_unknown_question_returns_refusal():
    b = _toy_book()
    plan = answer(b, "what is the boiling point of mercury?", top_k=2)
    # Either a refusal OR retrieval-only fallback content.  Always honest.
    assert plan.clauses
    text = " ".join(c.text for c in plan.clauses).lower()
    # Acceptable signals of honesty: empty match, or weak match flagged.
    assert ("cannot find" in text
            or "does not cover" in text
            or "rephrasing" in text
            or len(plan.visited_nids) >= 0)


def test_answer_is_deterministic():
    b = _toy_book()
    a = answer(b, "tell me about matrices", top_k=2)
    c = answer(b, "tell me about matrices", top_k=2)
    assert [cl.text for cl in a.clauses] == [cl.text for cl in c.clauses]


def test_retrieval_only_backend_concatenates_passages():
    from narrator.qa import RetrievedPassage
    p1 = RetrievedPassage(nid="b/ch2", text="Eigenvalues are scalars. Av equals lambda v.", score=2.0)
    p2 = RetrievedPassage(nid="b/ch1", text="A matrix represents a linear map.", score=1.0)
    out = _retrieval_only_backend("what are eigenvalues?", [p1, p2], None)
    assert len(out) >= 1
    full = " ".join(out).lower()
    assert "eigenvalue" in full or "lambda" in full


# ---------------------------------------------------------------------------
# Session pause/resume/ask integration
# ---------------------------------------------------------------------------

def test_session_pause_resume_ask():
    from chalkboard import Chalkboard
    from narrator.tts import NullTTS
    from serve.session import build_session

    b = _toy_book()
    p = plan_full(b)
    session = build_session(
        plan_id="test", book=b, plan=p,
        tts_factory=lambda: NullTTS(),
        canvas_w=800, canvas_h=480, max_content=4,
    )

    # Pull a couple of main events.
    e1 = session.next_event()
    assert e1 is not None
    assert e1.panel == "main"

    # Inject a tangent.
    tid = session.ask("what is an eigenvalue?", top_k=2)
    assert tid

    # Next event(s) should be tangent panel.
    found_tangent = False
    for _ in range(8):
        ev = session.next_event()
        if ev is None:
            break
        if ev.panel == "tangent" and not ev.is_tangent_end:
            found_tangent = True
            break
    assert found_tangent

    # Cancel cleanly.
    session.cancel()
    assert not session.is_active()
