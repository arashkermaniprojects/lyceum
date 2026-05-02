"""Regressions for the three RBM-query bugs.

Surfaced by asking ESLII "what is a restricted boltzmann machine":

  1. A 18×25 pt embedded Scream icon (page furniture, used in many
     places throughout the PDF) was being shown as the section's
     book figure.  Fix: drop figures with bbox area below
     ``FIGURE_MIN_BBOX_AREA``.
  2. Closed-book retrieval emitted no canonical formula cards because
     ``fetch_intro_formulas`` was gated on the LLM-intro branch.  Fix:
     fetch on every Q&A.
  3. The §17.4.4 narration was 2 sentences long because the
     retrieval-only backend took only ≤2 sentences per passage and
     they were the figure caption.  Fix: take ``_SENTS_PER_PASSAGE``
     non-caption sentences per passage.
"""
from __future__ import annotations

from book.ir import Book, BookNode, FigureRef
from serve.orchestrator import (
    FIGURE_MIN_BBOX_AREA,
    _figure_bbox_area,
    _figures_for_scope,
)
from narrator.qa import (
    RetrievedPassage,
    _is_caption_or_header,
    _retrieval_only_backend,
)


# ---------------------------------------------------------------------------
# Issue 1 — bbox area filter
# ---------------------------------------------------------------------------

def test_bbox_area_known_threshold_is_sane():
    # Tiny icon — area ≈ 18 * 25 = 450 pt²
    icon = FigureRef(
        fid="icon", home_nid="b/ch5/s5_8", page=186, caption="",
        bbox=(0.0, 0.0, 18.0, 25.0), image_path="icon.png",
    )
    # Real figure — width 400, height 250 → 100,000 pt²
    real = FigureRef(
        fid="real", home_nid="b/ch5/s5_8", page=186,
        caption="A real figure", bbox=(0.0, 0.0, 400.0, 250.0),
        image_path="real.png",
    )
    assert _figure_bbox_area(icon) < FIGURE_MIN_BBOX_AREA
    assert _figure_bbox_area(real) >= FIGURE_MIN_BBOX_AREA


def test_figures_for_scope_filters_tiny_icons():
    icon = FigureRef(
        fid="icon", home_nid="b/ch17/s17_4/ss17_4_4", page=660,
        caption="", bbox=(0.0, 0.0, 18.0, 25.0), image_path="i.png",
    )
    real = FigureRef(
        fid="real", home_nid="b/ch17/s17_4/ss17_4_4", page=661,
        caption="RBM",
        bbox=(0.0, 0.0, 400.0, 250.0), image_path="r.png",
    )
    root = BookNode(
        nid="b", kind="book", number=None, title="t",
        page_start=1, page_end=999,
    )
    book = Book(
        title="t", author=None, source="", root=root,
        concepts={}, pages=[], figures=[icon, real],
    )
    out = _figures_for_scope(book, "b/ch17/s17_4/ss17_4_4")
    assert {f.fid for f in out} == {"real"}


# ---------------------------------------------------------------------------
# Issue 2 — caption/header detector
# ---------------------------------------------------------------------------

def test_caption_detector_catches_typical_eslii_caption():
    samples = [
        "FIGURE 17.6. A restricted Boltzmann machine (RBM)",
        "Figure 5.18 shows the kernel.",
        "Table 7.4. Cross-validation results.",
    ]
    for s in samples:
        assert _is_caption_or_header(s), f"missed: {s!r}"


def test_caption_detector_keeps_real_prose():
    samples = [
        "In this section we consider a particular architecture for "
        "graphical models inspired by neural networks.",
        "The visible units are subdivided to allow the RBM to model "
        "the joint density of features and labels.",
        "Eigenvalues are scalars where A v equals lambda v.",
    ]
    for s in samples:
        assert not _is_caption_or_header(s), f"falsely flagged: {s!r}"


# ---------------------------------------------------------------------------
# Issue 3 — richer narration
# ---------------------------------------------------------------------------

def test_retrieval_only_emits_more_than_two_sentences_per_passage():
    """The Bagging-style filter used to emit only 2 sentences per
    passage even when the passage had 10+ sentences of real content."""
    passage_text = (
        "FIGURE 17.6. A restricted Boltzmann machine in which there "
        "are no connections between nodes in the same layer. "
        "In this section we consider a particular architecture for "
        "graphical models inspired by neural networks. "
        "The model is a generalisation of the Boltzmann machine, "
        "which has visible and hidden units. "
        "The joint distribution is defined by an energy function "
        "involving pairwise interactions. "
        "Training proceeds by stochastic gradient ascent on the "
        "log-likelihood, approximated using contrastive divergence. "
        "The hidden units are conditionally independent given the "
        "visible units, which makes inference tractable."
    )
    p = RetrievedPassage(
        nid="b/ch17/s17_4/ss17_4_4",
        text=passage_text,
        score=1.0, bm25_score=10.0, dense_score=0.7,
    )
    root = BookNode(nid="b", kind="book", number=None, title="t",
                    page_start=1, page_end=999)
    book = Book(title="t", author=None, source="", root=root,
                concepts={}, pages=[], figures=[])
    out = _retrieval_only_backend(
        "what is a restricted boltzmann machine", [p], book,
    )
    # Lead sentence + at least 5 content sentences.
    assert len(out) >= 6, (
        f"expected ≥ 6 sentences, got {len(out)}: {out}"
    )
    # The figure caption must NOT be among them.
    captions = [s for s in out if "FIGURE 17.6" in s]
    assert not captions, f"caption leaked into narration: {captions}"


def test_retrieval_only_with_no_passages_still_returns_apology():
    """Behaviour preserved when retrieval returned nothing."""
    out = _retrieval_only_backend("anything", [], book=None)  # type: ignore[arg-type]
    assert any("cannot find" in s.lower() for s in out)
