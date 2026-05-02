"""Visual context resolver — concept + book-context → SVG.

Given a concept mention (e.g., "matrix") and a current location in the
book (e.g., we are reading chapter 8 about tensor products), the resolver:

1.  Looks up the concept in ``book.concepts``.
2.  Picks the **best template** for the current location using path
    proximity scoring (same node > same parent > same chapter > anywhere).
3.  Builds a SeVim ``SceneNode`` from the template's ``primitive`` + ``meta``.
4.  Returns a ``ResolvedShape`` with everything needed to render through
    the existing SeVim S3→S5 pipeline.

If the concept is not in the corpus, the resolver falls back to
``sevim.math_lex.classify_math_label`` to produce a default-shape result —
honest about the gap (``ResolvedShape.from_corpus = False``).

The resolver is **deterministic**: same (concept_id, path) → same template.
Tie-breaking uses lexicographic order on ``(home_nid, primitive)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from book.ir import Book, ConceptEntry, ConceptTemplate
from sevim.math_lex import classify_math_label


# ---------------------------------------------------------------------------
# Path proximity scoring
# ---------------------------------------------------------------------------
# Both nids are slash-separated paths.  The "common prefix length" is the
# number of segments shared from the root.  Higher = closer.

def _path_segments(nid: str) -> list[str]:
    return [s for s in nid.split("/") if s]


def _common_prefix_len(a: str, b: str) -> int:
    sa, sb = _path_segments(a), _path_segments(b)
    n = min(len(sa), len(sb))
    for i in range(n):
        if sa[i] != sb[i]:
            return i
    return n


def _proximity_score(template_nid: str, current_nid: str) -> int:
    """Return a higher score when the template's home is closer to the
    current narration position.

    Score breakdown (higher better):
      common_prefix_len * 100        # the dominant signal
      - depth_diff                   # prefer the closer-depth match
    """
    common = _common_prefix_len(template_nid, current_nid)
    depth_diff = abs(len(_path_segments(template_nid))
                     - len(_path_segments(current_nid)))
    return common * 100 - depth_diff


# ---------------------------------------------------------------------------
# ResolvedShape — the output of the resolver
# ---------------------------------------------------------------------------

@dataclass
class ResolvedShape:
    """The result of resolving (concept_id, current_nid) → visual primitive.

    Attributes
    ----------
    cid:
        The concept id resolved.
    primitive:
        SeVim primitive name (matrix_bracket, set_blob, …).
    label:
        Display label for the shape.
    meta:
        Primitive-specific kwargs (cells for matrices, latex for equations,
        legs for tensor boxes, …).  Compatible with ``SceneNode.meta``.
    home_nid:
        The book node where this template was found (or empty when fallback).
    from_corpus:
        True when the template came from the book; False when fallback.
    score:
        Proximity score against the requested current_nid (higher = closer).
        0 when fallback.
    evidence:
        Provenance from the original ConceptTemplate, when available.
    """
    cid: str
    primitive: str
    label: str
    meta: dict
    home_nid: str = ""
    from_corpus: bool = True
    score: int = 0
    evidence: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resolution strategy
# ---------------------------------------------------------------------------

def _pick_best_template(
    entry: ConceptEntry, current_nid: str,
) -> Optional[ConceptTemplate]:
    """Return the template whose home is most proximate to *current_nid*.

    Tie-break (most-to-least specific):
      1. higher proximity score wins
      2. richer meta (more keys) wins
      3. lexicographic order on (home_nid, primitive) wins

    Returns None if the entry has no templates (which shouldn't happen in
    practice — the concept index doesn't store empty entries — but is
    defensive).
    """
    if not entry.templates:
        return None
    if not current_nid:
        # No location context — pick the first template by stable order.
        return min(entry.templates, key=lambda t: (t.home_nid, t.primitive))

    scored = sorted(
        entry.templates,
        key=lambda t: (
            -_proximity_score(t.home_nid, current_nid),
            -len(t.meta),
            t.home_nid, t.primitive,
        ),
    )
    return scored[0]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve(
    book: Book,
    concept_id: str,
    *,
    current_nid: str = "",
    label_override: Optional[str] = None,
) -> ResolvedShape:
    """Resolve a concept mention to a visual ResolvedShape.

    Parameters
    ----------
    book:
        Loaded corpus from ``book.load_corpus``.
    concept_id:
        Canonical concept id to look up (e.g. ``"matrix"``).
    current_nid:
        BookNode nid the narrator is currently visiting.  Drives the
        proximity-scored template selection.
    label_override:
        Force a specific display label (otherwise uses the entry's
        ``canonical`` field).
    """
    entry = book.concepts.get(concept_id)

    # Fallback path — concept not in corpus.
    if entry is None:
        prim = classify_math_label(concept_id) or "rect"
        label = label_override or concept_id.replace("_", " ")
        return ResolvedShape(
            cid=concept_id,
            primitive=prim,
            label=label,
            meta={"kind": prim},
            home_nid="",
            from_corpus=False,
            score=0,
            evidence={"reason": "not_in_corpus"},
        )

    template = _pick_best_template(entry, current_nid)
    if template is None:
        # Entry exists but has no templates (defensive).
        prim = classify_math_label(entry.canonical) or "rect"
        label = label_override or entry.canonical
        return ResolvedShape(
            cid=concept_id, primitive=prim, label=label,
            meta={"kind": prim}, home_nid="",
            from_corpus=False, score=0,
            evidence={"reason": "entry_has_no_templates"},
        )

    label = label_override or entry.canonical
    score = _proximity_score(template.home_nid, current_nid) if current_nid else 0
    # Ensure meta carries kind for downstream resolution in s3_map.
    meta = dict(template.meta)
    meta.setdefault("kind", template.primitive)
    return ResolvedShape(
        cid=concept_id,
        primitive=template.primitive,
        label=label,
        meta=meta,
        home_nid=template.home_nid,
        from_corpus=True,
        score=score,
        evidence=dict(template.evidence),
    )


# ---------------------------------------------------------------------------
# Bridge to SeVim — render a ResolvedShape through the S3→S5 pipeline.
# ---------------------------------------------------------------------------

def render_resolved(shape: ResolvedShape) -> str:
    """Render a single ResolvedShape to standalone SVG.

    Builds a one-node SceneGraph, runs map_visual → layout → render.
    The resulting SVG is sized to fit the shape with a small padding.
    """
    from sevim.ir import SceneGraph, SceneNode, SpanRef
    from sevim.s3_map import map_visual
    from sevim.s4_layout import layout
    from sevim.s5_render import render

    node = SceneNode(
        id=f"n_{shape.cid}",
        label=shape.label,
        node_type="entity",
        embedding=(),
        salience=0.5,
        src_spans=[SpanRef(0, len(shape.label), "u0")],
        meta=dict(shape.meta),
    )
    g = SceneGraph(nodes=[node], edges=[])
    vg = map_visual(g)
    pg = layout(vg)
    return render(pg)
