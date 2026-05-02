"""Cross-reference graph view — visualise the book's citation overlay.

ESL has 2554 cross-refs.  Drawing every edge produces a hairball, so we
expose three zoom levels:

  1.  ``render_chapter_flow`` — chapter-to-chapter aggregate.  Each chapter
      is a node; edge thickness is proportional to the number of citations
      between them.  Best for a one-screen overview.

  2.  ``render_top_cited`` — the N most-cited targets across the whole
      book, each with the chapters that cite it as adjacent labels.

  3.  ``render_node_neighbourhood`` — for a clicked node, show all of its
      citing nodes (incoming) and cited nodes (outgoing).

All three use SeVim's existing primitives (rect / set_blob / arrow) so
the strict-layout post-pass guarantees non-overlap.

Determinism
-----------
- Edge weights and rankings tie-break on (nid, target nid, label).
- Same book → same SVG bytes.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from book.ir import Book, BookNode, CrossRef
from sevim.ir import (
    SceneEdge, SceneGraph, SceneNode, SpanRef,
)
from sevim.s3_map import map_visual
from sevim.s4_layout import layout
from sevim.s5_render import render
from sevim.strict_layout import resolve_overlaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chapter_of(book: Book, nid: str) -> Optional[str]:
    """Return the chapter (or top-level part/intro) ancestor nid for *nid*."""
    if not nid:
        return None
    parts = nid.split("/")
    # "b/ch7/s7_3/..."  →  the ch7 segment is parts[1]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def _label_for_nid(book: Book, nid: str, max_chars: int = 32) -> str:
    n = book.find(nid)
    if n is None:
        return nid
    if n.number and n.title:
        s = f"{n.kind} {n.number}: {n.title}"
    elif n.number:
        s = f"{n.kind} {n.number}"
    elif n.title:
        s = n.title
    else:
        s = n.kind
    if len(s) > max_chars:
        s = s[:max_chars - 1] + "…"
    return s


# ---------------------------------------------------------------------------
# Layer 1: chapter-to-chapter aggregate
# ---------------------------------------------------------------------------

def _aggregate_chapter_flow(book: Book) -> tuple[Counter, Counter, Counter]:
    """Return (chapter_pair_counts, citing_counts, cited_counts).

    * pair_counts[(src_chapter, tgt_chapter)] = number of cross-refs
      that point from any node in src_chapter to any node in tgt_chapter
    * citing_counts[chapter] = total outgoing citations from this chapter
    * cited_counts[chapter] = total incoming citations to this chapter
    """
    pairs: Counter = Counter()
    out_c: Counter = Counter()
    in_c: Counter = Counter()
    for c in book.cross_refs:
        src = _chapter_of(book, c.from_nid)
        tgt = _chapter_of(book, c.to_nid)
        if not src or not tgt or src == tgt:
            continue
        pairs[(src, tgt)] += 1
        out_c[src] += 1
        in_c[tgt] += 1
    return pairs, out_c, in_c


def render_chapter_flow(
    book: Book, *,
    min_edge_weight: int = 3,
    max_edges: int = 60,
    canvas_w: float = 1800.0,
    canvas_h: float = 1100.0,
) -> str:
    """Render the chapter-to-chapter citation flow as an SVG.

    Each chapter is a labelled rect; edges with weight ≥ ``min_edge_weight``
    are drawn as arrows (relation ``causes``, which renders as a directed
    arrow in SeVim).  Arrow thickness isn't varied (SeVim uses fixed strokes),
    but the edge label includes the count.
    """
    pairs, out_c, in_c = _aggregate_chapter_flow(book)

    g = SceneGraph(nodes=[], edges=[])

    # Pull every chapter that participates in citations.
    used_chapters = set()
    for (a, b) in pairs:
        used_chapters.add(a)
        used_chapters.add(b)

    # Node per chapter, ordered by book position.
    chapters = [n for n in book.root.children
                if n.kind in ("chapter", "appendix", "introduction")
                and n.nid in used_chapters]

    for ch in chapters:
        label = _label_for_nid(book, ch.nid, max_chars=28)
        cited_in = in_c.get(ch.nid, 0)
        cited_out = out_c.get(ch.nid, 0)
        # Mark heavily-cited chapters with a stronger fill.
        salience = 0.5 + min(0.4, (cited_in + cited_out) / 200.0)
        g.nodes.append(SceneNode(
            id=ch.nid, label=label, node_type="entity",
            embedding=(), salience=salience,
            src_spans=[SpanRef(0, len(label), "u0")],
            meta={"in": cited_in, "out": cited_out},
        ))

    # Top-N edges by weight, filtered.
    sorted_edges = sorted(
        ((w, a, b) for (a, b), w in pairs.items() if w >= min_edge_weight),
        key=lambda wab: (-wab[0], wab[1], wab[2]),
    )[:max_edges]

    for w, a, b in sorted_edges:
        eid = f"e_xref_{a}_{b}"
        # Encode count in the edge id as suffix for trace logs.
        g.edges.append(SceneEdge(
            id=f"{eid}_{w}",
            from_id=a, to_id=b,
            relation="causes",
            src_spans=[],
        ))

    # Render through the existing pipeline.
    import os
    os.environ["SEVIM_CANVAS_W"] = str(canvas_w)
    os.environ["SEVIM_CANVAS_H"] = str(canvas_h)
    import sevim.s4_layout as L
    L.CANVAS_W = float(canvas_w); L.CANVAS_H = float(canvas_h)

    vg = map_visual(g)
    pg = layout(vg)
    pg = resolve_overlaps(pg, parent_of={}, min_gap=14.0)
    return render(pg)


# ---------------------------------------------------------------------------
# Layer 2: top-N most-cited targets across the whole book
# ---------------------------------------------------------------------------

def _top_cited_nids(book: Book, n: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for c in book.cross_refs:
        counter[c.to_nid] += 1
    return counter.most_common(n)


def render_top_cited(
    book: Book, *,
    top_n: int = 10,
    show_citers_per: int = 4,
    canvas_w: float = 1800.0,
    canvas_h: float = 1100.0,
) -> str:
    """Render the N most-cited nodes with their top citing chapters around them.

    Each cited target becomes a central rect; its top citing chapters become
    small adjacent rects with arrows pointing to it.
    """
    targets = _top_cited_nids(book, n=top_n)
    if not targets:
        return ""

    g = SceneGraph(nodes=[], edges=[])

    # Per-target citers ranked by chapter.
    for target_nid, total in targets:
        # Build target node.
        target_label = f"{_label_for_nid(book, target_nid, 28)}  (×{total})"
        g.nodes.append(SceneNode(
            id=target_nid, label=target_label, node_type="entity",
            embedding=(), salience=0.7,
            src_spans=[],
            meta={"role": "target", "incoming": total},
        ))
        citers: Counter = Counter()
        for c in book.cross_refs:
            if c.to_nid != target_nid:
                continue
            ch = _chapter_of(book, c.from_nid)
            if ch:
                citers[ch] += 1
        for ch_nid, cnt in citers.most_common(show_citers_per):
            citer_label = f"{_label_for_nid(book, ch_nid, 22)}  (×{cnt})"
            citer_node_id = f"{target_nid}::from::{ch_nid}"
            # Ensure each citer-target pair is a unique node id.
            g.nodes.append(SceneNode(
                id=citer_node_id, label=citer_label, node_type="entity",
                embedding=(), salience=0.4,
                src_spans=[],
                meta={"role": "citer", "outgoing": cnt,
                      "actual_chapter": ch_nid},
            ))
            g.edges.append(SceneEdge(
                id=f"e_xref_{ch_nid}_{target_nid}",
                from_id=citer_node_id, to_id=target_nid,
                relation="causes", src_spans=[],
            ))

    import os
    os.environ["SEVIM_CANVAS_W"] = str(canvas_w)
    os.environ["SEVIM_CANVAS_H"] = str(canvas_h)
    import sevim.s4_layout as L
    L.CANVAS_W = float(canvas_w); L.CANVAS_H = float(canvas_h)

    vg = map_visual(g)
    pg = layout(vg)
    pg = resolve_overlaps(pg, parent_of={}, min_gap=12.0)
    return render(pg)


# ---------------------------------------------------------------------------
# Layer 3: per-node neighbourhood
# ---------------------------------------------------------------------------

def render_node_neighbourhood(
    book: Book, nid: str, *,
    canvas_w: float = 1500.0,
    canvas_h: float = 900.0,
) -> str:
    """Render a star-shaped diagram with *nid* in the centre, citers as
    incoming arrows on the left, citees as outgoing arrows on the right.

    Useful for clicking a theorem or section and seeing its full citation
    surface in one view.
    """
    target = book.find(nid)
    if target is None:
        return ""

    g = SceneGraph(nodes=[], edges=[])

    # Centre node.
    label = _label_for_nid(book, nid, max_chars=40)
    g.nodes.append(SceneNode(
        id=nid, label=label, node_type="entity",
        embedding=(), salience=0.8,
        src_spans=[],
        meta={"role": "focus"},
    ))

    incoming = [c for c in book.cross_refs if c.to_nid == nid]
    outgoing = [c for c in book.cross_refs if c.from_nid == nid]

    # Group incoming by source — a chapter may cite multiple times.
    in_count: Counter = Counter()
    for c in incoming:
        in_count[c.from_nid] += 1
    out_count: Counter = Counter()
    for c in outgoing:
        out_count[c.to_nid] += 1

    for src_nid, cnt in in_count.most_common(8):
        src_label = _label_for_nid(book, src_nid, max_chars=28)
        if cnt > 1:
            src_label = f"{src_label}  (×{cnt})"
        node_id = f"in::{src_nid}"
        g.nodes.append(SceneNode(
            id=node_id, label=src_label, node_type="entity",
            embedding=(), salience=0.5,
            src_spans=[],
            meta={"role": "citer", "actual_nid": src_nid},
        ))
        g.edges.append(SceneEdge(
            id=f"e_xref_in_{src_nid}_{nid}",
            from_id=node_id, to_id=nid,
            relation="causes", src_spans=[],
        ))

    for tgt_nid, cnt in out_count.most_common(8):
        tgt_label = _label_for_nid(book, tgt_nid, max_chars=28)
        if cnt > 1:
            tgt_label = f"{tgt_label}  (×{cnt})"
        node_id = f"out::{tgt_nid}"
        g.nodes.append(SceneNode(
            id=node_id, label=tgt_label, node_type="entity",
            embedding=(), salience=0.5,
            src_spans=[],
            meta={"role": "citee", "actual_nid": tgt_nid},
        ))
        g.edges.append(SceneEdge(
            id=f"e_xref_out_{nid}_{tgt_nid}",
            from_id=nid, to_id=node_id,
            relation="causes", src_spans=[],
        ))

    import os
    os.environ["SEVIM_CANVAS_W"] = str(canvas_w)
    os.environ["SEVIM_CANVAS_H"] = str(canvas_h)
    import sevim.s4_layout as L
    L.CANVAS_W = float(canvas_w); L.CANVAS_H = float(canvas_h)

    vg = map_visual(g)
    pg = layout(vg)
    pg = resolve_overlaps(pg, parent_of={}, min_gap=12.0)
    return render(pg)
