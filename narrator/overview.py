"""Book overview — render the semantic network as a navigable SVG.

Two views:

  * **Table-of-contents (TOC)** — recursive BookNode tree as nested
    set_blob containers.  The whole book is one outer blob; chapters
    are blobs inside it; sections inside chapters; theorems inside
    sections; and so on, to whatever depth the book has.

  * **Cross-reference subgraph** — the same TOC tree, but with overlay
    edges drawn for every CrossRef (citations between theorems / lemmas).

Both views go through the existing SeVim pipeline so the strict-layout
post-pass guarantees non-overlap.  Every shape carries ``data-nid``
matching the BookNode id, so the frontend can click any node to start
narration from there.
"""
from __future__ import annotations

from typing import Optional

from book.ir import (
    Book, BookNode,
    is_environment, is_structural,
)
from sevim.ir import (
    SceneEdge, SceneGraph, SceneNode, SpanRef,
    PlacedGraph,
)
from sevim.s3_map import map_visual
from sevim.s4_layout import layout
from sevim.s5_render import render
from sevim.strict_layout import resolve_overlaps


# ---------------------------------------------------------------------------
# Kind → primitive mapping for the overview
# ---------------------------------------------------------------------------

# Structural nodes (book, part, chapter, section…) become set_blobs because
# they're meant to *contain* descendants.  Environments (theorem, lemma…)
# become proof_node so they read like inference cards inside their parents.
# Default for anything unrecognised: rect.

_KIND_TO_PRIMITIVE: dict[str, str] = {
    # Structural — containers.
    "book": "set_blob", "part": "set_blob",
    "chapter": "set_blob", "section": "set_blob",
    "subsection": "set_blob", "subsubsection": "set_blob",
    "appendix": "set_blob", "introduction": "set_blob",
    "preface": "set_blob",
    # Environments — leaves (mostly).
    "definition": "rect",
    "theorem": "proof_node", "lemma": "proof_node",
    "corollary": "proof_node", "proposition": "proof_node",
    "claim": "proof_node",
    "example": "rect", "exercise": "rect", "remark": "rect",
    "proof": "rect", "fact": "rect",
    "construction": "rect", "convention": "rect",
    # Auxiliary.
    "bibliography": "rect", "index": "rect", "glossary": "rect",
    "abstract": "rect",
}


def _primitive_for(kind: str) -> str:
    return _KIND_TO_PRIMITIVE.get(kind, "rect")


def _label_for(node: BookNode, *, max_chars: Optional[int] = None) -> str:
    """Short display label sized to the node's depth.

    Deep nodes (subsubsection and below) get *only* their number
    (``§3.2.1``) because the rendered ellipse at that depth is too
    small for any title text to fit without overlapping siblings.
    """
    short_kind = {
        "chapter": "Ch.", "section": "§", "subsection": "§",
        "subsubsection": "§", "part": "Part", "appendix": "App.",
    }.get(node.kind, "")

    # Number-only mode for any 3-or-more-deep numbered node (e.g. 3.2.1)
    # OR for environment leaves (theorem, definition, …).  These render
    # as tiny ellipses where titles never fit.
    deep_number = node.number and node.number.count(".") >= 2
    leaf_kinds = {"subsubsection", "theorem", "lemma", "proposition",
                  "corollary", "definition", "example", "exercise",
                  "proof", "remark"}
    if deep_number or node.kind in leaf_kinds:
        if node.number:
            return f"§{node.number}" if short_kind == "§" else \
                   f"{short_kind} {node.number}"
        if node.title:
            return node.title[:8] + ("…" if len(node.title) > 8 else "")
        return node.kind[:6]

    # Default budget shrinks with depth depth (if known via nid slashes).
    if max_chars is None:
        depth = node.nid.count("/")
        max_chars = {0: 18, 1: 14, 2: 14, 3: 10}.get(depth, 8)

    if node.number and node.title:
        if short_kind in ("Ch.", "Part", "App."):
            s = f"{short_kind}{node.number}  {node.title}"
        elif short_kind == "§":
            s = f"§{node.number} {node.title}"
        else:
            s = f"{node.kind} {node.number}: {node.title}"
    elif node.title:
        s = node.title
    elif node.number:
        s = f"{short_kind}{node.number}".strip()
    else:
        s = node.kind
    if len(s) > max_chars:
        s = s[:max_chars - 1] + "…"
    return s


# ---------------------------------------------------------------------------
# Tree → SceneGraph
# ---------------------------------------------------------------------------

def book_to_scene(
    book: Book,
    *,
    max_depth: Optional[int] = None,
    include_environments: bool = True,
    skip_kinds: Optional[set[str]] = None,
) -> SceneGraph:
    """Convert the recursive BookNode tree to a flat SceneGraph + container hierarchy.

    Parameters
    ----------
    max_depth:
        Cap depth of the rendered tree (None = render every level).
        Useful for large books — depth 3 gives a chapter-section view;
        depth 5 includes theorems.
    include_environments:
        Whether to render theorem/definition/proof leaves.  False gives
        a structure-only outline.
    skip_kinds:
        Specific kinds to omit (default: bibliography, index).
    """
    if skip_kinds is None:
        skip_kinds = {"bibliography", "index", "glossary"}

    graph = SceneGraph(nodes=[], edges=[])

    def visit(node: BookNode, depth: int) -> bool:
        """Add *node* and its eligible descendants.  Returns True iff the
        node was added (False = skipped due to filters)."""
        if node.kind in skip_kinds:
            return False
        if max_depth is not None and depth > max_depth:
            return False
        if not include_environments and is_environment(node.kind):
            return False

        prim = _primitive_for(node.kind)
        graph.nodes.append(SceneNode(
            id=node.nid, label=_label_for(node),
            node_type="entity", embedding=(),
            salience=0.5,
            src_spans=[SpanRef(0, len(node.title or ""), "u0")],
            meta={
                "kind": prim,
                "book_kind": node.kind,
                "book_number": node.number or "",
                "book_pages": (node.page_start, node.page_end),
            },
        ))

        # Visit children; for those that get added, emit a `contains`
        # edge so SeVim's container hierarchy renders them nested.
        for child in node.children:
            if visit(child, depth + 1):
                graph.edges.append(SceneEdge(
                    id=f"e_contains_{node.nid}_{child.nid}",
                    from_id=node.nid, to_id=child.nid,
                    relation="contains",
                    src_spans=[],
                ))
        return True

    visit(book.root, depth=0)
    return graph


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------

def render_toc(
    book: Book,
    *,
    max_depth: Optional[int] = 3,
    include_environments: bool = False,
    canvas_w: float = 2200.0,
    canvas_h: float = 1400.0,
    strict: bool = True,
) -> str:
    """Render the table-of-contents view as a complete SVG document.

    Default (``max_depth=3``, ``include_environments=False``) gives a
    book → chapter → section view — the natural overview.  Increase
    ``max_depth`` to drill into subsections; set ``include_environments``
    to also show every theorem/definition/etc.
    """
    import os
    os.environ["SEVIM_CANVAS_W"] = str(canvas_w)
    os.environ["SEVIM_CANVAS_H"] = str(canvas_h)
    # Force-reload layout module's CANVAS constants.
    import sevim.s4_layout as L
    L.CANVAS_W = float(canvas_w)
    L.CANVAS_H = float(canvas_h)

    g = book_to_scene(
        book, max_depth=max_depth,
        include_environments=include_environments,
    )
    vg = map_visual(g)
    pg = layout(vg)
    if strict:
        # Build parent_of from the visual graph's containers.
        parent_of: dict[str, str] = {}
        for parent, children in vg.containers:
            for child in children:
                parent_of.setdefault(child, parent)
        pg = resolve_overlaps(pg, parent_of, min_gap=24.0)
    return render(pg)


def render_chapter(
    book: Book, chapter_nid: str,
    *,
    max_depth: int = 4,
    canvas_w: float = 1400.0,
    canvas_h: float = 900.0,
) -> str:
    """Render a single chapter sub-tree at higher depth."""
    chapter = book.find(chapter_nid)
    if chapter is None:
        return ""
    # Build a synthetic Book whose root is the chapter, then reuse render_toc.
    sub = type(book)(
        title=book.title, author=book.author, source=book.source,
        root=chapter, figures=book.figures,
        cross_refs=book.cross_refs,
        concepts=book.concepts, pages=book.pages,
        meta=dict(book.meta),
    )
    return render_toc(
        sub, max_depth=max_depth, include_environments=True,
        canvas_w=canvas_w, canvas_h=canvas_h,
    )


# ---------------------------------------------------------------------------
# Stats helper — useful for the frontend to show "scope" before rendering.
# ---------------------------------------------------------------------------

def overview_stats(
    book: Book, *,
    max_depth: Optional[int] = 3,
    include_environments: bool = False,
) -> dict:
    """Return counts of nodes at each level so the frontend can warn
    when an overview will be very large (e.g., depth=8, 5000+ nodes)."""
    g = book_to_scene(
        book, max_depth=max_depth,
        include_environments=include_environments,
    )
    by_kind: dict[str, int] = {}
    by_depth: dict[int, int] = {}
    for n in g.nodes:
        kind = n.meta.get("book_kind", "?")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        depth = n.id.count("/")
        by_depth[depth] = by_depth.get(depth, 0) + 1
    return {
        "total_nodes": len(g.nodes),
        "total_edges": len(g.edges),
        "by_kind": by_kind,
        "by_depth": by_depth,
        "max_depth": max(by_depth.keys()) if by_depth else 0,
    }
