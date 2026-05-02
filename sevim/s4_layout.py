"""S4 Layout — rule-tree layout engine.

Assigns absolute (x, y) coordinates to every shape in the VisualGraph and
clips connector lines to shape boundaries.

Layout algorithm
----------------
1.  Container hierarchy (from S3) is processed bottom-up: leaf sizes are
    known first, then parents expand to fit their children.

2.  For each container (or the implicit top-level group), the dominant
    relation among the children's edges determines which sub-algorithm runs:

        sequence   → strip  (horizontal row, sorted by nid)
        DAG-type   → sugiyama (layered left-to-right, barycenter-sorted)
        no edges   → grid (square-ish grid)
        1 child    → stack (single column)

3.  The Sugiyama sub-algorithm runs:
      • Longest-path layering (topological level assignment)
      • 3-sweep barycenter crossing reduction (forward + backward + forward)
      • Column-width–aware x placement; vertically centred per column

4.  After all coordinates are computed, the whole layout is uniformly
    scaled down if it overflows CANVAS_W × CANVAS_H.

5.  Connectors that duplicate a container/child structural edge are
    suppressed.  Remaining connectors are clipped to shape bounding boxes.

Canvas size and spacing constants can be overridden with environment
variables SEVIM_CANVAS_W / SEVIM_CANVAS_H.
"""
from __future__ import annotations

import math

from .ir import (
    PlacedConn, PlacedGraph, PlacedShape, VisualConn, VisualGraph, VisualShape,
)

import os

CANVAS_W = float(os.environ.get("SEVIM_CANVAS_W", 700))
CANVAS_H = float(os.environ.get("SEVIM_CANVAS_H", 440))
PAD = 14.0
GAP = 30.0
LEVEL_GAP = 90.0
ROW_GAP = 40.0
CONTAINER_HEADER = 28.0   # must fit one line of font_size=16 with padding
CONTAINER_PAD = 10.0

_LAYOUT_RELATIONS: dict[str, str] = {
    # original 12
    "sequence": "strip",
    "causes": "sugiyama",
    "similar_to": "sugiyama",
    "instance_of": "sugiyama",
    "requires": "sugiyama",
    "reduces_to": "sugiyama",
    "used_for": "sugiyama",
    "attribute_of": "sugiyama",
    "measures": "sugiyama",
    "part_of": "sugiyama",
    "contains": "sugiyama",
    # math: relations carrying a left→right direction → layered DAG
    "maps_to": "sugiyama",
    "isomorphic_to": "sugiyama",
    "element_of": "sugiyama",
    "subset_of": "sugiyama",
    "points_to": "sugiyama",
    "connects": "sugiyama",
    # math: relations that are symmetric / annotative → grid is fine
    "equals": "grid",
    "congruent": "grid",
    "approximately_equal": "grid",
    "perpendicular": "grid",
    "parallel": "grid",
    "tangent_to": "grid",
    "lies_on": "grid",
    "between": "grid",
    "disjoint": "strip",
    "labels": "grid",
    "grouped_with": "grid",
    "aligned_with": "strip",
    # — university additions —
    # Logical entailment displays as a top-down proof tree.
    "implies": "proof_tree",
    # Categorical relations connect objects in a diagram → DAG layout.
    "adjoint_to": "sugiyama",
    "natural_transformation": "sugiyama",
    "commutes": "grid",
}


def _dominant_rule(children: list[str], connectors: list[VisualConn]) -> str:
    """Pick the layout sub-algorithm for a group of sibling nodes.

    Counts how many connectors between children carry each relation type, then
    picks the most-frequent relation that has a named layout rule.  Ties broken
    alphabetically for determinism.  Falls back to "grid" when there are no
    internal edges.
    """
    if len(children) <= 1:
        return "stack"
    child_set = set(children)
    counts: dict[str, int] = {}
    for c in connectors:
        if c.from_nid in child_set and c.to_nid in child_set:
            counts[c.relation] = counts.get(c.relation, 0) + 1
    if not counts:
        return "grid"

    def key(kv: tuple[str, int]) -> tuple[int, int, str]:
        rel, count = kv
        return (-int(rel in _LAYOUT_RELATIONS), -count, rel)

    best = sorted(counts.items(), key=key)[0][0]
    return _LAYOUT_RELATIONS.get(best, "grid")


Placement = tuple[str, float, float]  # (nid, rel_x, rel_y)
Block = tuple[float, float, list[Placement]]  # (w, h, placements)


def _strip(children: list[str], sizes: dict[str, tuple[float, float]]) -> Block:
    """Horizontal row layout for sequence relations."""
    ordered = sorted(children)
    x = 0.0
    max_h = 0.0
    out: list[Placement] = []
    for nid in ordered:
        w, h = sizes[nid]
        out.append((nid, x, 0.0))
        x += w + GAP
        if h > max_h:
            max_h = h
    total_w = max(0.0, x - GAP)
    return total_w, max_h, out


def _stack(children: list[str], sizes: dict[str, tuple[float, float]]) -> Block:
    """Single-column vertical stack for groups of one or two nodes."""
    ordered = sorted(children)
    y = 0.0
    max_w = 0.0
    out: list[Placement] = []
    for nid in ordered:
        w, h = sizes[nid]
        out.append((nid, 0.0, y))
        y += h + GAP
        if w > max_w:
            max_w = w
    total_h = max(0.0, y - GAP)
    return max_w, total_h, out


def _grid(children: list[str], sizes: dict[str, tuple[float, float]]) -> Block:
    """Square-ish grid layout for groups with no dominant relation."""
    ordered = sorted(children)
    n = len(ordered)
    if n == 0:
        return 0.0, 0.0, []
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, math.ceil(n / cols))
    cell_w = max(sizes[nid][0] for nid in ordered)
    cell_h = max(sizes[nid][1] for nid in ordered)
    out: list[Placement] = []
    for i, nid in enumerate(ordered):
        r, c = divmod(i, cols)
        out.append((nid, c * (cell_w + GAP), r * (cell_h + GAP)))
    total_w = cols * cell_w + max(0, cols - 1) * GAP
    total_h = rows * cell_h + max(0, rows - 1) * GAP
    return total_w, total_h, out


def _axes(
    children: list[str],
    sizes: dict[str, tuple[float, float]],
    coords: dict[str, tuple[float, float]],
    plot_w: float = 200.0,
    plot_h: float = 140.0,
) -> Block:
    """Coordinate-anchored layout for `axes` containers.

    Children whose `meta` carries math coords ``{"x": …, "y": …}`` get
    placed at the corresponding pixel offset inside a (plot_w × plot_h)
    frame.  Range is auto-fit from the coords.  Children without coords
    are appended below the frame in a horizontal strip so they remain
    visible (typically these are labels or auxiliary annotations).
    """
    if coords:
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xspan = max(xmax - xmin, 1e-9)
        yspan = max(ymax - ymin, 1e-9)
        # Pad 8% for breathing room; ensures point markers don't sit on the axis.
        xpad, ypad = xspan * 0.08, yspan * 0.08
        xmin -= xpad; xmax += xpad
        ymin -= ypad; ymax += ypad
        xspan = xmax - xmin
        yspan = ymax - ymin
    else:
        xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
        xspan = yspan = 2.0

    out: list[Placement] = []
    placed_set: set[str] = set()
    for nid in sorted(children):
        if nid not in coords:
            continue
        cx, cy = coords[nid]
        rx = (cx - xmin) / xspan * plot_w
        ry = plot_h - (cy - ymin) / yspan * plot_h  # SVG y is downward
        w, h = sizes[nid]
        out.append((nid, rx - w / 2, ry - h / 2))
        placed_set.add(nid)

    leftover = [c for c in children if c not in placed_set]
    cursor_x = 0.0
    bottom_y = plot_h + GAP
    max_h = 0.0
    for nid in sorted(leftover):
        w, h = sizes[nid]
        out.append((nid, cursor_x, bottom_y))
        cursor_x += w + GAP
        if h > max_h:
            max_h = h

    total_w = max(plot_w, cursor_x - GAP if leftover else 0.0)
    total_h = plot_h + ((max_h + GAP) if leftover else 0.0)
    return total_w, total_h, out


def _sugiyama(
    children: list[str],
    sizes: dict[str, tuple[float, float]],
    connectors: list[VisualConn],
) -> Block:
    """Deterministic layered layout, left→right by longest-path level.

    Tie-breaker: nid asc at every step. No crossing reduction in v0.1 —
    added when measurably needed (cf. ARCHITECTURE.md §6.1).
    """
    child_set = set(children)
    edges = [
        (c.from_nid, c.to_nid)
        for c in connectors
        if c.from_nid in child_set and c.to_nid in child_set
    ]
    out_adj: dict[str, list[str]] = {nid: [] for nid in children}
    in_adj: dict[str, list[str]] = {nid: [] for nid in children}
    for u, v in edges:
        out_adj[u].append(v)
        in_adj[v].append(u)

    remaining_in = {n: len(in_adj[n]) for n in children}
    frontier = sorted(n for n in children if remaining_in[n] == 0)
    topo: list[str] = []
    level = {nid: 0 for nid in children}
    while frontier:
        node = frontier.pop(0)
        topo.append(node)
        for nxt in sorted(out_adj[node]):
            remaining_in[nxt] -= 1
            if remaining_in[nxt] == 0:
                frontier.append(nxt)
                frontier.sort()
    for n in children:
        if n not in topo:
            topo.append(n)
    for n in topo:
        if in_adj[n]:
            level[n] = max(level[p] + 1 for p in in_adj[n])

    by_level: dict[int, list[str]] = {}
    for n, L in level.items():
        by_level.setdefault(L, []).append(n)
    max_level = max(by_level.keys(), default=0)

    # Barycenter crossing reduction (3 alternating sweeps).
    for _sweep in range(3):
        if _sweep % 2 == 0:
            # Forward: sort each level by barycenter of predecessors.
            for L in range(1, max_level + 1):
                pos = {nid: i for i, nid in enumerate(by_level[L - 1])}
                n_prev = len(pos)
                def _bc_fwd(n: str, _pos: dict = pos, _np: int = n_prev) -> float:
                    preds = [p for p in in_adj.get(n, []) if p in _pos]
                    return sum(_pos[p] for p in preds) / len(preds) if preds else _np / 2.0
                by_level[L].sort(key=lambda n: (_bc_fwd(n), n))
        else:
            # Backward: sort each level by barycenter of successors.
            for L in range(max_level - 1, -1, -1):
                pos = {nid: i for i, nid in enumerate(by_level[L + 1])}
                n_nxt = len(pos)
                def _bc_bwd(n: str, _pos: dict = pos, _nn: int = n_nxt) -> float:
                    succs = [s for s in out_adj.get(n, []) if s in _pos]
                    return sum(_pos[s] for s in succs) / len(succs) if succs else _nn / 2.0
                by_level[L].sort(key=lambda n: (_bc_bwd(n), n))

    row_h_max = max((sizes[nid][1] for nid in children), default=0.0)
    col_widths = {
        L: max((sizes[n][0] for n in by_level.get(L, [])), default=0.0)
        for L in range(max_level + 1)
    }
    level_x: dict[int, float] = {}
    cursor = 0.0
    for L in range(max_level + 1):
        level_x[L] = cursor
        cursor += col_widths[L] + LEVEL_GAP
    total_w = max(0.0, cursor - LEVEL_GAP)
    max_col_n = max((len(by_level.get(L, [])) for L in range(max_level + 1)), default=1)
    total_h = max_col_n * row_h_max + max(0, max_col_n - 1) * ROW_GAP

    out: list[Placement] = []
    for L in range(max_level + 1):
        col_nodes = sorted(by_level.get(L, []))
        n_in_col = len(col_nodes)
        col_h = n_in_col * row_h_max + max(0, n_in_col - 1) * ROW_GAP
        y_start = (total_h - col_h) / 2.0
        col_w = col_widths[L]
        for i, nid in enumerate(col_nodes):
            w, h = sizes[nid]
            x = level_x[L] + (col_w - w) / 2.0
            y = y_start + i * (row_h_max + ROW_GAP) + (row_h_max - h) / 2.0
            out.append((nid, x, y))
    out.sort(key=lambda p: p[0])
    return total_w, total_h, out


def _proof_tree(
    children: list[str],
    sizes: dict[str, tuple[float, float]],
    connectors: list[VisualConn],
) -> Block:
    """Top-to-bottom topological layout for proof trees.

    Edges with relation `implies` (premise → conclusion) define the partial
    order.  Roots (no incoming edges) sit at the top; conclusions (no
    outgoing edges) sit at the bottom.  Within a level, nodes are arranged
    horizontally and barycenter-sorted to reduce crossings.

    This is essentially the Sugiyama layout with axes swapped: levels become
    rows instead of columns.
    """
    child_set = set(children)
    edges = [
        (c.from_nid, c.to_nid)
        for c in connectors
        if c.from_nid in child_set and c.to_nid in child_set
        and c.relation == "implies"
    ]
    # Fall back to all DAG-style edges if no explicit `implies` edges exist.
    if not edges:
        edges = [
            (c.from_nid, c.to_nid)
            for c in connectors
            if c.from_nid in child_set and c.to_nid in child_set
        ]

    out_adj: dict[str, list[str]] = {nid: [] for nid in children}
    in_adj: dict[str, list[str]] = {nid: [] for nid in children}
    for u, v in edges:
        out_adj[u].append(v)
        in_adj[v].append(u)

    remaining_in = {n: len(in_adj[n]) for n in children}
    frontier = sorted(n for n in children if remaining_in[n] == 0)
    topo: list[str] = []
    level = {nid: 0 for nid in children}
    while frontier:
        node = frontier.pop(0)
        topo.append(node)
        for nxt in sorted(out_adj[node]):
            remaining_in[nxt] -= 1
            if remaining_in[nxt] == 0:
                frontier.append(nxt)
                frontier.sort()
    for n in children:
        if n not in topo:
            topo.append(n)
    for n in topo:
        if in_adj[n]:
            level[n] = max(level[p] + 1 for p in in_adj[n])

    by_level: dict[int, list[str]] = {}
    for n, L in level.items():
        by_level.setdefault(L, []).append(n)
    max_level = max(by_level.keys(), default=0)

    # Barycenter sort within each level to reduce edge crossings.
    for _sweep in range(2):
        for L in range(1, max_level + 1):
            pos = {nid: i for i, nid in enumerate(by_level[L - 1])}
            n_prev = len(pos)
            def _bc(n: str, _pos: dict = pos, _np: int = n_prev) -> float:
                preds = [p for p in in_adj.get(n, []) if p in _pos]
                return sum(_pos[p] for p in preds) / len(preds) if preds else _np / 2.0
            by_level[L].sort(key=lambda n: (_bc(n), n))

    # Compute row geometry.
    col_w_max = max((sizes[nid][0] for nid in children), default=0.0)
    row_y: dict[int, float] = {}
    cursor_y = 0.0
    row_heights: dict[int, float] = {}
    for L in range(max_level + 1):
        row_heights[L] = max((sizes[nid][1] for nid in by_level.get(L, [])), default=0.0)
        row_y[L] = cursor_y
        cursor_y += row_heights[L] + ROW_GAP
    total_h = max(0.0, cursor_y - ROW_GAP)
    max_row_n = max((len(by_level.get(L, [])) for L in range(max_level + 1)), default=1)
    total_w = max_row_n * col_w_max + max(0, max_row_n - 1) * GAP

    out: list[Placement] = []
    for L in range(max_level + 1):
        row_nodes = by_level.get(L, [])
        n_in_row = len(row_nodes)
        row_total_w = n_in_row * col_w_max + max(0, n_in_row - 1) * GAP
        x_start = (total_w - row_total_w) / 2.0
        for i, nid in enumerate(row_nodes):
            w, h = sizes[nid]
            x = x_start + i * (col_w_max + GAP) + (col_w_max - w) / 2.0
            y = row_y[L] + (row_heights[L] - h) / 2.0
            out.append((nid, x, y))
    out.sort(key=lambda p: p[0])
    return total_w, total_h, out


def _apply(
    rule: str,
    children: list[str],
    sizes: dict[str, tuple[float, float]],
    connectors: list[VisualConn],
) -> Block:
    """Dispatch to the sub-algorithm named by *rule*."""
    if rule == "strip":
        return _strip(children, sizes)
    if rule == "stack":
        return _stack(children, sizes)
    if rule == "sugiyama":
        return _sugiyama(children, sizes, connectors)
    if rule == "proof_tree":
        return _proof_tree(children, sizes, connectors)
    return _grid(children, sizes)


def _build_virtual_connectors(
    top_level: list[str],
    container_children: dict[str, list[str]],
    connectors: list[VisualConn],
) -> list[VisualConn]:
    """Lift every connector to a virtual edge between top-level ancestors.

    A connector from `point_p` (deep inside `universe`) to `line_l` (top-level)
    becomes a virtual edge `universe → line_l`.  Internal edges (both endpoints
    inside the same container) collapse to self-loops and are dropped.

    Why: the top-level layout otherwise can't see relationships that cross
    container boundaries, so it places connected groups arbitrarily far apart.

    Returns a fresh list of VisualConn objects with synthesised eids.  The
    original connectors are unchanged; they still drive in-container layout
    via `_apply` calls inside `size_item`.
    """
    # Build ancestor map: every node → its top-level container (or itself).
    ancestor: dict[str, str] = {}

    def collect(nid: str, root: str) -> None:
        ancestor[nid] = root
        for ch in container_children.get(nid, ()):
            collect(ch, root)

    for t in top_level:
        collect(t, t)

    seen: dict[tuple[str, str, str], int] = {}
    virt: list[VisualConn] = []
    for c in connectors:
        a = ancestor.get(c.from_nid)
        b = ancestor.get(c.to_nid)
        if a is None or b is None or a == b:
            continue
        # Only count edges that genuinely crossed a container boundary —
        # skip pairs where both endpoints are already top-level (those edges
        # are already in vg.connectors and the dominant_rule sees them).
        if c.from_nid == a and c.to_nid == b:
            continue
        key = (a, b, c.relation)
        if key in seen:
            continue  # de-dupe parallel virtual edges between the same pair
        seen[key] = 1
        virt.append(VisualConn(
            eid=f"v_{c.relation}_{a}_{b}",
            from_nid=a, to_nid=b,
            pattern=c.pattern, relation=c.relation,
        ))
    return virt


def _clip_to_rect(
    cx: float, cy: float, tx: float, ty: float,
    rx: float, ry: float, rw: float, rh: float,
) -> tuple[float, float]:
    """Return the point where the ray from (cx,cy) toward (tx,ty) exits the rectangle.

    Used to place connector endpoints on shape edges rather than at centres,
    so arrows do not overlap the shape body.
    (cx, cy) must lie inside or on the boundary of the rectangle.
    """
    dx = tx - cx
    dy = ty - cy
    if dx == 0 and dy == 0:
        return (cx, cy)
    t_candidates = [1.0]
    if dx > 0:
        t_candidates.append((rx + rw - cx) / dx)
    elif dx < 0:
        t_candidates.append((rx - cx) / dx)
    if dy > 0:
        t_candidates.append((ry + rh - cy) / dy)
    elif dy < 0:
        t_candidates.append((ry - cy) / dy)
    t = max(0.0, min(t_candidates))
    return (cx + t * dx, cy + t * dy)


def layout(vg: VisualGraph) -> PlacedGraph:
    """S4: assign absolute coordinates to all shapes and clip connector endpoints.

    Processes the container hierarchy bottom-up (size_item), then places the
    top-level group, then recursively absolutizes nested children.  Scales the
    whole layout down uniformly if it overflows the canvas.

    Parameters
    ----------
    vg:
        The VisualGraph from S3, with shapes, connectors, and container nesting.

    Returns
    -------
    PlacedGraph
        Every shape has (x, y); every non-redundant connector has clipped
        endpoint coordinates.
    """
    shape_by_id = {s.nid: s for s in vg.shapes}
    container_children = {p: list(c) for p, c in vg.containers}

    child_of_any: set[str] = set()
    for cs in container_children.values():
        child_of_any.update(cs)
    all_ids = {s.nid for s in vg.shapes}
    top_level = sorted(all_ids - child_of_any)

    sizes: dict[str, tuple[float, float]] = {}
    internal: dict[str, list[Placement]] = {}

    def size_item(nid: str) -> None:
        if nid in sizes:
            return
        if nid in container_children:
            children = container_children[nid]
            for ch in children:
                size_item(ch)
            own = shape_by_id.get(nid)
            # Math primitive overrides: an `axes` container uses coord-anchored
            # placement of its children when meta carries (x, y).
            if own is not None and own.primitive == "axes":
                coords: dict[str, tuple[float, float]] = {}
                for ch in children:
                    cs = shape_by_id.get(ch)
                    if cs and "x" in cs.meta and "y" in cs.meta:
                        coords[ch] = (float(cs.meta["x"]), float(cs.meta["y"]))
                inner_w, inner_h, placements = _axes(
                    children, sizes, coords,
                    plot_w=max(180.0, own.width - 40.0),
                    plot_h=max(120.0, own.height - 50.0),
                )
            else:
                rule = _dominant_rule(children, vg.connectors)
                inner_w, inner_h, placements = _apply(rule, children, sizes, vg.connectors)
            own_w = own.width if own else 0.0
            total_w = max(own_w, inner_w) + 2 * CONTAINER_PAD
            total_h = inner_h + CONTAINER_HEADER + 2 * CONTAINER_PAD
            # Ellipse-shaped containers (set_blob) inscribe a rectangle, so the
            # bounding rect's corners poke OUTSIDE the visible ellipse.  Inflate
            # by √2 so the ellipse circumscribes the children rect: an ellipse
            # with semi-axes (W/√2 · 1, H/√2 · 1) just contains the rect (W,H).
            # Equivalently, set width/height = inner × √2 (≈ 1.42×).
            if own is not None and own.primitive == "set_blob":
                total_w = max(own_w, inner_w * 1.42) + 2 * CONTAINER_PAD
                total_h = inner_h * 1.42 + CONTAINER_HEADER + 2 * CONTAINER_PAD
                # Re-centre children inside the inflated frame.
                extra_x = (total_w - 2 * CONTAINER_PAD - inner_w) / 2.0
                extra_y = (total_h - CONTAINER_HEADER - 2 * CONTAINER_PAD - inner_h) / 2.0
                placements = [(ch, rx + extra_x, ry + extra_y)
                              for ch, rx, ry in placements]
            sizes[nid] = (total_w, total_h)
            internal[nid] = [
                (ch, rx + CONTAINER_PAD, ry + CONTAINER_HEADER + CONTAINER_PAD)
                for ch, rx, ry in placements
            ]
        elif nid in shape_by_id:
            s = shape_by_id[nid]
            sizes[nid] = (s.width, s.height)

    for t in top_level:
        size_item(t)

    if len(top_level) >= 1:
        # Build virtual connectors between top-level groups by mapping every
        # original edge to its endpoints' top-level ancestors.  This makes the
        # top-level layout "see" cross-container relationships — without it,
        # a node that points into a deeply-nested container would sit far away
        # from its target because the layout doesn't know they're related.
        top_virt = _build_virtual_connectors(
            top_level, container_children, vg.connectors,
        )
        # Combine direct top-level edges with the virtual ones, so the layout
        # sees both kinds of relationship.
        top_set = set(top_level)
        direct_top = [c for c in vg.connectors
                      if c.from_nid in top_set and c.to_nid in top_set]
        top_edges = direct_top + top_virt
        # When there are cross-container relationships, force a layered layout
        # so connected groups end up adjacent.  The default `_dominant_rule`
        # would route most symmetric relations to `grid`, which doesn't reward
        # adjacency.  Use it only when the dominant_rule wouldn't produce a
        # sensible layered layout already.
        if top_virt and _dominant_rule(top_level, top_edges) == "grid":
            top_rule = "sugiyama"
        else:
            top_rule = _dominant_rule(top_level, top_edges)
        _, _, top_placements = _apply(top_rule, top_level, sizes, top_edges)
    else:
        top_placements = []

    abs_pos: dict[str, tuple[float, float]] = {}
    for nid, rx, ry in top_placements:
        abs_pos[nid] = (PAD + rx, PAD + ry)

    def absolutize(nid: str, x0: float, y0: float) -> None:
        for ch, rx, ry in internal.get(nid, ()):
            ax, ay = x0 + rx, y0 + ry
            abs_pos[ch] = (ax, ay)
            absolutize(ch, ax, ay)

    for nid, _, _ in top_placements:
        x0, y0 = abs_pos[nid]
        absolutize(nid, x0, y0)

    # Scale down the whole layout if it overflows the canvas.
    max_right = PAD
    max_bottom = PAD
    for nid, (x, y) in abs_pos.items():
        sh = shape_by_id.get(nid)
        w, h = sizes.get(nid, (sh.width if sh else 0.0, sh.height if sh else 0.0))
        max_right = max(max_right, x + w + PAD)
        max_bottom = max(max_bottom, y + h + PAD)
    sf = min(CANVAS_W / max(max_right, 1.0),
             CANVAS_H / max(max_bottom, 1.0),
             1.0)
    if sf < 1.0:
        abs_pos = {nid: (x * sf, y * sf) for nid, (x, y) in abs_pos.items()}
        sizes = {nid: (w * sf, h * sf) for nid, (w, h) in sizes.items()}

    placed_shapes: list[PlacedShape] = []
    for sh in vg.shapes:
        if sh.nid not in abs_pos:
            continue
        x, y = abs_pos[sh.nid]
        if sh.nid in container_children:
            w, h = sizes[sh.nid]
            placed_shapes.append(PlacedShape(
                shape=VisualShape(
                    nid=sh.nid, primitive=sh.primitive, label=sh.label,
                    width=w, height=h,
                    font_size=sh.font_size, stroke_width=sh.stroke_width,
                    fill_index=sh.fill_index,
                    is_container=True,
                    meta=dict(sh.meta),
                ),
                x=x, y=y,
            ))
        else:
            w, h = sizes.get(sh.nid, (sh.width, sh.height))
            placed_shapes.append(PlacedShape(
                shape=VisualShape(
                    nid=sh.nid, primitive=sh.primitive, label=sh.label,
                    width=w, height=h,
                    font_size=sh.font_size, stroke_width=sh.stroke_width,
                    fill_index=sh.fill_index,
                    is_container=sh.is_container,
                    meta=dict(sh.meta),
                ),
                x=x, y=y,
            ))

    redundant: set[tuple[str, str]] = set()
    for parent, children in container_children.items():
        for ch in children:
            redundant.add((ch, parent))
            redundant.add((parent, ch))

    lookup = {p.shape.nid: p for p in placed_shapes}
    conns: list[PlacedConn] = []
    for c in sorted(vg.connectors, key=lambda c: c.eid):
        if (c.from_nid, c.to_nid) in redundant:
            continue
        a = lookup.get(c.from_nid)
        b = lookup.get(c.to_nid)
        if not a or not b:
            continue
        ax = a.x + a.shape.width / 2.0
        ay = a.y + a.shape.height / 2.0
        bx = b.x + b.shape.width / 2.0
        by = b.y + b.shape.height / 2.0
        p_start = _clip_to_rect(ax, ay, bx, by, a.x, a.y, a.shape.width, a.shape.height)
        p_end = _clip_to_rect(bx, by, ax, ay, b.x, b.y, b.shape.width, b.shape.height)
        conns.append(PlacedConn(conn=c, points=[p_start, p_end]))

    return PlacedGraph(
        shapes=placed_shapes, conns=conns,
        canvas_w=CANVAS_W, canvas_h=CANVAS_H,
    )
