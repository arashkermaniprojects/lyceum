"""S3 Map — semantic graph → visual graph.

Two independent sub-mappings run in sequence:

  Symbolic mapping (relation → visual pattern)
  ---------------------------------------------
  Each edge relation type is assigned a named connector pattern from
  _RELATION_PATTERN.  The pattern name is later interpreted by S5 Render
  to produce the correct SVG stroke style and marker geometry.

  Numeric mapping  φ : (embedding, salience) → (w, h, font_size, stroke, fill)
  -------------------------------------------------------------------------------
  A frozen linear projection _W (seed=42) maps the node's embedding vector
  concatenated with its salience scalar to five visual dimensions.  tanh
  clips the projection into compact ranges so extreme embedding values cannot
  produce unusable geometry.

  When the embedding is empty (encoder-off mode), _phi falls back to a
  salience-only formula, so the pipeline degrades gracefully without Qwen.

  Container hierarchy
  -------------------
  part_of / contains edges are used to build a parent→children map that
  S4 Layout uses for nested rendering.
"""
from __future__ import annotations

import math
import re
import random

from .ir import RelationType, SceneGraph, VisualConn, VisualGraph, VisualShape
from .math_lex import classify_math_label

# ---------------------------------------------------------------------------
# Label → shape primitive
# Ordered from most-specific to least-specific; first match wins.
# ---------------------------------------------------------------------------
_SHAPE_RULES: list[tuple[re.Pattern[str], str]] = [
    # Neural-network architectures → hexagon (classic "computational block")
    (re.compile(
        r"\b(neural.network|multilayer.perceptron|\bmlp\b|convolutional.network"
        r"|\bcnn\b|\brnn\b|lstm|gru|transformer|deep.network"
        r"|recurrent.network|boltzmann.machine|auto.?encoder"
        r"|generative.adversarial|diffusion.model"
        r"|q.network|deep.q|value.network|policy.network)\b",
        re.IGNORECASE,
    ), "hexagon"),

    # Numeric parameters / weights / kernels → diamond (scalar/data value)
    (re.compile(
        r"\b(weight|parameter|kernel|filter|shared.weight|gradient"
        r"|bias|learning.rate|loss|error|reward|penalty)\b",
        re.IGNORECASE,
    ), "diamond"),

    # Layer / transform / projection concepts → parallelogram (data-flow block)
    (re.compile(
        r"\b(layer|fully.connected|hidden.layer|input.layer|output.layer"
        r"|embedding|projection|encoder|decoder|attention|pooling"
        r"|activation|softmax|relu|sigmoid|normalisation|normalization)\b",
        re.IGNORECASE,
    ), "parallelogram"),
]


def _label_to_primitive(label: str, node_type: str, meta: dict | None = None) -> str:
    """Return the SVG primitive name for a node given its label and type.

    Resolution order (first hit wins):
      1.  Explicit `meta["kind"]` from the math literal extractor.
      2.  Math vocabulary (matrix, point, axes, set, …) via `math_lex`.
      3.  Attribute nodes → ellipse.
      4.  Concept-diagram rules (_SHAPE_RULES) — neural net / parameter / layer.
      5.  Default → rect.
    """
    if meta and meta.get("kind"):
        return meta["kind"]
    math_kind = classify_math_label(label)
    if math_kind is not None:
        return math_kind
    if node_type == "attribute":
        return "ellipse"
    for pattern, prim in _SHAPE_RULES:
        if pattern.search(label):
            return prim
    return "rect"

# Maps each semantic relation to a named connector pattern consumed by S5.
# Math relations get descriptive pattern names; the actual SVG dispatch is
# done by `s5_render._render_conn` keyed on the relation literal, not this
# string.  These names are metadata for debugging / downstream tooling.
_RELATION_PATTERN: dict[str, str] = {
    # original 12
    "contains": "adjacent-dashed",
    "part_of": "adjacent-dashed",
    "causes": "arrow-directed",
    "sequence": "ordered-chain",
    "attribute_of": "adjacent-smaller",
    "similar_to": "parallel-adjacent",
    "opposes": "arrow-bar",
    "instance_of": "dashed-up",
    "used_for": "adjacent-dashed",
    "requires": "prereq-arrow",
    "reduces_to": "funnel",
    "measures": "dotted-annot",
    # math: incidence
    "lies_on": "thin-line",
    "between": "thin-line",
    "perpendicular": "label-perp",
    "parallel": "label-parallel",
    "tangent_to": "thin-line",
    # math: equivalence
    "equals": "label-equals",
    "congruent": "label-congruent",
    "approximately_equal": "label-approx",
    "isomorphic_to": "label-iso",
    # math: set/logical
    "element_of": "label-in",
    "subset_of": "label-subset",
    "disjoint": "label-disjoint",
    "maps_to": "label-mapsto",
    # math: layout
    "labels": "tether",
    "points_to": "arrow-plain",
    "connects": "thin-line",
    "grouped_with": "cluster-hint",
    "aligned_with": "alignment-hint",
    # university: logical entailment + category theory
    "implies": "implies-arrow",
    "adjoint_to": "adjoint-pair",
    "natural_transformation": "double-arrow",
    "commutes": "commutes-label",
    # animation: agent → animated-target (rotate/translate/scale)
    "transforms": "transform-arrow",
}

_PHI_SEED = 42
_PHI_ROWS = 5  # w, h, fs, sw, fill
_W: tuple[tuple[float, ...], ...] | None = None


def _init_W(dim: int) -> None:
    """Lazily initialise the frozen projection matrix for embedding dimension *dim*.

    Uses a fixed seed so the same model always produces the same visual output
    (invariant I1 — determinism).  Re-initialised automatically if the dimension
    changes (e.g. when switching encoder models).
    """
    global _W
    rng = random.Random(_PHI_SEED)
    scale = 1.0 / math.sqrt(dim + 1)
    _W = tuple(
        tuple(rng.gauss(0.0, 1.0) * scale for _ in range(dim + 1))
        for _ in range(_PHI_ROWS)
    )


def _math_size_override(
    primitive: str,
    meta: dict,
    w: float,
    h: float,
) -> tuple[float, float]:
    """Override width/height for math primitives whose visual envelope is
    not embedding-driven.

    Most concept-diagram nodes want size proportional to salience/content;
    math primitives want size proportional to *structure*: a point is always
    small, a matrix scales with its cell count, axes fill their container.
    """
    if primitive == "point":
        return 14.0, 14.0
    if primitive == "decoration_mark":
        return 18.0, 18.0
    if primitive == "matrix_bracket":
        nrows = max(1, int(meta.get("nrows", 2)))
        ncols = max(1, int(meta.get("ncols", 2)))
        cell_w, cell_h = 36.0, 28.0
        bracket_pad = 14.0
        return ncols * cell_w + 2 * bracket_pad, nrows * cell_h + 8.0
    if primitive == "equation_block":
        # Width grows with rendered length.  Structured constructs (\frac,
        # \sqrt, \sum_/^…) need vertical room — bump height when they appear.
        latex = meta.get("latex") or ""
        unicode_form = meta.get("unicode") or latex
        approx = max(80.0, 10.0 + 11.0 * len(unicode_form))
        from .equation import has_structured_constructs as _hsc
        if latex and _hsc(latex):
            # Multi-line: fraction stacks ~2x, sum/int stacks ~3x.
            line_h = 70.0
            if "\\\\" in latex:
                lines = latex.count("\\\\") + 1
                line_h = 30.0 * lines + 20.0
            return min(approx, 380.0), line_h
        return min(approx, 360.0), 38.0
    if primitive == "axes":
        return 220.0, 160.0
    if primitive == "set_blob":
        return max(w, 110.0), max(h, 70.0)
    if primitive == "circle":
        # Force square envelope so the circle is round, not elliptical.
        d = max(w, h, 80.0)
        return d, d
    if primitive == "polygon":
        # Polygons are often shown roughly square.
        d = max(w, 90.0)
        return d, d * 0.85
    if primitive == "segment":
        return max(w, 120.0), 18.0
    # — university —
    if primitive == "tensor_box":
        legs = max(1, int(meta.get("legs", 3)))
        return max(110.0, 28.0 + 22.0 * legs), 90.0
    if primitive == "string_wire":
        return 60.0, 110.0
    if primitive == "pullback_corner":
        return 56.0, 56.0
    if primitive == "axes_3d":
        return 200.0, 170.0
    if primitive == "pdf_curve":
        return 200.0, 110.0
    if primitive == "plate":
        return max(w, 160.0), max(h, 100.0)
    if primitive == "error_bar":
        return 80.0, 70.0
    if primitive == "proof_node":
        return max(w, 130.0), max(h, 36.0)
    return w, h


def _phi(
    embedding: tuple[float, ...],
    salience: float,
) -> tuple[float, float, float, float, int]:
    """Numeric mapping: (embedding, salience) → (w, h, font_size, stroke, fill).

    Linear projection with frozen W, clipped into compact ranges via tanh.
    Falls back to salience-only when embedding is empty.
    """
    d = len(embedding)
    if d == 0:
        return (100.0 + 60.0 * salience, 50.0 + 20.0 * salience, 14.0, 1.5, 0)

    global _W
    if _W is None or len(_W[0]) != d + 1:
        _init_W(d)
    assert _W is not None

    aug = list(embedding) + [salience]
    raw = [sum(a * b for a, b in zip(row, aug)) for row in _W]

    w = 150.0 + 50.0 * math.tanh(raw[0])
    h = 82.0 + 8.0 * math.tanh(raw[1])
    font_size = 16.0
    stroke_width = 1.2
    fill_index = abs(int(raw[4] * 1000)) % 5
    return (w, h, font_size, stroke_width, fill_index)


def map_visual(graph: SceneGraph) -> VisualGraph:
    """S3: map semantic SceneGraph to VisualGraph.

    For each node, calls _phi to get geometry parameters and _label_to_primitive
    to get the SVG shape type, producing a VisualShape.

    For each edge, looks up the connector pattern from _RELATION_PATTERN,
    producing a VisualConn.

    Also builds the container hierarchy (parent_of / children dicts) from
    part_of and contains edges so S4 Layout can nest shapes correctly.

    Parameters
    ----------
    graph:
        The SceneGraph from S2 (or S2b if the LLM improvement path is on).

    Returns
    -------
    VisualGraph
        Shapes, connectors, and container nesting ready for layout.
    """
    shapes: list[VisualShape] = []
    for n in graph.nodes:
        w, h, fs, sw, fill = _phi(n.embedding, n.salience)
        prim = _label_to_primitive(n.label, n.node_type, n.meta)
        # Math primitives often want different size envelopes than concept
        # nodes — e.g., a `point` should be a small disk regardless of the
        # embedding-driven width.
        w, h = _math_size_override(prim, n.meta, w, h)
        shapes.append(VisualShape(
            nid=n.id,
            primitive=prim,
            label=n.label,
            width=w, height=h, font_size=fs,
            stroke_width=sw, fill_index=fill,
            meta=dict(n.meta),
        ))

    conns: list[VisualConn] = []
    for e in graph.edges:
        conns.append(VisualConn(
            eid=e.id, from_nid=e.from_id, to_nid=e.to_id,
            pattern=_RELATION_PATTERN.get(e.relation, "plain-line"),
            relation=e.relation,
        ))

    # Build container hierarchy from part_of / contains edges.
    # Math relations also induce containment when one side is a set/space:
    #   element_of(a, S)  → a is rendered inside S
    #   subset_of(A, B)   → A is rendered inside B
    #   grouped_with      → siblings under a synthetic group (handled later)
    shape_kind = {n.id: _label_to_primitive(n.label, n.node_type, n.meta)
                  for n in graph.nodes}
    parent_of: dict[str, str] = {}
    for e in graph.edges:
        if e.relation == "part_of":
            parent_of.setdefault(e.from_id, e.to_id)
        elif e.relation == "contains":
            parent_of.setdefault(e.to_id, e.from_id)
        elif e.relation == "element_of":
            # Only nest when the parent is a set-like container.
            if shape_kind.get(e.to_id) == "set_blob":
                parent_of.setdefault(e.from_id, e.to_id)
        elif e.relation == "subset_of":
            if shape_kind.get(e.to_id) == "set_blob":
                parent_of.setdefault(e.from_id, e.to_id)
    children: dict[str, list[str]] = {}
    for child, parent in parent_of.items():
        children.setdefault(parent, []).append(child)
    containers = sorted(
        ((p, sorted(c)) for p, c in children.items()),
        key=lambda x: x[0],
    )

    return VisualGraph(shapes=shapes, connectors=conns, containers=containers)
