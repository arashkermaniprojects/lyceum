"""Intermediate Representation (IR) — shared data contracts for SeVim.

Every stage of the S1→S5 pipeline communicates exclusively through the
immutable dataclasses defined here.  Nothing outside this module should
define new IR types; cross-stage coupling is intentional and explicit.

Data flow:
    str  ──S1──▶  SpanToken[]
                    ──S2──▶  SceneGraph        (semantic layer)
                               ──S3──▶  VisualGraph    (visual layer)
                                          ──S4──▶  PlacedGraph   (geometry layer)
                                                     ──S5──▶  SVG string
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases — closed vocabularies enforced at the type-checker level.
# ---------------------------------------------------------------------------

NodeType = Literal["entity", "process", "attribute", "value", "group"]
"""Semantic category of a scene node.

- entity    : a named thing (noun phrase, concept)
- process   : an action or transformation (verb phrase)
- attribute : a property or quality of another node
- value     : a numeric or enumerated constant
- group     : a logical cluster with no single referent
"""

RelationType = Literal[
    # — original 12 (concept-diagram core) —
    "contains", "part_of", "causes", "sequence",
    "attribute_of", "similar_to", "opposes", "instance_of",
    "used_for", "requires", "reduces_to", "measures",
    # — math: incidence / geometric (5) —
    "lies_on", "between", "perpendicular", "parallel", "tangent_to",
    # — math: equivalence / comparison (4) —
    "equals", "congruent", "approximately_equal", "isomorphic_to",
    # — math: set / logical (4) —
    "element_of", "subset_of", "disjoint", "maps_to",
    # — math: layout / connective (5) —
    "labels", "points_to", "connects", "grouped_with", "aligned_with",
    # — university math (4): logical entailment + category theory —
    "implies", "adjoint_to", "natural_transformation", "commutes",
    # — animation (1): SMIL transform agent-of —
    "transforms",
]
"""Relation ontology — 12 concept-diagram + 18 math = 30 closed values.

Visual encoding (see s5_render.py):
  causes/sequence  → solid directed arrow
  used_for         → dashed arrow
  instance_of      → dashed arrow (hollow-triangle end)
  requires         → hollow triangle (UML dependency style)
  reduces_to       → filled funnel triangle
  similar_to       → double parallel line with ≈ label
  opposes          → bar–bar line
  measures         → dotted line with = label
  contains/part_of → rendered as container nesting (no explicit line)
  attribute_of     → dashed line, smaller node

  — math relations —
  lies_on              → thin solid segment, no markers (point-on-line)
  between              → ternary; rendered as two thin segments through midpoint
  perpendicular        → line with ⊥ midpoint label + small right-angle tick
  parallel             → line with ∥ midpoint label
  tangent_to           → line with single point of incidence
  equals               → solid line, '=' midpoint label
  congruent            → solid line, '≅' midpoint label
  approximately_equal  → solid line, '≈' midpoint label
  isomorphic_to        → solid arrow, '≅' midpoint label
  element_of           → solid arrow, '∈' midpoint label
  subset_of            → solid arrow, '⊆' midpoint label
  disjoint             → bar–bar line, '∅' midpoint label
  maps_to              → solid arrow, '→' midpoint label (function arrow)
  labels               → thin dotted line (annotation tether)
  points_to            → simple arrow, no markup
  connects             → plain line, no markup
  grouped_with         → very thin dashed line (cluster hint)
  aligned_with         → thin dotted line (alignment hint)
"""

# Relations present in v0.1 corpus seed; used as a fast membership check.
PHASE1_RELATIONS: set[str] = {"sequence", "causes", "part_of", "attribute_of"}

# Math-only subset, used by some renderers and the math test suite.
MATH_RELATIONS: frozenset[str] = frozenset({
    "lies_on", "between", "perpendicular", "parallel", "tangent_to",
    "equals", "congruent", "approximately_equal", "isomorphic_to",
    "element_of", "subset_of", "disjoint", "maps_to",
    "labels", "points_to", "connects", "grouped_with", "aligned_with",
    "implies", "adjoint_to", "natural_transformation", "commutes",
})

Primitive = Literal[
    # — original 9 (concept-diagram core) —
    "rect", "ellipse", "line", "arrow", "text", "group",
    "diamond", "hexagon", "parallelogram",
    # — math: 13 elementary primitives —
    "point", "segment", "curve", "polygon", "circle", "arc",
    "axes", "grid", "set_blob", "matrix_bracket", "equation_block",
    "decoration_mark", "bracket_brace",
    # — university math: 8 new primitives —
    "tensor_box", "string_wire", "pullback_corner", "axes_3d",
    "pdf_curve", "plate", "error_bar", "proof_node",
]
"""SVG primitive shape to use when rendering a node body.

The 13 math primitives:
  point           — small filled disk (geometric point with label)
  segment         — straight line between two anchors
  curve           — open path (function plot, parameterised curve)
  polygon         — n-gon (triangle, square, pentagon, …)
  circle          — geometric circle (distinct from `ellipse` attribute use)
  arc             — partial circle (angle marker, segment of a curve)
  axes            — coordinate axes (1-D number line or 2-D plot frame)
  grid            — regular grid background, parented to axes
  set_blob        — closed labeled region (Euler/Venn member)
  matrix_bracket  — bracketed/parenthesised cell array
  equation_block  — typeset formula box (LaTeX → Unicode at v1)
  decoration_mark — tick / right-angle / parallel-chevron / congruence hash
  bracket_brace   — under/over-brace
"""

# Math primitives exposed for downstream type checks.
MATH_PRIMITIVES: frozenset[str] = frozenset({
    "point", "segment", "curve", "polygon", "circle", "arc",
    "axes", "grid", "set_blob", "matrix_bracket", "equation_block",
    "decoration_mark", "bracket_brace",
    # — university extension —
    "tensor_box", "string_wire", "pullback_corner", "axes_3d",
    "pdf_curve", "plate", "error_bar", "proof_node",
})


# ---------------------------------------------------------------------------
# Semantic layer — output of S2 Extract.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpanRef:
    """Pointer to a character range in the original input text (I3 provenance)."""
    start: int          # inclusive byte offset in the utterance
    end: int            # exclusive byte offset
    utterance_id: str = "u0"  # which utterance (clause) this span belongs to


@dataclass(frozen=True)
class SpanToken:
    """One clause-unit token produced by S1 Parse.

    Embeddings are NOT attached here; they are added per-label in S2 Extract
    so that subject and object embeddings remain distinct (attaching a single
    clause embedding to both would collapse them under cosine merge).
    """
    text: str               # raw clause text
    start: int              # character offset of this clause in the full input
    end: int                # end offset
    utterance_id: str       # identifier for multi-turn streaming
    embedding: tuple[float, ...] = ()  # unused at S1; reserved for future use


@dataclass
class SceneNode:
    """One concept extracted from the text."""
    id: str                         # stable canonical ID, e.g. "n_gradient_descent"
    label: str                      # short display label, e.g. "gradient descent"
    node_type: NodeType
    embedding: tuple[float, ...]    # Qwen/MiniLM mean-pool vector; () when encoder is off
    salience: float                 # importance weight in [0, 1], derived from position/frequency
    src_spans: list[SpanRef] = field(default_factory=list)
    # Math metadata (free-form by primitive kind).  Examples:
    #   matrix:   {"nrows": 2, "ncols": 3, "cells": [["a","b","c"], …], "delim": "bracket"}
    #   equation: {"latex": r"\\frac{a}{b}", "unicode": "a/b"}
    #   axes:     {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1}
    #   point:    {"x": 1.0, "y": 2.0}  (math coords, not pixels)
    #   set_blob: {"members": ["a","b"], "kind": "venn"}
    #
    # Animation (optional, SMIL <animateTransform>) — purely declarative,
    # added only when the user explicitly requests motion.  Static SVG is
    # the default; absence of this key means no animation is rendered.
    #   meta["animate"] = {
    #       "kind":   "rotate",      # or "translate" or "scale"
    #       "from":   0.0,           # starting value
    #                                # (degrees for rotate, px for translate,
    #                                # multiplicative factor for scale)
    #       "to":     90.0,          # ending value (same units as `from`)
    #       "dur":    "2s",          # SMIL duration string ("2s", "500ms", …)
    #       "repeat": "indefinite",  # or a count string like "1", "3"
    #       "around": "center",      # rotate-only: "center" (default) |
    #                                # "origin" | (x, y) tuple of px
    #   }
    # Determinism (I1) is preserved: the animation element is order-stable
    # and contains no random IDs or timestamps.
    meta: dict = field(default_factory=dict)


@dataclass
class SceneEdge:
    """One directed relation between two SceneNodes."""
    id: str                 # e.g. "e_causes_n_overfitting_n_high_test_error"
    from_id: str            # source node ID
    to_id: str              # target node ID
    relation: RelationType
    src_spans: list[SpanRef] = field(default_factory=list)


@dataclass
class SceneGraph:
    """Full semantic graph for one diagram — the S2 output and S2b input/output."""
    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)
    revision: int = 0  # incremented by S2b each time it rewrites the graph


# ---------------------------------------------------------------------------
# Visual layer — output of S3 Map.
# ---------------------------------------------------------------------------

@dataclass
class VisualShape:
    """Visual parameters for one node, before geometry is assigned."""
    nid: str            # matches SceneNode.id
    primitive: Primitive
    label: str
    width: float        # px, from φ projection or fallback
    height: float       # px
    font_size: float    # px
    stroke_width: float # px
    fill_index: int     # index into the PALETTE list in s5_render.py
    is_container: bool = False  # True when this shape encloses children
    meta: dict = field(default_factory=dict)  # propagated from SceneNode.meta


@dataclass
class VisualConn:
    """Visual parameters for one edge, before geometry is assigned."""
    eid: str            # matches SceneEdge.id
    from_nid: str
    to_nid: str
    pattern: str        # e.g. "arrow-directed", "funnel" — drives renderer dispatch
    relation: RelationType


@dataclass
class VisualGraph:
    """Full visual description before layout (no x/y coordinates yet)."""
    shapes: list[VisualShape]
    connectors: list[VisualConn]
    containers: list[tuple[str, list[str]]]  # [(parent_nid, [child_nid, ...]), ...]


# ---------------------------------------------------------------------------
# Geometry layer — output of S4 Layout.
# ---------------------------------------------------------------------------

@dataclass
class PlacedShape:
    """A VisualShape with absolute canvas coordinates assigned by S4."""
    shape: VisualShape
    x: float  # top-left corner, px
    y: float  # top-left corner, px


@dataclass
class PlacedConn:
    """A VisualConn with routed waypoints assigned by S4."""
    conn: VisualConn
    points: list[tuple[float, float]]  # [start, ..., end] on shape boundaries


@dataclass
class PlacedGraph:
    """Fully placed scene, ready for S5 to serialise to SVG."""
    shapes: list[PlacedShape]
    conns: list[PlacedConn]
    canvas_w: float  # total SVG width in px
    canvas_h: float  # total SVG height in px


# ---------------------------------------------------------------------------
# Pipeline envelope — wraps the full run_pipeline() result.
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    """One diagnostic record emitted by a pipeline stage."""
    stage: str      # e.g. "S2", "S4.5"
    message: str    # human-readable summary
    refs: dict      # stage-specific structured payload (node ids, counts, etc.)


@dataclass
class PipelineResult:
    """Everything produced by a single run_pipeline() call."""
    svg: str            # canonical SVG string (byte-identical for identical inputs)
    graph: SceneGraph   # final semantic graph (after S2b if enabled)
    placed: PlacedGraph # laid-out geometry
    trace: list[TraceEvent]  # per-stage diagnostics
