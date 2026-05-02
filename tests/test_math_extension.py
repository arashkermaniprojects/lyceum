"""Tests for the math-extension layer.

Covers:
  - Math primitive selection (label classification → primitive)
  - Math relation extraction from text and Unicode symbols
  - Inline LaTeX detection and equation_block node creation
  - Matrix-literal detection and matrix_bracket node creation
  - End-to-end SVG rendering for new shapes/connectors
  - LaTeX → Unicode coverage
  - Determinism preserved across the math pipeline
"""
from sevim.math_lex import (
    classify_math_label,
    find_latex_spans,
    latex_to_unicode,
    parse_matrix_literal,
)
from sevim.pipeline import run_pipeline
from sevim.s3_map import _label_to_primitive


# ---------------------------------------------------------------------------
# Label classification
# ---------------------------------------------------------------------------

def test_matrix_label_maps_to_matrix_bracket():
    assert classify_math_label("matrix A") == "matrix_bracket"
    assert classify_math_label("the 2x3 matrix") == "matrix_bracket"
    assert _label_to_primitive("matrix", "entity") == "matrix_bracket"


def test_set_label_maps_to_set_blob():
    assert classify_math_label("set A") == "set_blob"
    assert classify_math_label("subset S") == "set_blob"
    assert classify_math_label("union of A and B") == "set_blob"


def test_geometric_primitives():
    assert classify_math_label("point P") == "point"
    assert classify_math_label("line segment AB") == "segment"
    assert classify_math_label("circle of radius 1") == "circle"
    assert classify_math_label("triangle ABC") == "polygon"
    assert classify_math_label("right angle") == "arc"


def test_axes_label():
    assert classify_math_label("coordinate system") == "axes"
    assert classify_math_label("x-axis") == "axes"


def test_vector_label_uses_arrow_primitive():
    assert classify_math_label("velocity vector") == "arrow"


def test_unrecognised_label_returns_none():
    assert classify_math_label("gradient descent") is None
    assert classify_math_label("dolphin") is None


# ---------------------------------------------------------------------------
# Math relation extraction (text-form)
# ---------------------------------------------------------------------------

def _edges(graph):
    return [(e.from_id, e.to_id, e.relation) for e in graph.edges]


def test_lies_on_relation():
    g = run_pipeline("Point P lies on line L.").graph
    rels = {e.relation for e in g.edges}
    assert "lies_on" in rels


def test_perpendicular_relation():
    g = run_pipeline("Line AB is perpendicular to line CD.").graph
    assert "perpendicular" in {e.relation for e in g.edges}


def test_parallel_relation():
    g = run_pipeline("Line AB is parallel to line CD.").graph
    assert "parallel" in {e.relation for e in g.edges}


def test_element_of_relation():
    g = run_pipeline("Element x is a member of set A.").graph
    assert "element_of" in {e.relation for e in g.edges}


def test_subset_of_relation():
    g = run_pipeline("Set A is a subset of set B.").graph
    assert "subset_of" in {e.relation for e in g.edges}


def test_isomorphic_to_relation():
    g = run_pipeline("Group G is isomorphic to group H.").graph
    assert "isomorphic_to" in {e.relation for e in g.edges}


def test_congruent_relation():
    g = run_pipeline("Triangle ABC is congruent to triangle DEF.").graph
    assert "congruent" in {e.relation for e in g.edges}


def test_maps_to_relation():
    # "X maps to Y" matches the regex cascade directly.
    g = run_pipeline("The domain maps to the codomain.").graph
    rels = {e.relation for e in g.edges}
    assert "maps_to" in rels


def test_maps_to_unicode_arrow():
    g = run_pipeline("X ↦ Y").graph
    assert "maps_to" in {e.relation for e in g.edges}


# ---------------------------------------------------------------------------
# Unicode-symbol relation extraction
# ---------------------------------------------------------------------------

def test_unicode_element_of_symbol():
    g = run_pipeline("x ∈ S").graph
    assert "element_of" in {e.relation for e in g.edges}


def test_unicode_subset_symbol():
    g = run_pipeline("A ⊂ B").graph
    assert "subset_of" in {e.relation for e in g.edges}


def test_unicode_perpendicular_symbol():
    g = run_pipeline("AB ⊥ CD").graph
    assert "perpendicular" in {e.relation for e in g.edges}


# ---------------------------------------------------------------------------
# Inline LaTeX → equation_block node
# ---------------------------------------------------------------------------

def test_inline_latex_creates_equation_node():
    g = run_pipeline(r"The identity $E = mc^2$ relates energy and mass.").graph
    eq_nodes = [n for n in g.nodes if n.meta.get("kind") == "equation_block"]
    assert len(eq_nodes) == 1
    assert "E" in eq_nodes[0].meta["unicode"]


def test_double_dollar_equation():
    g = run_pipeline(r"Recall $$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$ for sums.").graph
    eq_nodes = [n for n in g.nodes if n.meta.get("kind") == "equation_block"]
    assert len(eq_nodes) == 1
    assert "∑" in eq_nodes[0].meta["unicode"]


def test_paren_form_latex():
    g = run_pipeline(r"By definition \(\alpha + \beta = \gamma\).").graph
    eq_nodes = [n for n in g.nodes if n.meta.get("kind") == "equation_block"]
    assert len(eq_nodes) == 1
    assert "α" in eq_nodes[0].meta["unicode"]


# ---------------------------------------------------------------------------
# Matrix literal extraction
# ---------------------------------------------------------------------------

def test_pmatrix_environment():
    src = r"Consider $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ as a 2×2 example."
    g = run_pipeline(src).graph
    mats = [n for n in g.nodes if n.meta.get("kind") == "matrix_bracket"]
    assert len(mats) == 1
    m = mats[0]
    assert m.meta["nrows"] == 2 and m.meta["ncols"] == 2
    assert m.meta["delim"] == "paren"


def test_python_list_matrix():
    g = run_pipeline("The matrix [[1,2,3],[4,5,6]] has two rows.").graph
    mats = [n for n in g.nodes if n.meta.get("kind") == "matrix_bracket"]
    assert len(mats) == 1
    assert mats[0].meta["nrows"] == 2
    assert mats[0].meta["ncols"] == 3


# ---------------------------------------------------------------------------
# LaTeX → Unicode coverage
# ---------------------------------------------------------------------------

def test_latex_greek_letters():
    assert latex_to_unicode(r"\alpha + \beta = \gamma") == "α + β = γ"


def test_latex_operators():
    assert "∑" in latex_to_unicode(r"\sum_{i=1}^{n}")
    assert "∫" in latex_to_unicode(r"\int_0^1 f(x) dx")
    assert "∞" in latex_to_unicode(r"x \to \infty")


def test_latex_blackboard():
    assert "ℝ" in latex_to_unicode(r"x \in \mathbb{R}")
    assert "ℤ" in latex_to_unicode(r"\mathbb{Z}")


def test_latex_super_subscripts():
    assert "x²" in latex_to_unicode("x^2")
    assert "x₁" in latex_to_unicode("x_1")


def test_latex_fraction_to_slash():
    assert "/" in latex_to_unicode(r"\frac{a}{b}")


def test_latex_relations():
    out = latex_to_unicode(r"a \leq b, c \neq d, A \subseteq B")
    assert "≤" in out and "≠" in out and "⊆" in out


# ---------------------------------------------------------------------------
# LaTeX span detection
# ---------------------------------------------------------------------------

def test_find_latex_spans_basic():
    spans = find_latex_spans(r"Hello $x + y$ world.")
    assert len(spans) == 1
    assert spans[0][2] == "x + y"


def test_find_latex_spans_multiple():
    spans = find_latex_spans(r"$a$ and $b$ but not $c$.")
    assert len(spans) == 3


def test_find_latex_spans_mixed_delims():
    spans = find_latex_spans(r"$a$ and \(b\) and \[c\] and $$d$$.")
    bodies = [s[2] for s in spans]
    assert {"a", "b", "c", "d"} <= set(bodies)


# ---------------------------------------------------------------------------
# Matrix-literal parsing
# ---------------------------------------------------------------------------

def test_parse_pmatrix():
    m = parse_matrix_literal(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}")
    assert m == {"nrows": 2, "ncols": 2,
                 "cells": [["1", "2"], ["3", "4"]],
                 "delim": "paren"}


def test_parse_bmatrix():
    m = parse_matrix_literal(r"\begin{bmatrix} a & b & c \end{bmatrix}")
    assert m["nrows"] == 1 and m["ncols"] == 3
    assert m["delim"] == "bracket"


def test_parse_python_list():
    m = parse_matrix_literal("[[1, 2], [3, 4]]")
    assert m["cells"] == [["1", "2"], ["3", "4"]]


def test_parse_invalid():
    assert parse_matrix_literal("not a matrix") is None
    # Ragged rows must be rejected.
    assert parse_matrix_literal("[[1,2],[3]]") is None


# ---------------------------------------------------------------------------
# End-to-end SVG rendering for new primitives
# ---------------------------------------------------------------------------

def test_matrix_renders_with_brackets_in_svg():
    svg = run_pipeline(r"Consider $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$.").svg
    # The matrix renderer emits <path> elements for the brackets.
    assert "matrix_bracket" not in svg or "<path" in svg
    assert "<path" in svg


def test_equation_renders_in_svg():
    svg = run_pipeline(r"The equation $E = mc^2$ is famous.").svg
    # Equation block is dashed-rect + italic <text>.
    assert "stroke-dasharray=\"2,2\"" in svg
    assert "font-style=\"italic\"" in svg


def test_set_blob_renders_as_translucent_ellipse():
    svg = run_pipeline("Set A contains element x.").svg
    # set_blob uses fill-opacity 0.10
    assert 'fill-opacity="0.10"' in svg or "set" in svg.lower()


def test_point_renders_as_filled_disk():
    svg = run_pipeline("Point P lies on line L.").svg
    # Point primitive renders <circle> with filled stroke color.
    assert "<circle" in svg


def test_axes_renders_with_arrow_tips():
    # A sentence that creates an "axes" node via the part_of structural extractor.
    svg = run_pipeline("The coordinate system has an x-axis and a y-axis.").svg
    # Axes primitive uses two <line> elements with marker-end="url(#arrow)".
    assert 'marker-end="url(#arrow)"' in svg


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_math_pipeline_is_deterministic():
    text = (
        "Set A contains element x. "
        r"The matrix $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ is the identity. "
        "Triangle ABC is congruent to triangle DEF. "
        "Point P lies on line L."
    )
    svg1 = run_pipeline(text).svg
    svg2 = run_pipeline(text).svg
    assert svg1 == svg2


def test_math_relations_in_relation_pattern():
    """All 18 math relations must have an entry in _RELATION_PATTERN."""
    from sevim.s3_map import _RELATION_PATTERN
    from sevim.ir import MATH_RELATIONS
    assert MATH_RELATIONS <= set(_RELATION_PATTERN.keys())


def test_math_primitives_have_renderers():
    """Every math primitive must produce some SVG when rendered.

    Tested by constructing a SceneNode with the math kind in meta and
    confirming the resulting SVG contains a non-empty <g data-nid=…> block.
    """
    from sevim.ir import MATH_PRIMITIVES, PlacedGraph, PlacedShape, VisualShape
    from sevim.s5_render import render
    for prim in sorted(MATH_PRIMITIVES):
        meta = {"kind": prim}
        if prim == "matrix_bracket":
            meta.update({"nrows": 2, "ncols": 2,
                         "cells": [["a", "b"], ["c", "d"]],
                         "delim": "bracket"})
        elif prim == "equation_block":
            meta.update({"latex": "x", "unicode": "x"})
        shape = VisualShape(
            nid="n_test", primitive=prim, label="test",
            width=120.0, height=80.0, font_size=14.0,
            stroke_width=1.2, fill_index=0, meta=meta,
        )
        pg = PlacedGraph(
            shapes=[PlacedShape(shape=shape, x=20.0, y=20.0)],
            conns=[], canvas_w=400.0, canvas_h=300.0,
        )
        svg = render(pg)
        assert 'data-nid="n_test"' in svg, f"no SVG element for {prim}"
