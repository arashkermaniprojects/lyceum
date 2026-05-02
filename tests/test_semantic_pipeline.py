"""Tests for the deterministic Tier-3 semantic pipeline.

Covers:

  * parser → graph for the canonical inputs called out in the spec
  * SVG renderer produces valid primitive-only output
  * LaTeX renderer maps each node-type to canonical math source
  * empty-graph triggers fallback (orchestrator code path)
  * determinism: same input ⇒ byte-identical SVG and LaTeX
  * performance budget targets from the spec
"""
from __future__ import annotations

import re
import time

import pytest

from viz import semantic_parser, semantic_to_latex, semantic_to_svg
from viz.semantic_ir import SemanticEdge, SemanticGraph, SemanticNode


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_quadratic_function_recognised():
    g = semantic_parser.parse("The loss is a quadratic function")
    funcs = g.by_type("function")
    assert len(funcs) == 1
    assert funcs[0].params["form"] == "quadratic"
    assert funcs[0].id == "func_quadratic"


def test_linear_function_recognised():
    g = semantic_parser.parse("Use a linear model for the data")
    funcs = g.by_type("function")
    assert any(n.params.get("form") == "linear" for n in funcs)


def test_sigmoid_recognised():
    g = semantic_parser.parse("The sigmoid function squashes inputs")
    funcs = g.by_type("function")
    assert any(n.params.get("form") == "sigmoid" for n in funcs)


def test_matrix_with_dims():
    g = semantic_parser.parse("A is a 2x3 matrix")
    mats = g.by_type("matrix")
    assert mats, "expected a matrix node"
    assert mats[0].params["nrows"] == 2
    assert mats[0].params["ncols"] == 3


def test_inline_matrix_literal():
    g = semantic_parser.parse("Apply the matrix [[1, 2], [3, 4]] to v")
    mats = g.by_type("matrix")
    rows = mats[0].params.get("rows")
    assert rows == [["1", "2"], ["3", "4"]]


def test_vector_recognised():
    g = semantic_parser.parse("A vector v points east")
    vecs = g.by_type("vector")
    assert vecs and vecs[0].params["name"] == "v"


def test_vectors_plural_does_not_grab_letters():
    """Regression: \"orthogonal vectors\" must not capture 't' from
    \"vec(t)ors\" as a vector name."""
    g = semantic_parser.parse("They are orthogonal vectors")
    vecs = g.by_type("vector")
    bad = [v for v in vecs if v.params.get("name") in {"t", "to"}]
    assert not bad


def test_circle_shape_recognised():
    g = semantic_parser.parse("Draw a circle of radius r")
    shapes = g.by_type("shape")
    assert any(n.params.get("kind") == "circle" for n in shapes)


def test_relationship_edge_increases():
    g = semantic_parser.parse("Variance increases with model complexity")
    rels = {e.relation for e in g.edges}
    assert "increases_with" in rels


def test_relationship_edge_minimize():
    g = semantic_parser.parse("Minimize the loss over parameters")
    rels = {e.relation for e in g.edges}
    assert "minimizes" in rels


def test_dependency_edge():
    g = semantic_parser.parse("y depends on x")
    rels = {e.relation for e in g.edges}
    assert "depends_on" in rels


def test_operation_recognised():
    g = semantic_parser.parse("Compute the gradient of the function")
    ops = g.by_type("operation")
    assert ops and "gradient" in ops[0].label.lower()


def test_unknown_text_yields_empty_graph():
    g = semantic_parser.parse("aaa bbb ccc")
    assert g.is_empty()


def test_parse_meta_records_source_and_timing():
    g = semantic_parser.parse("quadratic function")
    assert g.meta["source"] == "regex"
    assert isinstance(g.meta["parse_ms"], float)
    assert g.meta["parse_ms"] >= 0


def test_parse_handles_empty_input():
    g = semantic_parser.parse("")
    assert g.is_empty()
    assert g.meta["source"] == "regex"


# ---------------------------------------------------------------------------
# SVG renderer
# ---------------------------------------------------------------------------

# Whitelisted SVG primitives — anything else means we slipped beyond the
# deterministic template set.  ``foreignObject`` + a single ``div`` are
# allowed because operation cards drop a KaTeX-targeted ``\[...\]`` block
# inside a foreignObject so the frontend's auto-render can compile the
# LaTeX into proper math glyphs (raw ``\sqrt{...}`` text would otherwise
# leak onto the chalkboard).
_ALLOWED_TAGS = {
    "g", "line", "circle", "rect", "text", "polyline", "ellipse",
    "foreignObject", "div",
}
_TAG_RE = re.compile(r"<\s*([a-zA-Z][\w\-]*)")


def _tags_in(svg: str) -> set[str]:
    return set(_TAG_RE.findall(svg))


def test_quadratic_renders_to_polyline_with_axes():
    g = semantic_parser.parse("The loss is a quadratic function")
    svg = semantic_to_svg.render_svg(g)
    assert "<polyline" in svg
    assert "<rect" in svg
    assert "x²" in svg or "f(x)" in svg


def test_svg_only_uses_allowed_primitives():
    cases = [
        "The loss is a quadratic function",
        "A is a 2x3 matrix",
        "A vector v points east",
        "Compute the gradient of the function",
        "Draw a circle of radius r",
        "y depends on x",
    ]
    for c in cases:
        g = semantic_parser.parse(c)
        svg = semantic_to_svg.render_svg(g)
        if not svg:
            continue
        tags = _tags_in(svg)
        unexpected = tags - _ALLOWED_TAGS
        assert not unexpected, f"unexpected tags {unexpected} in {c!r}"


def test_empty_graph_renders_empty_string():
    svg = semantic_to_svg.render_svg(SemanticGraph())
    assert svg == ""


def test_svg_renderer_is_pure_function():
    """No randomness — same graph must give byte-identical SVG."""
    g = semantic_parser.parse("The loss is a quadratic function")
    a = semantic_to_svg.render_svg(g)
    b = semantic_to_svg.render_svg(g)
    assert a == b


# ---------------------------------------------------------------------------
# LaTeX renderer
# ---------------------------------------------------------------------------

def test_quadratic_yields_canonical_latex():
    g = semantic_parser.parse("The loss is a quadratic function")
    out = semantic_to_latex.render_latex(g)
    assert any("x^{2}" in s for s in out)


def test_matrix_multiplication_renders_expansion():
    g = semantic_parser.parse(
        "Matrix multiplication combines rows and columns"
    )
    out = semantic_to_latex.render_latex(g)
    # The operations table emits the explicit expansion.
    assert any(r"\sum" in s and "AB" in s for s in out)


def test_inline_matrix_renders_bmatrix():
    g = semantic_parser.parse("Apply the matrix [[1, 2], [3, 4]] to v")
    out = semantic_to_latex.render_latex(g)
    assert any(s.startswith(r"\begin{bmatrix}") for s in out)
    assert any("1 & 2" in s for s in out)


def test_latex_renderer_dedupes_and_is_stable():
    g = semantic_parser.parse(
        "The sigmoid function. The sigmoid function again."
    )
    out_a = semantic_to_latex.render_latex(g)
    out_b = semantic_to_latex.render_latex(g)
    assert out_a == out_b
    # No exact duplicates: each LaTeX string appears at most once even
    # if the parser produced two nodes with the same canonical form.
    assert len(out_a) == len(set(out_a))


def test_empty_graph_yields_no_latex():
    assert semantic_to_latex.render_latex(SemanticGraph()) == []


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_unknown_input_signals_fallback():
    """Caller convention: an empty graph means the LLM fallback should
    be used.  This is the hook the orchestrator depends on."""
    g = semantic_parser.parse("zzqq nonsense token cluster 12345")
    assert g.is_empty()
    assert semantic_to_svg.render_svg(g) == ""
    assert semantic_to_latex.render_latex(g) == []


def test_known_input_does_not_signal_fallback():
    g = semantic_parser.parse("The loss is a quadratic function")
    assert not g.is_empty()


# ---------------------------------------------------------------------------
# Determinism end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The loss is a quadratic function",
    "The matrix [[1, 2], [3, 4]] is applied",
    "Compute the gradient of the function",
    "Draw a circle of radius r",
])
def test_pipeline_is_deterministic(text: str):
    g1 = semantic_parser.parse(text)
    g2 = semantic_parser.parse(text)
    # Graph contents (modulo timing meta) must match.
    assert g1.to_dict()["nodes"] == g2.to_dict()["nodes"]
    assert g1.to_dict()["edges"] == g2.to_dict()["edges"]
    assert semantic_to_svg.render_svg(g1) == semantic_to_svg.render_svg(g2)
    assert semantic_to_latex.render_latex(g1) == semantic_to_latex.render_latex(g2)


# ---------------------------------------------------------------------------
# Performance budgets
# ---------------------------------------------------------------------------

_PERF_CASES = [
    "The loss is a quadratic function",
    "A is a 2x3 matrix",
    "Compute the gradient of the function",
    "Draw a circle of radius r",
    "Vector v points east; matrix A acts on v",
]


def test_parser_under_20ms():
    for text in _PERF_CASES:
        # warm-up
        semantic_parser.parse(text)
        t0 = time.perf_counter()
        for _ in range(10):
            semantic_parser.parse(text)
        per_call = (time.perf_counter() - t0) / 10 * 1000
        assert per_call < 20.0, f"parse {per_call:.2f} ms for {text!r}"


def test_svg_render_under_10ms():
    graphs = [semantic_parser.parse(t) for t in _PERF_CASES]
    for g in graphs:
        # warm-up
        semantic_to_svg.render_svg(g)
        t0 = time.perf_counter()
        for _ in range(20):
            semantic_to_svg.render_svg(g)
        per_call = (time.perf_counter() - t0) / 20 * 1000
        assert per_call < 10.0, f"render {per_call:.2f} ms"


def test_latex_render_under_5ms():
    graphs = [semantic_parser.parse(t) for t in _PERF_CASES]
    for g in graphs:
        semantic_to_latex.render_latex(g)
        t0 = time.perf_counter()
        for _ in range(20):
            semantic_to_latex.render_latex(g)
        per_call = (time.perf_counter() - t0) / 20 * 1000
        assert per_call < 5.0, f"latex {per_call:.2f} ms"


# ---------------------------------------------------------------------------
# IR helpers
# ---------------------------------------------------------------------------

def test_graph_dedupes_nodes_by_id():
    g = SemanticGraph()
    g.add_node(SemanticNode(id="a", type="scalar", label="A", params={"x": 1}))
    g.add_node(SemanticNode(id="a", type="scalar", label="A2", params={"y": 2}))
    assert len(g.nodes) == 1
    # Last write merges params and replaces label.
    only = g.nodes[0]
    assert only.params == {"x": 1, "y": 2}
    assert only.label == "A2"


def test_graph_dedupes_edges():
    g = SemanticGraph()
    g.add_edge(SemanticEdge(source="a", target="b", relation="depends_on"))
    g.add_edge(SemanticEdge(source="a", target="b", relation="depends_on"))
    assert len(g.edges) == 1


def test_graph_to_dict_round_trip():
    g = SemanticGraph()
    g.add_node(SemanticNode(id="x", type="scalar", label="x"))
    g.add_node(SemanticNode(id="y", type="scalar", label="y"))
    g.add_edge(SemanticEdge(source="x", target="y", relation="depends_on"))
    d = g.to_dict()
    assert d["nodes"][0]["id"] == "x"
    assert d["edges"][0]["relation"] == "depends_on"


def test_node_ids_are_stable_across_parses():
    """Stable IDs are the contract that lets the frontend diff the DOM
    on incremental updates (Step 7)."""
    a = semantic_parser.parse("The loss is a quadratic function")
    b = semantic_parser.parse("The loss is a quadratic function")
    assert [n.id for n in a.nodes] == [n.id for n in b.nodes]
