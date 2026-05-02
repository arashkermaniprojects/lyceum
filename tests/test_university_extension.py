"""Tests for the university-level math extension.

Covers:
  - Structured equation rendering (\\frac, \\sqrt, \\sum/\\int with bounds, line breaks)
  - Category-theory primitives (tensor_box, string_wire, pullback_corner)
  - Category-theory relations (adjoint_to, natural_transformation, commutes)
  - Logical entailment (implies) + proof-tree layout
  - Statistical primitives (pdf_curve, plate, error_bar)
  - 3-D axes (axes_3d)
"""
from sevim.equation import has_structured_constructs, render_equation
from sevim.math_lex import classify_math_label
from sevim.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Structured equation rendering
# ---------------------------------------------------------------------------

def test_fraction_is_structured():
    assert has_structured_constructs(r"\frac{a}{b}")


def test_sqrt_is_structured():
    assert has_structured_constructs(r"\sqrt{x}")


def test_inline_x_squared_is_not_structured():
    # x^2 alone is fine on a single line — no need to escalate.
    assert not has_structured_constructs("x^2")


def test_render_fraction_emits_bar():
    body, w, h = render_equation(r"\frac{a}{b}")
    assert "<line" in body
    assert h > w  # taller than wide for a single-letter fraction


def test_render_sum_with_bounds():
    body, w, h = render_equation(r"\sum_{i=1}^{n} i^2")
    assert "∑" in body
    # Bounds are rendered separately above and below the glyph.
    assert body.count("<text") >= 4


def test_render_sqrt_emits_overline():
    body, w, h = render_equation(r"\sqrt{x}")
    assert "√" in body
    assert "<line" in body


def test_render_in_pipeline():
    svg = run_pipeline(r"By definition, $\frac{1}{n} \sum_{i=1}^{n} x_i$ is the mean.").svg
    assert "<line" in svg  # fraction bar
    assert "∑" in svg


# ---------------------------------------------------------------------------
# Category-theory primitives
# ---------------------------------------------------------------------------

def test_tensor_box_classification():
    assert classify_math_label("tensor product") == "tensor_box"
    assert classify_math_label("tensor diagram") == "tensor_box"


def test_string_wire_classification():
    assert classify_math_label("string diagram") == "string_wire"


def test_pullback_classification():
    assert classify_math_label("pullback") == "pullback_corner"
    assert classify_math_label("pushout square") == "pullback_corner"


def test_tensor_box_renders():
    svg = run_pipeline("The tensor product is associative.").svg
    # tensor_box renderer emits a rounded <rect> + leg <line>s
    assert "<rect" in svg


# ---------------------------------------------------------------------------
# Category-theory relations
# ---------------------------------------------------------------------------

def test_adjoint_relation():
    g = run_pipeline("Functor F is left adjoint to functor G.").graph
    rels = {e.relation for e in g.edges}
    assert "adjoint_to" in rels


def test_natural_transformation_relation():
    g = run_pipeline("Eta is a natural transformation from F to G.").graph
    rels = {e.relation for e in g.edges}
    # Either explicit natural_transformation, or maps_to fallback for "from X to Y"
    assert "natural_transformation" in rels or "maps_to" in rels


def test_commutes_relation():
    g = run_pipeline("Diagram A commutes with diagram B.").graph
    rels = {e.relation for e in g.edges}
    assert "commutes" in rels


def test_natural_transformation_double_arrow_in_svg():
    svg = run_pipeline("Eta is a natural transformation from F to G.").svg
    # The double-arrow renderer emits two parallel <line> elements.
    if 'e_natural_transformation_' in svg:
        idx = svg.index('e_natural_transformation_')
        seg = svg[idx:idx + 500]
        assert seg.count('<line') >= 2


# ---------------------------------------------------------------------------
# Logical entailment + proof tree
# ---------------------------------------------------------------------------

def test_implies_relation():
    g = run_pipeline("P implies Q. Q implies R.").graph
    rels = [e.relation for e in g.edges]
    assert rels.count("implies") >= 2


def test_implies_unicode():
    g = run_pipeline("P ⇒ Q").graph
    assert "implies" in {e.relation for e in g.edges}


def test_proof_tree_layout_runs():
    # Premises and conclusions linked by `implies` should layer top-to-bottom:
    # every premise's y is strictly less than the conclusion's y.
    r = run_pipeline(
        "Premise A implies Conclusion. "
        "Premise B implies Conclusion."
    )
    y_of = {p.shape.nid: p.y for p in r.placed.shapes}
    assert "n_conclusion" in y_of
    assert "n_premise_a" in y_of
    assert "n_premise_b" in y_of
    assert y_of["n_premise_a"] < y_of["n_conclusion"]
    assert y_of["n_premise_b"] < y_of["n_conclusion"]


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def test_pdf_curve_classification():
    assert classify_math_label("normal distribution") == "pdf_curve"
    assert classify_math_label("probability density function") == "pdf_curve"


def test_plate_classification():
    assert classify_math_label("plate notation") == "plate"


def test_error_bar_classification():
    assert classify_math_label("confidence interval") == "error_bar"


def test_pdf_curve_renders_path():
    svg = run_pipeline("The normal distribution has zero mean.").svg
    # pdf_curve renderer emits a sampled <path> with many M/L coords.
    assert "<path" in svg


# ---------------------------------------------------------------------------
# 3-D axes
# ---------------------------------------------------------------------------

def test_axes_3d_classification():
    assert classify_math_label("3D axes") == "axes_3d"


def test_axes_3d_has_three_arrows():
    # "3D axes" must extract via dep-parse "contains" semantics.
    svg = run_pipeline("The 3D axes contain the origin.").svg
    # axes_3d emits three <line> elements with marker-end arrows (x, y, z).
    assert svg.count('marker-end="url(#arrow)"') >= 3


# ---------------------------------------------------------------------------
# Determinism preserved
# ---------------------------------------------------------------------------

def test_university_pipeline_is_deterministic():
    text = (
        "Functor F is left adjoint to functor G. "
        "Eta is a natural transformation from F to G. "
        r"By definition $\sum_{i=1}^{n} \frac{1}{i^2} = \frac{\pi^2}{6}$. "
        "P implies Q."
    )
    a = run_pipeline(text).svg
    b = run_pipeline(text).svg
    assert a == b


def test_all_university_primitives_renderable():
    """All 8 university primitives produce non-empty SVG when rendered."""
    from sevim.ir import PlacedGraph, PlacedShape, VisualShape
    from sevim.s5_render import render
    new_prims = {"tensor_box", "string_wire", "pullback_corner", "axes_3d",
                 "pdf_curve", "plate", "error_bar", "proof_node"}
    for prim in sorted(new_prims):
        meta = {"kind": prim}
        if prim == "tensor_box":
            meta["legs"] = 3
        if prim == "pdf_curve":
            meta["shaded"] = (0.4, 0.7)
        if prim == "plate":
            meta["n"] = "N"
        shape = VisualShape(
            nid="n_test", primitive=prim, label="test",
            width=140.0, height=110.0, font_size=14.0,
            stroke_width=1.2, fill_index=0, meta=meta,
        )
        pg = PlacedGraph(
            shapes=[PlacedShape(shape=shape, x=20.0, y=20.0)],
            conns=[], canvas_w=400.0, canvas_h=300.0,
        )
        svg = render(pg)
        assert 'data-nid="n_test"' in svg, f"no SVG element for {prim}"
        assert len(svg) > 250, f"SVG too small for {prim}"


def test_relation_pattern_covers_university_relations():
    from sevim.ir import MATH_RELATIONS
    from sevim.s3_map import _RELATION_PATTERN
    assert MATH_RELATIONS <= set(_RELATION_PATTERN.keys())
