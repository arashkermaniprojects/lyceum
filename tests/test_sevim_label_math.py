"""SeVim concept-diagram label → KaTeX-renderable LaTeX (inspector A12).

Reproduces the bug where SeVim renders inner concept-card labels as
plain Unicode (``α0m``, ``αTmX``, ``Zm``) instead of typeset math, and
validates the new ``_label_to_math_html`` / ``_label_to_latex`` repair
in ``sevim.s5_render``.

Symptom in the field: SeVim diagram for ESLII Equation 11.5
``Z_m = \\sigma(\\alpha_{0m} + \\alpha_m^T X)`` shows the green
formula banner correctly typeset (KaTeX), but the inner concept
cards render raw glyphs because the diagram pipeline bypasses the
math-graph's ``serve.refcontent.to_latex`` repair stack.

Fix path: route node labels through ``_label_to_math_html`` before
SVG emission and emit ``<foreignObject>`` carrying the
``math-prose`` class so the frontend's ``runMathAutoRender`` runs
KaTeX over the ``\\(...\\)`` segments.
"""
from sevim.ir import (
    PlacedGraph,
    PlacedShape,
    SceneEdge,
    SceneNode,
    VisualShape,
)
from sevim.s5_render import (
    _label_to_latex,
    _label_to_math_html,
    _looks_like_math,
    render,
)


# ---------------------------------------------------------------------------
# _label_to_latex — Unicode → LaTeX repair
# ---------------------------------------------------------------------------

def test_glued_greek_subscript_becomes_latex():
    """``α0m`` → ``\\alpha_{0m}`` (PyMuPDF/LLM strip the ``_``)."""
    assert _label_to_latex("α0m") == r"\alpha_{0m}"


def test_glued_transpose_subscript_becomes_latex():
    """``αTmX`` → ``\\alpha_{m}^{T} X`` (transpose flattening)."""
    out = _label_to_latex("αTmX")
    assert out == r"\alpha_{m}^{T} X", out


def test_glued_uppercase_subscript_becomes_latex():
    """``Zm`` → ``Z_{m}`` (single-letter glued subscript)."""
    assert _label_to_latex("Zm") == "Z_{m}"


def test_mixed_prose_and_math_keeps_prose():
    """``bias term α0m`` → ``bias term \\alpha_{0m}`` (prose untouched)."""
    out = _label_to_latex("bias term α0m")
    assert out == r"bias term \alpha_{0m}", out


def test_short_english_words_are_not_subscripted():
    """``output of the m-th`` is pure prose; no ``th`` subscript."""
    out = _label_to_latex("output of the m-th")
    assert "_{th}" not in out
    # Should pass through untouched (or close to it).
    assert "output" in out and "m-th" in out


def test_unicode_subscript_glyph_becomes_latex_braces():
    """``xₘ`` (Unicode subscript) → ``x_{m}`` (LaTeX form)."""
    assert _label_to_latex("xₘ") == "x_{m}"


# ---------------------------------------------------------------------------
# _label_to_math_html — wraps math runs in ``\(...\)`` delimiters
# ---------------------------------------------------------------------------

def test_pure_prose_label_marks_no_math():
    html, has_math = _label_to_math_html("the m-th basis function")
    assert has_math is False
    assert html == "the m-th basis function"


def test_pure_math_label_wraps_in_inline_delimiters():
    html, has_math = _label_to_math_html("αTmX")
    assert has_math is True
    assert html == r"\(\alpha_{m}^{T} X\)"


def test_mixed_prose_math_wraps_only_math_run():
    html, has_math = _label_to_math_html("bias term α0m")
    assert has_math is True
    # Prose first, then ``\(...\)``-wrapped math.
    assert html == r"bias term \(\alpha_{0m}\)"


def test_looks_like_math_handles_greek_glyphs():
    assert _looks_like_math("α") is True
    assert _looks_like_math("Σ") is True
    assert _looks_like_math("xₘ") is True
    assert _looks_like_math("h_m") is True
    assert _looks_like_math("plain prose") is False


# ---------------------------------------------------------------------------
# End-to-end: a math-bearing label rides inside a foreignObject
# carrying the ``math-prose`` class so the frontend's KaTeX
# auto-render can compile it after injection.
# ---------------------------------------------------------------------------

def _placed_with_label(label: str) -> PlacedGraph:
    """Build a minimal ``PlacedGraph`` with a single ``rect`` shape."""
    vs = VisualShape(
        nid="n_test", primitive="rect", label=label,
        width=160.0, height=70.0, font_size=14.0, stroke_width=1.5,
        fill_index=0, meta={},
    )
    ps = PlacedShape(shape=vs, x=20.0, y=20.0)
    return PlacedGraph(shapes=[ps], conns=[], canvas_w=200.0, canvas_h=120.0)


def test_render_emits_math_prose_foreign_object_for_alpha_label():
    svg = render(_placed_with_label("αTmX"))
    assert "<foreignObject" in svg
    assert "math-prose" in svg
    # The LaTeX body lives inside ``\(...\)`` so the frontend's
    # ``runMathAutoRender`` (auto-render extension) finds and
    # compiles it.
    assert r"\(\alpha_{m}^{T} X\)" in svg


def test_render_keeps_plain_text_path_for_pure_prose():
    """Byte-determinism on existing relation tests requires that
    non-math labels stay on the cheap ``<text>`` path.
    """
    svg = render(_placed_with_label("forearm"))
    assert "<foreignObject" not in svg
    assert "<text " in svg
    assert ">forearm<" in svg


def test_render_mixed_label_uses_foreign_object():
    svg = render(_placed_with_label("bias term α0m"))
    assert "<foreignObject" in svg
    assert "math-prose" in svg
    assert r"\(\alpha_{0m}\)" in svg
    # Prose part must still appear before the math marker so KaTeX
    # auto-render only typesets the math run, not the whole label.
    assert svg.index("bias term") < svg.index(r"\(\alpha_{0m}\)")
