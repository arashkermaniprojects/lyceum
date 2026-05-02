"""Pin the de-duplication and citation-visibility rules.

  * ``_normalize_formula_key`` collapses whitespace + delimiters so
    cross-source dedup catches the same equation in different forms.
  * ``_render_reference_card`` falls back to the "see book" placeholder
    when an Equation reference's OCR body is garbled.  Earlier we
    returned ``None`` (silently dropping the card), but the user
    reported "the function is mentioned but not shown" — the citation
    now always surfaces something on the board, even when the body
    can't be rendered.
"""
from __future__ import annotations

from serve.orchestrator import (
    _balanced_brackets,
    _looks_garbled_equation,
    _normalize_formula_key,
    _render_reference_card,
)
from serve.refcontent import RefContent


def test_normalize_collapses_whitespace_and_delimiters():
    assert _normalize_formula_key("f(x) = a x + b") == "f(x)=ax+b"
    assert _normalize_formula_key("f(x)=ax+b") == "f(x)=ax+b"
    assert _normalize_formula_key(r"\[ y = m x + b \]") == "y=mx+b"


def test_normalize_handles_empty():
    assert _normalize_formula_key("") == ""
    assert _normalize_formula_key("   ") == ""


def test_balanced_brackets():
    assert _balanced_brackets("f(x) + g(y)")
    assert _balanced_brackets("[a, b]")
    assert _balanced_brackets("{x | x > 0}")
    assert not _balanced_brackets("L(yi")
    assert not _balanced_brackets("f(x] + b")
    assert not _balanced_brackets(")")


def test_garbled_equation_renders_actual_body_as_text():
    """Garbled OCR equations skip KaTeX but still render the *actual*
    extracted body as monospace lines on the card.

    The user asked for "the actual functions from the book" — a
    "see book" stub is unwanted, even for messy OCR.
    """
    content = RefContent(
        kind="Equation", ref_label="3.47",
        text="j=1\nuj\nd2\nj\nd2\nj + λuT\njy,",
        latex="",
    )
    rendered = _render_reference_card(content)
    assert rendered is not None
    body, w, h = rendered
    assert "Equation 3.47" in body
    # Don't fall back to the "see book" stub.
    assert "see book" not in body
    # KaTeX foreignObject is skipped for garbled OCR — lines render
    # as plain monospace text instead.
    assert "foreignObject" not in body
    assert "ui-monospace" in body
    # The actual OCR'd content shows up in the card.
    assert "λuT" in body or "uj" in body
    assert w > 0 and h > 0


def test_garbled_equation_with_only_artefacts_falls_back_to_stub():
    """When stripping artefacts leaves nothing, we still emit a
    minimal card so the citation has a marker on the board."""
    content = RefContent(
        kind="Equation", ref_label="9.99",
        text='"\n#\n(9.99)',
        latex="",
    )
    rendered = _render_reference_card(content)
    assert rendered is not None
    body, _w, _h = rendered
    assert "Equation 9.99" in body


def test_clean_equation_card_still_renders():
    """Equations that survive the garbled detector still render."""
    content = RefContent(
        kind="Equation", ref_label="3.10",
        text="y = X beta + epsilon",
        latex="",
    )
    assert _looks_garbled_equation(content.text) is False
    rendered = _render_reference_card(content)
    assert rendered is not None
    body, w, h = rendered
    assert "Equation 3.10" in body
    assert w > 0 and h > 0


def test_figure_reference_unaffected_by_garbled_check():
    """Garbled-equation suppression only applies to Equation refs."""
    content = RefContent(
        kind="Section", ref_label="3.5",
        text="Methods using derived input directions",
        latex="",
    )
    rendered = _render_reference_card(content)
    assert rendered is not None
