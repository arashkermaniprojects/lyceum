"""Tests for inline equation-citation + variable-definition extraction.

When the narrator says ``f(x) = a x + b, where b is the bias term, as
in Equation 5.42``, the formula card for ``f(x) = a x + b`` should
carry both the ``Equation 5.42`` citation and the ``b: the bias term``
definition — instead of emitting a separate reference card *and* a
formula card with no context.
"""
from __future__ import annotations

from serve.orchestrator import (
    _equation_citations_in,
    _variable_definitions_in,
    _formula_card_size,
    _formula_card_svg,
)


def test_citation_extraction_canonical_forms():
    cases = [
        ("As shown in Equation 5.42, we have ...",
         ["Equation 5.42"]),
        ("This is Theorem 3.2 of the book",
         ["Theorem 3.2"]),
        ("The result follows from (5.42).",
         ["Equation 5.42"]),
        # Multiple citations in one clause.
        ("By Equation 5.42 and Theorem 3.2, ...",
         ["Equation 5.42", "Theorem 3.2"]),
    ]
    for text, expected in cases:
        got = _equation_citations_in(text)
        assert got == expected, f"{text!r} → {got}, expected {expected}"


def test_citation_dedupes_repeated_label():
    text = "We use Equation 5.42, and again Equation 5.42 below."
    assert _equation_citations_in(text) == ["Equation 5.42"]


def test_variable_definition_extraction():
    cases = [
        ("z = W x + b, where b is the bias term.",
         [("b", "the bias term")]),
        ("f maps inputs to outputs, where f denotes the model.",
         [("f", "the model")]),
        # Two definitions in one clause.
        ("y = sigma(z), where z is the linear pre-activation "
         "and sigma denotes the activation function.",
         [("z", "the linear pre-activation"),
          ("sigma", "the activation function")]),
    ]
    for text, expected in cases:
        got = _variable_definitions_in(text)
        assert got == expected, (
            f"{text!r} → {got}, expected {expected}"
        )


def test_variable_definition_skips_unrelated_where_clauses():
    """Don't grab arbitrary 'where' patterns."""
    samples = [
        "We discuss the algorithm, where the third step adds noise.",
        "For models where parameters are sparse",
    ]
    for text in samples:
        got = _variable_definitions_in(text)
        # Either nothing extracted, or the symbol is at most a single
        # token and the definition is short — never a full sentence.
        assert all(
            len(sym) <= 8 and len(defn) <= 80
            for sym, defn in got
        ), f"unexpected greedy match: {got}"


def test_formula_card_grows_with_annotations():
    base_w, base_h = _formula_card_size("y = a x + b")
    # With citation tag — same height: chips render inline inside the
    # header band, not on their own row.  The card stays the same size.
    cited_w, cited_h = _formula_card_size(
        "y = a x + b", cite_labels=["Equation 5.42"],
    )
    assert cited_h == base_h
    # With three variable defs — height grows because each row is laid
    # out below the math.
    defs_w, defs_h = _formula_card_size(
        "y = a x + b",
        var_defs=[("a", "slope"), ("b", "intercept"), ("x", "input")],
    )
    assert defs_h > base_h
    # With a multi-line meaning string — height also grows.
    meaning_w, meaning_h = _formula_card_size(
        "y = a x + b",
        meaning=("a linear model in one variable, where the parameters "
                "a and b control slope and offset"),
    )
    assert meaning_h > base_h


def test_formula_card_svg_includes_citation_chip():
    body = _formula_card_svg(
        "y = a x + b", "y = a x + b",
        w=400, h=120,
        cite_labels=["Equation 5.42"],
        var_defs=[("b", "the bias term")],
    )
    assert "Equation 5.42" in body
    assert "b: the bias term" in body
