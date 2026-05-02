"""Regression tests for the pseudocode/prose rejection in equation
detection paths.

ESLII Algorithm 8.7 (Bagging) contains a step ``b = 1 to B: (a) Draw
a bootstrap sample Z* of size N from the training data``.  Two paths
were treating this as math:

  * ``viz.semantic_parser._extract_equations`` produced an equation
    node, which the deterministic Tier-3 renderer rendered as a giant
    overflowing text card.
  * ``serve.orchestrator._detect_math_fragments`` produced a "spoken
    formula" fragment, which the orchestrator passed to KaTeX.  KaTeX
    in math mode strips whitespace and concatenates every alphabetic
    token, producing a run-on like "b=1toB:(a)Drawabootstrapsample…".

Both paths now reject pseudocode-shaped text.
"""
from __future__ import annotations

from serve.orchestrator import _detect_math_fragments, _looks_like_pseudocode
from viz import semantic_parser


# ---------------------------------------------------------------------------
# semantic_parser path
# ---------------------------------------------------------------------------

def test_parser_rejects_bagging_pseudocode():
    text = (
        "b = 1 to B: (a) Draw a bootstrap sample Z* of size N from "
        "the training data."
    )
    g = semantic_parser.parse(text)
    eqs = g.by_type("equation")
    assert eqs == [], f"unexpected equation node: {eqs}"


def test_parser_keeps_real_equations():
    """Self-contained equations (terminated by punctuation) survive."""
    cases = [
        "z = a x + b.",                 # affine
        "y = x^2 + 2*x + 1.",           # quadratic
        "P(A) = 0.5;",                  # probability
    ]
    for text in cases:
        g = semantic_parser.parse(text)
        assert g.by_type("equation"), f"real equation rejected: {text!r}"


def test_parser_drops_prose_extension_after_equation():
    """``z = a x + b is the affine transform`` should be rejected
    rather than capturing the prose trail as part of the rhs.  The
    semantic_parser path is for clean math-first clauses; mixed
    prose+math goes through the orchestrator's inline-formula detector
    which clusters math tokens separately."""
    text = "z = a x + b is the affine transform"
    g = semantic_parser.parse(text)
    assert not g.by_type("equation")


def test_parser_rejects_random_pseudocode_steps():
    bad = [
        "for i = 1 to N do",
        "set x = max value of the array",
        "let y = sum of squared residuals",
    ]
    for text in bad:
        g = semantic_parser.parse(text)
        assert not g.by_type("equation"), \
            f"pseudocode passed: {text!r}"


# ---------------------------------------------------------------------------
# orchestrator inline-formula detector
# ---------------------------------------------------------------------------

def test_pseudocode_classifier_basic():
    assert _looks_like_pseudocode(
        "b = 1 to B: (a) Draw a bootstrap sample Z* of size N from "
        "the training data"
    )
    assert not _looks_like_pseudocode("z = W x + b")
    assert not _looks_like_pseudocode("E[X] = sum x p(x)")
    assert _looks_like_pseudocode("(a) sample x from the data set")


def test_inline_detector_rejects_bagging_step():
    text = (
        "Now consider the following algorithm. "
        "For b = 1 to B: (a) Draw a bootstrap sample Z* of size N from "
        "the training data."
    )
    frags = _detect_math_fragments(text)
    # Any returned fragment must NOT be pseudocode.
    for frag, _off in frags:
        assert not _looks_like_pseudocode(frag), (
            f"inline detector kept pseudocode: {frag!r}"
        )


def test_inline_detector_keeps_real_inline_formula():
    text = "We compute the loss z = W x + b. Other prose follows."
    frags = _detect_math_fragments(text)
    assert any("=" in f for f, _ in frags), \
        "real inline formula should still be detected"
