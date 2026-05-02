"""Regressions for the §3.5.2 PLS / Equation 3.47 dual misfire.

User reported the §3.5.2 Partial Least Squares passage rendering a
``Regularization path`` Tier-1 diagram (wrong topic) plus an
Equation 3.47 card whose KaTeX rendering was a wall of italicised
single-letter fragments.

Two independent fixes:
  * Drop the body-preview rescue from
    ``viz.registry.find_visualization`` — it was strictly more
    permissive than the main check and resurrected misfires whenever
    the passage body shared keywords with the wrong curated topic.
  * Detect garbled OCR-extracted equation text and render a plain
    "see book" card instead of feeding the fragments to KaTeX.
"""
from __future__ import annotations

from serve.orchestrator import _looks_garbled_equation
from viz.registry import find_visualization


# ---------------------------------------------------------------------------
# Fix 1 — body-preview rescue dropped
# ---------------------------------------------------------------------------

def test_pls_passage_does_not_match_regularization_path():
    """Title alone scores 0.64 (below 0.65 cutoff); the body mentions
    'ridge' and 'shrinks' so the old body-rescue path pushed cosine
    to 0.67 and let the misfire through.  With body-rescue removed,
    Tier-1 returns None for this passage."""
    body = (
        "80\n3. Linear Methods for Regression\nIndex\nShrinkage Factor\n"
        "FIGURE 3.17. Ridge regression shrinks the regression "
        "coefficients of the principal components, using shrinkage "
        "factors d2 j / (d2 j + lambda) as in (3.47)."
    )
    result = find_visualization(
        title="Partial Least Squares", body_preview=body,
    )
    assert result is None


def test_query_alone_still_works_for_strong_matches():
    """The body-rescue removal should not affect queries that already
    pass the main threshold."""
    cases = [
        ("what is overfitting",         "overfitting"),
        ("what is the ROC curve",       "roc_curve"),
        ("what is k-fold cross validation", "kfold_split"),
    ]
    for q, expected in cases:
        m = find_visualization(question=q)
        assert m is not None and m[0] == expected, \
            f"{q!r} → {m and m[0]!r} (expected {expected!r})"


# ---------------------------------------------------------------------------
# Fix 2 — garbled equation detector
# ---------------------------------------------------------------------------

def test_garbled_detector_catches_eslii_fragmented_equation():
    """The β^ridge equation extracted from ESLII p.66 area gets
    flattened into one-token-per-line garbage."""
    text = "j=1\nuj\nd2\nj\nd2\nj + λuT\njy,"
    assert _looks_garbled_equation(text)


def test_garbled_detector_keeps_real_equations():
    samples = [
        "y = X beta + epsilon",
        "MSE = (1/n) sum_i (y_i - hat y_i)^2",
        "f(x) = sigma(W x + b)",
        # Multi-line but each line is substantive math:
        "x_1^2 + x_2^2 = r^2\nx = r cos theta\ny = r sin theta",
    ]
    for s in samples:
        assert not _looks_garbled_equation(s), f"falsely flagged: {s!r}"
