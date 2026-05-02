"""Pin: ``_sanitize_for_narration`` rewrites LaTeX as spoken English.

The TTS synthesises whatever text we hand it.  If the LLM emits raw
``\\sum_{i=1}^{N} x_i``, Kokoro will read it as 'sigma i 1 N x i' —
unintelligible for a student listening in audio.  These tests pin that
the sanitizer always converts mathematical notation into the form the
human ear expects ('the sum from i = 1 to N of x sub i').
"""
from __future__ import annotations

from narrator.qa import _sanitize_for_narration as san


def test_sum_with_full_bounds():
    out = san(r"Consider \sum_{i=1}^{N} x_i for the loss.")
    low = out.lower()
    assert "sum from i=1 to n" in low or "sum from i = 1 to n" in low, out
    assert "x sub i" in low, out
    # No raw LaTeX commands or braces leak through.
    assert "\\" not in out
    assert "{" not in out and "}" not in out


def test_integral_with_bounds():
    out = san(r"\int_a^b f(x) dx").lower()
    assert "integral from a to b" in out, out


def test_product_with_bounds():
    out = san(r"\prod_{k=1}^{K} p_k").lower()
    assert "product from k=1 to k" in out or "product from k = 1 to k" in out, out


def test_bare_sum_says_sum_of():
    out = san(r"\sum x_i").lower()
    assert "sum of" in out, out


def test_frac_becomes_over():
    out = san(r"the gradient is \frac{\partial L}{\partial w}").lower()
    assert "partial" in out
    assert "over" in out, out


def test_sqrt_becomes_square_root_of():
    out = san(r"\sqrt{x^2 + y^2}").lower()
    assert "square root of" in out, out
    # Inner powers also get verbalized.
    assert "squared" in out, out


def test_hat_becomes_y_hat():
    out = san(r"the prediction \hat{y} is").lower()
    assert "y hat" in out, out


def test_bar_becomes_x_bar():
    out = san(r"the mean \bar{x}").lower()
    assert "x bar" in out, out


def test_x_squared_and_cubed():
    assert "x squared" in san("x^2").lower()
    assert "x cubed" in san("x^3").lower()
    assert "x to the 4" in san("x^4").lower()
    assert "x inverse" in san(r"x^{-1}").lower()
    assert "x transpose" in san(r"x^T").lower()


def test_norm_and_squared_norm():
    out = san(r"the squared norm \|x\|^2 of x").lower()
    assert "squared norm" in out, out
    out2 = san(r"the norm \|w\|").lower()
    assert "norm of w" in out2, out2


def test_greek_and_operators_still_work():
    out = san(r"\alpha + \beta = \gamma").lower()
    assert "alpha" in out and "beta" in out and "gamma" in out


def test_no_backslashes_or_braces_leak():
    """Catch-all: after sanitization, the spoken stream never carries
    raw LaTeX delimiters that would make Kokoro pronounce 'backslash'."""
    samples = [
        r"\sum_{i=1}^N \alpha_i x_i + \beta",
        r"\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2",
        r"\int_0^\infty e^{-t} dt",
        r"\nabla f(x) = 0",
        r"\mathbb{E}[X] = \mu",
    ]
    for raw in samples:
        out = san(raw)
        assert "\\" not in out, f"backslash leaked in {out!r}"
        assert "{" not in out and "}" not in out, (
            f"braces leaked in {out!r}"
        )


def test_complex_loss_function_round_trip():
    """The §5.42 case: kernel ridge / RKHS objective."""
    raw = (
        r"\min_{f \in H} \sum_{i=1}^N L(y_i, f(x_i)) + \lambda J(f)"
    )
    out = san(raw).lower()
    # Bound forms verbalized.
    assert "sum from i" in out, out
    # Greek lambda spelled out.
    assert "lambda" in out, out
    # Subscripts read as "sub".
    assert "y sub i" in out and "x sub i" in out, out
