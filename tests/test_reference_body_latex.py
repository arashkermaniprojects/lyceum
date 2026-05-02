"""Pin the LLM-driven LaTeX rendering for reference-card bodies.

Exercise / Theorem / Lemma / etc. references used to render their
full body as plain monospace, mangling math fragments into runs of
text like ``f(x) =\\nN\\nX\\ni=1\\nαiK(x, xi).``.  After session
warm-up the orchestrator now caches an LLM-cleaned body with math
wrapped in ``\\(..\\)`` / ``\\[..\\]`` delimiters; the reference
card switches to a ``foreignObject`` carrying ``class="math-prose"``
so the frontend's auto-render compiles the math inline.
"""
from __future__ import annotations

from serve.eq_latex import _strip_wrappers, _PREAMBLE_RE
from serve.orchestrator import _render_reference_card
from serve.refcontent import RefContent


# ---------------------------------------------------------------------------
# Wrapper / preamble stripping
# ---------------------------------------------------------------------------

def test_strip_wrappers_removes_code_fences():
    raw = "```latex\nx = 1\n```"
    assert _strip_wrappers(raw) == "x = 1"


def test_strip_wrappers_removes_display_delimiters():
    raw = r"\[ y = m x + b \]"
    assert _strip_wrappers(raw) == "y = m x + b"


def test_strip_wrappers_removes_inline_delimiters():
    raw = r"\( a + b \)"
    assert _strip_wrappers(raw) == "a + b"


def test_strip_wrappers_removes_dollars():
    assert _strip_wrappers("$x + 1$") == "x + 1"
    assert _strip_wrappers("$$x + 1$$") == "x + 1"


def test_preamble_re_drops_here_is():
    text = "Here is the reformatted body. \\(x = 1\\)"
    cleaned = _PREAMBLE_RE.sub("", text).strip()
    assert cleaned.startswith(r"\(x = 1\)")


def test_preamble_re_leaves_body_starting_with_math():
    text = r"\[ f(x) = ax + b \] further explanation."
    cleaned = _PREAMBLE_RE.sub("", text).strip()
    assert cleaned == text


# ---------------------------------------------------------------------------
# Reference card rendering with prose_html
# ---------------------------------------------------------------------------

def test_reference_card_uses_foreign_object_with_prose_html():
    content = RefContent(
        kind="Exercise", ref_label="5.15",
        text="solution to (5.48) is finite-dimensional, and has the form "
             "f(x) = Σ αi K(x, xi).",
    )
    prose_html = (
        r"Exercise 5.15 shows that the solution is finite-dimensional and "
        r"has the form \[ f(x) = \sum_{i=1}^{N} \alpha_i K(x, x_i). \]"
    )
    rendered = _render_reference_card(content, prose_html=prose_html)
    assert rendered is not None
    body, w, h = rendered
    # foreignObject + math-prose class + LaTeX delimiters preserved.
    assert "<foreignObject" in body
    assert 'class="math-prose"' in body
    assert r"\[ f(x) = \sum_{i=1}^{N} \alpha_i K(x, x_i). \]" in body
    # Title and kind tag still present.
    assert "Exercise 5.15" in body
    # Card width should be the wide reading column, not the monospace size.
    assert w == 560.0


def test_reference_card_falls_back_to_monospace_without_cache():
    """When the body cache missed (LLM down or text too short),
    the reference card stays in the legacy monospace format."""
    content = RefContent(
        kind="Exercise", ref_label="5.15",
        text="solution to (5.48) is finite-dimensional",
    )
    rendered = _render_reference_card(content)
    assert rendered is not None
    body, _w, _h = rendered
    assert "math-prose" not in body
    assert "ui-monospace" in body
    assert "<foreignObject" not in body


def test_reference_card_prose_html_escapes_html_chars():
    """``<`` and ``>`` in the cleaned body must not break the SVG."""
    content = RefContent(kind="Theorem", ref_label="3.2",
                         text="some prose")
    prose_html = "if a < b and c > d then \\(x = 1\\)"
    rendered = _render_reference_card(content, prose_html=prose_html)
    assert rendered is not None
    body, _w, _h = rendered
    # Raw "<" / ">" must be escaped in the foreignObject content.
    # The KaTeX delimiters survive because they live behind backslashes.
    assert "a &lt; b" in body
    assert "c &gt; d" in body
    assert r"\(x = 1\)" in body
