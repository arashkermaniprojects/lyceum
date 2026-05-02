"""Deterministic LaTeX renderer for :class:`SemanticGraph`.

For every node-type the parser can emit, this module returns one
canonical LaTeX string.  The orchestrator pairs each entry with the
matching SVG card so the frontend gets both visual and symbolic forms.

No LLM is consulted here; mappings are pure-Python lookup tables.
Target budget: < 5 ms per graph (in practice well under 1 ms).
"""
from __future__ import annotations

import re

from .semantic_ir import SemanticGraph, SemanticNode


# ---------------------------------------------------------------------------
# Function-form → canonical LaTeX
# ---------------------------------------------------------------------------

_FUNCTION_LATEX = {
    "linear":      r"f(x) = a x + b",
    "quadratic":   r"f(x) = a x^{2} + b x + c",
    "parabola":    r"y = x^{2}",
    "cubic":       r"f(x) = a x^{3} + b x^{2} + c x + d",
    "polynomial":  r"f(x) = \sum_{k=0}^{n} a_k\, x^{k}",
    "exp":         r"f(x) = e^{x}",
    "log":         r"f(x) = \log x",
    "sin":         r"f(x) = \sin x",
    "cos":         r"f(x) = \cos x",
    "sigmoid":     r"\sigma(x) = \frac{1}{1 + e^{-x}}",
    "relu":        r"\mathrm{ReLU}(x) = \max(0, x)",
    "tanh":        r"\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}",
    "softmax":     r"\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}",
}

_SHAPE_LATEX = {
    "circle":    r"x^{2} + y^{2} = r^{2}",
    "ellipse":   r"\frac{x^{2}}{a^{2}} + \frac{y^{2}}{b^{2}} = 1",
    "line":      r"y = m x + b",
    "triangle":  r"\triangle ABC",
    "rectangle": r"\text{rectangle}",
    "square":    r"\text{square}",
    "polygon":   r"\text{polygon}",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_latex(graph: SemanticGraph) -> list[str]:
    """Return one LaTeX string per renderable node, in stable order.

    Order: equations first, then functions, vectors, matrices,
    operations, shapes, and finally label/optimization tags.  Within
    each group, nodes appear in the order they were added to the graph
    so the output is deterministic for any given parser run.
    """
    if graph.is_empty():
        return []
    out: list[str] = []

    for n in graph.by_type("equation"):
        out.append(_equation_latex(n))

    for n in graph.by_type("function"):
        out.append(_function_latex(n))

    for n in graph.by_type("vector"):
        out.append(_vector_latex(n))

    for n in graph.by_type("matrix"):
        out.append(_matrix_latex(n))

    for n in graph.by_type("operation"):
        out.append(_operation_latex(n))

    for n in graph.by_type("shape"):
        out.append(_shape_latex(n))

    for n in graph.by_type("scalar"):
        out.append(_scalar_latex(n, graph))

    for n in graph.by_type("label"):
        out.append(_label_latex(n))

    # Drop empty entries and dedupe while preserving order.
    seen: set[str] = set()
    final: list[str] = []
    for item in out:
        item = (item or "").strip()
        if not item:
            continue
        norm = re.sub(r"\s+", " ", item)
        if norm in seen:
            continue
        seen.add(norm)
        final.append(item)
    return final


# ---------------------------------------------------------------------------
# Per-type renderers
# ---------------------------------------------------------------------------

def _equation_latex(node: SemanticNode) -> str:
    lhs = _to_latex_token(str(node.params.get("lhs") or "").strip())
    rhs = _to_latex_token(str(node.params.get("rhs") or "").strip())
    if not lhs or not rhs:
        return ""
    return f"{lhs} = {rhs}"


def _function_latex(node: SemanticNode) -> str:
    form = node.params.get("form") or ""
    return _FUNCTION_LATEX.get(form, "")


def _vector_latex(node: SemanticNode) -> str:
    name = node.params.get("name") or node.label or "v"
    name = name.strip()
    if not name:
        return r"\vec{v}"
    if len(name) == 1:
        return rf"\vec{{{name}}}"
    return rf"\vec{{{name[0]}}}"


def _matrix_latex(node: SemanticNode) -> str:
    rows = node.params.get("rows") or []
    name = node.params.get("name") or ""
    if rows:
        body = " \\\\ ".join(" & ".join(str(c) for c in row) for row in rows)
        bracket = rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"
        if name:
            return f"{name} = {bracket}"
        return bracket
    nrows = node.params.get("nrows") or 2
    ncols = node.params.get("ncols") or 2
    cells = [
        " & ".join(rf"a_{{{i + 1}{j + 1}}}" for j in range(ncols))
        for i in range(nrows)
    ]
    body = " \\\\ ".join(cells)
    bracket = rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"
    if name:
        return f"{name} = {bracket}"
    return bracket


def _operation_latex(node: SemanticNode) -> str:
    return str(node.params.get("latex") or "").strip()


def _shape_latex(node: SemanticNode) -> str:
    kind = node.params.get("kind") or ""
    return _SHAPE_LATEX.get(kind, "")


def _scalar_latex(node: SemanticNode, g: SemanticGraph) -> str:
    name = node.params.get("name") or node.label
    if not name:
        return ""
    # If the scalar participates in a depends_on edge, emit f-of form.
    for e in g.edges:
        if e.source == node.id and e.relation == "depends_on":
            tgt = g.node(e.target)
            tname = (tgt.params.get("name") if tgt else "") or (tgt.label if tgt else "x")
            return rf"{name} = f({tname})"
    return rf"{name}"


def _label_latex(node: SemanticNode) -> str:
    role = node.params.get("role") or ""
    label = node.label or node.id
    if role == "minimized":
        return rf"\min\; \mathrm{{{_text_safe(label)}}}"
    if role == "maximized":
        return rf"\max\; \mathrm{{{_text_safe(label)}}}"
    return rf"\mathrm{{{_text_safe(label)}}}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Greek letters that appear in PDF-extracted equations as Unicode glyphs.
_UNICODE_TO_LATEX = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "λ": r"\lambda", "μ": r"\mu", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "φ": r"\varphi", "ω": r"\omega",
    "Σ": r"\Sigma", "Δ": r"\Delta", "Π": r"\Pi", "Ω": r"\Omega",
    "·": r"\cdot", "×": r"\times", "÷": r"\div",
    "≤": r"\le", "≥": r"\ge", "≠": r"\ne", "≈": r"\approx",
    "→": r"\to", "∈": r"\in", "∑": r"\sum", "∏": r"\prod",
    "∫": r"\int", "∂": r"\partial", "∇": r"\nabla", "√": r"\sqrt",
    "∞": r"\infty", "±": r"\pm",
}


def _to_latex_token(s: str) -> str:
    """Best-effort conversion of a captured equation token to LaTeX."""
    if not s:
        return ""
    out: list[str] = []
    for ch in s:
        out.append(_UNICODE_TO_LATEX.get(ch, ch))
    s = "".join(out)
    # Naive ``a^2`` / ``x_i`` handling: ASCII ``^`` and ``_`` are valid
    # LaTeX as-is, but make sure single-token operands are braced when
    # they're alphanumeric multi-char so KaTeX handles them.
    s = re.sub(r"\^([A-Za-z0-9]{2,})", r"^{\1}", s)
    s = re.sub(r"_([A-Za-z0-9]{2,})", r"_{\1}", s)
    # Wrap multi-char identifiers like "log", "sin" in \operatorname
    # only when followed by an opening parenthesis — keeps prose
    # "x = a + b" untouched.
    s = re.sub(r"\b(sin|cos|tan|log|ln|exp|max|min)\s*\(",
               lambda m: rf"\{m.group(1)}(", s)
    return s.strip()


def _text_safe(s: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9 _\-]", "", s).strip()
    safe = safe.replace(" ", r"\ ")
    return safe or "x"


__all__ = ["render_latex"]
