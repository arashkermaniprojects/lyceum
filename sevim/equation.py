"""Structured equation typesetter — LaTeX subset → AST → SVG.

Used by `s5_render._render_equation` when the equation contains constructs
that don't fit on a single inline line: fractions, square roots, integrals
or sums with stacked bounds, multi-line aligned environments.

Pipeline
--------
    LaTeX source  ──tokenize──▶  Token stream
                  ──parse────▶   Box tree   (Atom, Frac, Sqrt, BigOp, Sup, Sub)
                  ──measure──▶   Each box gets (w, h, baseline)
                  ──render───▶   SVG <g> with absolute-positioned <text>/<line>

This is *not* a full TeX implementation — it covers the constructs that show
up in K-12 → undergraduate math:

    \\frac{a}{b}         vertical stacked numerator over denominator
    \\sqrt{x}            radical with overline
    \\sum_{lo}^{up}      Σ with bounds stacked above/below
    \\int_{lo}^{up}      ∫ with bounds stacked above/below (and \\prod, \\oint, …)
    x^{n}, x_{n}         super/subscripts on any atom
    \\\\ inside the source        line break (rendered as separate lines)
    plain math via latex_to_unicode for everything else (Greek, ≤, ∈, …)

Determinism: pure-Python, no measurement of real glyph metrics — widths come
from char_count × em_factor.  Identical inputs always produce identical SVG.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .math_lex import latex_to_unicode

# ---------------------------------------------------------------------------
# Approximate font metrics (em-based, no glyph table).
# ---------------------------------------------------------------------------
# Width per character at font_size = 1.  Italic serif averages ~0.55 em wide.
_CHAR_W = 0.55
# Vertical advance for a single line.
_LINE_H = 1.20
# How far the baseline sits below the top of the box.
_BASELINE = 0.80


# ---------------------------------------------------------------------------
# Box AST
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """One layout box.  Coordinates are filled in during the measure pass."""
    kind: str
    text: str = ""
    children: list["Box"] = field(default_factory=list)
    fs: float = 16.0       # font size in px
    w: float = 0.0
    h: float = 0.0
    baseline: float = 0.0  # offset from box top to baseline
    # Filled in by render pass:
    x: float = 0.0
    y: float = 0.0


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
# Token kinds: COMMAND, LBRACE, RBRACE, SUP, SUB, NEWLINE, ATOM (single char or run)

_TOKEN_RE = re.compile(
    r"\\\\"                              # explicit line break
    r"|\\[A-Za-z]+"                      # \command
    r"|\{|\}"                            # braces
    r"|\^|_"                             # super / sub
    r"|[A-Za-z](?:[A-Za-z]*)"            # identifier run
    r"|[0-9]+(?:\.[0-9]+)?"              # number
    r"|[+\-=*/<>≤≥≠≈≅∈∉⊂⊆∪∩→↦↔⇒⇔]"      # math operators (Unicode)
    r"|."                                # everything else, single char
)


def _tokenize(src: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(src) if t.strip() or t in (" ",)]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# LaTeX big-operator commands → the Unicode glyph used to render them.
_BIG_OPS = {
    r"\sum": "∑", r"\prod": "∏", r"\coprod": "∐",
    r"\int": "∫", r"\iint": "∬", r"\iiint": "∭", r"\oint": "∮",
    r"\bigcup": "⋃", r"\bigcap": "⋂",
    r"\bigvee": "⋁", r"\bigwedge": "⋀",
    r"\lim": "lim",
}


def _parse(tokens: list[str], i: int = 0,
           stop: Optional[set[str]] = None) -> tuple[list[Box], int]:
    """Parse tokens until end-of-stream or a stop token.  Returns (boxes, idx)."""
    out: list[Box] = []
    stop = stop or set()
    while i < len(tokens):
        tok = tokens[i]
        if tok in stop:
            return out, i

        if tok == "\\\\":
            out.append(Box(kind="newline"))
            i += 1
            continue

        if tok == r"\frac":
            num, i = _parse_brace_arg(tokens, i + 1)
            den, i = _parse_brace_arg(tokens, i)
            out.append(Box(kind="frac", children=[
                Box(kind="row", children=num),
                Box(kind="row", children=den),
            ]))
            continue

        if tok == r"\sqrt":
            arg, i = _parse_brace_arg(tokens, i + 1)
            out.append(Box(kind="sqrt", children=[
                Box(kind="row", children=arg),
            ]))
            continue

        if tok in _BIG_OPS:
            glyph = _BIG_OPS[tok]
            i += 1
            lower: list[Box] = []
            upper: list[Box] = []
            # Optional _ {…} and ^ {…} in either order.
            for _ in range(2):
                if i < len(tokens) and tokens[i] == "_":
                    lower, i = _parse_super_sub_arg(tokens, i + 1)
                elif i < len(tokens) and tokens[i] == "^":
                    upper, i = _parse_super_sub_arg(tokens, i + 1)
            out.append(Box(kind="bigop", text=glyph, children=[
                Box(kind="row", children=upper),
                Box(kind="row", children=lower),
            ]))
            continue

        if tok == "{":
            inner, i = _parse(tokens, i + 1, stop={"}"})
            if i < len(tokens):
                i += 1  # consume "}"
            out.append(Box(kind="row", children=inner))
            continue

        if tok in ("^", "_"):
            kind = "sup" if tok == "^" else "sub"
            arg, i = _parse_super_sub_arg(tokens, i + 1)
            base = out.pop() if out else Box(kind="atom", text="")
            out.append(Box(kind=kind, children=[
                base,
                Box(kind="row", children=arg),
            ]))
            continue

        # Plain atom — let math_lex turn LaTeX commands into Unicode glyphs.
        out.append(Box(kind="atom", text=latex_to_unicode(tok)))
        i += 1

    return out, i


def _parse_brace_arg(tokens: list[str], i: int) -> tuple[list[Box], int]:
    """Parse a single braced argument or one token if no braces."""
    if i >= len(tokens):
        return [], i
    if tokens[i] == "{":
        inner, j = _parse(tokens, i + 1, stop={"}"})
        if j < len(tokens):
            j += 1
        return inner, j
    inner, j = _parse([tokens[i]], 0)
    return inner, i + 1


def _parse_super_sub_arg(tokens: list[str], i: int) -> tuple[list[Box], int]:
    """Sup/sub argument: either a brace group, a \\command, or a single char."""
    return _parse_brace_arg(tokens, i)


# ---------------------------------------------------------------------------
# Measure pass — fill in (w, h, baseline) bottom-up.
# ---------------------------------------------------------------------------

def _measure(box: Box, fs: float) -> None:
    box.fs = fs
    if box.kind == "atom":
        box.w = max(0.0, _CHAR_W * fs * len(box.text))
        box.h = _LINE_H * fs
        box.baseline = _BASELINE * fs
        return

    if box.kind == "newline":
        box.w = 0.0
        box.h = _LINE_H * fs
        box.baseline = _BASELINE * fs
        return

    if box.kind == "row":
        for c in box.children:
            _measure(c, fs)
        # Row has line-break support: split into actual visual lines.
        lines = _split_rows(box.children)
        line_widths = [sum(c.w for c in line) for line in lines]
        line_heights = [max((c.h for c in line), default=_LINE_H * fs) for line in lines]
        box.w = max(line_widths) if line_widths else 0.0
        box.h = sum(line_heights) if line_heights else _LINE_H * fs
        # Baseline = first-line baseline.
        box.baseline = (max((c.baseline for c in lines[0]), default=_BASELINE * fs)
                        if lines else _BASELINE * fs)
        # Stash the line split for the renderer.
        box.children = [Box(kind="line", children=line, w=lw, h=lh,
                            baseline=max((c.baseline for c in line), default=_BASELINE * fs))
                        for line, lw, lh in zip(lines, line_widths, line_heights)]
        return

    if box.kind == "line":
        # Already measured in the row pass.
        return

    if box.kind == "frac":
        num, den = box.children
        _measure(num, fs * 0.92)
        _measure(den, fs * 0.92)
        box.w = max(num.w, den.w) + 4.0
        box.h = num.h + den.h + 4.0
        box.baseline = num.h + 2.0  # bar sits at the baseline-ish position
        return

    if box.kind == "sqrt":
        inner, = box.children
        _measure(inner, fs)
        # Add room for the radical sign (≈0.6 em wide) and the overline.
        box.w = inner.w + 0.7 * fs
        box.h = inner.h + 4.0
        box.baseline = inner.baseline + 4.0
        return

    if box.kind == "bigop":
        upper, lower = box.children
        _measure(upper, fs * 0.7)
        _measure(lower, fs * 0.7)
        glyph_w = _CHAR_W * fs * 1.6  # big operators are slightly wider
        glyph_h = _LINE_H * fs * 1.3
        box.w = max(glyph_w, upper.w, lower.w)
        box.h = upper.h + glyph_h + lower.h
        box.baseline = upper.h + glyph_h * 0.7
        return

    if box.kind in ("sup", "sub"):
        base, exp = box.children
        _measure(base, fs)
        _measure(exp, fs * 0.72)
        box.w = base.w + exp.w + 1.0
        if box.kind == "sup":
            box.h = base.h
            box.baseline = base.baseline
        else:
            box.h = base.h + exp.h * 0.4
            box.baseline = base.baseline
        return


def _split_rows(children: list[Box]) -> list[list[Box]]:
    """Split a row's children at every 'newline' box."""
    out: list[list[Box]] = [[]]
    for c in children:
        if c.kind == "newline":
            out.append([])
        else:
            out[-1].append(c)
    return [line for line in out if line]


# ---------------------------------------------------------------------------
# Render pass — emit SVG with absolute positions.
# ---------------------------------------------------------------------------

def _render(box: Box, x: float, y: float) -> str:
    """Return SVG fragment for *box* placed with top-left at (x, y)."""
    box.x, box.y = x, y
    parts: list[str] = []

    if box.kind == "atom":
        # Italic serif for variables, upright for digits/operators.
        is_var = bool(re.fullmatch(r"[A-Za-z][A-Za-z]*", box.text))
        style = ' font-style="italic"' if is_var else ""
        baseline_y = y + box.baseline
        parts.append(
            f'<text x="{x:g}" y="{baseline_y:g}" font-size="{box.fs:g}" '
            f'font-family="serif"{style} fill="#222">{_xmlesc(box.text)}</text>'
        )
        return "".join(parts)

    if box.kind == "newline":
        return ""

    if box.kind == "row":
        cy = y
        for line in box.children:
            cx = x
            for c in line.children:
                # Vertically align children on the row's baseline.
                child_top = cy + (line.baseline - c.baseline)
                parts.append(_render(c, cx, child_top))
                cx += c.w
            cy += line.h
        return "".join(parts)

    if box.kind == "frac":
        num, den = box.children
        bar_y = y + num.h + 2.0
        bar_x1 = x
        bar_x2 = x + box.w
        # Numerator centered.
        parts.append(_render(num, x + (box.w - num.w) / 2, y))
        # Denominator centered.
        parts.append(_render(den, x + (box.w - den.w) / 2, bar_y + 2.0))
        # Fraction bar.
        parts.append(
            f'<line x1="{bar_x1:g}" y1="{bar_y:g}" x2="{bar_x2:g}" y2="{bar_y:g}" '
            f'stroke="#222" stroke-width="1.0"/>'
        )
        return "".join(parts)

    if box.kind == "sqrt":
        inner, = box.children
        radical_w = 0.6 * box.fs
        inner_x = x + radical_w
        inner_y = y + 4.0
        # Radical sign: an unfilled "√" glyph followed by an overline.
        parts.append(
            f'<text x="{x:g}" y="{y + box.baseline:g}" font-size="{box.fs:g}" '
            f'font-family="serif" fill="#222">√</text>'
        )
        parts.append(
            f'<line x1="{inner_x - 1:g}" y1="{y + 1:g}" '
            f'x2="{inner_x + inner.w + 1:g}" y2="{y + 1:g}" '
            f'stroke="#222" stroke-width="1.0"/>'
        )
        parts.append(_render(inner, inner_x, inner_y))
        return "".join(parts)

    if box.kind == "bigop":
        upper, lower = box.children
        glyph_h = _LINE_H * box.fs * 1.3
        glyph_w = _CHAR_W * box.fs * 1.6
        glyph_x = x + (box.w - glyph_w) / 2
        glyph_y = y + upper.h + glyph_h * 0.78  # baseline of the big op glyph
        parts.append(_render(upper, x + (box.w - upper.w) / 2, y))
        parts.append(
            f'<text x="{glyph_x:g}" y="{glyph_y:g}" '
            f'font-size="{box.fs * 1.6:g}" font-family="serif" '
            f'fill="#222">{_xmlesc(box.text)}</text>'
        )
        parts.append(_render(lower, x + (box.w - lower.w) / 2,
                             y + upper.h + glyph_h))
        return "".join(parts)

    if box.kind == "sup":
        base, exp = box.children
        parts.append(_render(base, x, y))
        # Place the exponent above the base's mid-x-height.
        parts.append(_render(exp, x + base.w + 0.5, y - exp.h * 0.3))
        return "".join(parts)

    if box.kind == "sub":
        base, exp = box.children
        parts.append(_render(base, x, y))
        parts.append(_render(exp, x + base.w + 0.5,
                             y + base.h - exp.h * 0.6))
        return "".join(parts)

    return ""


def _xmlesc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def has_structured_constructs(latex: str) -> bool:
    """True when *latex* contains a construct that benefits from structured
    rendering (fractions, roots, sums/integrals with bounds, line breaks)."""
    return any(tok in latex
               for tok in (r"\frac", r"\sqrt", r"\\\\",
                           r"\sum_", r"\sum^",
                           r"\int_", r"\int^",
                           r"\prod_", r"\prod^",
                           r"\oint_", r"\oint^"))


def render_equation(
    latex: str, font_size: float = 16.0,
) -> tuple[str, float, float]:
    """Render *latex* to an SVG `<g>` fragment.

    Returns
    -------
    (body, width, height)
        ``body`` is the SVG string (without an outer wrapper); width and
        height are the bounding-box dimensions in pixels.  The body uses
        absolute coordinates starting at (0, 0) so callers can wrap it in
        a `<g transform="translate(…)">` to position the equation.
    """
    tokens = _tokenize(latex)
    boxes, _ = _parse(tokens)
    root = Box(kind="row", children=boxes)
    _measure(root, font_size)
    body = _render(root, 0.0, 0.0)
    return body, root.w, root.h
