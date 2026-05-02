"""Math vocabulary, label classification, and LaTeX-to-Unicode utilities.

This module is the single source of truth for math-specific token recognition.
Other stages (s2_extract, s3_map, s5_render) delegate here so the math
vocabulary can be tuned without touching pipeline plumbing.

Three layers:

1.  Label classification — `classify_math_label(label)` returns one of the 13
    math `Primitive` literals or None.  Driven by ordered keyword regexes that
    mirror Inter-GPS / PGDP5K / OntoMathPRO vocabularies.

2.  LaTeX detection / extraction — `find_latex_spans(text)` finds inline math
    delimited by `$ … $`, `\\( … \\)`, `\\[ … \\]`, or `$$ … $$`.

3.  LaTeX → Unicode — `latex_to_unicode(src)` converts a small but useful
    subset of LaTeX (Greek letters, common operators, super/subscripts, common
    symbols) to inline Unicode for fast SVG `<text>` rendering.  It does NOT
    attempt to render fractions, roots, or matrices — those are handled by
    higher-level renderers via `parse_matrix_literal` and the equation block.

The conversion table is the same one used by `unicode-math-table.tex`, scoped
to the symbols that appear in K-12 → lower-undergrad math.  When a richer
renderer (MathJax) is needed, it can drop in by replacing `latex_to_unicode`.
"""
from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Label classification — label string → math primitive kind (or None)
# ---------------------------------------------------------------------------
# Ordered: most specific first. First match wins.
_MATH_LABEL_RULES: list[tuple[re.Pattern[str], str]] = [
    # Matrices & arrays (must come before "vector" because "row vector" exists)
    (re.compile(r"\b(matri(?:x|ces)|array|tableau|table\s+of\s+values)\b",
                re.IGNORECASE), "matrix_bracket"),

    # Equation / formula blocks (caught here when the *label* itself names one)
    (re.compile(r"\b(equation|formula|identity|expression|congruence|inequality)\b",
                re.IGNORECASE), "equation_block"),

    # 3-D coordinate systems — must run before plain "axes" rule.
    # Match both plural ("axes") and singular ("axis") because s2_extract's
    # `_normalize` singularises "axes" → "axis" before this rule sees the label.
    (re.compile(r"\b(3.?d.ax(?:e?s|is)|3.?d.coordinate|"
                r"three.dimensional.ax(?:e?s|is)|"
                r"xyz.ax(?:e?s|is)|spatial.ax(?:e?s|is))\b",
                re.IGNORECASE), "axes_3d"),

    # Coordinate axes / plots
    (re.compile(r"\b(axis|axes|x.axis|y.axis|z.axis|coordinate.system|"
                r"plot|graph\s+of|cartesian.plane|number.line)\b",
                re.IGNORECASE), "axes"),

    # Grids / lattices
    (re.compile(r"\b(grid|lattice|mesh|tessellation)\b",
                re.IGNORECASE), "grid"),

    # Sets — drawn as labeled blobs (Euler/Venn circles).
    (re.compile(
        r"\b(set|subset|superset|universe|universal.set|"
        r"empty.set|intersection|union|complement|"
        r"power.set|family\s+of\s+sets|class\s+of\s+sets|"
        r"venn\s+diagram|euler\s+diagram)\b",
        re.IGNORECASE), "set_blob"),

    # Geometric primitives — points first (most specific common case).
    (re.compile(r"\b(point|vertex|node|origin|midpoint|centroid|"
                r"intersection.point|fixed.point)\b",
                re.IGNORECASE), "point"),

    # Circles BEFORE segment — "circle of radius 1" should classify as circle,
    # not segment (radius is a chord-like primitive of a circle).
    (re.compile(r"\b(circle|disk|disc|unit.circle)\b",
                re.IGNORECASE), "circle"),

    # Arcs (angle markers, partial circles)
    (re.compile(r"\b(arc|sector|angle|right.angle|acute.angle|obtuse.angle)\b",
                re.IGNORECASE), "arc"),

    # Lines / segments / rays
    (re.compile(r"\b(line.segment|segment|ray|chord|edge|"
                r"diagonal|diameter|radius|altitude|median|bisector|"
                r"transversal|tangent.line|secant)\b",
                re.IGNORECASE), "segment"),

    # Plain "line" as a geometric object (not a "line of code", "line of reasoning")
    (re.compile(r"^(?:the\s+|a\s+)?line(?:\s|$)", re.IGNORECASE), "segment"),

    # Curves / function plots
    (re.compile(r"\b(curve|parabola|hyperbola|sine\s+wave|cosine\s+wave|"
                r"function.graph|trajectory|spline|bezier)\b",
                re.IGNORECASE), "curve"),

    # Pullback / pushout corners — must run BEFORE polygon, or "pullback
    # square" / "pushout square" would fire the polygon rule on "square".
    (re.compile(r"\b(pullback|pushout|pullback.square|pushout.square|"
                r"fiber.product|fibre.product)\b",
                re.IGNORECASE), "pullback_corner"),

    # Polygons (triangle, square, rectangle-as-shape, polygon, n-gon)
    (re.compile(
        r"\b(triangle|right.triangle|isosceles|equilateral|scalene|"
        r"quadrilateral|rectangle|square|rhombus|parallelogram\s+\(shape\)|"
        r"trapezoid|trapezium|pentagon|hexagon\s+\(shape\)|heptagon|"
        r"octagon|n.gon|polygon|polytope)\b",
        re.IGNORECASE), "polygon"),

    # Vector — render as arrow (uses existing 'arrow' primitive)
    (re.compile(r"\b(vector|displacement|velocity\s+vector|force\s+vector|"
                r"position\s+vector|unit\s+vector|basis\s+vector)\b",
                re.IGNORECASE), "arrow"),

    # Decoration marks — small geometric annotations
    (re.compile(r"\b(tick.mark|right.angle.mark|congruence.hash|"
                r"congruence.mark|parallel.chevron)\b",
                re.IGNORECASE), "decoration_mark"),

    # Brackets / braces (under/over)
    (re.compile(r"\b(under.?brace|over.?brace|brace\s+notation|"
                r"set.builder.bracket)\b",
                re.IGNORECASE), "bracket_brace"),

    # — university-level primitives (other than ones moved above) —

    # Tensor diagrams (Penrose graphical notation)
    (re.compile(r"\b(tensor|tensor.product|tensor.network|tensor.diagram)\b",
                re.IGNORECASE), "tensor_box"),

    # String diagrams (monoidal categories)
    (re.compile(r"\b(string.diagram|string.wire|monoidal.diagram)\b",
                re.IGNORECASE), "string_wire"),

    # Probability / statistics primitives
    (re.compile(r"\b(probability.density|density.function|"
                r"normal.distribution|gaussian|bell.curve|pdf|"
                r"distribution.curve)\b",
                re.IGNORECASE), "pdf_curve"),
    (re.compile(r"\b(plate|plate.notation|plate.diagram)\b",
                re.IGNORECASE), "plate"),
    (re.compile(r"\b(error.bar|confidence.interval|standard.error)\b",
                re.IGNORECASE), "error_bar"),

    # Proof tree / natural deduction
    (re.compile(r"\b(premise|hypothesis|assumption|conclusion|"
                r"proof.step|inference.step)\b",
                re.IGNORECASE), "proof_node"),
]

# Math objects that should remain as plain `rect` nodes (graph-style boxes)
# but trigger math-aware *connector* defaults: groups, fields, rings, spaces.
# These are recognised by s3_map but kept as rect for now.
ALGEBRAIC_OBJECTS = re.compile(
    r"\b(group|ring|field|module|algebra|vector.space|"
    r"topological.space|metric.space|manifold|lie.group|"
    r"hilbert.space|banach.space|category|functor|natural.transformation|"
    r"morphism|homomorphism|isomorphism|automorphism|homeomorphism|"
    r"diffeomorphism)\b",
    re.IGNORECASE,
)


def classify_math_label(label: str) -> Optional[str]:
    """Return the math primitive name for *label*, or None if not recognised.

    Returned values are valid `Primitive` literals from `ir.py`.  Callers in
    s3_map.py treat None as "fall through to the existing concept-diagram
    rules", so unrecognised labels are unaffected.
    """
    for pattern, prim in _MATH_LABEL_RULES:
        if pattern.search(label):
            return prim
    return None


def is_algebraic_object(label: str) -> bool:
    """True when the label names a category-theoretic / algebraic object.

    These are not given a math-specific primitive (they render as `rect`) but
    their *outgoing edges* default to morphism-style connectors.
    """
    return bool(ALGEBRAIC_OBJECTS.search(label))


# ---------------------------------------------------------------------------
# LaTeX inline-math detection
# ---------------------------------------------------------------------------
# Matches any of: $...$, $$...$$, \( ... \), \[ ... \].  Non-greedy.
_LATEX_RE = re.compile(
    r"\$\$(?P<dd>.+?)\$\$"
    r"|\$(?P<single>[^$\n]+?)\$"
    r"|\\\((?P<paren>.+?)\\\)"
    r"|\\\[(?P<bracket>.+?)\\\]",
    re.DOTALL,
)


def find_latex_spans(text: str) -> list[tuple[int, int, str]]:
    """Return all inline-math regions as (start, end, body) triples.

    *body* is the LaTeX source without delimiters.  Spans are sorted by
    start offset and never overlap.
    """
    out: list[tuple[int, int, str]] = []
    for m in _LATEX_RE.finditer(text):
        body = (m.group("dd") or m.group("single")
                or m.group("paren") or m.group("bracket") or "")
        out.append((m.start(), m.end(), body.strip()))
    return out


# ---------------------------------------------------------------------------
# LaTeX → Unicode — small but useful subset
# ---------------------------------------------------------------------------
# Built from unicode-math-table.tex coverage of K-12 / lower-undergrad symbols.
# Order matters: longer keys first so e.g. \alpha matches before \alp.

_LATEX_GREEK: dict[str, str] = {
    # lowercase
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "ϑ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π",
    r"\varpi": "ϖ", r"\rho": "ρ", r"\varrho": "ϱ", r"\sigma": "σ",
    r"\varsigma": "ς", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
    r"\varphi": "ϕ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    # uppercase
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
    r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}

_LATEX_OPERATORS: dict[str, str] = {
    r"\sum": "∑", r"\prod": "∏", r"\coprod": "∐",
    r"\int": "∫", r"\iint": "∬", r"\iiint": "∭", r"\oint": "∮",
    r"\partial": "∂", r"\nabla": "∇", r"\infty": "∞",
    r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\circ": "∘", r"\bullet": "•", r"\star": "⋆", r"\ast": "∗",
    r"\sqrt": "√", r"\surd": "√",
    r"\to": "→", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\leftrightarrow": "↔", r"\Rightarrow": "⇒", r"\Leftarrow": "⇐",
    r"\Leftrightarrow": "⇔", r"\mapsto": "↦", r"\hookrightarrow": "↪",
    r"\uparrow": "↑", r"\downarrow": "↓",
}

_LATEX_RELATIONS: dict[str, str] = {
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\neq": "≠", r"\ne": "≠", r"\equiv": "≡", r"\sim": "∼",
    r"\simeq": "≃", r"\cong": "≅", r"\approx": "≈", r"\propto": "∝",
    r"\ll": "≪", r"\gg": "≫", r"\prec": "≺", r"\succ": "≻",
    r"\subset": "⊂", r"\supset": "⊃", r"\subseteq": "⊆", r"\supseteq": "⊇",
    r"\in": "∈", r"\notin": "∉", r"\ni": "∋",
    r"\cup": "∪", r"\cap": "∩", r"\setminus": "∖",
    r"\emptyset": "∅", r"\varnothing": "∅",
    r"\forall": "∀", r"\exists": "∃", r"\nexists": "∄",
    r"\land": "∧", r"\wedge": "∧", r"\lor": "∨", r"\vee": "∨",
    r"\neg": "¬", r"\lnot": "¬",
    r"\perp": "⊥", r"\parallel": "∥", r"\angle": "∠",
}

_LATEX_BLACKBOARD: dict[str, str] = {
    r"\mathbb{R}": "ℝ", r"\mathbb{Z}": "ℤ", r"\mathbb{N}": "ℕ",
    r"\mathbb{Q}": "ℚ", r"\mathbb{C}": "ℂ", r"\mathbb{P}": "ℙ",
    r"\mathbb{F}": "𝔽", r"\mathbb{H}": "ℍ",
}

_LATEX_TABLE: dict[str, str] = {
    **_LATEX_GREEK, **_LATEX_OPERATORS, **_LATEX_RELATIONS, **_LATEX_BLACKBOARD,
}

# Sorted longest-first so e.g. "\subseteq" matches before "\subset".
_LATEX_KEYS_SORTED = sorted(_LATEX_TABLE.keys(), key=len, reverse=True)
_LATEX_REPLACE_RE = re.compile(
    "|".join(re.escape(k) for k in _LATEX_KEYS_SORTED)
)

# Super/subscript single-char Unicode mappings.
_SUPERSCRIPT: dict[str, str] = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵",
    "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ",
}
_SUBSCRIPT: dict[str, str] = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅",
    "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
    "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ",
    "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
}


def _convert_run(table: dict[str, str], chars: str) -> str:
    """Map every char in *chars* through *table*, leaving unknowns unchanged."""
    return "".join(table.get(c, c) for c in chars)


def _expand_super_sub(s: str) -> str:
    """Replace x^{n}, x^n, x_{n}, x_n with Unicode superscripts/subscripts.

    Best-effort: characters without a Unicode form pass through as-is.
    """
    # Superscripts: ^{...} or ^X
    def _sup_brace(m: re.Match) -> str:
        return _convert_run(_SUPERSCRIPT, m.group(1))
    s = re.sub(r"\^\{([^{}]+)\}", _sup_brace, s)
    s = re.sub(r"\^(.)",
               lambda m: _SUPERSCRIPT.get(m.group(1), "^" + m.group(1)), s)

    # Subscripts: _{...} or _X
    def _sub_brace(m: re.Match) -> str:
        return _convert_run(_SUBSCRIPT, m.group(1))
    s = re.sub(r"_\{([^{}]+)\}", _sub_brace, s)
    s = re.sub(r"_(.)",
               lambda m: _SUBSCRIPT.get(m.group(1), "_" + m.group(1)), s)
    return s


def latex_to_unicode(src: str) -> str:
    """Convert a small subset of LaTeX math to inline Unicode.

    Covered:
      - Greek letters (lowercase + uppercase)
      - Common operators: ∑ ∏ ∫ ∞ × ÷ → ↦ ∂ √ …
      - Common relations: ≤ ≥ ≠ ≡ ≈ ⊂ ⊆ ∈ ∪ ∩ ∅ ∀ ∃ ∧ ∨ ¬ ⊥ ∥ …
      - Blackboard letters: ℝ ℤ ℕ ℚ ℂ
      - Single-character super/subscripts and braced runs

    Out of scope (return as-is, surrounding tokens stripped):
      - Fractions \\frac{a}{b}     → "a/b"   (rendered approximately)
      - Square roots \\sqrt{a}     → "√a"
      - Matrices, alignment envs, Bigg-style sizing, font commands beyond \\mathbb

    The output is suitable for `<text>` placement; for richer typesetting,
    swap in a MathJax-Node call at this seam.
    """
    if not src:
        return ""
    s = src

    # Special multi-arg commands first.
    # \frac{a}{b} → a/b
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
               lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", s)
    # \sqrt{a} → √a   (no proper overline — best effort)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}",
               lambda m: f"√{m.group(1).strip()}", s)
    # \mathbb{X} handled via _LATEX_BLACKBOARD if X ∈ {R,Z,N,Q,C,P,F,H}; for
    # any other letter, drop the wrapper.
    s = re.sub(r"\\mathbb\s*\{([A-Za-z])\}",
               lambda m: _LATEX_TABLE.get(f"\\mathbb{{{m.group(1)}}}",
                                          m.group(1)), s)
    # \text{...}, \mathrm{...} etc. — strip the wrapper, keep the body.
    s = re.sub(r"\\(?:text|mathrm|mathit|mathbf|mathsf|mathtt)\s*\{([^{}]*)\}",
               lambda m: m.group(1), s)

    # Token-table replace.
    s = _LATEX_REPLACE_RE.sub(lambda m: _LATEX_TABLE[m.group(0)], s)

    # Super/subscripts.
    s = _expand_super_sub(s)

    # Strip residual braces and single backslash-letter sequences we couldn't
    # translate (best-effort cleanup so output is at least readable).
    s = re.sub(r"\{|\}", "", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)

    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Matrix literal parsing
# ---------------------------------------------------------------------------
# Recognises:
#   \begin{pmatrix} a & b \\ c & d \end{pmatrix}
#   \begin{bmatrix} … \end{bmatrix}
#   \begin{matrix}  … \end{matrix}
#   [[a, b], [c, d]]    (Python-list literal, common in narration)
#   ((1,2,3),(4,5,6))   (parenthesised tuple-of-tuples)

_MATRIX_ENV_RE = re.compile(
    r"\\begin\{(?P<env>p|b|v|V|B)?matrix\}(?P<body>.*?)\\end\{(?P=env)?matrix\}",
    re.DOTALL,
)
_PYLIST_MATRIX_RE = re.compile(
    r"\[\s*(\[[^\[\]]+\](?:\s*,\s*\[[^\[\]]+\])*)\s*\]"
)
_TUPLE_MATRIX_RE = re.compile(
    r"\(\s*(\([^()]+\)(?:\s*,\s*\([^()]+\))*)\s*\)"
)


def parse_matrix_literal(src: str) -> Optional[dict]:
    """Parse a matrix literal from *src*. Returns dict with cells/nrows/ncols.

    Returns None if no matrix structure is found.  Cells are stripped strings.
    """
    if not src:
        return None
    m = _MATRIX_ENV_RE.search(src)
    if m:
        env = m.group("env") or ""
        body = m.group("body")
        delim = {"p": "paren", "b": "bracket", "v": "vbar",
                 "V": "double_vbar", "B": "brace"}.get(env, "bracket")
        rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
        cells = [[c.strip() for c in row.split("&")] for row in rows]
    else:
        # Python-list form: [[1,2],[3,4]]
        m = _PYLIST_MATRIX_RE.search(src)
        if m:
            inner = m.group(1)
            rows = re.findall(r"\[([^\[\]]+)\]", inner)
            cells = [[c.strip() for c in row.split(",")] for row in rows]
            delim = "bracket"
        else:
            # Parenthesised tuple form: ((1,2),(3,4))
            m = _TUPLE_MATRIX_RE.search(src)
            if not m:
                return None
            inner = m.group(1)
            rows = re.findall(r"\(([^()]+)\)", inner)
            cells = [[c.strip() for c in row.split(",")] for row in rows]
            delim = "paren"
    if not cells or not all(len(r) == len(cells[0]) for r in cells):
        return None
    return {
        "nrows": len(cells),
        "ncols": len(cells[0]),
        "cells": cells,
        "delim": delim,
    }
