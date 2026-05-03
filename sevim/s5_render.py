"""S5 Render — PlacedGraph → canonical SVG string.

Converts the geometry-complete PlacedGraph into well-formed SVG 1.1.  The
output is deterministic: identical inputs always produce byte-identical SVG
(invariant I1), because all coordinates and identifiers come from the placed
graph and all string concatenation is order-stable.

Shape primitives supported
--------------------------
  rect          — rounded rectangle (default; also used for containers)
  ellipse       — attribute nodes
  diamond       — numeric parameter / weight nodes
  hexagon       — neural-network architecture nodes (flat-top hexagon)
  parallelogram — layer / transform nodes

Connector styles
----------------
  arrow-directed   — cubic bezier + arrowhead  (causes, sequence, …)
  dashed           — dashed bezier  (used_for, instance_of)
  dotted-annot     — dotted line with '=' label  (measures)
  parallel-lines   — two offset lines + '≈' label  (similar_to)
  opposes          — bar at both ends  (opposes)
  prereq-arrow     — hollow triangle  (requires)
  funnel           — filled triangle  (reduces_to)

All node labels are XML-escaped; node IDs are embedded as data-nid attributes
for downstream JavaScript interactivity (I4 — bidirectional provenance).
"""
from __future__ import annotations

import re
from xml.sax.saxutils import escape

from .ir import MATH_PRIMITIVES, PlacedConn, PlacedGraph, PlacedShape
from .equation import has_structured_constructs, render_equation
from .math_lex import latex_to_unicode

PALETTE = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

# ---------------------------------------------------------------------------
# Label → KaTeX repair
# ---------------------------------------------------------------------------
# `tools/build_sevim_diagrams.py` typesets node labels at build time —
# Greek words become Unicode glyphs ("alpha" → "α") and ``x_m`` →
# ``xₘ``.  But the LLM (or PyMuPDF-extracted prose feeding the LLM)
# regularly emits subscripts WITHOUT an explicit underscore, so glyphs
# arrive glued: ``α0m`` (= α₀ₘ), ``Zm`` (= Zₘ), ``αTmX`` (= α_m^T X).
# These then render as raw Unicode in SVG ``<text>`` and the user
# sees ``"bias term α0m"`` instead of typeset math.
#
# Strategy used by `_label_to_math_html`:
#   1. Detect glued sub/transpose patterns and rewrite them into LaTeX
#      with proper ``_{...}`` / ``^{T}`` braces.
#   2. Convert Greek + math glyphs back into LaTeX commands so KaTeX
#      can typeset them.
#   3. Wrap the math fragment in ``\(...\)`` so the frontend's
#      ``runMathAutoRender`` (which already handles ``.math-prose``
#      blocks for reference cards) compiles it inline with prose.

# Greek lowercase Unicode glyphs we typeset back into LaTeX commands.
_GREEK_UNI_TO_LATEX: dict[str, str] = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\varphi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}

# Unicode subscript/superscript reverse map for the cases where
# `_typeset` already collapsed the underscore-form into the glyph.
_SUB_UNI_TO_CHAR: dict[str, str] = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ₐ": "a", "ₑ": "e", "ₕ": "h", "ᵢ": "i", "ⱼ": "j", "ₖ": "k",
    "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₒ": "o", "ₚ": "p", "ᵣ": "r",
    "ₛ": "s", "ₜ": "t", "ᵤ": "u", "ᵥ": "v", "ₓ": "x",
    "₊": "+", "₋": "-", "₌": "=",
}
_SUP_UNI_TO_CHAR: dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "ⁿ": "n",
    # Modifier-Letter Capital block — used for transpose ``ᵀ`` and
    # other one-letter superscripts that show up in typeset labels.
    "ᵀ": "T", "ᴬ": "A", "ᴮ": "B", "ᴰ": "D", "ᴱ": "E",
    "ᴳ": "G", "ᴴ": "H", "ᴵ": "I", "ᴶ": "J", "ᴷ": "K",
    "ᴸ": "L", "ᴹ": "M", "ᴺ": "N", "ᴼ": "O", "ᴾ": "P",
    "ᴿ": "R", "ᵁ": "U", "ⱽ": "V", "ᵂ": "W",
}

_GREEK_CHARS = "".join(_GREEK_UNI_TO_LATEX)
_SUB_CHARS = "".join(_SUB_UNI_TO_CHAR)
_SUP_CHARS = "".join(_SUP_UNI_TO_CHAR)

# Math-bearing characters: any glyph that signals "this is math, not prose".
# We DON'T include arithmetic operators ``+``/``-``/``/``/``*`` here
# because ``-`` appears in English ordinals (``m-th``) and would
# false-trigger.  ``=`` is included because it never appears in plain
# prose.
_MATH_HINT_RE = re.compile(
    rf"[{_GREEK_CHARS}{_SUB_CHARS}{_SUP_CHARS}=\^]"
    r"|[A-Za-z]_[A-Za-z0-9]"
    r"|\\[A-Za-z]"
)

# Tokens we treat as PART of a math run when they sit between two
# math-bearing tokens (so ``α0m + αTmX`` is one run, not two).
# Pure-prose fillers (``of``, ``the``…) DON'T extend the run.
_MATH_GLUE_RE = re.compile(r"^[+\-*/(),.|·×÷]+$")

# Operator names that look like glued ``head + tail`` but must not be
# subscripted (mirrors `serve.refcontent.to_latex`).
_OP_NAMES = {
    "log", "ln", "exp", "sin", "cos", "tan", "cot", "sec", "csc",
    "min", "max", "arg", "sup", "inf", "lim", "det", "tr", "diag",
    "var", "cov", "rank", "sgn", "sign", "mod",
}
# Short prose words that MAY look like glued sub patterns but are not.
# Lowercased.  Padded with the ones we observed misfire on real labels
# ("the m-th" → "th" got subscripted previously).
_PROSE_GLUE = {
    "is", "in", "on", "at", "to", "of", "an", "or", "no", "be",
    "as", "by", "we", "us", "my", "if", "do", "go", "so", "it",
    "th", "st", "nd", "rd",  # ordinals
    "the", "and", "for", "all", "any", "are", "can", "has", "had",
    "her", "his", "let", "may", "not", "now", "off", "one", "our",
    "out", "say", "see", "set", "she", "two", "who", "why", "you",
}


def _looks_like_math(label: str) -> bool:
    """Return True when *label* contains any math-typography hint."""
    if not label:
        return False
    return bool(_MATH_HINT_RE.search(label))


def _label_to_latex(label: str) -> str:
    """Convert a SeVim label string into KaTeX-renderable LaTeX source.

    Repairs the common PyMuPDF / LLM artefacts:
      * ``α0m``  → ``\\alpha_{0m}``
      * ``Zm``   → ``Z_{m}``
      * ``αTmX`` → ``\\alpha_{m}^{T} X``   (transpose-with-subscript)
      * Unicode subscript/superscript glyphs → ``_{…}`` / ``^{…}``

    Conservative: only applies the joining rules to tokens that don't
    look like English short words (see ``_PROSE_GLUE``) so labels like
    ``"output of the m-th"`` survive unchanged.
    """
    if not label:
        return ""
    s = label

    # Step 1 — Unicode sub/sup runs back into LaTeX braces.  Process
    # superscripts first then subscripts.  Run-collapsed: ``Z₂ ₃`` →
    # ``Z_{2 3}`` would be wrong, so each run goes into its own group.
    def _sup_run(m: "re.Match[str]") -> str:
        run = "".join(_SUP_UNI_TO_CHAR[c] for c in m.group(0))
        return f"^{{{run}}}"

    def _sub_run(m: "re.Match[str]") -> str:
        run = "".join(_SUB_UNI_TO_CHAR[c] for c in m.group(0))
        return f"_{{{run}}}"

    s = re.sub(rf"[{_SUP_CHARS}]+", _sup_run, s)
    s = re.sub(rf"[{_SUB_CHARS}]+", _sub_run, s)

    # Step 2 — Greek glyphs → LaTeX command + space (so the next
    # repair pass can spot ``\alpha 0m`` style joins).
    s = re.sub(
        rf"[{_GREEK_CHARS}]",
        lambda m: _GREEK_UNI_TO_LATEX[m.group(0)] + " ",
        s,
    )

    # Step 3 — transpose-with-subscript joined: ``\alpha TmX`` → ``\alpha_{m}^{T} X``.
    # Pattern: ``\command`` + space + uppercase ``T`` + lowercase index +
    # uppercase letter (the operand the transposed object multiplies).
    # The classic shape that PyMuPDF produces for ``\alpha_m^T X``.
    def _join_transpose(m: "re.Match[str]") -> str:
        head, sub, var = m.group(1), m.group(2), m.group(3)
        return f"{head}_{{{sub}}}^{{T}} {var}"

    s = re.sub(
        r"(\\[A-Za-z]+)\s+T([a-z])([A-Z])(?![A-Za-z])",
        _join_transpose, s,
    )
    # ASCII head version (no LaTeX command): ``ZT mX`` etc.  Rare but
    # symmetrical with the above.
    s = re.sub(
        r"(?<![A-Za-z\\])([A-Z])T([a-z])([A-Z])(?![A-Za-z])",
        _join_transpose, s,
    )

    # Step 4 — bare glued subscripts on a LaTeX-command head:
    # ``\alpha 0m`` → ``\alpha_{0m}``.  Require the tail to start with
    # a digit so we don't break ``\alpha x`` (two distinct variables).
    s = re.sub(
        r"(\\[A-Za-z]+)\s+(\d[0-9a-z]{0,2})(?![A-Za-z])",
        lambda m: f"{m.group(1)}_{{{m.group(2)}}}",
        s,
    )

    # Step 5 — bare glued subscripts on an ASCII-letter head: ``Zm`` →
    # ``Z_{m}``.  Single-letter head, 1-2 lowercase-letter or 1-3 digit
    # tail, no whitespace, no preceding letter or backslash.  Skips
    # operator names (``log``, ``min``…) and short English words.
    def _join_sub_ascii(m: "re.Match[str]") -> str:
        head, sub = m.group(1), m.group(2)
        whole = (head + sub).lower()
        if whole in _OP_NAMES:
            return "\\" + whole
        if whole in _PROSE_GLUE:
            return m.group(0)
        return f"{head}_{{{sub}}}"

    s = re.sub(
        r"(?<![A-Za-z\\])([A-Za-z])([a-z]{1,2}|\d{1,3})(?![A-Za-z])",
        _join_sub_ascii, s,
    )

    # Step 6 — collapse double whitespace introduced by the Greek pass.
    s = re.sub(r" {2,}", " ", s).strip()
    return s


def _label_to_math_html(label: str) -> tuple[str, bool]:
    """Return ``(html, has_math)`` for *label*.

    When *label* mixes prose and math (e.g. ``"bias term α0m"``), the
    math-bearing tokens are wrapped in ``\\(...\\)`` delimiters with
    LaTeX-converted bodies; the prose tokens stay as plain text.

    When *label* is all prose, returns ``(escape(label), False)`` so
    the caller can keep the existing flat ``<text>`` SVG path.

    The token classification splits on whitespace and tags each token
    as MATH (carries a math hint), GLUE (math operators / parens that
    don't count alone but glue adjacent math tokens), or PROSE.  Then
    consecutive MATH/GLUE runs collapse into one ``\\(...\\)`` block.
    """
    if not _looks_like_math(label):
        return escape(label), False

    # Tokenise on whitespace, preserving the leading whitespace before
    # each token so we can reconstruct the original spacing.
    tokens: list[tuple[str, str, str]] = []  # (kind, leading_ws, body)
    pos = 0
    for m in re.finditer(r"\S+", label):
        ws = label[pos:m.start()]
        body = m.group(0)
        if _MATH_HINT_RE.search(body):
            kind = "math"
        elif _MATH_GLUE_RE.match(body):
            kind = "glue"
        else:
            kind = "prose"
        tokens.append((kind, ws, body))
        pos = m.end()
    trailing_ws = label[pos:]

    # Sweep: any GLUE token wedged between two MATH tokens is promoted
    # to MATH.  This keeps ``α0m + αTmX`` as a single run.
    for i, (kind, ws, body) in enumerate(tokens):
        if kind != "glue":
            continue
        prev_math = i > 0 and tokens[i - 1][0] == "math"
        next_math = i + 1 < len(tokens) and tokens[i + 1][0] == "math"
        if prev_math and next_math:
            tokens[i] = ("math", ws, body)

    # Emit: collapse consecutive math tokens into one ``\(...\)`` block.
    parts: list[str] = []
    i = 0
    while i < len(tokens):
        kind, ws, body = tokens[i]
        if kind == "math":
            # Greedy run.
            run = [(ws, body)]
            j = i + 1
            while j < len(tokens) and tokens[j][0] == "math":
                run.append((tokens[j][1], tokens[j][2]))
                j += 1
            # Re-stitch the run with its internal whitespace.
            run_text = "".join(w + b for w, b in run)
            # Leading whitespace BEFORE the run (escaped) stays as
            # plain text; only the body goes inside ``\(...\)``.
            leading_ws = run[0][0]
            run_body = run_text[len(leading_ws):]
            parts.append(escape(leading_ws))
            parts.append(r"\(" + _label_to_latex(run_body) + r"\)")
            i = j
        else:
            parts.append(escape(ws + body))
            i += 1
    parts.append(escape(trailing_ws))
    html = "".join(parts).strip()
    if not html:
        return escape(label), False
    return html, True

# Which relations get a terminal arrow.
_ARROW_END = {
    "causes", "sequence", "instance_of",
    "used_for", "reduces_to",
    "transforms",
}
# Which relations use dashed / dotted strokes.
_DASHED = {"used_for", "instance_of"}
_DOTTED = {"measures"}

_DEFS = (
    '<defs>'
    '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    '<path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>'
    '<marker id="hollow-tri" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="10" markerHeight="10" orient="auto-start-reverse">'
    '<path d="M0,0 L10,5 L0,10 z" fill="#fff" stroke="#333" stroke-width="1.2"/></marker>'
    # Bar: solid vertical rectangle, rendered thick enough to be visible at
    # normal zoom. Used for the opposes relation at both line ends.
    '<marker id="bar" viewBox="0 0 10 10" refX="5" refY="5" '
    'markerWidth="12" markerHeight="16" orient="auto">'
    '<rect x="3" y="0" width="4" height="10" fill="#333"/></marker>'
    '</defs>'
)


def render(pg: PlacedGraph) -> str:
    """S5: serialise a PlacedGraph to a canonical SVG string.

    Renders connectors first (so shape bodies draw on top), then shapes sorted
    containers-first (so container backgrounds are behind their children).

    Parameters
    ----------
    pg:
        The fully-placed graph from S4.

    Returns
    -------
    str
        A well-formed SVG 1.1 string.  Byte-identical for identical inputs.
    """
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{pg.canvas_w:g}" height="{pg.canvas_h:g}" '
        f'viewBox="0 0 {pg.canvas_w:g} {pg.canvas_h:g}">',
        _DEFS,
    ]
    lookup = {p.shape.nid: p for p in pg.shapes}
    for pc in pg.conns:
        parts.append(_render_conn(pc, lookup))
    # Containers rendered first (larger, background), children on top.
    for ps in sorted(pg.shapes, key=lambda p: (0 if p.shape.is_container else 1,
                                               -p.shape.width * p.shape.height)):
        parts.append(_wrap_with_animation(ps, _render_shape(ps)))
    parts.append("</svg>")
    return "".join(parts)


def _animate_transform(ps: PlacedShape) -> str:
    """Build the SMIL ``<animateTransform>`` element for a PlacedShape.

    Reads ``ps.shape.meta["animate"]`` (see :class:`SceneNode.meta` docstring
    for the schema).  Returns the empty string when no animation is configured;
    otherwise returns a fully-formed SMIL element.

    The output is purely declarative: no random IDs, no timestamps, so it
    preserves invariant I1 (deterministic SVG bytes).
    """
    s = ps.shape
    spec = s.meta.get("animate") if isinstance(s.meta, dict) else None
    if not spec or not isinstance(spec, dict):
        return ""
    kind = spec.get("kind")
    if kind not in ("rotate", "translate", "scale"):
        return ""
    dur = str(spec.get("dur", "2s"))
    repeat = str(spec.get("repeat", "indefinite"))
    frm = spec.get("from", 0.0)
    to = spec.get("to", 0.0)

    if kind == "rotate":
        around = spec.get("around", "center")
        if around == "origin":
            cx, cy = 0.0, 0.0
        elif isinstance(around, (tuple, list)) and len(around) == 2:
            cx, cy = float(around[0]), float(around[1])
        else:  # "center" — pivot around the shape's centre
            cx = ps.x + s.width / 2.0
            cy = ps.y + s.height / 2.0
        from_attr = f"{float(frm):g} {cx:g} {cy:g}"
        to_attr = f"{float(to):g} {cx:g} {cy:g}"
    elif kind == "translate":
        # `from`/`to` may be a single scalar (x-translation) or an (x, y) pair.
        from_attr = _xy_pair(frm)
        to_attr = _xy_pair(to)
    else:  # scale
        from_attr = _xy_pair(frm, default_y=None)
        to_attr = _xy_pair(to, default_y=None)

    return (
        f'<animateTransform attributeName="transform" '
        f'attributeType="XML" type="{kind}" '
        f'from="{from_attr}" to="{to_attr}" '
        f'dur="{escape(dur)}" repeatCount="{escape(repeat)}"/>'
    )


def _xy_pair(val, default_y: float | None = 0.0) -> str:
    """Format a scalar or (x, y) pair as an SMIL ``from``/``to`` attribute.

    For ``translate`` we default the y-component to 0 when missing.  For
    ``scale`` we omit the y-component (uniform scale) when missing.
    """
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return f"{float(val[0]):g} {float(val[1]):g}"
    if default_y is None:
        return f"{float(val):g}"
    return f"{float(val):g} {float(default_y):g}"


def _wrap_with_animation(ps: PlacedShape, body: str) -> str:
    """Wrap *body* in a ``<g>`` carrying an ``<animateTransform>`` when the
    shape's meta requests animation.  Pass-through otherwise.
    """
    anim = _animate_transform(ps)
    if not anim:
        return body
    return f"<g>{body}{anim}</g>"


def _wrap_label(label: str, node_width: float = 160.0, font_size: float = 16.0) -> tuple[list[str], float]:
    """Wrap a label to fit node_width. Returns (lines, effective_font_size).

    Tries up to 3 lines at font_size; if still too many lines, reduces font
    size in steps until the label fits in 3 lines.
    """
    for fs in (font_size, font_size * 0.85, font_size * 0.70):
        chars_per_line = max(8, int(node_width / (fs * 0.60)))
        words = label.split()
        lines: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in words:
            add_len = (1 if current else 0) + len(word)
            if current and current_len + add_len > chars_per_line:
                lines.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += add_len
        if current:
            lines.append(" ".join(current))
        if len(lines) <= 3:
            return lines, fs
    # Fallback: cap at 3 lines, truncate last
    lines = lines[:3]
    return lines, font_size * 0.70


def _render_shape(ps: PlacedShape) -> str:
    s = ps.shape
    stroke = PALETTE[s.fill_index % len(PALETTE)]
    # Math primitives have their own renderers — dispatch first.
    if s.primitive in MATH_PRIMITIVES:
        return _render_math_shape(ps, stroke)
    lines, eff_fs = _wrap_label(s.label, node_width=s.width, font_size=s.font_size)
    if s.primitive == "ellipse":
        cx = ps.x + s.width / 2.0
        cy = ps.y + s.height / 2.0
        body = (
            f'<ellipse data-nid="{escape(s.nid)}" '
            f'cx="{cx:g}" cy="{cy:g}" rx="{s.width / 2:g}" ry="{s.height / 2:g}" '
            f'fill="#fff" stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        )
    elif s.primitive == "diamond":
        cx = ps.x + s.width / 2.0
        cy = ps.y + s.height / 2.0
        pts = (f"{cx:g},{ps.y:g} {ps.x + s.width:g},{cy:g} "
               f"{cx:g},{ps.y + s.height:g} {ps.x:g},{cy:g}")
        body = (
            f'<polygon data-nid="{escape(s.nid)}" points="{pts}" '
            f'fill="#fff" stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        )
    elif s.primitive == "hexagon":
        # Flat-top hexagon: 6 points, corner cut = w/4
        x, y, w, h = ps.x, ps.y, s.width, s.height
        q = w / 4.0
        pts = (f"{x+q:g},{y:g} {x+w-q:g},{y:g} {x+w:g},{y+h/2:g} "
               f"{x+w-q:g},{y+h:g} {x+q:g},{y+h:g} {x:g},{y+h/2:g}")
        body = (
            f'<polygon data-nid="{escape(s.nid)}" points="{pts}" '
            f'fill="#fff" stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        )
    elif s.primitive == "parallelogram":
        x, y, w, h = ps.x, ps.y, s.width, s.height
        skew = h * 0.25
        pts = (f"{x+skew:g},{y:g} {x+w:g},{y:g} "
               f"{x+w-skew:g},{y+h:g} {x:g},{y+h:g}")
        body = (
            f'<polygon data-nid="{escape(s.nid)}" points="{pts}" '
            f'fill="#fff" stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        )
    else:
        if s.is_container:
            fill = "#f5f5fa"
            extra = ' fill-opacity="0.55"'
        else:
            fill = "#fff"
            extra = ""
        body = (
            f'<rect data-nid="{escape(s.nid)}" '
            f'x="{ps.x:g}" y="{ps.y:g}" '
            f'width="{s.width:g}" height="{s.height:g}" rx="6" ry="6" '
            f'fill="{fill}"{extra} stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        )
    # Container label: italic small font at top-left, well clear of children.
    # Regular shapes: centred label.
    if s.is_container:
        container_fs = max(10.0, s.font_size * 0.75)
        label = _render_container_label(ps, container_fs)
        return body + label

    label = _render_inner_label(ps, lines, eff_fs)
    return body + label


def _render_container_label(ps: PlacedShape, container_fs: float) -> str:
    """Italic title at the top-left of a container.

    When the label contains math glyphs, route through a
    ``<foreignObject>`` carrying the ``math-prose`` class so KaTeX
    typesets it AND HTML word-wrapping prevents the title from
    clipping past the container's right edge.  Pure prose stays on
    the cheap ``<text>`` path so deterministic-bytes-out tests
    keep their reference signatures.
    """
    s = ps.shape
    html, has_math = _label_to_math_html(s.label)
    # Always use foreignObject for container labels — this is the
    # cheapest fix for the "long title clips at the box edge"
    # problem because HTML naturally word-wraps to width.
    label_x = ps.x + 8
    label_y = ps.y + 4
    fo_w = max(40.0, s.width - 16.0)
    # Two-line cap so the title stays compact even on tiny containers.
    fo_h = max(container_fs * 2.4, 30.0)
    css_class = "sevim-label sevim-container-label"
    if has_math:
        css_class += " math-prose"
    return (
        f'<foreignObject data-nid="{escape(s.nid)}" '
        f'x="{label_x:g}" y="{label_y:g}" '
        f'width="{fo_w:g}" height="{fo_h:g}" '
        f'style="overflow:visible;">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" class="{css_class}" '
        f'style="font-size:{container_fs:g}px; '
        f'font-family:ui-sans-serif,sans-serif; font-style:italic; '
        f'color:#555; line-height:1.2; '
        f'overflow:hidden; word-break:break-word; '
        f'display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; '
        f'text-overflow:ellipsis;">'
        f'{html}'
        f'</div>'
        f'</foreignObject>'
    )


def _render_inner_label(
    ps: PlacedShape, lines: list[str], eff_fs: float,
) -> str:
    """Centred body label for a regular concept-card shape.

    Pure prose continues to render via flat ``<text>`` (preserving
    byte-determinism for the existing relation tests).  When the
    label contains math typography (Greek glyphs, sub/superscripts,
    glued patterns like ``α0m`` or ``Zm``), we switch to a
    ``<foreignObject>`` that carries a ``.math-prose`` div — the
    frontend then runs KaTeX over the ``\\(...\\)`` segments inside
    so the user sees ``α₀ₘ`` typeset, not ``α0m`` raw.
    """
    s = ps.shape
    html, has_math = _label_to_math_html(s.label)
    if not has_math:
        return _render_inner_label_text(ps, lines, eff_fs)
    return _render_inner_label_html(ps, html, eff_fs)


def _render_inner_label_text(
    ps: PlacedShape, lines: list[str], eff_fs: float,
) -> str:
    """Original flat ``<text>`` path — unchanged, kept byte-deterministic."""
    s = ps.shape
    n = len(lines)
    label_x = ps.x + s.width / 2
    lh = eff_fs * 1.15
    total_h = n * lh
    label_y = ps.y + s.height / 2 - total_h / 2 + eff_fs * 0.8
    anchor = "middle"
    if n == 1:
        inner = escape(lines[0])
    else:
        tspans = [f'<tspan x="{label_x:g}" dy="0">{escape(lines[0])}</tspan>']
        for line in lines[1:]:
            tspans.append(f'<tspan x="{label_x:g}" dy="{lh:g}">{escape(line)}</tspan>')
        inner = "".join(tspans)
    return (
        f'<text data-nid="{escape(s.nid)}" '
        f'x="{label_x:g}" y="{label_y:g}" '
        f'font-size="{eff_fs:g}" font-family="sans-serif" '
        f'text-anchor="{anchor}" fill="#222">{inner}</text>'
    )


def _render_inner_label_html(
    ps: PlacedShape, html: str, eff_fs: float,
) -> str:
    """Math-aware label via ``<foreignObject>`` + KaTeX auto-render."""
    s = ps.shape
    fo_w = max(40.0, s.width - 12.0)
    fo_h = max(28.0, s.height - 12.0)
    fo_x = ps.x + 6.0
    fo_y = ps.y + 6.0
    return (
        f'<foreignObject data-nid="{escape(s.nid)}" '
        f'x="{fo_x:g}" y="{fo_y:g}" '
        f'width="{fo_w:g}" height="{fo_h:g}" '
        f'style="overflow:visible;">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" '
        f'class="sevim-label sevim-inner-label math-prose" '
        f'style="font-size:{eff_fs:g}px; '
        f'font-family:ui-sans-serif,sans-serif; '
        f'color:#222; line-height:1.2; '
        f'width:100%; height:100%; '
        f'display:flex; align-items:center; justify-content:center; '
        f'text-align:center; word-break:break-word; '
        f'overflow:hidden;">'
        f'{html}'
        f'</div>'
        f'</foreignObject>'
    )


def _render_conn(pc: PlacedConn, lookup: dict[str, PlacedShape]) -> str:
    """Return the SVG element(s) for one connector, dispatching on relation type."""
    rel = pc.conn.relation
    (x1, y1), (x2, y2) = pc.points[0], pc.points[-1]
    eid = escape(pc.conn.eid)

    if rel == "reduces_to":
        return _funnel(pc, lookup, eid)
    if rel == "similar_to":
        return _parallel(x1, y1, x2, y2, eid)
    if rel == "measures":
        return _measures_annot(x1, y1, x2, y2, eid)
    if rel == "opposes":
        return _opposes(x1, y1, x2, y2, eid)
    if rel == "requires":
        return _requires(x1, y1, x2, y2, eid)

    # Math relations: most are line/arrow + Unicode midpoint label.
    label = _MATH_REL_LABELS.get(rel)
    if rel in _MATH_REL_ARROW:
        return _label_line(x1, y1, x2, y2, eid, label,
                           dashed=False, arrow_end=True)
    if rel in _MATH_REL_LINE:
        return _label_line(x1, y1, x2, y2, eid, label,
                           dashed=False, arrow_end=False)
    if rel == "labels":
        return _label_line(x1, y1, x2, y2, eid, None,
                           dashed=True, arrow_end=False, dotted=True)
    if rel == "grouped_with":
        return _label_line(x1, y1, x2, y2, eid, None,
                           dashed=True, arrow_end=False, thin=True)
    if rel == "aligned_with":
        return _label_line(x1, y1, x2, y2, eid, None,
                           dashed=False, arrow_end=False, dotted=True, thin=True)
    if rel == "points_to":
        return _label_line(x1, y1, x2, y2, eid, None,
                           dashed=False, arrow_end=True)
    if rel == "connects":
        return _label_line(x1, y1, x2, y2, eid, None,
                           dashed=False, arrow_end=False)
    if rel == "disjoint":
        # Bar-bar with ∅ midpoint label.
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        return (
            f'<g data-eid="{eid}">'
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
            f'stroke="#333" stroke-width="1.5" '
            f'marker-start="url(#bar)" marker-end="url(#bar)"/>'
            f'<text x="{mx:g}" y="{my - 6:g}" font-size="13" '
            f'font-family="sans-serif" text-anchor="middle" fill="#333">∅</text>'
            f'</g>'
        )
    if rel == "implies":
        # Bold double-shafted arrow (⇒).
        return _label_line(x1, y1, x2, y2, eid, "⇒",
                           dashed=False, arrow_end=True)
    if rel == "natural_transformation":
        # Doubled-shaft arrow drawn as two parallel lines.
        return _double_arrow(x1, y1, x2, y2, eid)
    if rel == "adjoint_to":
        # Adjoint pair drawn as two opposing arrows with ⊣ label.
        return _adjoint_pair(x1, y1, x2, y2, eid)
    if rel == "commutes":
        # Commuting region — connector with ↻ symbol at midpoint.
        return _label_line(x1, y1, x2, y2, eid, "↻",
                           dashed=False, arrow_end=False)

    dash = ""
    if rel in _DASHED:
        dash = ' stroke-dasharray="6,4"'
    elif rel in _DOTTED:
        dash = ' stroke-dasharray="1,3"'
    marker = ' marker-end="url(#arrow)"' if rel in _ARROW_END else ""
    # Cubic bezier: control points pull horizontally from each endpoint so
    # crossing edges form distinguishable S-curves rather than straight diagonals.
    dx = x2 - x1
    cx1, cy1 = x1 + dx * 0.4, y1
    cx2, cy2 = x2 - dx * 0.4, y2
    return (
        f'<path data-eid="{eid}" '
        f'd="M{x1:g},{y1:g} C{cx1:g},{cy1:g} {cx2:g},{cy2:g} {x2:g},{y2:g}" '
        f'fill="none" stroke="#333" stroke-width="1.5"{dash}{marker}/>'
    )


def _parallel(x1: float, y1: float, x2: float, y2: float, eid: str) -> str:
    """Two clearly-separated parallel lines with a '~' label — similar_to."""
    dx, dy = x2 - x1, y2 - y1
    L = (dx * dx + dy * dy) ** 0.5 or 1.0
    ox, oy = -dy / L * 7.0, dx / L * 7.0
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        f'<g data-eid="{eid}">'
        f'<line x1="{x1 + ox:g}" y1="{y1 + oy:g}" '
        f'x2="{x2 + ox:g}" y2="{y2 + oy:g}" '
        f'stroke="#333" stroke-width="1.5"/>'
        f'<line x1="{x1 - ox:g}" y1="{y1 - oy:g}" '
        f'x2="{x2 - ox:g}" y2="{y2 - oy:g}" '
        f'stroke="#333" stroke-width="1.5"/>'
        f'<text x="{mx:g}" y="{my - 14:g}" font-size="14" '
        f'font-family="sans-serif" text-anchor="middle" fill="#333">&#8776;</text>'
        f'</g>'
    )


def _measures_annot(x1: float, y1: float, x2: float, y2: float, eid: str) -> str:
    """Dotted line with '=' label at midpoint — measures."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        f'<g data-eid="{eid}">'
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="#333" stroke-width="1" stroke-dasharray="1,3"/>'
        f'<text x="{mx:g}" y="{my - 4:g}" font-size="11" '
        f'font-family="sans-serif" text-anchor="middle" fill="#333">=</text>'
        f'</g>'
    )


def _opposes(x1: float, y1: float, x2: float, y2: float, eid: str) -> str:
    """Bar at source, arrow at target — opposes."""
    return (
        f'<line data-eid="{eid}" '
        f'x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="#333" stroke-width="1.5" '
        f'marker-start="url(#bar)" marker-end="url(#bar)"/>'
    )


def _requires(x1: float, y1: float, x2: float, y2: float, eid: str) -> str:
    """Bezier curve with hollow triangle marker — requires (UML dependency style)."""
    dx = x2 - x1
    cx1, cy1 = x1 + dx * 0.4, y1
    cx2, cy2 = x2 - dx * 0.4, y2
    return (
        f'<path data-eid="{eid}" '
        f'd="M{x1:g},{y1:g} C{cx1:g},{cy1:g} {cx2:g},{cy2:g} {x2:g},{y2:g}" '
        f'fill="none" stroke="#333" stroke-width="1.5" '
        f'marker-end="url(#hollow-tri)"/>'
    )


def _funnel(pc: PlacedConn, lookup: dict[str, PlacedShape], eid: str) -> str:
    """Triangle path, source edge full-height wide → target point — reduces_to."""
    src = lookup.get(pc.conn.from_nid)
    (x1, y1), (x2, y2) = pc.points[0], pc.points[-1]
    if src is None:
        return (
            f'<line data-eid="{eid}" '
            f'x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
            f'stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )
    # Use the full source bounding dimension so the funnel is unmistakable.
    h = max(src.shape.height, src.shape.width * 0.5)
    dx, dy = x2 - x1, y2 - y1
    L = (dx * dx + dy * dy) ** 0.5 or 1.0
    nx, ny = -dy / L, dx / L
    p1 = (x1 + nx * h * 0.5, y1 + ny * h * 0.5)
    p2 = (x1 - nx * h * 0.5, y1 - ny * h * 0.5)
    return (
        f'<path data-eid="{eid}" '
        f'd="M{p1[0]:g},{p1[1]:g} L{x2:g},{y2:g} L{p2[0]:g},{p2[1]:g} Z" '
        f'fill="#999" fill-opacity="0.55" stroke="#333" stroke-width="1.2"/>'
    )


# ===========================================================================
# MATH CONNECTORS — 18 new relations
# ===========================================================================

# Relations that draw a labeled solid line (no arrow).
_MATH_REL_LINE = frozenset({
    "equals", "congruent", "approximately_equal",
    "perpendicular", "parallel",
    "lies_on", "between", "tangent_to",
})
# Relations that draw a labeled arrow.
_MATH_REL_ARROW = frozenset({
    "isomorphic_to", "element_of", "subset_of", "maps_to",
})
# Midpoint Unicode glyphs.
_MATH_REL_LABELS: dict[str, str | None] = {
    "equals": "=",
    "congruent": "≅",
    "approximately_equal": "≈",
    "perpendicular": "⊥",
    "parallel": "∥",
    "tangent_to": None,    # geometric, no symbol on the line itself
    "lies_on": None,
    "between": None,
    "isomorphic_to": "≅",
    "element_of": "∈",
    "subset_of": "⊆",
    "maps_to": "↦",
}


def _label_line(
    x1: float, y1: float, x2: float, y2: float, eid: str,
    label: str | None,
    *, dashed: bool = False, dotted: bool = False,
    arrow_end: bool = False, thin: bool = False,
) -> str:
    """Generic labeled line/arrow — used by most math relations."""
    sw = 0.8 if thin else 1.5
    dash = ""
    if dashed:
        dash = ' stroke-dasharray="6,4"'
    elif dotted:
        dash = ' stroke-dasharray="1,3"'
    marker = ' marker-end="url(#arrow)"' if arrow_end else ""
    line = (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="#333" stroke-width="{sw:g}"{dash}{marker}/>'
    )
    if not label:
        return f'<g data-eid="{eid}">{line}</g>'
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    text = (
        f'<text x="{mx:g}" y="{my - 6:g}" font-size="13" '
        f'font-family="sans-serif" text-anchor="middle" fill="#333">'
        f'{escape(label)}</text>'
    )
    return f'<g data-eid="{eid}">{line}{text}</g>'


# ===========================================================================
# MATH SHAPES — 13 new primitives
# ===========================================================================

def _render_math_shape(ps: PlacedShape, stroke: str) -> str:
    """Dispatch a math primitive to its specialised renderer."""
    s = ps.shape
    p = s.primitive
    if p == "point":
        return _render_point(ps, stroke)
    if p == "segment":
        return _render_segment(ps, stroke)
    if p == "curve":
        return _render_curve(ps, stroke)
    if p == "polygon":
        return _render_polygon(ps, stroke)
    if p == "circle":
        return _render_circle(ps, stroke)
    if p == "arc":
        return _render_arc(ps, stroke)
    if p == "axes":
        return _render_axes(ps, stroke)
    if p == "grid":
        return _render_grid(ps, stroke)
    if p == "set_blob":
        return _render_set_blob(ps, stroke)
    if p == "matrix_bracket":
        return _render_matrix(ps, stroke)
    if p == "equation_block":
        return _render_equation(ps, stroke)
    if p == "decoration_mark":
        return _render_decoration(ps, stroke)
    if p == "bracket_brace":
        return _render_bracket(ps, stroke)
    # — university —
    if p == "tensor_box":
        return _render_tensor_box(ps, stroke)
    if p == "string_wire":
        return _render_string_wire(ps, stroke)
    if p == "pullback_corner":
        return _render_pullback(ps, stroke)
    if p == "axes_3d":
        return _render_axes_3d(ps, stroke)
    if p == "pdf_curve":
        return _render_pdf_curve(ps, stroke)
    if p == "plate":
        return _render_plate(ps, stroke)
    if p == "error_bar":
        return _render_error_bar(ps, stroke)
    if p == "proof_node":
        return _render_proof_node(ps, stroke)
    return ""


def _render_point(ps: PlacedShape, stroke: str) -> str:
    """Small filled disk + label below."""
    s = ps.shape
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2
    r = min(s.width, s.height) * 0.25
    label_y = cy + r + s.font_size
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" '
        f'fill="{stroke}" stroke="{stroke}" stroke-width="1"/>'
        f'<text x="{cx:g}" y="{label_y:g}" font-size="{s.font_size:g}" '
        f'font-family="serif" font-style="italic" text-anchor="middle" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_segment(ps: PlacedShape, stroke: str) -> str:
    """Horizontal line with endpoint ticks + label above the midpoint."""
    s = ps.shape
    y_mid = ps.y + s.height / 2
    x1, x2 = ps.x + 4, ps.x + s.width - 4
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<line x1="{x1:g}" y1="{y_mid:g}" x2="{x2:g}" y2="{y_mid:g}" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width * 1.3:g}"/>'
        f'<line x1="{x1:g}" y1="{y_mid - 4:g}" x2="{x1:g}" y2="{y_mid + 4:g}" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        f'<line x1="{x2:g}" y1="{y_mid - 4:g}" x2="{x2:g}" y2="{y_mid + 4:g}" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        f'<text x="{(x1+x2)/2:g}" y="{y_mid - 6:g}" font-size="{s.font_size*0.85:g}" '
        f'font-family="serif" font-style="italic" text-anchor="middle" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_curve(ps: PlacedShape, stroke: str) -> str:
    """Smooth cubic-bezier arc inscribed in the bounding box. Default sinusoidal."""
    s = ps.shape
    x, y, w, h = ps.x, ps.y, s.width, s.height
    # Default: a sigmoid-ish S-curve from bottom-left to top-right.
    p1 = (x + 4, y + h - 4)
    p2 = (x + w * 0.4, y + h * 0.7)
    p3 = (x + w * 0.6, y + h * 0.3)
    p4 = (x + w - 4, y + 4)
    path = f'M{p1[0]:g},{p1[1]:g} C{p2[0]:g},{p2[1]:g} {p3[0]:g},{p3[1]:g} {p4[0]:g},{p4[1]:g}'
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<path d="{path}" fill="none" stroke="{stroke}" '
        f'stroke-width="{s.stroke_width * 1.4:g}"/>'
        f'<text x="{x + w/2:g}" y="{y + h + 14:g}" font-size="{s.font_size*0.85:g}" '
        f'font-family="serif" font-style="italic" text-anchor="middle" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_polygon(ps: PlacedShape, stroke: str) -> str:
    """Inscribed regular n-gon. Reads `meta["sides"]` (default 3 → triangle)."""
    s = ps.shape
    sides = max(3, int(s.meta.get("sides", 3)))
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2
    rx = s.width / 2 - 4
    ry = s.height / 2 - 4
    import math
    pts: list[str] = []
    # Start angle: -π/2 so a triangle points up.
    for i in range(sides):
        theta = -math.pi / 2 + 2 * math.pi * i / sides
        pts.append(f"{cx + rx * math.cos(theta):g},{cy + ry * math.sin(theta):g}")
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<polygon points="{" ".join(pts)}" fill="#fff" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        f'<text x="{cx:g}" y="{cy + s.font_size*0.35:g}" '
        f'font-size="{s.font_size:g}" font-family="sans-serif" '
        f'text-anchor="middle" fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_circle(ps: PlacedShape, stroke: str) -> str:
    s = ps.shape
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2
    r = min(s.width, s.height) / 2 - 2
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" '
        f'fill="#fff" stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        f'<text x="{cx:g}" y="{cy + s.font_size*0.35:g}" '
        f'font-size="{s.font_size:g}" font-family="sans-serif" '
        f'text-anchor="middle" fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_arc(ps: PlacedShape, stroke: str) -> str:
    """Quarter-circle arc with a small right-angle tick if angle == 90."""
    s = ps.shape
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2
    r = min(s.width, s.height) * 0.35
    # Default: 60° arc starting at 0°.
    import math
    start = math.radians(s.meta.get("start_deg", 10))
    sweep = math.radians(s.meta.get("sweep_deg", 70))
    end = start + sweep
    p_start = (cx + r * math.cos(start), cy + r * math.sin(start))
    p_end = (cx + r * math.cos(end), cy + r * math.sin(end))
    large_arc = 1 if sweep > math.pi else 0
    path = (f'M{p_start[0]:g},{p_start[1]:g} '
            f'A{r:g},{r:g} 0 {large_arc} 1 {p_end[0]:g},{p_end[1]:g}')
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<path d="{path}" fill="none" stroke="{stroke}" '
        f'stroke-width="{s.stroke_width * 1.3:g}"/>'
        f'<text x="{cx:g}" y="{cy + r + 14:g}" font-size="{s.font_size*0.85:g}" '
        f'font-family="serif" font-style="italic" text-anchor="middle" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_axes(ps: PlacedShape, stroke: str) -> str:
    """2-D Cartesian axes with arrow tips and tick marks. Container-friendly."""
    s = ps.shape
    pad_top = 28.0
    pad_left = 20.0
    pad_right = 14.0
    pad_bot = 22.0
    x0 = ps.x + pad_left
    y0 = ps.y + s.height - pad_bot
    x1 = ps.x + s.width - pad_right
    y1 = ps.y + pad_top
    return (
        f'<g data-nid="{escape(s.nid)}">'
        # frame
        f'<rect x="{ps.x:g}" y="{ps.y:g}" width="{s.width:g}" height="{s.height:g}" '
        f'rx="4" ry="4" fill="#fafafe" stroke="#ddd" stroke-width="1"/>'
        # x-axis
        f'<line x1="{x0:g}" y1="{y0:g}" x2="{x1:g}" y2="{y0:g}" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}" '
        f'marker-end="url(#arrow)"/>'
        # y-axis
        f'<line x1="{x0:g}" y1="{y0:g}" x2="{x0:g}" y2="{y1:g}" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}" '
        f'marker-end="url(#arrow)"/>'
        # axis labels
        f'<text x="{x1 - 8:g}" y="{y0 + 14:g}" font-size="{s.font_size*0.8:g}" '
        f'font-family="serif" font-style="italic" text-anchor="end" fill="#555">'
        f'{escape(s.meta.get("xlabel", "x"))}</text>'
        f'<text x="{x0 + 6:g}" y="{y1 + 8:g}" font-size="{s.font_size*0.8:g}" '
        f'font-family="serif" font-style="italic" text-anchor="start" fill="#555">'
        f'{escape(s.meta.get("ylabel", "y"))}</text>'
        # title
        f'<text x="{ps.x + s.width/2:g}" y="{ps.y + 14:g}" '
        f'font-size="{s.font_size*0.85:g}" font-family="sans-serif" '
        f'text-anchor="middle" fill="#444">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_grid(ps: PlacedShape, stroke: str) -> str:
    """Light gridlines inside the bounding box (background utility primitive)."""
    s = ps.shape
    cell = float(s.meta.get("cell", 20.0))
    lines: list[str] = []
    x = ps.x
    while x <= ps.x + s.width:
        lines.append(
            f'<line x1="{x:g}" y1="{ps.y:g}" x2="{x:g}" y2="{ps.y + s.height:g}" '
            f'stroke="#e8e8ee" stroke-width="0.7"/>'
        )
        x += cell
    y = ps.y
    while y <= ps.y + s.height:
        lines.append(
            f'<line x1="{ps.x:g}" y1="{y:g}" x2="{ps.x + s.width:g}" y2="{y:g}" '
            f'stroke="#e8e8ee" stroke-width="0.7"/>'
        )
        y += cell
    return f'<g data-nid="{escape(s.nid)}">{"".join(lines)}</g>'


def _render_set_blob(ps: PlacedShape, stroke: str) -> str:
    """Rounded-rect 'blob' with translucent fill — represents a labeled set."""
    s = ps.shape
    rx = s.width / 2
    ry = s.height / 2
    cx = ps.x + rx
    cy = ps.y + ry
    label_y = ps.y + s.font_size + 4
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<ellipse cx="{cx:g}" cy="{cy:g}" rx="{rx:g}" ry="{ry:g}" '
        f'fill="{stroke}" fill-opacity="0.10" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}"/>'
        f'<text x="{cx:g}" y="{label_y:g}" font-size="{s.font_size:g}" '
        f'font-family="sans-serif" text-anchor="middle" fill="#333" '
        f'font-weight="bold">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_matrix(ps: PlacedShape, stroke: str) -> str:
    """Bracketed matrix with cells from `meta["cells"]`."""
    s = ps.shape
    cells = s.meta.get("cells") or [[s.label]]
    nrows = len(cells)
    ncols = len(cells[0]) if cells else 1
    delim = s.meta.get("delim", "bracket")
    cell_w = (s.width - 28.0) / max(ncols, 1)
    cell_h = max(20.0, (s.height - 8.0) / max(nrows, 1))
    inner_x = ps.x + 14.0
    inner_y = ps.y + 4.0
    inner_w = cell_w * ncols
    inner_h = cell_h * nrows
    parts: list[str] = [f'<g data-nid="{escape(s.nid)}">']

    # Brackets.
    bx_l = inner_x - 4
    bx_r = inner_x + inner_w + 4
    by_t = inner_y
    by_b = inner_y + inner_h
    if delim == "paren":
        parts.append(
            f'<path d="M{bx_l:g},{by_t:g} Q{bx_l-8:g},{(by_t+by_b)/2:g} {bx_l:g},{by_b:g}" '
            f'fill="none" stroke="{stroke}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<path d="M{bx_r:g},{by_t:g} Q{bx_r+8:g},{(by_t+by_b)/2:g} {bx_r:g},{by_b:g}" '
            f'fill="none" stroke="{stroke}" stroke-width="1.5"/>'
        )
    else:  # bracket / vbar / brace — render bracket for all v1
        parts.append(
            f'<path d="M{bx_l:g},{by_t:g} L{bx_l-6:g},{by_t:g} '
            f'L{bx_l-6:g},{by_b:g} L{bx_l:g},{by_b:g}" '
            f'fill="none" stroke="{stroke}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<path d="M{bx_r:g},{by_t:g} L{bx_r+6:g},{by_t:g} '
            f'L{bx_r+6:g},{by_b:g} L{bx_r:g},{by_b:g}" '
            f'fill="none" stroke="{stroke}" stroke-width="1.5"/>'
        )

    # Cells.
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            tx = inner_x + c * cell_w + cell_w / 2
            ty = inner_y + r * cell_h + cell_h / 2 + s.font_size * 0.30
            parts.append(
                f'<text x="{tx:g}" y="{ty:g}" font-size="{s.font_size:g}" '
                f'font-family="serif" font-style="italic" text-anchor="middle" '
                f'fill="#222">{escape(latex_to_unicode(str(val)))}</text>'
            )
    parts.append('</g>')
    return "".join(parts)


def _render_equation(ps: PlacedShape, stroke: str) -> str:
    """Equation block: structured SVG typesetting for fractions/roots/big-ops,
    or single-line Unicode for simpler expressions.

    The structured renderer handles \\frac, \\sqrt, \\sum/\\int with stacked
    bounds, and explicit line breaks.  Everything else falls through to the
    inline LaTeX→Unicode path for sub-millisecond rendering.

    For higher-fidelity typesetting (real glyph metrics, full LaTeX), this
    function is the documented swap-in seam: replace the structured branch
    with a MathJax-Node subprocess call that returns an SVG fragment.
    """
    s = ps.shape
    latex = s.meta.get("latex") or ""
    # Structured path: fraction / root / big-op / multi-line.
    if latex and has_structured_constructs(latex):
        body, eq_w, eq_h = render_equation(latex, font_size=s.font_size)
        # Re-fit the rectangle to the structured equation's actual size,
        # adding padding.  The shape's stored width/height become the
        # frame; the structured equation is centered inside.
        pad = 8.0
        frame_w = max(eq_w + 2 * pad, s.width)
        frame_h = max(eq_h + 2 * pad, s.height)
        eq_x = ps.x + (frame_w - eq_w) / 2.0
        eq_y = ps.y + (frame_h - eq_h) / 2.0
        return (
            f'<g data-nid="{escape(s.nid)}">'
            f'<rect x="{ps.x:g}" y="{ps.y:g}" '
            f'width="{frame_w:g}" height="{frame_h:g}" '
            f'rx="4" ry="4" fill="#fbfbff" stroke="{stroke}" '
            f'stroke-width="{s.stroke_width:g}" stroke-dasharray="2,2"/>'
            f'<g transform="translate({eq_x:g},{eq_y:g})">{body}</g>'
            f'</g>'
        )
    # Single-line path.
    text = s.meta.get("unicode") or latex_to_unicode(latex) or s.label
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2 + s.font_size * 0.30
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<rect x="{ps.x:g}" y="{ps.y:g}" width="{s.width:g}" height="{s.height:g}" '
        f'rx="4" ry="4" fill="#fbfbff" stroke="{stroke}" '
        f'stroke-width="{s.stroke_width:g}" stroke-dasharray="2,2"/>'
        f'<text x="{cx:g}" y="{cy:g}" font-size="{s.font_size:g}" '
        f'font-family="serif" font-style="italic" text-anchor="middle" '
        f'fill="#222">{escape(text)}</text>'
        f'</g>'
    )


def _render_decoration(ps: PlacedShape, stroke: str) -> str:
    """Small geometric annotation: tick / right-angle / hash mark."""
    s = ps.shape
    kind = s.meta.get("mark", "tick")
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2
    if kind == "right_angle":
        body = (
            f'<path d="M{cx-6:g},{cy+6:g} L{cx-6:g},{cy-6:g} L{cx+6:g},{cy-6:g}" '
            f'fill="none" stroke="{stroke}" stroke-width="1.2"/>'
        )
    elif kind == "parallel_chevron":
        body = (
            f'<path d="M{cx-4:g},{cy-5:g} L{cx+4:g},{cy:g} L{cx-4:g},{cy+5:g}" '
            f'fill="none" stroke="{stroke}" stroke-width="1.2"/>'
        )
    else:  # tick / congruence hash
        body = (
            f'<line x1="{cx:g}" y1="{cy-6:g}" x2="{cx:g}" y2="{cy+6:g}" '
            f'stroke="{stroke}" stroke-width="1.2"/>'
        )
    return f'<g data-nid="{escape(s.nid)}">{body}</g>'


def _double_arrow(
    x1: float, y1: float, x2: float, y2: float, eid: str,
) -> str:
    """Double-shaft arrow — natural transformation (⇒)."""
    dx, dy = x2 - x1, y2 - y1
    L = (dx * dx + dy * dy) ** 0.5 or 1.0
    ox, oy = -dy / L * 3.0, dx / L * 3.0
    return (
        f'<g data-eid="{eid}">'
        f'<line x1="{x1+ox:g}" y1="{y1+oy:g}" x2="{x2+ox:g}" y2="{y2+oy:g}" '
        f'stroke="#333" stroke-width="1.4"/>'
        f'<line x1="{x1-ox:g}" y1="{y1-oy:g}" x2="{x2-ox:g}" y2="{y2-oy:g}" '
        f'stroke="#333" stroke-width="1.4" marker-end="url(#arrow)"/>'
        f'</g>'
    )


def _adjoint_pair(
    x1: float, y1: float, x2: float, y2: float, eid: str,
) -> str:
    """Adjoint pair — opposing arrows above and below the line, with ⊣ label."""
    dx, dy = x2 - x1, y2 - y1
    L = (dx * dx + dy * dy) ** 0.5 or 1.0
    ox, oy = -dy / L * 5.0, dx / L * 5.0
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        f'<g data-eid="{eid}">'
        # Upper arrow: source → target
        f'<line x1="{x1+ox:g}" y1="{y1+oy:g}" x2="{x2+ox:g}" y2="{y2+oy:g}" '
        f'stroke="#333" stroke-width="1.4" marker-end="url(#arrow)"/>'
        # Lower arrow: target → source
        f'<line x1="{x2-ox:g}" y1="{y2-oy:g}" x2="{x1-ox:g}" y2="{y1-oy:g}" '
        f'stroke="#333" stroke-width="1.4" marker-end="url(#arrow)"/>'
        f'<text x="{mx:g}" y="{my - 4:g}" font-size="13" '
        f'font-family="sans-serif" text-anchor="middle" fill="#333">⊣</text>'
        f'</g>'
    )


def _render_tensor_box(ps: PlacedShape, stroke: str) -> str:
    """Penrose-graphical-style tensor: rounded box with N labelled wires.

    `meta["legs"]` (default 3) controls wire count.  Wires emerge from the
    bottom edge by default; an "in_legs" entry can route some to the top.
    """
    s = ps.shape
    legs = max(1, int(s.meta.get("legs", 3)))
    in_legs = max(0, int(s.meta.get("in_legs", 0)))
    out_legs = legs - in_legs
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2
    box_w = s.width * 0.7
    box_h = s.height * 0.5
    bx = cx - box_w / 2
    by = cy - box_h / 2
    parts = [
        f'<g data-nid="{escape(s.nid)}">',
        f'<rect x="{bx:g}" y="{by:g}" width="{box_w:g}" height="{box_h:g}" '
        f'rx="6" ry="6" fill="#fff" stroke="{stroke}" '
        f'stroke-width="{s.stroke_width * 1.3:g}"/>',
        f'<text x="{cx:g}" y="{cy + s.font_size*0.35:g}" '
        f'font-size="{s.font_size:g}" font-family="serif" font-style="italic" '
        f'text-anchor="middle" fill="#222">{escape(s.label)}</text>',
    ]
    # Outgoing legs from bottom edge.
    if out_legs > 0:
        for i in range(out_legs):
            t = (i + 1) / (out_legs + 1)
            wx = bx + t * box_w
            parts.append(
                f'<line x1="{wx:g}" y1="{by + box_h:g}" '
                f'x2="{wx:g}" y2="{ps.y + s.height - 2:g}" '
                f'stroke="{stroke}" stroke-width="1.2"/>'
            )
    # Incoming legs from top edge.
    if in_legs > 0:
        for i in range(in_legs):
            t = (i + 1) / (in_legs + 1)
            wx = bx + t * box_w
            parts.append(
                f'<line x1="{wx:g}" y1="{ps.y + 2:g}" '
                f'x2="{wx:g}" y2="{by:g}" '
                f'stroke="{stroke}" stroke-width="1.2"/>'
            )
    parts.append('</g>')
    return "".join(parts)


def _render_string_wire(ps: PlacedShape, stroke: str) -> str:
    """Single curved wire — primitive of monoidal-category string diagrams."""
    s = ps.shape
    x, y, w, h = ps.x, ps.y, s.width, s.height
    # Vertical S-curve from top to bottom.
    p1 = (x + w * 0.2, y + 4)
    p2 = (x + w * 0.8, y + h * 0.4)
    p3 = (x + w * 0.2, y + h * 0.6)
    p4 = (x + w * 0.8, y + h - 4)
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<path d="M{p1[0]:g},{p1[1]:g} C{p2[0]:g},{p2[1]:g} '
        f'{p3[0]:g},{p3[1]:g} {p4[0]:g},{p4[1]:g}" '
        f'fill="none" stroke="{stroke}" stroke-width="1.6"/>'
        f'<text x="{x + w + 4:g}" y="{y + h/2:g}" '
        f'font-size="{s.font_size*0.85:g}" font-family="serif" font-style="italic" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_pullback(ps: PlacedShape, stroke: str) -> str:
    """Pullback / pushout corner: small L-shaped mark with optional ⌟ glyph."""
    s = ps.shape
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2
    glyph = "⌜" if s.meta.get("kind_corner") == "pushout" else "⌟"
    side = 14
    # Draw a small square frame plus an inset corner symbol.
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<path d="M{cx - side:g},{cy + side:g} L{cx - side:g},{cy - side:g} '
        f'L{cx + side:g},{cy - side:g}" '
        f'fill="none" stroke="{stroke}" stroke-width="1.3"/>'
        f'<text x="{cx + side + 3:g}" y="{cy + side + s.font_size:g}" '
        f'font-size="{s.font_size:g}" font-family="serif" font-style="italic" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_axes_3d(ps: PlacedShape, stroke: str) -> str:
    """3-D axes drawn in isometric projection (x-right, y-up-right, z-up)."""
    s = ps.shape
    import math
    cx = ps.x + s.width * 0.30
    cy = ps.y + s.height * 0.70
    L = min(s.width, s.height) * 0.55
    # Standard isometric: x at -30°, y at +30° below horizontal, z straight up.
    ang_x = math.radians(-25)
    ang_y = math.radians(25)
    x_end = (cx + L * math.cos(ang_x), cy + L * math.sin(ang_x))
    y_end = (cx + L * math.cos(ang_y) * 0.9, cy + L * math.sin(ang_y) * 0.9)
    z_end = (cx, cy - L)
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<rect x="{ps.x:g}" y="{ps.y:g}" width="{s.width:g}" height="{s.height:g}" '
        f'rx="4" ry="4" fill="#fafafe" stroke="#ddd" stroke-width="1"/>'
        f'<line x1="{cx:g}" y1="{cy:g}" x2="{x_end[0]:g}" y2="{x_end[1]:g}" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}" '
        f'marker-end="url(#arrow)"/>'
        f'<line x1="{cx:g}" y1="{cy:g}" x2="{y_end[0]:g}" y2="{y_end[1]:g}" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}" '
        f'marker-end="url(#arrow)"/>'
        f'<line x1="{cx:g}" y1="{cy:g}" x2="{z_end[0]:g}" y2="{z_end[1]:g}" '
        f'stroke="{stroke}" stroke-width="{s.stroke_width:g}" '
        f'marker-end="url(#arrow)"/>'
        f'<text x="{x_end[0] + 4:g}" y="{x_end[1] + 4:g}" '
        f'font-size="{s.font_size*0.8:g}" font-family="serif" '
        f'font-style="italic" fill="#555">x</text>'
        f'<text x="{y_end[0] + 4:g}" y="{y_end[1] + 4:g}" '
        f'font-size="{s.font_size*0.8:g}" font-family="serif" '
        f'font-style="italic" fill="#555">y</text>'
        f'<text x="{z_end[0] - 4:g}" y="{z_end[1] - 4:g}" '
        f'font-size="{s.font_size*0.8:g}" font-family="serif" '
        f'font-style="italic" text-anchor="end" fill="#555">z</text>'
        f'<text x="{ps.x + s.width/2:g}" y="{ps.y + 14:g}" '
        f'font-size="{s.font_size*0.85:g}" font-family="sans-serif" '
        f'text-anchor="middle" fill="#444">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_pdf_curve(ps: PlacedShape, stroke: str) -> str:
    """Bell curve (normal density approximation) inside the bounding box.

    `meta["shaded"]` may give (lo, hi) as fractions of width to shade — used
    to depict probability mass / confidence regions.
    """
    s = ps.shape
    import math
    x, y, w, h = ps.x, ps.y, s.width, s.height
    # Sample the standard normal on [-3, 3] mapped onto [x, x+w].
    n = 32
    pts: list[str] = []
    for i in range(n + 1):
        t = -3.0 + 6.0 * i / n
        density = math.exp(-0.5 * t * t)
        px = x + (i / n) * w
        py = y + h - 6 - density * (h - 14)
        pts.append(f"{px:g},{py:g}")
    path = "M" + " L".join(pts)
    parts = [f'<g data-nid="{escape(s.nid)}">']
    # Optional shaded region.
    shade = s.meta.get("shaded")
    if shade and isinstance(shade, (list, tuple)) and len(shade) == 2:
        lo, hi = float(shade[0]), float(shade[1])
        i_lo = max(0, int(round(lo * n)))
        i_hi = min(n, int(round(hi * n)))
        sub_pts = pts[i_lo:i_hi + 1]
        if sub_pts:
            base_left = f"{x + (i_lo / n) * w:g},{y + h - 6:g}"
            base_right = f"{x + (i_hi / n) * w:g},{y + h - 6:g}"
            parts.append(
                f'<path d="M{base_left} L{" L".join(sub_pts)} L{base_right} Z" '
                f'fill="{stroke}" fill-opacity="0.18" stroke="none"/>'
            )
    parts.append(
        f'<line x1="{x:g}" y1="{y + h - 6:g}" '
        f'x2="{x + w:g}" y2="{y + h - 6:g}" stroke="#888" stroke-width="0.8"/>'
    )
    parts.append(
        f'<path d="{path}" fill="none" stroke="{stroke}" '
        f'stroke-width="{s.stroke_width * 1.4:g}"/>'
    )
    parts.append(
        f'<text x="{x + w/2:g}" y="{y + h:g}" '
        f'font-size="{s.font_size*0.85:g}" font-family="sans-serif" '
        f'text-anchor="middle" fill="#222">{escape(s.label)}</text>'
    )
    parts.append('</g>')
    return "".join(parts)


def _render_plate(ps: PlacedShape, stroke: str) -> str:
    """Plate notation — rounded rect with N badge in the bottom-right corner."""
    s = ps.shape
    n = s.meta.get("n", "N")
    x, y, w, h = ps.x, ps.y, s.width, s.height
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
        f'rx="10" ry="10" fill="none" stroke="{stroke}" '
        f'stroke-width="{s.stroke_width:g}"/>'
        # Badge with N
        f'<rect x="{x + w - 26:g}" y="{y + h - 22:g}" width="22" height="18" '
        f'rx="4" ry="4" fill="#fff" stroke="{stroke}" stroke-width="1"/>'
        f'<text x="{x + w - 15:g}" y="{y + h - 9:g}" '
        f'font-size="{s.font_size*0.85:g}" font-family="serif" '
        f'font-style="italic" text-anchor="middle" fill="#333">'
        f'{escape(str(n))}</text>'
        f'<text x="{x + 8:g}" y="{y + s.font_size + 4:g}" '
        f'font-size="{s.font_size*0.85:g}" font-family="sans-serif" '
        f'fill="#444">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_error_bar(ps: PlacedShape, stroke: str) -> str:
    """Point with vertical error bars — `meta["err"]` is the half-height in px."""
    s = ps.shape
    err = float(s.meta.get("err", min(s.width, s.height) * 0.4))
    cx = ps.x + s.width / 2
    cy = ps.y + s.height / 2
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<line x1="{cx:g}" y1="{cy - err:g}" x2="{cx:g}" y2="{cy + err:g}" '
        f'stroke="{stroke}" stroke-width="1.4"/>'
        f'<line x1="{cx - 5:g}" y1="{cy - err:g}" '
        f'x2="{cx + 5:g}" y2="{cy - err:g}" '
        f'stroke="{stroke}" stroke-width="1.4"/>'
        f'<line x1="{cx - 5:g}" y1="{cy + err:g}" '
        f'x2="{cx + 5:g}" y2="{cy + err:g}" '
        f'stroke="{stroke}" stroke-width="1.4"/>'
        f'<circle cx="{cx:g}" cy="{cy:g}" r="3" fill="{stroke}"/>'
        f'<text x="{cx + 8:g}" y="{cy + s.font_size*0.35:g}" '
        f'font-size="{s.font_size*0.85:g}" font-family="serif" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_proof_node(ps: PlacedShape, stroke: str) -> str:
    """Proof-tree node: label box with a horizontal inference bar above."""
    s = ps.shape
    x, y, w, h = ps.x, ps.y, s.width, s.height
    bar_y = y + 2
    text_y = y + h / 2 + s.font_size * 0.35
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<line x1="{x:g}" y1="{bar_y:g}" x2="{x + w:g}" y2="{bar_y:g}" '
        f'stroke="{stroke}" stroke-width="1.4"/>'
        f'<text x="{x + w/2:g}" y="{text_y + 4:g}" font-size="{s.font_size:g}" '
        f'font-family="serif" font-style="italic" text-anchor="middle" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )


def _render_bracket(ps: PlacedShape, stroke: str) -> str:
    """Under-brace / over-brace path. `meta["side"]` ∈ {under, over}."""
    s = ps.shape
    side = s.meta.get("side", "under")
    x = ps.x
    y = ps.y + s.height / 2
    w = s.width
    if side == "under":
        path = (
            f'M{x:g},{y-4:g} Q{x+w*0.25:g},{y+4:g} {x+w*0.5:g},{y:g} '
            f'Q{x+w*0.75:g},{y+4:g} {x+w:g},{y-4:g}'
        )
        text_y = y + s.font_size + 4
    else:
        path = (
            f'M{x:g},{y+4:g} Q{x+w*0.25:g},{y-4:g} {x+w*0.5:g},{y:g} '
            f'Q{x+w*0.75:g},{y-4:g} {x+w:g},{y+4:g}'
        )
        text_y = y - 6
    return (
        f'<g data-nid="{escape(s.nid)}">'
        f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="1.4"/>'
        f'<text x="{x+w/2:g}" y="{text_y:g}" font-size="{s.font_size*0.85:g}" '
        f'font-family="serif" font-style="italic" text-anchor="middle" '
        f'fill="#222">{escape(s.label)}</text>'
        f'</g>'
    )
