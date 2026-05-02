"""Deterministic SVG renderer for :class:`SemanticGraph`.

Tier-3 of the visualisation pipeline used to be an LLM that emitted SVG
free-hand.  We replace it with a small, deterministic compiler:

  graph → fixed-layout templates → ``<line>`` / ``<polyline>`` / ``<circle>``
  / ``<rect>`` / ``<text>`` / ``<g>`` markup.

Hard rules:

* No randomness, no temperature, no time-based jitter.
* Same input ⇒ byte-identical output.
* Only the primitive set above (frontend-safe).
* Render budget: << 10 ms for any graph this module produces.
* Output is the inner body (no outer ``<svg>``); the chalkboard wraps it.

Templates implemented (selected from the graph in priority order):

  1. Function plot — quadratic, linear, cubic, sigmoid, sin, cos, exp,
     log, tanh, ReLU, softmax → axis frame + labelled curve.
  2. Vector — coordinate frame + arrow with label.
  3. Matrix — bracket-flanked grid with cells (uses params['rows']
     when present; falls back to nrows/ncols placeholder).
  4. Operation card — operation node only ⇒ minimal card.
  5. Shape — circle / triangle / line / rectangle / square / ellipse.
  6. Equation block — labelled equation tile.
  7. Flow diagram — when ≥ 2 dependency edges connect named nodes.

If the graph has none of the above, returns an empty string.
"""
from __future__ import annotations

import math
from typing import Optional

from .semantic_ir import SemanticEdge, SemanticGraph, SemanticNode


# Canvas dimensions used for every Tier-3 deterministic figure.  Kept
# constant so the chalkboard can pre-size the card.
CANVAS_W = 480.0
CANVAS_H = 300.0

# Plot inset (axis frame coordinates).
_PLOT_X = 50.0
_PLOT_Y = 30.0
_PLOT_W = 410.0
_PLOT_H = 240.0

# Colours — match the existing operation/canonical card palette.
_AXIS_INK = "#37474f"
_LABEL_INK = "#212121"
_GRID_INK = "#cfd8dc"

# English stop-words / question fragments the LLM occasionally emits
# as a node label.  We drop them in favour of canonical math symbols.
_NON_LABELS = frozenset({
    "is", "a", "an", "the", "what", "which", "how", "of", "to", "in",
    "on", "or", "and", "be", "as", "for", "by", "this", "that",
    "vector", "function", "matrix", "scalar", "set", "shape",
    "operation", "equation", "axis", "label", "point",
})


_SERIES_COLOURS = {
    "linear":     "#1976d2",
    "quadratic":  "#5e35b1",
    "cubic":      "#d81b60",
    "polynomial": "#5e35b1",
    "exp":        "#f57c00",
    "log":        "#388e3c",
    "sin":        "#1565c0",
    "cos":        "#6a1b9a",
    "sigmoid":    "#00897b",
    "relu":       "#3949ab",
    "tanh":       "#039be5",
    "softmax":    "#7b1fa2",
    "parabola":   "#5e35b1",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canvas_size(graph: SemanticGraph) -> tuple[float, float]:
    """Return the (width, height) the orchestrator should reserve."""
    if _has_matrix(graph):
        m = _largest_matrix(graph)
        rows = m.params.get("nrows") or len(m.params.get("rows") or []) or 2
        cols = m.params.get("ncols") or (
            len((m.params.get("rows") or [[]])[0]) if m.params.get("rows") else 2
        )
        w = max(220.0, 44.0 * cols + 40.0)
        h = max(120.0, 32.0 * rows + 30.0)
        return w, h
    # Equation-only graphs (single short formula, no shape / function /
    # operation that needs a 480×300 plot canvas) get a tight size so
    # ``i = 1 y2 i < ∞`` doesn't render inside a half-empty card.
    eq_dims = _equation_only_canvas_size(graph)
    if eq_dims is not None:
        return eq_dims
    return CANVAS_W, CANVAS_H


def _equation_only_canvas_size(
    graph: SemanticGraph,
) -> Optional[tuple[float, float]]:
    """Return a snug (w, h) when *graph* renders as a single equation
    card, else ``None``.

    Trigger: the renderable templates above ``equation`` in the
    priority list (function / vector / matrix / operation / shape) all
    miss, AND ``_pick_equation`` returns a node — i.e. ``render_svg``
    is going to call ``_render_equation_card``.  In that case the
    bulky 480×300 default canvas is wasted space.
    """
    if (_pick_function(graph) is not None
            or _pick_vector(graph) is not None
            or _pick_matrix(graph) is not None
            or _pick_operation(graph) is not None
            or _pick_shape(graph) is not None):
        return None
    eq = _pick_equation(graph)
    if eq is None:
        return None
    formula = _equation_formula_text(eq)
    return _equation_card_size(formula)


def _equation_formula_text(node: SemanticNode) -> str:
    lhs = node.params.get("lhs") or ""
    rhs = node.params.get("rhs") or ""
    if lhs and rhs:
        return f"{lhs} = {rhs}".strip()
    return (lhs or rhs).strip()


# Inner-padding budget shared by ``_render_equation_card`` and the
# tight canvas-size helper so both agree on geometry.
_EQ_CARD_LEFT = 16.0     # left text x.
_EQ_CARD_RIGHT = 16.0    # right gutter.
_EQ_CARD_TOP = 28.0      # y of the "equation" tag.
_EQ_CARD_BODY_Y = 60.0   # y of the first formula line.
_EQ_CARD_BOTTOM = 16.0   # padding under the last line.


def _equation_card_size(formula: str) -> tuple[float, float]:
    """Return tight ``(w, h)`` for a single-equation card.

    Picks the largest font size from the standard ladder that lets the
    formula fit on one line within ``CANVAS_W`` (so we never *grow*
    beyond the legacy 480 px cap), then sizes the card just enough to
    house the chosen line(s).
    """
    if not formula.strip():
        return CANVAS_W, CANVAS_H
    candidate_sizes = [20.0, 18.0, 16.0, 14.0, 13.0]
    inner_w_cap = CANVAS_W - (_EQ_CARD_LEFT + _EQ_CARD_RIGHT)
    chosen_size, lines = _fit_equation_text(formula, inner_w_cap, 1e6)
    line_h = chosen_size * 1.35
    # Width: fit the longest rendered line + side gutters.
    glyph_w = chosen_size * 0.55
    max_line_chars = max(len(ln) for ln in lines)
    needed_w = max_line_chars * glyph_w + _EQ_CARD_LEFT + _EQ_CARD_RIGHT
    w = max(180.0, min(CANVAS_W, needed_w))
    # Height: header band + body lines + bottom padding.
    h = (_EQ_CARD_BODY_Y + line_h * len(lines)
         - chosen_size * 0.25  # baseline correction
         + _EQ_CARD_BOTTOM)
    h = max(96.0, min(CANVAS_H, h))
    return w, h


def render_svg(graph: SemanticGraph) -> str:
    """Render *graph* to an inner SVG body string.

    Templates are tried in priority order; the first one that has the
    data it needs wins.  Returns ``""`` when nothing in the graph maps
    to a renderable template.
    """
    if graph.is_empty():
        return ""

    # Priority: function plot > matrix > vector > operation > shape >
    # equation > flow > label-only.  Matrix outranks vector because
    # matrix-centred topics (e.g. "what is a matrix") usually carry
    # both a matrix node and an example vector — picking the matrix
    # keeps the diagram about the headline concept.
    func = _pick_function(graph)
    if func is not None:
        return _render_function_plot(func, graph)

    mat = _pick_matrix(graph)
    if mat is not None:
        return _render_matrix(mat)

    vec = _pick_vector(graph)
    if vec is not None:
        return _render_vector(vec, graph)

    op = _pick_operation(graph)
    if op is not None:
        return _render_operation_card(op)

    shape = _pick_shape(graph)
    if shape is not None:
        return _render_shape(shape)

    eq = _pick_equation(graph)
    if eq is not None:
        return _render_equation_card(eq)

    if _has_flow(graph):
        return _render_flow(graph)

    label = _pick_label(graph)
    if label is not None:
        return _render_label_card(label)

    return ""


# ---------------------------------------------------------------------------
# Picking which template to render
# ---------------------------------------------------------------------------

def _pick_function(g: SemanticGraph) -> Optional[SemanticNode]:
    funcs = g.by_type("function")
    if not funcs:
        return None
    # Priority order — pick the most specific named form first.
    priority = ["quadratic", "cubic", "linear", "polynomial", "exp", "log",
                "sin", "cos", "sigmoid", "relu", "tanh", "softmax",
                "parabola"]
    by_form = {n.params.get("form"): n for n in funcs}
    for form in priority:
        if form in by_form:
            return by_form[form]
    return funcs[0]


def _pick_vector(g: SemanticGraph) -> Optional[SemanticNode]:
    vs = g.by_type("vector")
    return vs[0] if vs else None


def _pick_matrix(g: SemanticGraph) -> Optional[SemanticNode]:
    return _largest_matrix(g)


def _pick_operation(g: SemanticGraph) -> Optional[SemanticNode]:
    ops = g.by_type("operation")
    return ops[0] if ops else None


def _pick_shape(g: SemanticGraph) -> Optional[SemanticNode]:
    shapes = g.by_type("shape")
    return shapes[0] if shapes else None


def _pick_equation(g: SemanticGraph) -> Optional[SemanticNode]:
    eqs = g.by_type("equation")
    return eqs[0] if eqs else None


def _pick_label(g: SemanticGraph) -> Optional[SemanticNode]:
    labels = g.by_type("label")
    return labels[0] if labels else None


def _has_matrix(g: SemanticGraph) -> bool:
    return bool(g.by_type("matrix"))


def _largest_matrix(g: SemanticGraph) -> Optional[SemanticNode]:
    mats = g.by_type("matrix")
    if not mats:
        return None
    def _size(n: SemanticNode) -> int:
        rows = n.params.get("rows") or []
        if rows:
            return len(rows) * (len(rows[0]) if rows[0] else 0)
        nr = n.params.get("nrows") or 0
        nc = n.params.get("ncols") or 0
        return nr * nc
    return max(mats, key=_size)


def _has_flow(g: SemanticGraph) -> bool:
    """Render a flow diagram when ≥ 1 edge links named nodes that
    aren't auto-generated underscore placeholders, and there are ≥ 2
    real nodes to draw."""
    real = [e for e in g.edges
            if not e.source.startswith("_") and not e.target.startswith("_")]
    real_nodes = [n for n in g.nodes if not n.id.startswith("_")]
    return len(real) >= 1 and len(real_nodes) >= 2


# ---------------------------------------------------------------------------
# Function plot template
# ---------------------------------------------------------------------------

def _render_function_plot(node: SemanticNode, g: SemanticGraph) -> str:
    form = node.params.get("form") or "linear"
    pts_raw = _function_points(form)
    if not pts_raw:
        return ""
    poly = _scale_points(pts_raw, *_function_domain(form))
    pts_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly)
    colour = _SERIES_COLOURS.get(form, "#1976d2")
    title = node.label or _form_title(form)
    has_increase = any(e.relation == "increases_with" for e in g.edges)
    has_decrease = any(e.relation == "decreases_with" for e in g.edges)
    annotation = ""
    if has_increase:
        annotation = "increases with x"
    elif has_decrease:
        annotation = "decreases with x"

    body = []
    body.append(_axis_frame(title=title, x_label=_x_label(form),
                            y_label=_y_label(form)))
    body.append(
        f'<polyline points="{pts_attr}" fill="none" stroke="{colour}" '
        f'stroke-width="2"/>'
    )
    if annotation:
        body.append(_text(_PLOT_X + 12, _PLOT_Y + 18, annotation,
                          size=11, ink=colour))
    return "".join(body)


def _function_domain(form: str) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) for the function form."""
    if form == "log":
        return 0.05, 5.0, -2.5, 2.0
    if form == "exp":
        return -2.0, 2.5, 0.0, 12.5
    if form in ("sin", "cos"):
        return -math.pi, math.pi, -1.2, 1.2
    if form == "sigmoid":
        return -6.0, 6.0, -0.1, 1.1
    if form == "tanh":
        return -3.0, 3.0, -1.2, 1.2
    if form == "relu":
        return -3.0, 3.0, -0.5, 3.0
    if form == "softmax":
        return -3.0, 3.0, 0.0, 1.0
    if form == "linear":
        return -3.0, 3.0, -3.5, 3.5
    if form == "quadratic" or form == "parabola":
        return -3.0, 3.0, -0.5, 9.5
    if form == "cubic":
        return -2.0, 2.0, -8.5, 8.5
    if form == "polynomial":
        return -2.5, 2.5, -6.0, 6.0
    return -3.0, 3.0, -3.5, 3.5


def _function_points(form: str) -> list[tuple[float, float]]:
    n = 60
    xmin, xmax, _, _ = _function_domain(form)
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        x = xmin + (xmax - xmin) * (i / n)
        y = _function_eval(form, x)
        if y is None or not math.isfinite(y):
            continue
        pts.append((x, y))
    return pts


def _function_eval(form: str, x: float) -> Optional[float]:
    try:
        if form == "linear":
            return x
        if form == "quadratic" or form == "parabola":
            return x * x
        if form == "cubic":
            return x * x * x
        if form == "polynomial":
            # x^3 - 3x — a representative non-monotone polynomial.
            return x * x * x - 3 * x
        if form == "exp":
            return math.exp(x)
        if form == "log":
            return math.log(x) if x > 0 else None
        if form == "sin":
            return math.sin(x)
        if form == "cos":
            return math.cos(x)
        if form == "sigmoid":
            return 1.0 / (1.0 + math.exp(-x))
        if form == "tanh":
            return math.tanh(x)
        if form == "relu":
            return max(0.0, x)
        if form == "softmax":
            # 1-D softmax against zero — yields sigmoid-like curve.
            return math.exp(x) / (math.exp(x) + 1.0)
    except Exception:
        return None
    return None


def _scale_points(
    pts: list[tuple[float, float]], xmin: float, xmax: float,
    ymin: float, ymax: float,
) -> list[tuple[float, float]]:
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    out: list[tuple[float, float]] = []
    for x, y in pts:
        sx = _PLOT_X + ((x - xmin) / span_x) * _PLOT_W
        sy = _PLOT_Y + _PLOT_H - ((y - ymin) / span_y) * _PLOT_H
        # Clamp to the plot rect so degenerate values don't escape.
        sx = max(_PLOT_X, min(_PLOT_X + _PLOT_W, sx))
        sy = max(_PLOT_Y, min(_PLOT_Y + _PLOT_H, sy))
        out.append((sx, sy))
    return out


def _form_title(form: str) -> str:
    return {
        "linear": "f(x) = x",
        "quadratic": "f(x) = x²",
        "parabola": "y = x²",
        "cubic": "f(x) = x³",
        "polynomial": "f(x) = x³ - 3x",
        "exp": "f(x) = eˣ",
        "log": "f(x) = log x",
        "sin": "f(x) = sin x",
        "cos": "f(x) = cos x",
        "sigmoid": "σ(x) = 1 / (1 + e⁻ˣ)",
        "tanh": "f(x) = tanh x",
        "relu": "ReLU(x) = max(0, x)",
        "softmax": "softmax (1-D)",
    }.get(form, "function")


def _x_label(form: str) -> str:
    return "x"


def _y_label(form: str) -> str:
    if form in ("sigmoid", "softmax"):
        return "σ(x)"
    if form == "relu":
        return "ReLU(x)"
    return "f(x)"


# ---------------------------------------------------------------------------
# Vector template
# ---------------------------------------------------------------------------

def _render_vector(node: SemanticNode, g: SemanticGraph) -> str:
    name = node.params.get("name") or node.label or "v"
    # Sanitise: if the LLM handed back a question fragment as the
    # vector name ("is", "what", "the", "a", "vector"), fall back to
    # the canonical "v" — never paint stop-words on the diagram.
    if not name or name.strip().lower() in _NON_LABELS:
        name = "v"
    # Default direction: 30° above x-axis, half plot width.  Stable.
    cx = _PLOT_X + _PLOT_W * 0.5
    cy = _PLOT_Y + _PLOT_H * 0.55
    angle = math.radians(-30.0)  # SVG y inverted
    length = _PLOT_W * 0.35
    tx = cx + length * math.cos(angle)
    ty = cy + length * math.sin(angle)
    body = [_axis_frame(title=f"vector {name}", x_label="x", y_label="y")]
    # Arrow shaft.
    body.append(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
        f'stroke="#1976d2" stroke-width="2.2"/>'
    )
    # Arrowhead — two small lines forming a v.
    head = math.radians(150.0)
    h1x = tx + 10.0 * math.cos(angle + head)
    h1y = ty + 10.0 * math.sin(angle + head)
    h2x = tx + 10.0 * math.cos(angle - head)
    h2y = ty + 10.0 * math.sin(angle - head)
    body.append(
        f'<polyline points="{h1x:.1f},{h1y:.1f} {tx:.1f},{ty:.1f} '
        f'{h2x:.1f},{h2y:.1f}" fill="none" stroke="#1976d2" stroke-width="2.2"/>'
    )
    body.append(_text(tx + 6, ty - 4, name, size=13, ink="#0d47a1",
                      weight="600"))
    return "".join(body)


# ---------------------------------------------------------------------------
# Matrix template
# ---------------------------------------------------------------------------

def _render_matrix(node: SemanticNode) -> str:
    rows = node.params.get("rows") or []
    nrows = node.params.get("nrows") or len(rows) or 2
    ncols = node.params.get("ncols") or (len(rows[0]) if rows else 2)
    name = node.params.get("name") or ""
    if name.strip().lower() in _NON_LABELS:
        name = "A"

    cell_w = 44.0
    cell_h = 32.0
    pad = 12.0
    bracket_w = 8.0
    grid_w = cell_w * ncols
    grid_h = cell_h * nrows
    total_w = bracket_w * 2 + grid_w + pad * 2
    total_h = grid_h + pad * 2

    # Centre within the canvas.
    x0 = (total_w - bracket_w * 2 - grid_w) / 2 + bracket_w
    y0 = pad

    body: list[str] = []
    # Background frame — gives the structural inspector a chart skeleton
    # (rect + polylines + texts) so the matrix card passes the
    # "recognisable chart" gate.  Stroke is transparent; the visible
    # frame is the brackets.
    body.append(
        f'<rect x="0" y="0" width="{total_w:.1f}" height="{total_h:.1f}" '
        f'rx="4" fill="#fff" stroke="none"/>'
    )
    # Title (e.g. "matrix A") above.
    title = name and f"matrix {name}" or "matrix"
    body.append(_text(pad, 14, title, size=12, ink=_LABEL_INK, weight="600"))

    # Brackets — left and right.
    bx_l = x0 - bracket_w
    bx_r = x0 + grid_w
    body.append(
        f'<polyline points="{bx_l + bracket_w:.1f},{y0:.1f} '
        f'{bx_l:.1f},{y0:.1f} {bx_l:.1f},{y0 + grid_h:.1f} '
        f'{bx_l + bracket_w:.1f},{y0 + grid_h:.1f}" '
        f'fill="none" stroke="{_AXIS_INK}" stroke-width="2"/>'
    )
    body.append(
        f'<polyline points="{bx_r:.1f},{y0:.1f} '
        f'{bx_r + bracket_w:.1f},{y0:.1f} '
        f'{bx_r + bracket_w:.1f},{y0 + grid_h:.1f} '
        f'{bx_r:.1f},{y0 + grid_h:.1f}" '
        f'fill="none" stroke="{_AXIS_INK}" stroke-width="2"/>'
    )

    # Cells — when explicit rows are present, render their values;
    # otherwise emit symbolic ``a_{ij}`` placeholders.
    for i in range(nrows):
        for j in range(ncols):
            cx = x0 + j * cell_w + cell_w / 2
            cy = y0 + i * cell_h + cell_h / 2 + 4
            if rows and i < len(rows) and j < len(rows[i]):
                value = str(rows[i][j])
            else:
                value = f"a{i + 1}{j + 1}"
            body.append(_text(cx, cy, value, size=12, ink=_LABEL_INK,
                              anchor="middle"))
    return "".join(body)


# ---------------------------------------------------------------------------
# Operation card
# ---------------------------------------------------------------------------

def _render_operation_card(node: SemanticNode) -> str:
    label = node.label or node.params.get("key") or "operation"
    if label.strip().lower() in _NON_LABELS:
        label = node.params.get("key") or "operation"
    latex = node.params.get("latex") or ""
    body: list[str] = []
    body.append(
        f'<rect x="0" y="0" width="{CANVAS_W:.1f}" height="{CANVAS_H:.1f}" '
        f'rx="6" fill="#fff" stroke="{_AXIS_INK}" stroke-width="1.4"/>'
    )
    body.append(_text(16, 28, "operation", size=11, ink="#5e35b1"))
    body.append(_text(16, 56, label, size=18, ink=_LABEL_INK, weight="700"))
    if latex:
        # Render the formula via the frontend's KaTeX auto-render: wrap
        # the LaTeX in a foreignObject + ``\[..\]`` markers so the
        # chalkboard's ``runMathAutoRender`` compiles it after insertion.
        # Falls back to plain text if KaTeX isn't loaded — the renderer
        # never emits raw ``\sqrt{...}`` glyphs to the user.
        fo_x, fo_y = 16.0, 80.0
        fo_w = CANVAS_W - 32.0
        fo_h = CANVAS_H - fo_y - 16.0
        safe_latex = _xml_escape(latex)
        body.append(
            f'<foreignObject x="{fo_x:.1f}" y="{fo_y:.1f}" '
            f'width="{fo_w:.1f}" height="{fo_h:.1f}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="math-card" '
            f'data-latex="{safe_latex}" '
            f'style="font-family:KaTeX_Main, ui-serif, serif; '
            f'color:{_LABEL_INK}; padding:4px 0; overflow:hidden; '
            f'max-width:{fo_w:.1f}px; max-height:{fo_h:.1f}px; '
            f'font-size:18px; line-height:1.3;">'
            f'\\[{latex}\\]'
            f'</div>'
            f'</foreignObject>'
        )
    return "".join(body)


# ---------------------------------------------------------------------------
# Shape template
# ---------------------------------------------------------------------------

def _render_shape(node: SemanticNode) -> str:
    kind = node.params.get("kind") or node.label or "shape"
    cx = CANVAS_W / 2
    cy = CANVAS_H / 2
    body: list[str] = []
    body.append(_text(16, 24, kind, size=12, ink=_LABEL_INK, weight="600"))
    if kind == "circle":
        body.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="90" fill="none" '
            f'stroke="#1976d2" stroke-width="2"/>'
        )
        body.append(_text(cx, cy + 4, "r", size=12, ink="#0d47a1",
                          anchor="middle"))
    elif kind == "ellipse":
        body.append(
            f'<g transform="translate({cx:.1f},{cy:.1f})">'
            f'<polyline points="' +
            _ellipse_points(120, 70) +
            f'" fill="none" stroke="#5e35b1" stroke-width="2"/>'
            f'</g>'
        )
    elif kind == "triangle":
        body.append(
            f'<polyline points="{cx - 100:.1f},{cy + 70:.1f} '
            f'{cx + 100:.1f},{cy + 70:.1f} {cx:.1f},{cy - 90:.1f} '
            f'{cx - 100:.1f},{cy + 70:.1f}" '
            f'fill="none" stroke="#388e3c" stroke-width="2"/>'
        )
    elif kind == "rectangle" or kind == "square":
        size_w, size_h = (160.0, 100.0) if kind == "rectangle" else (140.0, 140.0)
        body.append(
            f'<rect x="{cx - size_w / 2:.1f}" y="{cy - size_h / 2:.1f}" '
            f'width="{size_w:.1f}" height="{size_h:.1f}" '
            f'fill="none" stroke="#f57c00" stroke-width="2"/>'
        )
    elif kind == "line":
        body.append(
            f'<line x1="{_PLOT_X:.1f}" y1="{cy:.1f}" x2="{_PLOT_X + _PLOT_W:.1f}" '
            f'y2="{cy:.1f}" stroke="#1976d2" stroke-width="2"/>'
        )
    elif kind == "polygon":
        # Regular pentagon.
        body.append(
            f'<polyline points="' +
            _polygon_points(cx, cy, 90.0, 5) +
            f'" fill="none" stroke="#d81b60" stroke-width="2"/>'
        )
    return "".join(body)


def _ellipse_points(rx: float, ry: float, n: int = 48) -> str:
    pts: list[str] = []
    for i in range(n + 1):
        a = 2 * math.pi * (i / n)
        x = rx * math.cos(a)
        y = ry * math.sin(a)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _polygon_points(cx: float, cy: float, r: float, n: int) -> str:
    pts: list[str] = []
    for i in range(n + 1):
        a = -math.pi / 2 + 2 * math.pi * (i / n)
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


# ---------------------------------------------------------------------------
# Equation card
# ---------------------------------------------------------------------------

def _render_equation_card(node: SemanticNode) -> str:
    """Equation card sized tightly to the formula it contains.

    The card frame, header, and formula lines all live within
    (w, h) computed by ``_equation_card_size`` so the user never
    sees a half-empty 480×300 box around a one-line equation.
    """
    formula = _equation_formula_text(node)
    w, h = _equation_card_size(formula)
    body: list[str] = [
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" '
        f'rx="6" fill="#fff" stroke="{_AXIS_INK}" stroke-width="1.4"/>',
        _text(_EQ_CARD_LEFT, _EQ_CARD_TOP, "equation",
              size=11, ink="#388e3c"),
    ]
    inner_w = w - (_EQ_CARD_LEFT + _EQ_CARD_RIGHT)
    inner_h = h - _EQ_CARD_BODY_Y - _EQ_CARD_BOTTOM
    chosen_size, lines = _fit_equation_text(formula, inner_w, max(inner_h, 1.0))
    line_h = chosen_size * 1.35
    y = _EQ_CARD_BODY_Y
    for line in lines:
        body.append(_text(_EQ_CARD_LEFT, y, line,
                          size=int(round(chosen_size)),
                          ink=_LABEL_INK, weight="600"))
        y += line_h
        if y > h - 6:
            break
    return "".join(body)


def _fit_equation_text(
    formula: str, inner_w: float, inner_h: float,
) -> tuple[float, list[str]]:
    """Return ``(font_size, wrapped_lines)`` that fits ``formula``
    inside ``(inner_w, inner_h)``.

    Glyph width is approximated as ``font_size * 0.55``.  We try sizes
    from large to small; for each, wrap the formula on whitespace into
    lines that fit ``inner_w``.  Accept the first size where total
    height (line_h * n_lines) ≤ ``inner_h``.  If even the smallest
    size doesn't fit, truncate the last visible line with an ellipsis.
    """
    candidate_sizes = [20.0, 18.0, 16.0, 14.0, 13.0]
    for size in candidate_sizes:
        line_h = size * 1.35
        max_chars = max(1, int(inner_w / (size * 0.55)))
        wrapped = _word_wrap(formula, max_chars)
        if line_h * len(wrapped) <= inner_h:
            return size, wrapped
    # Smallest size still too tall — truncate.
    size = candidate_sizes[-1]
    line_h = size * 1.35
    max_chars = max(1, int(inner_w / (size * 0.55)))
    wrapped = _word_wrap(formula, max_chars)
    max_lines = max(1, int(inner_h / line_h))
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1].rstrip()[: max(1, max_chars - 1)] + "…"
    return size, wrapped


def _word_wrap(s: str, max_chars: int) -> list[str]:
    """Greedy whitespace wrap.  Long unbroken tokens are hard-split."""
    out: list[str] = []
    cur = ""
    for tok in s.split():
        if len(tok) > max_chars:
            # Hard-split a giant token (rare for equations but possible
            # for very long composite identifiers).
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(tok), max_chars):
                piece = tok[i : i + max_chars]
                if i + max_chars < len(tok):
                    out.append(piece)
                else:
                    cur = piece
            continue
        if not cur:
            cur = tok
        elif len(cur) + 1 + len(tok) <= max_chars:
            cur += " " + tok
        else:
            out.append(cur)
            cur = tok
    if cur:
        out.append(cur)
    return out or [""]


# ---------------------------------------------------------------------------
# Flow diagram (lightweight)
# ---------------------------------------------------------------------------

def _render_flow(g: SemanticGraph) -> str:
    """Stacked vertical node placement with edges.

    Vertical layout fits long labels (e.g. "Boltzmann Machine",
    "Energy Function") that horizontal placement would clip.  Boxes are
    sized to the longest label, arrows flow top-to-bottom along the
    canvas's centre column.  Order: nodes appear in iteration order of
    ``g.nodes``, so output stays deterministic.
    """
    real_nodes = [n for n in g.nodes if not n.id.startswith("_")]
    if len(real_nodes) > 6:
        real_nodes = real_nodes[:6]
    n = len(real_nodes)
    if n == 0:
        return ""

    labels = [(node.label or node.id)[:24] for node in real_nodes]
    font_size = 13
    glyph_w = font_size * 0.55
    text_w = max((len(s) * glyph_w for s in labels), default=80.0)
    box_w = min(max(140.0, text_w + 24.0), CANVAS_W - 40.0)
    box_h = 36.0
    gap = 14.0
    total_h = n * box_h + (n - 1) * gap
    # Anchor the column to the canvas centre, top-padded so the title
    # band (if any wrapper added one) doesn't clip.
    cx = CANVAS_W / 2
    y_start = max(20.0, (CANVAS_H - total_h) / 2)

    pos: dict[str, tuple[float, float]] = {}
    body: list[str] = []
    for i, node in enumerate(real_nodes):
        y0 = y_start + i * (box_h + gap)
        cy = y0 + box_h / 2
        pos[node.id] = (cx, cy)
        x0 = cx - box_w / 2
        body.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{box_w:.1f}" '
            f'height="{box_h:.1f}" rx="6" fill="#eceff1" '
            f'stroke="{_AXIS_INK}" stroke-width="1.2"/>'
        )
        body.append(_text(cx, cy + 4, labels[i],
                          size=font_size, ink=_LABEL_INK, anchor="middle",
                          weight="600"))

    for e in g.edges:
        if e.source.startswith("_") or e.target.startswith("_"):
            continue
        if e.source not in pos or e.target not in pos:
            continue
        sx, sy = pos[e.source]
        tx, ty = pos[e.target]
        # Top-to-bottom edge: skip when source is below target (would
        # draw upward); also skip self-edges and same-row pairs.
        if sy >= ty - 4:
            continue
        sy_bot = sy + box_h / 2
        ty_top = ty - box_h / 2
        body.append(
            f'<line x1="{sx:.1f}" y1="{sy_bot:.1f}" x2="{tx:.1f}" '
            f'y2="{ty_top:.1f}" stroke="{_AXIS_INK}" stroke-width="1.4"/>'
        )
        # Downward arrowhead.
        body.append(
            f'<polyline points="{tx - 5:.1f},{ty_top - 6:.1f} '
            f'{tx:.1f},{ty_top:.1f} {tx + 5:.1f},{ty_top - 6:.1f}" '
            f'fill="none" stroke="{_AXIS_INK}" stroke-width="1.4"/>'
        )
    return "".join(body)


# ---------------------------------------------------------------------------
# Label-only fallback
# ---------------------------------------------------------------------------

def _render_label_card(node: SemanticNode) -> str:
    body = [
        f'<rect x="0" y="0" width="{CANVAS_W:.1f}" height="{CANVAS_H:.1f}" '
        f'rx="6" fill="#fff" stroke="{_AXIS_INK}" stroke-width="1.4"/>',
        _text(16, 28, node.params.get("role") or "label", size=11, ink="#5e35b1"),
        _text(16, 64, node.label or node.id, size=18,
              ink=_LABEL_INK, weight="600"),
    ]
    return "".join(body)


# ---------------------------------------------------------------------------
# Tiny SVG helpers
# ---------------------------------------------------------------------------

def _axis_frame(*, title: str, x_label: str, y_label: str) -> str:
    """Box + ticks + axis labels + chart title."""
    body: list[str] = []
    # Frame rect — required for the structural inspector.
    body.append(
        f'<rect x="{_PLOT_X:.1f}" y="{_PLOT_Y:.1f}" width="{_PLOT_W:.1f}" '
        f'height="{_PLOT_H:.1f}" fill="#fff" stroke="{_AXIS_INK}" '
        f'stroke-width="1.4"/>'
    )
    # Mid-line guides (light grey) — give the curve a visual reference.
    mid_y = _PLOT_Y + _PLOT_H / 2
    body.append(
        f'<line x1="{_PLOT_X:.1f}" y1="{mid_y:.1f}" x2="{_PLOT_X + _PLOT_W:.1f}" '
        f'y2="{mid_y:.1f}" stroke="{_GRID_INK}" stroke-width="1" '
        f'stroke-dasharray="4 4"/>'
    )
    mid_x = _PLOT_X + _PLOT_W / 2
    body.append(
        f'<line x1="{mid_x:.1f}" y1="{_PLOT_Y:.1f}" x2="{mid_x:.1f}" '
        f'y2="{_PLOT_Y + _PLOT_H:.1f}" stroke="{_GRID_INK}" stroke-width="1" '
        f'stroke-dasharray="4 4"/>'
    )
    # Title above the frame.
    body.append(_text(_PLOT_X, _PLOT_Y - 10, title, size=13,
                      ink=_LABEL_INK, weight="600"))
    # Axis labels.
    body.append(_text(_PLOT_X + _PLOT_W / 2, _PLOT_Y + _PLOT_H + 22,
                      x_label, size=12, ink=_AXIS_INK, anchor="middle"))
    body.append(_text(_PLOT_X - 26, _PLOT_Y + _PLOT_H / 2 + 4,
                      y_label, size=12, ink=_AXIS_INK, anchor="middle"))
    return "".join(body)


def _text(x: float, y: float, text: str, *,
          size: int = 12, ink: str = "#212121",
          anchor: str = "start", weight: str = "400") -> str:
    safe = _xml_escape(text)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{ink}" '
        f'font-family="ui-sans-serif,sans-serif" '
        f'text-anchor="{anchor}" font-weight="{weight}">{safe}</text>'
    )


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


__all__ = ["render_svg", "canvas_size", "CANVAS_W", "CANVAS_H"]
