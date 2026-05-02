"""Squarified treemap layout + SVG renderer for the chapter-map primitive.

Includes a small OCR-LaTeX cleaner so the canonical-formula strings the
math graph collected from PyMuPDF's text extraction render through KaTeX.
PyMuPDF emits well-known artefacts: `\\sum` becomes "M X m=1" with the
upper-bound capital and the body capital `X` substituting for the
big-Sigma glyph; transposes drop the caret ("NT" instead of "N^T");
inverses drop the brace ("-1" instead of "^{-1}"); and equation labels
like "(5.42)" trail the body verbatim.  ``_clean_ocr_latex`` walks the
common patterns and rewrites them; KaTeX renders the result.

Algorithm: the squarified treemap by Bruls, Huijsen, van Wijk (2000),
which packs nested rectangles so each child's aspect ratio stays close
to 1.  Pure Python, deterministic — same input → same layout, byte-for-byte.

Public surface:

    layout(weights, x, y, w, h)
        -> [(x, y, w, h), …]            one rect per child weight
    render_chapter_map(map_root, *, w, h)
        -> str                          one SVG <g data-nid="…"> body

The SVG body is dropped onto the chalkboard as a single shape with
``primitive="chapter_map"``; every nested cell is itself a
``<g data-nid="…">`` so the existing per-shape highlight, focus-dim,
and snapshot machinery treat it like any other card.
"""
from __future__ import annotations

import re
from html import escape as _esc
from typing import Iterable


# ---------------------------------------------------------------------------
# OCR-LaTeX cleaner
# ---------------------------------------------------------------------------

def _clean_ocr_latex(latex: str) -> str:
    """Best-effort cleanup of PyMuPDF-extracted LaTeX so KaTeX accepts it.

    Patterns observed on ESLII Ch.5 chapter-map data:

      * ``M X m=1`` → ``\\sum_{m=1}^{M}`` (and the unbounded variant
        ``X i=1``, where the upper bound is implicit).
      * ``ΩN``     → ``\\Omega_N``  (Greek + subscript).
      * ``\\beta mhm(X)`` → ``\\beta_m h_m(X)``  (PDF text loses the
        ``_`` between a Greek/Roman variable and its subscript letter).
      * ``\\theta T`` → ``\\theta^T``  (transpose lost its caret).
      * ``NT``     → ``N^T``         (same, but on bare uppercase).
      * ``)- 1``   → ``)^{-1}``      (matrix inverse).
      * ``||...||2`` → ``\\|...\\|^2``,  ``||...||1`` → ``\\|...\\|_1``.
      * trailing ``, (5.42)``, ``\\#``, control chars → dropped.

    The cleanup is a sequence of regex substitutions; each rule was
    added when a real string in the chapter map didn't render.  We
    keep the original string when KaTeX-incompatible bits remain —
    KaTeX silently drops what it can't parse, so the worst case is
    a blank typeset block instead of a wrong one.
    """
    if not latex:
        return ""
    s = latex

    # Drop OCR artefacts and trailing equation labels.
    s = re.sub(r"[\x00-\x08\x0b-\x1f]", "", s)
    s = s.replace(r"\#", "")
    s = re.sub(r",?\s*\(\d+\.\d+\)\s*$", "", s)

    # Unicode → LaTeX for the few glyphs PyMuPDF leaves verbatim.
    s = s.replace("Σ", r"\sum")
    s = s.replace("Π", r"\prod")
    s = s.replace("∫", r"\int")
    s = s.replace("∂", r"\partial")
    s = s.replace("∇", r"\nabla")
    s = s.replace("∞", r"\infty")
    s = s.replace("∥", r"\|")
    s = s.replace("≤", r"\le")
    s = s.replace("≥", r"\ge")
    s = s.replace("≈", r"\approx")
    s = s.replace("Ω", r"\Omega")
    # Subscripted Omega: "\Omega N" / "\Omega B" → "\Omega_N", "\Omega_B".
    s = re.sub(r"\\Omega\s*([A-Z])", r"\\Omega_\1", s)

    # Rebuild summations.  PyMuPDF turns
    #     \sum_{m=1}^{M}            into    "M X m=1"
    #     \sum_{i=1}                into    "X i=1"
    # The capital `X` is the substituting glyph; the digit chooser
    # is the lower-bound, the leading capital is the upper-bound.
    s = re.sub(r"\b([A-Z])\s+X\s+([a-zA-Z]+)\s*=\s*(\d+)\b",
               r"\\sum_{\2=\3}^{\1}", s)
    s = re.sub(r"\bX\s+([a-zA-Z]+)\s*=\s*(\d+)\b",
               r"\\sum_{\1=\2}", s)

    # Norms with order suffix.  ``||...||2 2`` (PyMuPDF doubles the
    # exponent) → ``\|...\|_2^2``.  ``||...||2`` → ``\|...\|^2``.
    s = re.sub(r"\|\|([^|]+)\|\|\s*(\d+)\s+\2\b",
               lambda m: rf"\|{m.group(1)}\|_{m.group(2)}^{m.group(2)}", s)
    s = re.sub(r"\|\|([^|]+)\|\|\s*(\d+)\b",
               lambda m: rf"\|{m.group(1)}\|^{m.group(2)}", s)
    s = re.sub(r"\|\|([^|]+)\|\|", r"\\|\1\\|", s)

    # Inverse: ")- 1"  →  ")^{-1}" .  Common after a parenthesised
    # matrix expression.
    s = re.sub(r"\)\s*-\s*1\b", r")^{-1}", s)

    # Transpose patterns.  "NT N" / "BT B" / "\theta T D\theta" all
    # mean the leading symbol is transposed.  Word-boundary on T so
    # we don't catch tokens like "Trace" or "Then".
    s = re.sub(r"(\\[a-zA-Z]+)\s+T(?=[\s\W]|$)", r"\1^T", s)
    s = re.sub(r"\b([A-Z])T(?=\s+[A-Z])", r"\1^T", s)

    # Subscripts on Greek letters and bare lower-case names.
    #   "\beta m"      →  "\beta_m"
    #   "\alpha i"     →  "\alpha_i"
    #   "\hat{\alpha}i" →  "\hat{\alpha}_i"
    #
    # Triple-letter run, "<greek> idx var idx" with matching index
    # letters bracketing one variable letter — a very common PyMuPDF
    # squashing pattern for "\beta_m h_m(X)" → "\beta mhm(X)" and
    # "\alpha_i x_i" → "\alpha ixi".  Recognised when the first and
    # third letters are equal and look like subscript indices
    # (i,j,k,l,m,n).  The lookahead after the third letter rejects
    # cases where the run continues into an actual word.  Apply
    # *before* the single-letter rule below so the longer pattern
    # wins.
    s = re.sub(
        r"(\\[a-zA-Z]+)\s+([ijklmn])([a-z])\2(?=[\s\W(]|$)",
        r"\1_\2 \3_\2",
        s,
    )
    s = re.sub(r"(\\[a-zA-Z]+)\s+([a-z])(?![a-zA-Z])", r"\1_\2", s)
    s = re.sub(r"(\\hat\{\\?[a-zA-Z]+\})([a-z])\b", r"\1_\2", s)

    # Variable+single-letter without space already squashed: "yi"/"xi"
    # → "y_i"/"x_i".  Apply only on lower-case+lower-case pairs that
    # aren't real two-letter words.
    _PAIR_KEEP = {"is", "in", "on", "of", "as", "or", "to", "be", "an",
                   "we", "us", "by", "do", "go", "no", "if", "so"}
    def _split_pair(m):
        s = m.group(0)
        if s.lower() in _PAIR_KEEP:
            return s
        return f"{s[0]}_{s[1]}"
    s = re.sub(r"\b([yxh])([ijklmn])\b", _split_pair, s)

    # \tau i+1 → \tau_{i+1}
    s = re.sub(r"(\\[a-zA-Z]+)\s+([a-z]\+\d+)", r"\1_{\2}", s)

    # Collapse runs of whitespace.
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;:")

    # Force inline-style limits on the big operators so the formula
    # renders short.  Inline mode in KaTeX puts ``\sum_{m=1}^{M}``'s
    # bounds *above and below* the symbol by default, which blows up
    # the vertical extent past the slim parent-band height we
    # reserved.  ``\nolimits`` keeps them at the side.  Skip when
    # the cleaner already inserted ``\nolimits`` (idempotent).
    # Note: ``\b`` after ``\sum`` doesn't match when the next char is
    # ``_`` (both ``m`` and ``_`` are word chars in the regex sense),
    # so use ``(?![a-zA-Z])`` to terminate the operator name without
    # depending on word boundaries.  Same for the negative lookahead
    # checking we haven't already inserted \\nolimits.
    s = re.sub(
        r"\\(sum|prod|int|iint|iiint|oint|max|min|argmax|argmin|"
        r"inf|sup|lim|liminf|limsup)(?![a-zA-Z])(?!\\nolimits)",
        r"\\\1\\nolimits",
        s,
    )

    return s


# ---------------------------------------------------------------------------
# Squarified treemap (Bruls et al., 2000)
# ---------------------------------------------------------------------------

def _aspect_worst(row_sum: float, side: float, w: float) -> float:
    """Worst aspect ratio in a row of rectangles whose ``side`` is fixed
    (the short side of the remaining region) and which together carry
    ``row_sum`` of the parent's area."""
    if row_sum <= 0 or w <= 0 or side <= 0:
        return float("inf")
    s2 = side * side
    rs2 = row_sum * row_sum
    # max(side^2 * w / row_sum^2, row_sum^2 / (side^2 * w))
    a = s2 * w / rs2
    b = rs2 / (s2 * w) if w > 0 else float("inf")
    # ``w`` here is the largest item; squarify uses
    #   max(s^2 * w / rs^2,  rs^2 / (s^2 * smallest))
    # but for our purposes both bounds suffice — same as Bruls's original.
    return max(a, b)


def _layout_row(row_weights: list[float], x: float, y: float,
                w: float, h: float, *, horizontal: bool
                ) -> list[tuple[float, float, float, float]]:
    """Place a single row along the short side of the remaining region."""
    rsum = sum(row_weights)
    if rsum <= 0:
        return [(x, y, 0.0, 0.0) for _ in row_weights]
    rects: list[tuple[float, float, float, float]] = []
    if horizontal:
        # row stacked left→right, height occupies full ``h`` of the strip
        cx = x
        for w_i in row_weights:
            frac = w_i / rsum
            rects.append((cx, y, w * frac, h))
            cx += w * frac
    else:
        cy = y
        for w_i in row_weights:
            frac = w_i / rsum
            rects.append((x, cy, w, h * frac))
            cy += h * frac
    return rects


def layout(weights: Iterable[float], x: float, y: float, w: float, h: float
           ) -> list[tuple[float, float, float, float]]:
    """Pack *weights* into rectangles inside the box ``(x, y, w, h)``.

    Returns one (x, y, w, h) per input weight in the same order.  Implements
    Bruls et al.'s squarified algorithm: process weights in descending
    order, accumulate them into a row along the shorter edge while the
    worst aspect ratio improves, then commit and recurse on the
    remaining region.
    """
    weights = list(weights)
    if not weights:
        return []
    total = sum(weights)
    if total <= 0:
        return [(x, y, 0.0, 0.0) for _ in weights]

    # Scale weights to an area equal to w*h so they tile exactly.
    scale = (w * h) / total
    items = [(scale * wt, idx) for idx, wt in enumerate(weights)]
    items.sort(key=lambda p: (-p[0], p[1]))

    # Output buffer indexed by original position.
    out: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)
                                                     for _ in weights]

    cx, cy, cw, ch = x, y, w, h
    i = 0
    while i < len(items):
        # Always pack along the *short* side of the remaining region.
        side = min(cw, ch)
        long_side = max(cw, ch)
        horizontal = (cw >= ch)   # short side is height

        # Greedily extend the row while the aspect ratio improves.
        row_areas: list[float] = []
        row_idxs: list[int] = []
        worst = float("inf")
        j = i
        while j < len(items):
            row_areas_try = row_areas + [items[j][0]]
            rs = sum(row_areas_try)
            # The row's shorter edge length is rs / side; longest item's
            # other edge is side.  Bruls's worst() over the row:
            biggest = max(row_areas_try)
            smallest = min(row_areas_try)
            row_thickness = rs / side if side > 0 else 0
            new_worst = max(
                (side * side * biggest) / (rs * rs) if rs > 0 else float("inf"),
                (rs * rs) / (side * side * smallest) if smallest > 0 else float("inf"),
            )
            if new_worst > worst:
                break
            row_areas = row_areas_try
            row_idxs = row_idxs + [items[j][1]]
            worst = new_worst
            j += 1

        # Place the committed row.
        rs = sum(row_areas)
        thickness = rs / side if side > 0 else 0
        if horizontal:
            row_rects = _layout_row(
                row_areas, x=cx, y=cy, w=cw, h=thickness,
                horizontal=True,
            )
            cy += thickness
            ch -= thickness
        else:
            row_rects = _layout_row(
                row_areas, x=cx, y=cy, w=thickness, h=ch,
                horizontal=False,
            )
            cx += thickness
            cw -= thickness

        for idx, rect in zip(row_idxs, row_rects):
            out[idx] = rect

        i = j
        if not row_idxs:
            # Defensive: never possible to make progress; place the
            # remaining items as a final naive split.
            break

    # Pack any leftovers (the algorithm above handles all items in
    # practice; the loop guard is for floating-point edge cases).
    while i < len(items):
        out[items[i][1]] = (cx, cy, cw, ch)
        i += 1

    return out


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

# Material-design-ish palette indexed by depth.  Easy on the eye and
# distinct enough that nested cells are visually separable.
_DEPTH_FILL = {
    0: "#fff8e1",     # amber 50  — chapter
    1: "#e8f5e9",     # green 50  — section
    2: "#e3f2fd",     # blue 50   — subsection
    3: "#f3e5f5",     # purple 50 — subsubsection
    4: "#fbe9e7",     # deep-orange 50
}
_DEPTH_STROKE = {
    0: "#ff8f00",
    1: "#2e7d32",
    2: "#1565c0",
    3: "#6a1b9a",
    4: "#bf360c",
}
_TITLE_BAR_H = 36.0
_PAD = 8.0
_GIST_LINE_HEIGHT = 22.0


def _rect(x, y, w, h, fill, stroke, sw=1.0, rx=4):
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}" '
        f'rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d(])")


def _split_sentences_simple(text: str) -> list[str]:
    """Lightweight sentence splitter for chapter-zoom cell bodies.
    Doesn't have to be perfect — the frontend tolerates approximate
    matching when locating the currently-spoken sentence."""
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _wrap(text: str, chars_per_line: int) -> list[str]:
    out, line = [], ""
    for word in (text or "").split():
        if len(line) + len(word) + 1 <= chars_per_line:
            line = (line + " " + word).strip()
        else:
            if line:
                out.append(line)
            line = word
    if line:
        out.append(line)
    return out


_PARENT_FORMULA_H = 48.0


def _cell_svg(rect: tuple[float, float, float, float],
              node: dict, *, font_px: float = 17.0,
              render_formula: bool = True,
              sevim_svg: str = "",
              sevim_bbox: tuple[float, float, float, float] | None
              = None) -> str:
    x, y, w, h = rect
    if w < 6 or h < 14:
        return ""
    depth = max(0, int(node.get("depth", 0)))
    fill = _DEPTH_FILL.get(depth, "#fafafa")
    stroke = _DEPTH_STROKE.get(depth, "#90a4ae")
    # The frontend's ``highlightActiveClause`` matches a clause's
    # ``home_nid`` (after slash-to-underscore normalisation) against the
    # shape's ``data-nid`` via substring inclusion.  Cells therefore
    # store the same flattened nid the orchestrator's passage-card
    # emitter uses ("b/ch5/s5_2" → "b_ch5_s5_2") so the rAF highlight
    # walker can recognise which subtree the active clause lives in.
    nid = (node.get("nid", "") or "").replace("/", "_")
    kind = (node.get("kind") or "").replace("_", " ")
    number = (node.get("number") or "").strip()
    title = (node.get("title") or "").strip()
    label = (
        f"§{number} {title}" if number and title
        else (title or kind or "—")
    )

    parts: list[str] = [
        f'<g data-nid="{_esc(nid)}" class="chmap-cell" '
        f'data-depth="{depth}">'
    ]
    parts.append(_rect(x, y, w, h, fill, stroke, sw=1.4 if depth == 0 else 1.0))
    # Title bar — fixed height drawn from constants so the caller can
    # compute the cell's exact h needed to fit its contents.
    title_h = _TITLE_BAR_H + (4.0 if depth == 0 else 0.0)
    parts.append(_rect(x, y, w, title_h, stroke, stroke, sw=0, rx=4))
    title_font = font_px + (4 if depth == 0 else 2)
    chars_per_line = max(8, int(w / (title_font * 0.55)))
    label_short = label
    if len(label_short) > chars_per_line:
        label_short = label_short[: chars_per_line - 1] + "…"
    parts.append(
        f'<text x="{x + 10:.2f}" y="{y + title_h - 9:.2f}" '
        f'font-family="-apple-system, system-ui, sans-serif" '
        f'font-weight="600" fill="#fff" font-size="{title_font:.1f}">'
        f'{_esc(label_short)}</text>'
    )

    # Parent-cell formula band — sits immediately under the title bar
    # when ``render_formula=False`` (i.e. the cell will nest children).
    # Lets every cell carry visible math: the leaves get the larger
    # bottom box, the parents get this slim band.  We also suppress
    # the band when the node merely *inherits* its parent's formula
    # (the chapter-map builder's parent-fallback gave §5.1, §5.2,
    # §5.3 the same Eq 5.1, which would render the same equation
    # over and over down the treemap).
    parent_band_h = 0.0
    parent_band_latex = ""
    parent_band_label = ""
    inherits_parent = bool(node.get("_inherits_parent_formula"))
    if not render_formula and not inherits_parent:
        raw = (node.get("canonical_formula_latex") or "").strip()
        cleaned = _clean_ocr_latex(raw)
        if cleaned and w >= 160 and h >= title_h + _PARENT_FORMULA_H + 80:
            parent_band_h = _PARENT_FORMULA_H
            parent_band_latex = cleaned
            parent_band_label = (
                node.get("canonical_formula_label") or ""
            ).strip()
    if parent_band_h > 0:
        band_y = y + title_h
        band_x = x
        band_w = w
        # Pale-yellow band so the formula stands out against the
        # cell fill but doesn't fight the ``shape-highlight`` red
        # border the active-clause pulse adds.
        parts.append(
            f'<rect x="{band_x:.2f}" y="{band_y:.2f}" '
            f'width="{band_w:.2f}" height="{parent_band_h:.2f}" '
            f'fill="#ffffff" opacity="0.92" '
            f'stroke="{stroke}" stroke-width="0.6" '
            f'data-formula-band="1"/>'
        )
        if parent_band_label:
            parts.append(
                f'<text x="{band_x + 6:.2f}" '
                f'y="{band_y + 13:.2f}" '
                f'font-family="-apple-system, system-ui, sans-serif" '
                f'font-size="10" fill="{stroke}" font-weight="600">'
                f'{_esc(parent_band_label)}</text>'
            )
        safe_latex = (parent_band_latex.replace("&", "&amp;")
                                         .replace("<", "&lt;")
                                         .replace(">", "&gt;")
                                         .replace('"', "&quot;"))
        xhtml_latex = (parent_band_latex.replace("&", "&amp;")
                                         .replace("<", "&lt;")
                                         .replace(">", "&gt;"))
        # ``overflow="hidden"`` clamps any KaTeX glyphs that would
        # otherwise spill into the cell below — prior version used
        # ``visible`` and the upper-bound of ``\sum_M`` reached up
        # into the title bar above and the formula body dropped
        # into the children's title rows.  The font is small enough
        # at 12px that even \sum + sub/super bounds fit inside.
        # ``data-eq-label`` lets ``highlightActiveClause`` find this
        # band by the equation label that the audio just named.
        eq_attr = (parent_band_label.replace('"', "&quot;")
                   if parent_band_label else "")
        parts.append(
            f'<foreignObject x="{band_x + (66 if parent_band_label else 6):.2f}" '
            f'y="{band_y + 4:.2f}" '
            f'width="{band_w - (72 if parent_band_label else 12):.2f}" '
            f'height="{parent_band_h - 8:.2f}" overflow="hidden" '
            f'data-eq-label="{eq_attr}" class="chmap-formula-band">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="math-card" '
            f'data-latex="{safe_latex}" '
            f'style="font-family:KaTeX_Main, ui-serif, serif; '
            f'color:#212121; padding:0; '
            f'font-size:12px; line-height:1; text-align:center; '
            f'overflow:hidden;">'
            f'\\({xhtml_latex}\\)'
            f'</div></foreignObject>'
        )

    # Body text — prefer the multi-sentence ``story_paragraph`` over the
    # single-sentence ``gist`` so each cell carries enough content to
    # justify its slot in the stack.  When the cell shows a formula,
    # append the per-equation parameter walk-through (the same
    # paragraph the narrator reads after the pin sentence) so what
    # the listener hears is also written in the box at eye level.
    # Rendered as HTML inside a ``<foreignObject>`` with one
    # ``<span class="chmap-sent">`` per sentence — the frontend's
    # rAF loop adds ``is-active`` to the span whose text matches the
    # currently-spoken clause, so the listener sees the said
    # sentence highlighted in the cell itself.  Cell height is
    # derived from the same ``_wrap`` line-count the SVG path used,
    # so the foreignObject's reserved height matches the original
    # SVG-rendered version.
    body_top = y + title_h + parent_band_h + _PAD
    story_text = (node.get("story_paragraph")
                   or node.get("gist") or "").strip()
    body_text = story_text
    show_explanation = (node.get("_show_formula")
                         if node.get("_show_formula") is not None
                         else not inherits_parent)
    expl = (node.get("formula_explanation") or "").strip()
    if expl and show_explanation:
        body_text = (story_text + "  " + expl).strip() if story_text else expl
    chars_per_line = max(20, int((w - 24) / (font_px * 0.55)))
    body_lines = _wrap(body_text, chars_per_line) if body_text else []
    body_text_h = len(body_lines) * _GIST_LINE_HEIGHT
    if body_text and w > 80:
        sents = _split_sentences_simple(body_text)
        if not sents:
            sents = [body_text]
        spans_html: list[str] = []
        for sent in sents:
            safe_t = (sent.replace("&", "&amp;")
                          .replace("<", "&lt;")
                          .replace(">", "&gt;")
                          .replace('"', "&quot;"))
            spans_html.append(
                f'<span class="chmap-sent" data-text="{safe_t}">'
                f'{safe_t}</span> '
            )
        parts.append(
            f'<foreignObject x="{x + 12:.2f}" '
            f'y="{body_top:.2f}" '
            f'width="{w - 24:.2f}" '
            f'height="{max(body_text_h, _GIST_LINE_HEIGHT):.2f}" '
            f'overflow="visible">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" '
            f'class="chmap-body" '
            f'style="font-family:-apple-system, system-ui, sans-serif; '
            f'font-size:{font_px:.0f}px; '
            f'line-height:{_GIST_LINE_HEIGHT:.0f}px; '
            f'color:#212121; padding:0;">'
            f'{"".join(spans_html)}'
            f'</div></foreignObject>'
        )
    # Canonical formula pill at the bottom — drawn only when this node
    # owns the equation (math-graph attribution succeeded for it),
    # never when ``_inherits_parent_formula`` is set.  The chapter-map
    # builder propagates a parent's canonical formula down to children
    # the math-graph couldn't attribute, so every section in a
    # chapter would otherwise render the same Equation 5.1 pill 25
    # times down the stack.  Inheritance fallback exists for narration
    # context, not visual repetition; the parent cell (sitting just
    # above in the stack) already shows the inherited equation.
    cf_latex_raw = (node.get("canonical_formula_latex") or "").strip()
    cf_label = (node.get("canonical_formula_label") or "").strip()
    cf_latex = _clean_ocr_latex(cf_latex_raw) if cf_latex_raw else ""
    # ``_show_formula`` is the per-cell decision computed by
    # ``_mark_visible_formulas``: every cell whose narration names
    # this equation paints the pill, every other inheriting cell
    # leaves the math to the parent above it in the stack.
    show_flag = node.get("_show_formula")
    if show_flag is None:
        # Caller didn't run ``_mark_visible_formulas`` — fall back to
        # the conservative "own formula only" rule.
        show_flag = not inherits_parent
    has_formula = bool(cf_latex) and render_formula and bool(show_flag)
    if has_formula and w > 140:
        formula_h = _STACK_FORMULA_H
        formula_box_x = x + 8
        formula_box_y = body_top + body_text_h + _PAD
        formula_box_w = max(60.0, w - 16)
        formula_box_h = formula_h
        safe_latex = (
            cf_latex.replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;").replace('"', "&quot;")
        )
        # Background pill so the formula sits visibly inside the cell.
        parts.append(
            f'<rect x="{formula_box_x:.2f}" y="{formula_box_y:.2f}" '
            f'width="{formula_box_w:.2f}" height="{formula_box_h:.2f}" '
            f'rx="6" fill="#ffffff" '
            f'stroke="{stroke}" stroke-width="1" opacity="0.96" '
            f'data-formula-pill="1"/>'
        )
        if cf_label:
            parts.append(
                f'<text x="{formula_box_x + 10:.2f}" '
                f'y="{formula_box_y + 18:.2f}" '
                f'font-family="-apple-system, system-ui, sans-serif" '
                f'font-size="12" fill="{stroke}" font-weight="600">'
                f'{_esc(cf_label)}</text>'
            )
        xhtml_latex = (cf_latex.replace("&", "&amp;")
                                 .replace("<", "&lt;")
                                 .replace(">", "&gt;"))
        eq_attr = (cf_label.replace('"', "&quot;") if cf_label else "")
        parts.append(
            f'<foreignObject x="{formula_box_x + 8:.2f}" '
            f'y="{formula_box_y + 22:.2f}" '
            f'width="{formula_box_w - 16:.2f}" '
            f'height="{formula_box_h - 26:.2f}" overflow="hidden" '
            f'data-eq-label="{eq_attr}" class="chmap-formula-box">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="math-card" '
            f'data-latex="{safe_latex}" '
            f'style="font-family:KaTeX_Main, ui-serif, serif; '
            f'color:#212121; padding:0; overflow:hidden; '
            f'font-size:20px; line-height:1.15; text-align:center;">'
            f'\\[{xhtml_latex}\\]'
            f'</div></foreignObject>'
        )

    # Concept diagram for this cell, embedded as a nested SVG so the
    # listener sees the relationships next to the prose and equation
    # without scanning to a side panel.  We position it directly
    # below the formula pill (or below the body text when the cell
    # has no formula) — the cell height already reserved
    # ``_STACK_DIAGRAM_H`` worth of space when ``sevim_svg`` is
    # non-empty (see ``_stack_cell_height``).
    if sevim_svg:
        diag_top = body_top + body_text_h + _PAD
        if has_formula and w > 140:
            diag_top += _STACK_FORMULA_H + _PAD
        diag_x = x + 8
        diag_w = w - 16
        diag_h = _diagram_h_for_width(w, sevim_bbox)
        parts.append(_embed_sevim_svg(
            sevim_svg, x=diag_x, y=diag_top,
            w=diag_w, h=diag_h,
            bbox=sevim_bbox,
        ))

    parts.append("</g>")
    return "".join(parts)


def _walk_layout(node: dict, x: float, y: float, w: float, h: float,
                 *, parts: list[str], font_px: float = 11.0) -> None:
    """Recursively layout *node* and its children; append cell SVGs to
    ``parts`` in *outer-first* order so children render on top of their
    parent rectangle (last-in-z-order wins in SVG)."""
    children = list(node.get("children") or [])
    # Parents whose subtree fits inside their own cell skip formula
    # rendering so the inner area stays for the children.  Only leaves
    # (and parents whose children would be too small to lay out) carry
    # the formula box.
    will_nest = bool(children)
    parts.append(_cell_svg((x, y, w, h), node,
                            font_px=font_px,
                            render_formula=not will_nest))
    if not children:
        return

    # Reserve title strip + small inner padding for children.
    depth = max(0, int(node.get("depth", 0)))
    title_h = (min(_TITLE_BAR_H, h * 0.22) if depth > 0
               else min(28.0, h * 0.18))
    pad = 4.0
    # If the parent cell will carry a formula band right under its
    # title bar, push the children's inner area down so they don't
    # overlap the math.  Mirrors the qualification logic in
    # ``_cell_svg`` exactly.
    raw_latex = (node.get("canonical_formula_latex") or "").strip()
    has_parent_band = bool(_clean_ocr_latex(raw_latex)
                            ) and w >= 160 and h >= title_h + _PARENT_FORMULA_H + 80
    parent_band_h = _PARENT_FORMULA_H if has_parent_band else 0.0
    inner_x = x + pad
    inner_y = y + title_h + parent_band_h + pad
    inner_w = max(0.0, w - 2 * pad)
    inner_h = max(0.0, h - title_h - parent_band_h - 2 * pad)
    if inner_w < 30 or inner_h < 30:
        return  # too small to nest further

    # Weight every child by ``1 + 0.3 * subtree_size`` so the leaves
    # always get a visible cell — the previous leaf-count weighting
    # collapsed sections without subsections (§5.1, §5.3, §5.6, §5.7,
    # §5.9 etc.) into thin slivers that rendered as title-only or
    # were filtered out at the < 14 px threshold.  The 0.3 boost
    # still gives meatier subtrees more area without starving the
    # leaves.
    def _weight(n: dict) -> float:
        cs = n.get("children") or []
        if not cs:
            return 1.0
        return 1.0 + 0.3 * sum(_weight(c) for c in cs)
    weights = [_weight(c) for c in children]
    rects = layout(weights, inner_x, inner_y, inner_w, inner_h)
    for child, rect in zip(children, rects):
        _walk_layout(child, *rect, parts=parts, font_px=font_px)


def _mark_inherited_formulas(node: dict, parent_formula_id: str) -> None:
    """Walk the chapter map tree and tag every node that merely
    inherits its parent's canonical formula.  The render code reads
    ``_inherits_parent_formula`` to suppress the duplicate band/pill
    on those nodes, so the visible math doesn't repeat down the
    treemap when the math-graph builder fell back to the parent
    formula because the node's own equations weren't extracted."""
    own_id = (node.get("canonical_formula_id") or "").strip()
    inherits = bool(parent_formula_id and own_id == parent_formula_id)
    node["_inherits_parent_formula"] = inherits
    propagated = own_id or parent_formula_id
    for c in node.get("children", []) or []:
        _mark_inherited_formulas(c, propagated)


def _mark_visible_formulas(root: dict) -> None:
    """Walk the chapter-map in DFS order and mark each node with
    ``_show_formula`` so the renderer paints the formula pill on
    *exactly* the cells whose narration names the equation.

    Mirrors ``plan_chapter_zoom``'s ``last_pin_label`` rule:

      * Chapter root always shows (the headline of the chapter map).
      * Any node that *owns* its formula (math-graph attribution
        succeeded) always shows — this is the cell the narrator's
        story is built around.
      * A node that *inherits* its parent's formula shows only when
        the inherited label hasn't been pinned yet by an earlier
        sibling; that mirrors the planner's pin de-dup, so §5.1
        (the first L1 child to inherit Eq 5.1 after the chapter
        root) shows the equation while §5.2…§5.4 (who would inherit
        the same label) don't.
    """
    last_label = ""

    def _walk(n: dict, *, is_root: bool) -> None:
        nonlocal last_label
        cf_label = (n.get("canonical_formula_label") or "").strip()
        if is_root:
            n["_show_formula"] = bool(cf_label)
            # Root paints visually but doesn't update ``last_label``
            # — the planner doesn't call ``_emit_node`` for the
            # chapter root, so the narrator's first pin happens at
            # the first L1 child even if it inherits the same label.
        elif cf_label and cf_label != last_label:
            n["_show_formula"] = True
            last_label = cf_label
        else:
            n["_show_formula"] = False
        for c in n.get("children", []) or []:
            _walk(c, is_root=False)

    _walk(root, is_root=True)


_STACK_GAP = 14.0
_STACK_MARGIN_X = 8.0
_STACK_FORMULA_H = 110.0
_STACK_DIAGRAM_VIEWBOX_W = 700.0
_STACK_DIAGRAM_VIEWBOX_H = 440.0
# Diagram now fills the cell's inner width and grows in height
# proportionally to the SeVim canvas's 700×440 aspect (about
# 0.629).  Keeps the picture readable on wide layouts instead of
# rendering as a thumbnail with dead space on either side.  The
# height is capped at ``_STACK_DIAGRAM_MAX_H`` so an unusually
# wide column can't blow past sensible vertical space.
_STACK_DIAGRAM_ASPECT = (
    _STACK_DIAGRAM_VIEWBOX_H / _STACK_DIAGRAM_VIEWBOX_W
)
_STACK_DIAGRAM_MIN_H = 160.0
_STACK_DIAGRAM_MAX_H = 560.0


def _diagram_h_for_width(inner_cell_w: float,
                         bbox: tuple[float, float, float, float] | None
                         = None) -> float:
    """Cell-width-aware diagram height — derived from the *actual*
    content bounding box's aspect ratio (so a wide-flat graph gets
    a short diagram band, a tall graph gets a tall one).  Falls
    back to the canonical 700:440 canvas aspect when no bbox is
    available."""
    diag_w = max(0.0, inner_cell_w - 16.0)
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
        cw = max(1.0, (bx1 - bx0))
        ch = max(1.0, (by1 - by0))
        aspect = ch / cw
    else:
        aspect = _STACK_DIAGRAM_ASPECT
    raw = diag_w * aspect
    if raw < _STACK_DIAGRAM_MIN_H:
        return _STACK_DIAGRAM_MIN_H
    if raw > _STACK_DIAGRAM_MAX_H:
        return _STACK_DIAGRAM_MAX_H
    return raw


import re as _re_diag

_RE_SVG_OPEN = _re_diag.compile(r"<svg\b[^>]*>", _re_diag.DOTALL)


def _sevim_content_bbox(svg: str) -> tuple[float, float, float, float] | None:
    """Approximate bounding box of all the visible shapes in a SeVim
    diagram SVG.  We use it as the nested ``viewBox`` so the embed
    fills its container with the actual content rather than SeVim's
    fixed 700×440 canvas (which is mostly empty for short graphs).

    Conservative: walks ``rect``, ``circle``, ``ellipse``, ``line``
    and ``text`` attributes only — paths and polygons are rare in
    SeVim output and the bbox margin we apply absorbs any
    contribution they'd make.
    """
    if not svg:
        return None
    xs: list[float] = []
    ys: list[float] = []
    # rects
    for m in _re_diag.finditer(
        r'<rect\b[^>]*?\bx="([\-0-9.]+)"[^>]*?\by="([\-0-9.]+)"'
        r'[^>]*?\bwidth="([\-0-9.]+)"[^>]*?\bheight="([\-0-9.]+)"',
        svg,
    ):
        x, y, w, h = (float(m.group(1)), float(m.group(2)),
                      float(m.group(3)), float(m.group(4)))
        xs.extend([x, x + w])
        ys.extend([y, y + h])
    # circles
    for m in _re_diag.finditer(
        r'<circle\b[^>]*?\bcx="([\-0-9.]+)"[^>]*?\bcy="([\-0-9.]+)"'
        r'[^>]*?\br="([\-0-9.]+)"',
        svg,
    ):
        cx, cy, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
        xs.extend([cx - r, cx + r])
        ys.extend([cy - r, cy + r])
    # ellipses
    for m in _re_diag.finditer(
        r'<ellipse\b[^>]*?\bcx="([\-0-9.]+)"[^>]*?\bcy="([\-0-9.]+)"'
        r'[^>]*?\brx="([\-0-9.]+)"[^>]*?\bry="([\-0-9.]+)"',
        svg,
    ):
        cx, cy = float(m.group(1)), float(m.group(2))
        rx, ry = float(m.group(3)), float(m.group(4))
        xs.extend([cx - rx, cx + rx])
        ys.extend([cy - ry, cy + ry])
    # lines
    for m in _re_diag.finditer(
        r'<line\b[^>]*?\bx1="([\-0-9.]+)"[^>]*?\by1="([\-0-9.]+)"'
        r'[^>]*?\bx2="([\-0-9.]+)"[^>]*?\by2="([\-0-9.]+)"',
        svg,
    ):
        xs.extend([float(m.group(1)), float(m.group(3))])
        ys.extend([float(m.group(2)), float(m.group(4))])
    # text x/y
    for m in _re_diag.finditer(
        r'<text\b[^>]*?\bx="([\-0-9.]+)"[^>]*?\by="([\-0-9.]+)"',
        svg,
    ):
        xs.append(float(m.group(1)))
        ys.append(float(m.group(2)))
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _embed_sevim_svg(sevim_svg: str, *, x: float, y: float,
                     w: float, h: float,
                     bbox: tuple[float, float, float, float] | None
                     = None) -> str:
    """Strip a SeVim diagram's outer ``<svg …>`` wrapper and re-wrap
    with our own positioned ``<svg>`` whose ``viewBox`` is the
    *content* bounding box, not SeVim's canonical 700×440 canvas.
    That removes the empty bands the canonical canvas leaves around
    short graphs, so the embed fills the cell's diagram band.
    """
    m = _RE_SVG_OPEN.search(sevim_svg or "")
    if m is None:
        return ""
    inner = sevim_svg[m.end():]
    if inner.endswith("</svg>"):
        inner = inner[: -len("</svg>")]
    if bbox is None:
        bbox = _sevim_content_bbox(sevim_svg)
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
        pad = 12.0
        vb_x = bx0 - pad
        vb_y = by0 - pad
        vb_w = (bx1 - bx0) + 2 * pad
        vb_h = (by1 - by0) + 2 * pad
    else:
        vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, 700.0, 440.0
    return (
        f'<svg x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}" '
        f'viewBox="{vb_x:.2f} {vb_y:.2f} {vb_w:.2f} {vb_h:.2f}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'class="chmap-cell-diagram">'
        f'{inner}</svg>'
    )


def _flatten_dfs(node: dict) -> list[dict]:
    out = [node]
    for c in node.get("children", []) or []:
        out.extend(_flatten_dfs(c))
    return out


def _stack_cell_height(node: dict, w: float, *,
                       font_px: float,
                       has_diagram: bool = False,
                       diagram_bbox: tuple[float, float, float, float]
                       | None = None) -> float:
    """Height needed for one stack cell to fit its title, body text and
    optional canonical-formula pill exactly — no padding wasted, no
    content clipped.  Mirrors the layout ``_cell_svg`` performs when
    ``render_formula=True``: title bar at top, body text in the
    middle (every wrapped line of ``story_paragraph`` or ``gist``),
    formula pill at the bottom when the node carries one and isn't
    just inheriting the parent's equation.
    """
    depth = max(0, int(node.get("depth", 0)))
    title_h = _TITLE_BAR_H + (4.0 if depth == 0 else 0.0)
    story_text = (node.get("story_paragraph")
                   or node.get("gist") or "").strip()
    body_text = story_text
    show_explanation = (node.get("_show_formula")
                         if node.get("_show_formula") is not None
                         else not bool(node.get("_inherits_parent_formula")))
    expl = (node.get("formula_explanation") or "").strip()
    if expl and show_explanation:
        body_text = (story_text + "  " + expl).strip() if story_text else expl
    chars_per_line = max(20, int((w - 24) / (font_px * 0.55)))
    body_lines = _wrap(body_text, chars_per_line) if body_text else []
    body_h = len(body_lines) * _GIST_LINE_HEIGHT
    cf_latex = (node.get("canonical_formula_latex") or "").strip()
    show_flag = node.get("_show_formula")
    if show_flag is None:
        show_flag = not bool(node.get("_inherits_parent_formula"))
    has_formula = bool(cf_latex) and bool(show_flag)
    formula_h = _STACK_FORMULA_H if has_formula else 0.0
    formula_pad = _PAD if has_formula else 0.0
    diagram_h = (_diagram_h_for_width(w, diagram_bbox)
                 if has_diagram else 0.0)
    diagram_pad = _PAD if has_diagram else 0.0
    return (title_h + _PAD + body_h
             + formula_pad + formula_h
             + diagram_pad + diagram_h
             + _PAD)


def render_chapter_map(map_root: dict, *, w: float = 1080.0,
                        h: float = 700.0,
                        font_px: float = 17.0,
                        sevim_diagrams: dict[str, str] | None = None,
                        ) -> tuple[str, float]:
    """Render the chapter map as a vertical stack of full-width cells.

    Each node in the chapter-map tree becomes a single full-width cell,
    laid out top-to-bottom in depth-first order.  The active cell is
    centered in the viewport by the frontend's ``scrollIntoView`` hook
    in ``highlightActiveClause``, so the listener follows the
    narration like reading down a page rather than zooming through a
    treemap.  Earlier squarified-treemap layout was hard to read once
    sections nested 3+ levels deep — cells were small and the active
    one's perimeter highlight got lost in neighbour borders.

    The ``h`` parameter is the *suggested* viewport height; the actual
    rendered stack is taller and the function returns its real
    bounding-box height alongside the SVG so the caller can size the
    chalkboard shape correctly.

    Returns ``(svg_body, total_height)``.  Empty input yields
    ``("", 0.0)``.
    """
    if not isinstance(map_root, dict):
        return ("", 0.0)
    _mark_inherited_formulas(map_root, parent_formula_id="")
    _mark_visible_formulas(map_root)
    nodes = _flatten_dfs(map_root)
    inner_w = max(200.0, w - 2 * _STACK_MARGIN_X)
    parts: list[str] = []
    y = 0.0
    for node in nodes:
        nid = (node.get("nid") or "").strip()
        sevim_svg = ""
        if sevim_diagrams and nid:
            sevim_svg = sevim_diagrams.get(nid, "") or ""
        has_diagram = bool(sevim_svg)
        # Compute the SeVim diagram's content bbox once and feed it
        # to both the height calc and the embed so they agree on
        # exactly how much vertical space the diagram needs.
        sevim_bbox = _sevim_content_bbox(sevim_svg) if has_diagram else None
        cell_h = _stack_cell_height(
            node, inner_w, font_px=font_px,
            has_diagram=has_diagram,
            diagram_bbox=sevim_bbox,
        )
        rect = (_STACK_MARGIN_X, y, inner_w, cell_h)
        parts.append(_cell_svg(
            rect, node, font_px=font_px,
            render_formula=True,
            sevim_svg=sevim_svg,
            sevim_bbox=sevim_bbox,
        ))
        y += cell_h + _STACK_GAP
    total_h = max(0.0, y - _STACK_GAP)
    return ("".join(parts), total_h)
