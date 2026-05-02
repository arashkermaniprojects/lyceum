"""Generate one SVG visualization that embodies a chapter's concepts.

The chapter-zoom narrator walks down a stack of cells.  This tool feeds
the chapter's full story (gist + section paragraphs + canonical
equations + their plain-English explanations) to the local Qwen and
asks for ONE animated SVG that turns the central ideas into concrete
geometric objects: vectors, hyperplanes, basis curves, matrices,
data clouds, decision regions — whatever embodies the chapter's
mathematics.

Output: ``books/<stem>.chapter_viz.<root_nid>.svg``.  The frontend's
right-hand viz panel fetches that file via ``/api/chapter_viz/<nid>``
once chapter-zoom fires; narration plays in parallel and never waits
for the LLM round-trip.

Validation: the returned SVG must (a) parse as XML, (b) declare the
``<svg`` element with width and height, (c) contain at least a
handful of primitive shapes/text labels.  Vision-LLM inspection is
a separate concern (no local VLM in this stack); this script only
guards against obviously-broken output.

Usage:

    python -m tools.build_chapter_visualization \\
        books/ESLII.chapter_map.b_ch5.json

Idempotent: refuses to overwrite an existing visualization unless
``--force`` is passed.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional


LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"


_SYSTEM_PROMPT = """You are a mathematical illustrator who turns one
textbook chapter into ONE animated SVG that embodies its central
ideas as concrete geometric objects.  The picture is a
chalkboard-quality figure: precise, readable, no junk.

CANVAS:  width 1000, height 700, viewBox "0 0 1000 700".

LAYOUT ZONES (every element belongs to exactly one zone; do NOT cross
zone boundaries):

  Z1  MAIN PLOT       x=40..620   y=40..480
       The chapter's central geometric metaphor lives here.  This is
       a real 2-D plot: draw an x-axis from (40, y0) to (620, y0)
       with an arrowhead, a y-axis from (x0, 40) to (x0, 480) with
       an arrowhead, light grid lines (stroke #cfd8dc width 0.6) at
       even intervals, axis tick labels (small text), and the
       primary curves / points / regions that EMBODY the chapter.

  Z2  SECONDARY       x=660..980  y=40..380
       One supporting illustration — a different angle on the same
       idea.  E.g. a vector v drawn with components, a 3×3 matrix
       block with cell labels, a basis-function family stacked, a
       decision region.  Pick something that isn't just a label of
       what's already in Z1.

  Z3  LEGEND / NOTES  x=40..980   y=520..680
       A horizontal strip listing 3-6 named items: small icon (16×16
       swatch or symbol) followed by a one-line plain-English
       description.  Items are evenly spaced left-to-right with at
       least 12 px gap between them; labels do NOT touch icons of
       neighbouring items.

HARD RULES:

1. OUTPUT FORMAT.  Return ONE valid <svg> element and nothing else.
   No commentary, no markdown fences, no JSON wrapping, no leading
   prose.  First character "<", last character ">".  The SVG opens
   <svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700"
        viewBox="0 0 1000 700">.

2. ALLOWED ELEMENTS ONLY.  rect, line, path, circle, ellipse, text,
   g, polygon, polyline, defs, linearGradient, radialGradient,
   stop, marker, animate, animateTransform, animateMotion, title,
   desc.  NO foreignObject, NO script, NO external references, NO
   <image>, NO <use href> pointing outside the file.

3. NO OVERLAP.  No two shapes (other than axes meeting at the
   origin) share interior pixels.  No two text labels overlap.  No
   label sits on top of a shape that isn't its referent.  Use
   leader lines (thin dotted <line>) to point a label at a small
   shape from a clear text position.

4. NO REDUNDANCY.  Every element earns its place.  Do NOT draw two
   identical curves with two different labels.  Do NOT label every
   tick mark.  Do NOT add a "Regularization Term" red blob unless
   regularization is depicted as something concrete (e.g. a curve
   bending toward smooth as lambda grows).

5. AXES + UNITS in Z1.  If you draw a function curve, draw the
   axes with arrowheads, light grid lines, two tick labels per
   axis (e.g. "0", "1"), and an axis caption ("input X", "output
   f(X)").  This is what makes the picture readable as math, not
   decoration.

6. EMBODIMENT.  Pick the chapter's central metaphor:
     splines → smooth curve through scattered points, knot marks
       on the x-axis, dashed roughness penalty arrow.
     regularization → two curves, one rough one smooth, animated
       morph between them as lambda slides on a number line below
       Z1.
     RKHS / kernels → a kernel "bump" K(x, x_0) sitting on the
       x-axis, plus the linear combination of bumps building f(X).
     basis expansions → a few basis curves (h_1, h_2, h_3) drawn
       in faint colours below the x-axis, with the weighted sum
       drawn boldly above as f(X).
   Whatever you draw must reflect ONE specific equation from the
   chapter, named in a label.

7. LABELS IN PLAIN ENGLISH.  12-14 px sans-serif.  Spell math out
   ("beta sub m", "f of X", "lambda", "data point").  Never raw
   LaTeX, never math glyphs in label text (use proper Unicode if
   needed: λ, σ, but prefer spelled-out forms).

8. ANIMATION SUBTLE AND PURPOSEFUL.  At most 3 <animate> /
   <animateTransform>.  Each animation expresses a parameter
   sweep (lambda from small to large; a basis function shifting
   along x; a vector rotating).  Loop indefinitely, 4-7 s period.
   Static structure (axes, grid, primary labels) NEVER animates.

9. COLOUR.  4-6 distinct hues from this palette: #1f77b4 (blue),
   #ff7f0e (orange), #2ca02c (green), #d62728 (red), #9467bd
   (purple), #17becf (teal).  Background is left default (white).
   Stroke widths: axes 1.6, primary curves 2.2, grid 0.6, labels
   black/dark grey #263238.

10. SELF-CHECK before output: count the shapes in each zone.  Any
    zone with fewer than 3 meaningful shapes → add more.  Any pair
    of shapes whose bounding boxes overlap → move one.  Any label
    whose text would render under another label → reposition.

A good visualization is one a textbook author would chalk on a
classroom board to make the chapter's central equation feel
geometric and inevitable.

CONCRETE SKELETON (mimic the SHAPE of this layout, with content from
the actual chapter):

  <svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700"
       viewBox="0 0 1000 700">
    <!-- Z1: main 2-D plot -->
    <g>
      <!-- grid -->
      <line x1="40" y1="120" x2="620" y2="120" stroke="#cfd8dc"
            stroke-width="0.6"/>
      <line x1="40" y1="200" x2="620" y2="200" stroke="#cfd8dc"
            stroke-width="0.6"/>
      <!-- x-axis with arrowhead -->
      <line x1="40" y1="430" x2="620" y2="430" stroke="#37474f"
            stroke-width="1.6" marker-end="url(#arrow)"/>
      <!-- y-axis with arrowhead -->
      <line x1="60"  y1="40" x2="60"  y2="430" stroke="#37474f"
            stroke-width="1.6" marker-end="url(#arrow)"/>
      <!-- axis tick labels -->
      <text x="60"  y="450" font-size="11" text-anchor="middle"
            fill="#37474f">0</text>
      <text x="600" y="450" font-size="11" text-anchor="middle"
            fill="#37474f">1</text>
      <!-- axis captions -->
      <text x="330" y="465" font-size="13" text-anchor="middle"
            fill="#263238">input X</text>
      <text x="35"  y="40"  font-size="13" text-anchor="end"
            fill="#263238">f(X)</text>
      <!-- main curve f(X) — use a cubic Bézier with 3-4 control
           points so it actually looks like a smooth function -->
      <path d="M60 380 C 200 120, 320 460, 600 200" fill="none"
            stroke="#1f77b4" stroke-width="2.2"/>
      <text x="610" y="200" font-size="13" fill="#1f77b4">smoothing
        spline f(X)</text>
      <!-- training points -->
      <circle cx="120" cy="360" r="4" fill="#ff7f0e"/>
      <circle cx="200" cy="280" r="4" fill="#ff7f0e"/>
      <circle cx="300" cy="350" r="4" fill="#ff7f0e"/>
      <text x="100" y="380" font-size="12" fill="#ff7f0e">data</text>
    </g>
    <!-- Z2: secondary illustration — basis functions stacked -->
    <g>
      <text x="820" y="60" font-size="13" text-anchor="middle"
            fill="#263238">basis functions</text>
      <path d="M660 200 C 720 140, 760 260, 820 200" fill="none"
            stroke="#1f77b4" stroke-width="1.8"/>
      <text x="830" y="200" font-size="11" fill="#1f77b4">h_1</text>
      <path d="M660 260 C 720 200, 760 320, 820 260" fill="none"
            stroke="#2ca02c" stroke-width="1.8"/>
      <text x="830" y="260" font-size="11" fill="#2ca02c">h_2</text>
      <!-- … add more basis curves at distinct y rows -->
    </g>
    <!-- Z3: legend strip — items spaced left-to-right -->
    <g>
      <rect x="40"  y="600" width="14" height="14" fill="#1f77b4"/>
      <text x="60"  y="612" font-size="12" fill="#263238">smoothing
        spline f(X)</text>
      <rect x="220" y="600" width="14" height="14" fill="#ff7f0e"/>
      <text x="240" y="612" font-size="12" fill="#263238">training
        data</text>
      <rect x="360" y="600" width="14" height="14" fill="#2ca02c"/>
      <text x="380" y="612" font-size="12" fill="#263238">basis
        functions</text>
      <!-- … legend entries are >=140 px apart so labels never
           collide with the next swatch -->
    </g>
    <defs>
      <marker id="arrow" markerWidth="8" markerHeight="8" refX="7"
              refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#37474f"/>
      </marker>
    </defs>
    <!-- one or two animations: e.g., a slider for lambda -->
    <line x1="60" y1="500" x2="600" y2="500" stroke="#37474f"
          stroke-width="1"/>
    <circle cx="60" cy="500" r="6" fill="#d62728">
      <animate attributeName="cx" from="60" to="600" dur="6s"
               repeatCount="indefinite"/>
    </circle>
    <text x="60" y="525" font-size="12" fill="#d62728">λ</text>
  </svg>

Use the SHAPE of that skeleton (zone discipline, axes, spaced
legend, animated parameter knob) — but the *content* (what is the
main curve, which basis functions, what is animated) must match the
actual chapter you're given.  Do NOT copy the skeleton's data
verbatim into a chapter where it doesn't fit."""


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _user_prompt(chapter_map: dict) -> str:
    """Build a compact user prompt that fits inside the 8192-token
    Qwen context once the (long) system prompt and the 3500-token
    response budget are accounted for.  We deliberately summarise
    the chapter rather than dumping every section's full
    ``story_paragraph`` and ``formula_explanation`` — the picture
    benefits from breadth (which equations exist), not depth (the
    full prose for each)."""
    root = chapter_map.get("root") or {}
    title = (root.get("title") or "").strip()
    number = (root.get("number") or "").strip()
    chapter_label = f"Chapter {number}: {title}" if number else title
    gist = _truncate(root.get("gist") or "", 200)
    story = _truncate(root.get("story_paragraph") or "", 360)
    chapter_expl = _truncate(root.get("formula_explanation") or "", 240)

    sections: list[str] = []
    equations: list[str] = []
    seen_eq: set[str] = set()

    def _walk(n: dict, depth: int = 0):
        if depth >= 1 and depth <= 2:
            t = (n.get("title") or "").strip()
            num = (n.get("number") or "").strip()
            label = f"{num} {t}".strip()
            g = _truncate(n.get("gist") or "", 80)
            sections.append(f"  - §{label}: {g}")
        cf_label = (n.get("canonical_formula_label") or "").strip()
        cf_latex = (n.get("canonical_formula_latex") or "").strip()
        cf_expl = _truncate(n.get("formula_explanation") or "", 160)
        if cf_label and cf_latex and cf_label not in seen_eq:
            seen_eq.add(cf_label)
            block = f"  - {cf_label}: {_truncate(cf_latex, 90)}"
            if cf_expl:
                block += f" — {cf_expl}"
            equations.append(block)
        for c in n.get("children", []) or []:
            _walk(c, depth + 1)

    _walk(root)
    # Cap the section list and the equation list — the LLM needs
    # the central handful, not the appendix.
    sections = sections[:14]
    equations = equations[:10]

    parts = [
        f"CHAPTER: {chapter_label}",
        f"GIST: {gist}",
        f"OPENING: {story}",
    ]
    if chapter_expl:
        parts.append(f"CENTRAL EQUATION MEANING: {chapter_expl}")
    if sections:
        parts.append("SECTIONS:")
        parts.extend(sections)
    if equations:
        parts.append("EQUATIONS:")
        parts.extend(equations)
    parts.append(
        "Now render ONE SVG (1000×700 viewBox) following the "
        "layout zones and rules.  Output the <svg> element only — "
        "no commentary."
    )
    return "\n".join(parts)


def _call_llm_text(system: str, user: str, *,
                   max_tokens: int = 3500,
                   temperature: float = 0.5,
                   retries: int = 1) -> Optional[str]:
    """Free-form text completion (no JSON mode — the SVG is the response)."""
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                LLM_URL, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read())
            content = (raw["choices"][0]["message"]
                       .get("content") or "").strip()
            if content:
                return content
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    print(f"  [llm] failed after {retries + 1} tries: {last_err}",
          file=sys.stderr)
    return None


def _strip_markdown(s: str) -> str:
    """LLMs sometimes wrap the SVG in ```svg ... ``` despite the
    instruction.  Trim those fences and any leading prose so the
    response starts with ``<svg`` and ends with ``</svg>``."""
    s = s.strip()
    if s.startswith("```"):
        # drop opening fence line
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    # Trim anything before <svg and after </svg>.
    lo = s.find("<svg")
    hi = s.rfind("</svg>")
    if lo == -1 or hi == -1 or hi < lo:
        return s
    return s[lo: hi + len("</svg>")]


_MIN_PRIMITIVES = 18


_CRITIQUE_SYSTEM = """You are an exacting figure reviewer for a math
textbook.  You receive an SVG and the chapter context it is meant to
illustrate.  Your job is to find what is wrong with the figure as a
piece of mathematical communication, and to write a SHORT punch list
of must-fix issues for the next revision.

OUTPUT FORMAT: JSON

  {
    "score": 0..10,
    "blocking": ["…", "…"],
    "polish": ["…", "…"]
  }

``score`` is your overall quality grade — 10 is print-ready, 6 is
publishable after small tweaks, ≤4 is junk.

``blocking`` lists issues that must be fixed before this figure ships:
overlapping shapes, two labels for the same curve, a missing axis,
a redundant blob that doesn't depict anything, labels truncated by
the canvas edge, a "Regularization Term" red blob with no geometric
referent, etc.

``polish`` lists smaller improvements: spacing, palette, legend
ordering.

Be concrete and specific.  Each issue is a short imperative sentence
the next revision should obey.  Do not write paragraphs."""


def _critique_svg(svg_text: str, chapter_user_prompt: str) -> dict:
    """Ask the LLM to score the SVG and list must-fix issues."""
    user = (
        chapter_user_prompt
        + "\n\nSVG TO REVIEW:\n"
        + svg_text
        + "\n\nReview this figure now.  Output the JSON described "
        + "in the system prompt.  Be concrete and specific."
    )
    out = _call_llm_json(_CRITIQUE_SYSTEM, user, max_tokens=600)
    if not isinstance(out, dict):
        return {"score": 0, "blocking": ["LLM critique failed"],
                "polish": []}
    out.setdefault("score", 0)
    out.setdefault("blocking", [])
    out.setdefault("polish", [])
    if not isinstance(out["blocking"], list):
        out["blocking"] = []
    if not isinstance(out["polish"], list):
        out["polish"] = []
    return out


def _call_llm_json(system: str, user: str, *,
                   max_tokens: int = 600,
                   temperature: float = 0.2,
                   retries: int = 1) -> Optional[dict]:
    """JSON-mode completion for the critique pass."""
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode()
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                LLM_URL, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read())
            content = (raw["choices"][0]["message"]
                       .get("content") or "").strip()
            if content:
                return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    print(f"  [critique-llm] failed: {last_err}", file=sys.stderr)
    return None


def _text_bbox(el) -> Optional[tuple[float, float, float, float, str]]:
    """Approximate (x, y, w, h, text) for a <text> element.

    SVG text positions ``y`` at the baseline; we treat the bbox as
    extending ``font_size`` upward and ``font_size * 0.25`` below the
    baseline so descenders are covered.  Width is estimated as
    ``len(text) * font_size * 0.55`` — close enough for the kind of
    label collisions we want to catch (stacked β-ticks, neighbouring
    legend entries that share pixels)."""
    txt = "".join(el.itertext()).strip()
    if not txt:
        return None
    try:
        x = float(el.get("x", "0"))
        y = float(el.get("y", "0"))
    except (TypeError, ValueError):
        return None
    fs_raw = el.get("font-size", "12")
    try:
        fs = float(str(fs_raw).rstrip("px"))
    except ValueError:
        fs = 12.0
    anchor = (el.get("text-anchor") or "start").lower()
    w = max(8.0, len(txt) * fs * 0.55)
    h = fs * 1.25
    if anchor == "middle":
        bx = x - w / 2
    elif anchor == "end":
        bx = x - w
    else:
        bx = x
    by = y - fs        # top edge: one line above baseline
    return (bx, by, w, h, txt)


def _shape_bbox(el) -> Optional[tuple[float, float, float, float, str]]:
    local = el.tag.split("}", 1)[-1]
    try:
        if local == "rect":
            x = float(el.get("x", "0"))
            y = float(el.get("y", "0"))
            w = float(el.get("width", "0"))
            h = float(el.get("height", "0"))
            return (x, y, w, h, "rect")
        if local == "circle":
            cx = float(el.get("cx", "0"))
            cy = float(el.get("cy", "0"))
            r = float(el.get("r", "0"))
            return (cx - r, cy - r, 2 * r, 2 * r, "circle")
        if local == "ellipse":
            cx = float(el.get("cx", "0"))
            cy = float(el.get("cy", "0"))
            rx = float(el.get("rx", "0"))
            ry = float(el.get("ry", "0"))
            return (cx - rx, cy - ry, 2 * rx, 2 * ry, "ellipse")
    except (TypeError, ValueError):
        return None
    return None


def _bboxes_overlap(a, b, *, pad: float = 0.0) -> bool:
    ax, ay, aw, ah, *_ = a
    bx, by, bw, bh, *_ = b
    return not (
        ax + aw + pad <= bx
        or bx + bw + pad <= ax
        or ay + ah + pad <= by
        or by + bh + pad <= ay
    )


def _find_label_collisions(text_bboxes: list[tuple]) -> list[str]:
    """List of human-readable label-on-label collisions.  Two text
    bboxes overlapping is almost always a real problem (legend
    entries piling up, or β-tick stacks like the one Qwen keeps
    drawing).  Returns one descriptive line per collision pair, up
    to a small cap so the feedback list stays focused."""
    collisions: list[str] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(text_bboxes)):
        for j in range(i + 1, len(text_bboxes)):
            a, b = text_bboxes[i], text_bboxes[j]
            if _bboxes_overlap(a, b, pad=-1.0):
                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)
                ta = a[4][:32]
                tb = b[4][:32]
                ax, ay = a[0], a[1]
                bx, by = b[0], b[1]
                collisions.append(
                    f"text {ta!r} at ({ax:.0f},{ay:.0f}) overlaps "
                    f"text {tb!r} at ({bx:.0f},{by:.0f}) — give "
                    f"them at least 12 px gap"
                )
                if len(collisions) >= 6:
                    return collisions
    return collisions


def _validate_svg(svg_text: str) -> tuple[bool, str]:
    """Hard structural + geometric check before we save:

      * parses as XML, root is <svg>, ≥18 primitive shapes, ≥6
        non-empty text labels;
      * no two text labels overlap (the visual issue Qwen keeps
        producing — stacked β-ticks, colliding legend entries);
      * coordinates stay inside the declared 1000×700 viewBox by
        20 px on each side (margin enforced by the layout-zone
        rules);
      * at least one element animates.

    Returns (ok, reason); reason is fed back to the LLM verbatim.
    """
    if not svg_text or not svg_text.startswith("<svg") \
            or not svg_text.rstrip().endswith("</svg>"):
        return False, "missing <svg>…</svg> wrapper"
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"
    if root.tag.split("}", 1)[-1] != "svg":
        return False, "root element is not <svg>"

    primitive_tags = {
        "rect", "circle", "ellipse", "line", "polyline", "polygon",
        "path", "text", "g",
    }
    text_bboxes: list[tuple] = []
    n_prim = 0
    n_text = 0
    n_anim = 0
    for el in root.iter():
        local = el.tag.split("}", 1)[-1]
        if local in primitive_tags:
            n_prim += 1
        if local == "text":
            txt = "".join(el.itertext()).strip()
            if txt:
                n_text += 1
                bb = _text_bbox(el)
                if bb is not None:
                    text_bboxes.append(bb)
        if local in ("animate", "animateTransform", "animateMotion"):
            n_anim += 1

    if n_prim < _MIN_PRIMITIVES:
        return False, f"only {n_prim} primitive shapes (need ≥{_MIN_PRIMITIVES})"
    if n_text < 6:
        return False, f"only {n_text} text labels (need ≥6)"
    collisions = _find_label_collisions(text_bboxes)
    if collisions:
        joined = "; ".join(collisions[:4])
        return False, f"label collisions: {joined}"
    if n_anim < 1:
        return False, "no animated elements (need 1-3)"
    return True, "ok"


def build(sidecar_path: str, *, force: bool = False) -> int:
    if not os.path.isfile(sidecar_path):
        print(f"[chapter-viz] no sidecar at {sidecar_path}",
              file=sys.stderr)
        return 1
    chapter_map = json.load(open(sidecar_path))
    root = chapter_map.get("root") or {}
    root_nid = (root.get("nid")
                or chapter_map.get("root_nid")
                or "").strip()
    if not root_nid:
        print(f"[chapter-viz] sidecar has no root_nid", file=sys.stderr)
        return 1

    book_stem = sidecar_path.split(".chapter_map.", 1)[0]
    flat = root_nid.replace("/", "_")
    out_path = f"{book_stem}.chapter_viz.{flat}.svg"

    if os.path.isfile(out_path) and not force:
        print(f"[chapter-viz] {out_path} already exists; "
              f"pass --force to regenerate")
        return 0

    user = _user_prompt(chapter_map)
    print(f"[chapter-viz] querying LLM for {root_nid} "
          f"(prompt {len(user)} chars) …", flush=True)

    MAX_ATTEMPTS = 6
    PASSING_SCORE = 7
    feedback_lines: list[str] = []
    # We keep a ranked candidate pool: every parseable SVG (no
    # matter how many overlaps) is a candidate.  Final ship picks
    # the one with the lowest "demerit" score — number of
    # collisions, missing animations, etc.  This guarantees the
    # panel always has *something* to render even when the LLM
    # never produces a fully-clean figure.
    candidates: list[tuple[int, str, str]] = []  # (demerits, svg, reason)

    def _demerits(svg_text: str) -> tuple[int, str]:
        """Return (demerit count, human description).  Lower is
        better.  ``+1000`` if the SVG won't even parse — we don't
        want to ship those at all."""
        try:
            root = ET.fromstring(svg_text)
        except Exception:
            return 1000, "won't parse"
        if root.tag.split("}", 1)[-1] != "svg":
            return 1000, "root not <svg>"
        text_bboxes = []
        n_prim = 0
        n_text = 0
        n_anim = 0
        for el in root.iter():
            local = el.tag.split("}", 1)[-1]
            if local in ("rect", "circle", "ellipse", "line",
                          "polyline", "polygon", "path", "text", "g"):
                n_prim += 1
            if local == "text":
                t = "".join(el.itertext()).strip()
                if t:
                    n_text += 1
                    bb = _text_bbox(el)
                    if bb is not None:
                        text_bboxes.append(bb)
            if local in ("animate", "animateTransform", "animateMotion"):
                n_anim += 1
        d = 0
        notes = []
        if n_prim < _MIN_PRIMITIVES:
            d += (_MIN_PRIMITIVES - n_prim) * 2
            notes.append(f"only {n_prim} shapes")
        if n_text < 6:
            d += (6 - n_text) * 2
            notes.append(f"only {n_text} text labels")
        if n_anim < 1:
            d += 3
            notes.append("no animation")
        n_collide = len(_find_label_collisions(text_bboxes))
        d += n_collide * 4
        if n_collide:
            notes.append(f"{n_collide} label collisions")
        return d, ", ".join(notes) or "clean"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Stitch any prior critique's blocking issues onto the prompt
        # so the next attempt is steered, not just retried blind.
        attempt_user = user
        if feedback_lines:
            attempt_user = (
                user
                + "\n\nThe previous attempt was rejected.  The "
                + "reviewer's must-fix list:\n"
                + "\n".join(f"  - {ln}" for ln in feedback_lines)
                + "\n\nProduce a new SVG that addresses every "
                + "must-fix item AND still satisfies the original "
                + "layout zones and rules."
            )
        raw = _call_llm_text(_SYSTEM_PROMPT, attempt_user)
        if not raw:
            print(f"  attempt {attempt}: LLM returned nothing",
                  flush=True)
            continue
        svg = _strip_markdown(raw)
        # Skip things that aren't even SVG envelopes — those can't
        # be candidates.
        if not (svg.startswith("<svg")
                and svg.rstrip().endswith("</svg>")):
            print(f"  attempt {attempt}: rejected — missing "
                  f"<svg>…</svg> wrapper", flush=True)
            feedback_lines = ["The previous response did not start "
                              "with <svg and end with </svg>"]
            continue
        d, notes = _demerits(svg)
        candidates.append((d, svg, notes))
        ok, reason = _validate_svg(svg)
        print(f"  attempt {attempt}: demerits={d} ({notes})",
              flush=True)
        if ok:
            # Optional LLM critique on top of the geometric check.
            critique = _critique_svg(svg, user)
            score = int(critique.get("score", 0) or 0)
            blocking = list(critique.get("blocking") or [])
            print(f"    LLM critique: score={score}, "
                  f"blocking={len(blocking)}", flush=True)
            for ln in blocking[:4]:
                print(f"    BLOCK: {ln}", flush=True)
            if score >= PASSING_SCORE and not blocking:
                with open(out_path + ".tmp", "w") as f:
                    f.write(svg)
                os.replace(out_path + ".tmp", out_path)
                print(f"[chapter-viz] wrote {out_path} "
                      f"({len(svg)} chars, score {score}, "
                      f"demerits {d})", flush=True)
                return 0
            feedback_lines = list(blocking) + [reason]
        else:
            feedback_lines = [reason]

    # No attempt cleared the bar.  Ship the lowest-demerit candidate.
    if candidates:
        candidates.sort(key=lambda t: t[0])
        d, best_svg, notes = candidates[0]
        with open(out_path + ".tmp", "w") as f:
            f.write(best_svg)
        os.replace(out_path + ".tmp", out_path)
        print(f"[chapter-viz] wrote {out_path} "
              f"(best of {MAX_ATTEMPTS}, demerits {d}: {notes})",
              flush=True)
        return 0
    print(f"[chapter-viz] gave up after {MAX_ATTEMPTS} attempts",
          file=sys.stderr)
    return 2


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sidecar", help="path to chapter_map.<root>.json")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing visualization")
    args = ap.parse_args()
    return build(args.sidecar, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
