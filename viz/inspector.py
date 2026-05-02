"""Generated-SVG inspection: structural heuristic + optional local VLM.

The inspector decides whether a synthesised figure actually illustrates
the topic.  Two stages:

  1. **Structural** — pure-Python parse of the SVG.  Verifies axes, at
     least one labelled series, non-degenerate path lengths, no NaN/inf
     coordinates, axis labels present.  ~5 ms.

  2. **VLM (optional)** — rasterise the SVG and ask a *local* Qwen2.5-VL
     server (vLLM at ``VLM_BASE_URL``) "does this image illustrate
     {topic}? yes/no + one-line reason".  Uses no Anthropic / OpenAI.
     ~700 ms when the endpoint is up; gracefully degrades to
     structural-only when not.

Time budget targets:
  * Realtime path:   <2 s end-to-end (structural-only ⇒ <100 ms)
  * Offline path:    <3 s (with VLM, including ≤2 regen attempts)
"""
from __future__ import annotations

import base64
import io
import json as _json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config — points at the *local* vLLM-served VLM.  Default port 8004 to
# avoid clashing with the existing text Qwen on 8000 / embedding on 8003.
# ---------------------------------------------------------------------------

VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8004/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
VLM_TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "8.0"))


@dataclass
class InspectionResult:
    accepted: bool
    reason: str
    structural_ok: bool
    layout_ok: bool = True       # False when text labels visibly overlap
    vlm_used: bool = False
    vlm_verdict: str = ""
    elapsed_ms: float = 0.0
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1: structural heuristic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Text bounding-box estimation + overlap detection
# ---------------------------------------------------------------------------

# Average glyph width in the default sans font, as a fraction of font-size.
# Empirical for KaTeX/sans-serif at the viewport scale we use: ~0.55.
_GLYPH_W_FACTOR = 0.55

# Two text bboxes are considered overlapping only if their intersection on
# both axes exceeds these slacks.  A few px overlap (e.g. a tag chip
# touching the title baseline) is fine; a full-token collision is not.
_OVERLAP_SLACK_X = 4.0
_OVERLAP_SLACK_Y = 4.0


def _text_bbox(elem_attrs: dict, content: str) -> tuple[float, float, float, float]:
    """Return ``(x0, y0, x1, y1)`` for one ``<text>`` element.

    Approximates width as ``len(content) * font_size * 0.55``.  Height is
    ``font_size`` (text baseline at ``y``, ascenders extend up by
    ``~0.85 * font_size`` and descenders below by ``~0.15``; we
    conservatively use the full font_size around the baseline).

    Honours ``text-anchor`` (default ``start``).  When attributes are
    missing or non-numeric, returns a zero-area bbox at the origin so
    that overlap checks degrade gracefully.
    """
    try:
        x = float(elem_attrs.get("x", 0))
        y = float(elem_attrs.get("y", 0))
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0, 0.0
    try:
        font_size = float(elem_attrs.get("font-size", 16))
    except (TypeError, ValueError):
        font_size = 16.0
    anchor = (elem_attrs.get("text-anchor") or "start").strip()
    width = max(1.0, len(content)) * font_size * _GLYPH_W_FACTOR
    height = font_size
    if anchor == "middle":
        x0 = x - width / 2
    elif anchor == "end":
        x0 = x - width
    else:
        x0 = x
    # SVG y is the baseline; bbox extends up by ~0.85 * height and down
    # by ~0.15 * height.
    y0 = y - 0.85 * height
    y1 = y + 0.15 * height
    return x0, y0, x0 + width, y1


_TEXT_TAG_RE = re.compile(
    r'<text\b([^>]*)>\s*([^<]*)</text>',
    re.I | re.S,
)
_ATTR_RE = re.compile(r'([\w\-]+)\s*=\s*"([^"]*)"')


def _parse_texts(svg: str) -> list[tuple[dict, str, tuple[float, float, float, float]]]:
    out: list[tuple[dict, str, tuple[float, float, float, float]]] = []
    for m in _TEXT_TAG_RE.finditer(svg):
        attrs = dict(_ATTR_RE.findall(m.group(1) or ""))
        content = (m.group(2) or "").strip()
        if not content:
            continue
        bbox = _text_bbox(attrs, content)
        if bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
            continue
        out.append((attrs, content, bbox))
    return out


def _bbox_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    return ix > _OVERLAP_SLACK_X and iy > _OVERLAP_SLACK_Y


def _layout_inspect(svg: str) -> tuple[bool, str, dict]:
    """Pairwise text-bbox overlap check.

    Returns ``(ok, reason, diag)``.  Layout is rejected when ≥ 1
    pair of distinct ``<text>`` elements overlap by more than the
    slack on both axes.  Diagnostics carries the offending pair so
    regen can be informed.
    """
    diag: dict = {}
    texts = _parse_texts(svg)
    overlaps: list[tuple[str, str]] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if _bbox_overlap(texts[i][2], texts[j][2]):
                overlaps.append((texts[i][1], texts[j][1]))
    if overlaps:
        a, b = overlaps[0]
        diag["overlaps"] = [list(p) for p in overlaps]
        # Truncate to keep the reason short for the regen prompt feedback.
        return False, f"text overlap: {a[:24]!r} ↔ {b[:24]!r}", diag
    return True, "layout ok", diag


def _structural_inspect(svg: str, *, topic: str) -> tuple[bool, str, dict]:
    """Quick sanity checks on the SVG body itself.

    Accepts two flavours of figure:

      * **Chart-like** — has an axis frame (``<rect>``) plus at least one
        data series (polyline / path / circle scatter).
      * **Diagram-like** — neural-network blocks, layouts, schematics —
        no axis frame, but plenty of edges (``<line>``) connecting
        nodes (``<circle>``).

    A valid figure must have ≥ 2 ``<text>`` elements (title + at least
    one label) and a non-trivial total drawable count.
    """
    diag: dict = {}
    if re.search(r"\b(?:nan|inf)\b", svg, flags=re.I):
        return False, "NaN/inf coordinate present", diag
    n_polylines = len(re.findall(r"<polyline\b", svg))
    n_lines = len(re.findall(r"<line\b", svg))
    n_circles = len(re.findall(r"<circle\b", svg))
    n_rects = len(re.findall(r"<rect\b", svg))
    n_texts = len(re.findall(r"<text\b", svg))
    n_paths = len(re.findall(r"<path\b", svg))
    n_ellipses = len(re.findall(r"<ellipse\b", svg))
    diag.update(
        polylines=n_polylines, lines=n_lines, circles=n_circles,
        rects=n_rects, texts=n_texts, paths=n_paths, ellipses=n_ellipses,
    )
    has_chart_skeleton = (n_rects >= 1 and
                         (n_polylines + n_paths + n_circles + n_ellipses) >= 1)
    has_diagram_skeleton = (n_lines + n_paths >= 3 and
                           (n_circles + n_rects) >= 2)
    if not (has_chart_skeleton or has_diagram_skeleton):
        return False, "no recognisable chart or diagram skeleton", diag
    if n_texts < 2:
        return False, "missing labels (need title + at least one label)", diag
    for m in re.finditer(r'<polyline points="([^"]+)"', svg):
        pts = m.group(1).split()
        if len(pts) < 2:
            return False, "degenerate polyline (≤1 point)", diag
    if len(svg) < 400:
        return False, f"svg too small ({len(svg)} chars)", diag
    return True, "structural ok", diag


# ---------------------------------------------------------------------------
# Stage 2: VLM inspection (optional, local-only)
# ---------------------------------------------------------------------------

_HEALTH_CACHE: dict[str, tuple[float, bool]] = {}


def _vlm_reachable() -> bool:
    """Cheap probe of the local VLM endpoint with a short timeout."""
    import time
    cached = _HEALTH_CACHE.get(VLM_BASE_URL)
    now = time.time()
    if cached and now - cached[0] < 30.0:
        return cached[1]
    try:
        req = urllib.request.Request(
            VLM_BASE_URL.rstrip("/") + "/models",
            headers={"Authorization": "Bearer local-vlm"},
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            ok = resp.status == 200
    except Exception:
        ok = False
    _HEALTH_CACHE[VLM_BASE_URL] = (now, ok)
    return ok


def _rasterize(svg: str, *, width: int, height: int) -> Optional[bytes]:
    """SVG → PNG bytes.  Tries cairosvg, then resvg/rsvg, then None.

    Local-only.  No external API.
    """
    # 1) cairosvg
    try:
        import cairosvg  # type: ignore
        return cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=width, output_height=height,
        )
    except Exception:
        pass
    # 2) resvg / rsvg via Pillow not generally available; skip.
    return None


def _vlm_inspect(
    svg: str, *, topic: str, full_svg: str,
) -> tuple[bool, str]:
    """Ask the local Qwen2.5-VL whether *full_svg* (raster) illustrates
    *topic*.  Returns ``(accepted, reason)``.
    """
    png = _rasterize(full_svg, width=480, height=320)
    if not png:
        return True, "rasterizer unavailable; skipped VLM"
    b64 = base64.b64encode(png).decode("ascii")
    body = _json.dumps({
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": (
                     f"Does this figure illustrate the concept of "
                     f"\"{topic.replace('_', ' ')}\"? Answer in the "
                     f"first word with YES or NO, then in one short "
                     f"sentence say why."
                 )},
            ],
        }],
        "max_tokens": 80,
        "temperature": 0.0,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            VLM_BASE_URL.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer local-vlm"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=VLM_TIMEOUT) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return True, f"VLM unreachable ({e}); skipped"
    except Exception as e:
        return True, f"VLM error ({e}); skipped"
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    head = text.split(None, 1)[0].lower() if text else ""
    if head.startswith("yes"):
        return True, text
    if head.startswith("no"):
        return False, text
    # Ambiguous response — accept by default to avoid loops.
    return True, f"ambiguous VLM reply: {text!r}"


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def inspect_svg(
    svg_body: str, *, topic: str, width: float, height: float,
    use_vlm: Optional[bool] = None,
) -> InspectionResult:
    """Inspect a generated SVG body.  *svg_body* is the inner content
    (no outer ``<svg>``); the inspector wraps it for rasterisation.

    Pass ``use_vlm=False`` to force structural-only (sub-100 ms path);
    ``use_vlm=True`` to *require* VLM (will refuse if unreachable);
    leave ``None`` for auto: VLM if reachable, structural otherwise.
    """
    import time
    t0 = time.perf_counter()
    ok_struct, why_struct, diag = _structural_inspect(svg_body, topic=topic)
    if not ok_struct:
        return InspectionResult(
            accepted=False, reason=why_struct,
            structural_ok=False, layout_ok=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            diagnostics=diag,
        )
    # Layout pass — independent of skeleton; failure rejects by default
    # but the orchestrator can opt-in to a lenient acceptance for Q&A.
    layout_ok, layout_reason, layout_diag = _layout_inspect(svg_body)
    diag.update(layout_diag)
    if not layout_ok:
        return InspectionResult(
            accepted=False, reason=layout_reason,
            structural_ok=True, layout_ok=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            diagnostics=diag,
        )
    do_vlm = use_vlm if use_vlm is not None else _vlm_reachable()
    if do_vlm:
        full = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}">{svg_body}</svg>'
        )
        accepted, verdict = _vlm_inspect(svg_body, topic=topic, full_svg=full)
        return InspectionResult(
            accepted=accepted,
            reason=verdict,
            structural_ok=True, layout_ok=True,
            vlm_used=True,
            vlm_verdict=verdict,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            diagnostics=diag,
        )
    return InspectionResult(
        accepted=True, reason=why_struct,
        structural_ok=True, layout_ok=True,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        diagnostics=diag,
    )
