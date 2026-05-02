"""Deterministic geometry checker for placed scenes.

Reports shape-vs-shape intersection (excluding legitimate containment)
and label bleed where the text at its font-size is wider than its box.

No VLM, no subjective verdict — deterministic, byte-stable, cheap.
Designed to back-stop I1 (determinism) with an I4-adjacent property:
*nothing in the rendered scene visually occludes another element.*
"""
from __future__ import annotations

from .ir import PlacedGraph, PlacedShape

# Average glyph width relative to font-size, for a sans-serif face at the
# scales we use. Tuned empirically to match Chromium's default sans-serif
# rendering; intentionally slightly conservative so borderline cases flag.
_AVG_CHAR_W = 0.55


def _rect(p: PlacedShape) -> tuple[float, float, float, float]:
    return (p.x, p.y, p.x + p.shape.width, p.y + p.shape.height)


def _intersect(a, b, tol: float = 0.5) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1 < bx2 - tol and bx1 < ax2 - tol
            and ay1 < by2 - tol and by1 < ay2 - tol)


def _contains(outer, inner) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def _longest_line_chars(label: str, max_chars: int = 16) -> int:
    if len(label) <= max_chars:
        return len(label)
    # Matches the two-line wrap in s5_render._wrap_label.
    words = label.split()
    if len(words) == 1:
        return max_chars
    line1, tally = [], 0
    for i, w in enumerate(words):
        if line1 and tally + 1 + len(w) > max_chars:
            rest = " ".join(words[i:])
            if len(rest) > max_chars * 2:
                rest = rest[: max_chars * 2 - 1] + "…"
            return max(len(" ".join(line1)), len(rest))
        line1.append(w)
        tally += len(w) + (1 if line1 else 0)
    return len(" ".join(line1))


def detect_overlaps(pg: PlacedGraph) -> list[dict]:
    """Return deterministic list of overlap/bleed findings.

    Each finding is a dict with a ``kind`` and the offending node id(s).
    Ordering is stable (sorted by nid) so the output is reproducible.
    """
    findings: list[dict] = []
    placed = sorted(pg.shapes, key=lambda p: p.shape.nid)
    rects = {p.shape.nid: _rect(p) for p in placed}

    # 1. Shape-vs-shape intersection. Containment (a legitimate nesting
    # where one shape is flagged is_container and the other's rect is
    # fully inside the container's rect) is NOT reported.
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            ra, rb = rects[a.shape.nid], rects[b.shape.nid]
            if not _intersect(ra, rb):
                continue
            if (a.shape.is_container and _contains(ra, rb)) or \
               (b.shape.is_container and _contains(rb, ra)):
                continue
            findings.append({
                "kind": "shape_overlap",
                "a": a.shape.nid, "b": b.shape.nid,
                "a_rect": ra, "b_rect": rb,
            })

    # 2. Label bleed: longest rendered line exceeds available box width.
    for p in placed:
        s = p.shape
        if s.is_container:
            continue  # container labels ride at the top; handled by layout
        line_chars = _longest_line_chars(s.label)
        approx_text_w = line_chars * _AVG_CHAR_W * s.font_size
        # Inner padding: ~8 px total for non-container rects.
        avail = max(0.0, s.width - 8)
        if approx_text_w > avail:
            findings.append({
                "kind": "label_overflow",
                "a": s.nid,
                "label": s.label,
                "approx_text_w": round(approx_text_w, 1),
                "box_w": s.width,
            })
    return findings


class OverlapError(RuntimeError):
    """Raised by assert_no_overlaps when strict mode is on."""

    def __init__(self, findings: list[dict]):
        super().__init__(
            f"overlap detector found {len(findings)} finding(s): "
            f"{findings[:3]}{'…' if len(findings) > 3 else ''}"
        )
        self.findings = findings


def assert_no_overlaps(pg: PlacedGraph) -> None:
    findings = detect_overlaps(pg)
    if findings:
        raise OverlapError(findings)
