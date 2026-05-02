"""Regression tests for the SVG layout overlap detector.

The structural inspector grew a layout pass that estimates each
``<text>`` element's bounding box and rejects the SVG if any pair of
labels overlap.  This was driven by the user's "Functional" diagram
where 'Function Space', 'Φ(f(x)) = Φ(g(x))', and 'Scalar Value' all
landed on the same y-line and visibly collided on the chalkboard.
"""
from __future__ import annotations

from viz.inspector import _layout_inspect, _text_bbox, inspect_svg


# ---------------------------------------------------------------------------
# Bounding-box estimation
# ---------------------------------------------------------------------------

def test_text_bbox_default_anchor_is_start():
    bbox = _text_bbox({"x": "100", "y": "50", "font-size": "14"}, "hello")
    x0, y0, x1, y1 = bbox
    # Default text-anchor=start → bbox starts at x.
    assert x0 == 100
    # Width ≈ 5 chars * 14 * 0.55 = 38.5
    assert 35 < (x1 - x0) < 42
    # Height ≈ font_size; baseline at y; bbox extends mostly upward.
    assert y0 < 50 < y1


def test_text_bbox_middle_anchor_centers_horizontally():
    bbox = _text_bbox(
        {"x": "240", "y": "20", "font-size": "14",
         "text-anchor": "middle"},
        "Title"
    )
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    assert abs(cx - 240) < 0.1


# ---------------------------------------------------------------------------
# _layout_inspect
# ---------------------------------------------------------------------------

def test_overlap_rejects_collision():
    """Three centred labels stacked horizontally on the same y collide."""
    svg = (
        '<rect x="10" y="200" width="460" height="40" fill="none" '
        'stroke="#000"/>'
        '<text x="200" y="225" font-size="14" '
        'text-anchor="middle">Function Space</text>'
        '<text x="240" y="225" font-size="14" '
        'text-anchor="middle">Phi(f) = Phi(g)</text>'
        '<text x="280" y="225" font-size="14" '
        'text-anchor="middle">Scalar Value</text>'
    )
    ok, reason, diag = _layout_inspect(svg)
    assert not ok
    assert "overlap" in reason.lower()
    assert "overlaps" in diag


def test_layout_accepts_well_spaced_labels():
    """Same canvas, labels spaced 100 px apart — no overlap."""
    svg = (
        '<text x="80"  y="225" font-size="14" '
        'text-anchor="middle">Function Space</text>'
        '<text x="240" y="225" font-size="14" '
        'text-anchor="middle">Phi map</text>'
        '<text x="400" y="225" font-size="14" '
        'text-anchor="middle">Scalar</text>'
    )
    ok, reason, _ = _layout_inspect(svg)
    assert ok, f"unexpected rejection: {reason}"


def test_label_under_node_does_not_collide_with_title():
    svg = (
        '<text x="240" y="20"  font-size="14" '
        'text-anchor="middle">What is a Functional?</text>'
        '<circle cx="120" cy="160" r="20" fill="#1976d2"/>'
        '<text x="120" y="200" font-size="13" '
        'text-anchor="middle">f(x)</text>'
    )
    ok, _, _ = _layout_inspect(svg)
    assert ok


# ---------------------------------------------------------------------------
# Wired into the structural inspector
# ---------------------------------------------------------------------------

def test_inspect_svg_rejects_overlapping_layout():
    """Body that has a real chart skeleton but overlapping labels gets
    rejected by the structural pass."""
    svg = (
        '<rect x="40" y="30" width="420" height="240" '
        'fill="#fff" stroke="#37474f"/>'
        '<polyline points="50,260 200,180 350,120 460,60" '
        'fill="none" stroke="#1976d2" stroke-width="2"/>'
        '<text x="200" y="225" font-size="14" '
        'text-anchor="middle">Function Space</text>'
        '<text x="240" y="225" font-size="14" '
        'text-anchor="middle">Phi(f) = Phi(g)</text>'
        '<text x="280" y="225" font-size="14" '
        'text-anchor="middle">Scalar Value</text>'
        '<text x="240" y="20"  font-size="14" '
        'text-anchor="middle">Functional</text>'
    )
    result = inspect_svg(
        svg, topic="functional", width=480, height=300, use_vlm=False,
    )
    assert not result.accepted
    # Skeleton is fine (rect + polyline + ≥ 2 texts), but layout
    # collides — so structural_ok stays True while layout_ok is False.
    assert result.structural_ok
    assert not result.layout_ok
    assert "overlap" in result.reason.lower()


def test_inspect_svg_accepts_clean_layout():
    """Same skeleton with non-overlapping labels passes."""
    svg = (
        '<rect x="40" y="30" width="420" height="240" '
        'fill="#ffffff" stroke="#37474f" stroke-width="1.4"/>'
        '<polyline points="50,260 100,210 200,180 300,140 350,120 460,60" '
        'fill="none" stroke="#1976d2" stroke-width="2"/>'
        '<line x1="40" y1="150" x2="460" y2="150" '
        'stroke="#cfd8dc" stroke-dasharray="4 4"/>'
        '<text x="60"  y="225" font-size="13" '
        'font-family="ui-sans-serif">Function space</text>'
        '<text x="240" y="20" font-size="14" '
        'font-family="ui-sans-serif" text-anchor="middle">Title</text>'
        '<text x="240" y="290" font-size="12" '
        'font-family="ui-sans-serif" text-anchor="middle">x axis</text>'
    )
    result = inspect_svg(
        svg, topic="functional", width=480, height=300, use_vlm=False,
    )
    assert result.accepted, result.reason
    assert result.structural_ok
