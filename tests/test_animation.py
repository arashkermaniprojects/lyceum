"""Tests for SMIL ``<animateTransform>`` support.

Animation is opt-in: it appears in the rendered SVG only when a node's
``meta["animate"]`` is set, either by hand (low-level API) or via the
animation-aware extraction layer ("rotates X by N degrees", etc.).

These tests exercise both paths.
"""
from __future__ import annotations

from sevim.ir import (
    PlacedGraph, PlacedShape, VisualShape,
)
from sevim.pipeline import run_pipeline
from sevim.s5_render import _animate_transform, render


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_placed(meta: dict | None = None, *, primitive: str = "rect",
                 nid: str = "n_test") -> PlacedShape:
    """Construct a single placed shape for direct render tests."""
    return PlacedShape(
        shape=VisualShape(
            nid=nid, primitive=primitive, label="X",
            width=100.0, height=60.0, font_size=14.0, stroke_width=1.5,
            fill_index=0, meta=dict(meta or {}),
        ),
        x=20.0, y=30.0,
    )


def _make_pg(shape: PlacedShape) -> PlacedGraph:
    return PlacedGraph(shapes=[shape], conns=[], canvas_w=200.0, canvas_h=200.0)


# ---------------------------------------------------------------------------
# Direct renderer / helper tests
# ---------------------------------------------------------------------------

def test_no_animation_meta_produces_no_smil_element():
    """Regression: scenes without `meta["animate"]` must contain zero SMIL."""
    svg = render(_make_pg(_make_placed()))
    assert "<animateTransform" not in svg


def test_rotate_meta_emits_animate_transform_element():
    ps = _make_placed({
        "animate": {"kind": "rotate", "from": 0.0, "to": 90.0,
                    "dur": "2s", "repeat": "indefinite", "around": "center"},
    })
    svg = render(_make_pg(ps))
    assert "<animateTransform" in svg
    # Centre = (x + w/2, y + h/2) = (70, 60).
    assert 'type="rotate"' in svg
    assert 'from="0 70 60"' in svg
    assert 'to="90 70 60"' in svg
    assert 'dur="2s"' in svg
    assert 'repeatCount="indefinite"' in svg


def test_rotate_around_origin_uses_zero_pivot():
    ps = _make_placed({
        "animate": {"kind": "rotate", "from": 10.0, "to": 50.0,
                    "dur": "1s", "repeat": "3", "around": "origin"},
    })
    svg = render(_make_pg(ps))
    assert 'from="10 0 0"' in svg
    assert 'to="50 0 0"' in svg
    assert 'repeatCount="3"' in svg


def test_translate_meta_emits_translate_type():
    ps = _make_placed({
        "animate": {"kind": "translate", "from": 0.0, "to": 40.0,
                    "dur": "1.5s", "repeat": "1"},
    })
    svg = render(_make_pg(ps))
    assert "<animateTransform" in svg
    assert 'type="translate"' in svg
    assert 'from="0 0"' in svg
    assert 'to="40 0"' in svg


def test_scale_meta_emits_scale_type():
    ps = _make_placed({
        "animate": {"kind": "scale", "from": 1.0, "to": 2.0,
                    "dur": "2s", "repeat": "indefinite"},
    })
    svg = render(_make_pg(ps))
    assert "<animateTransform" in svg
    assert 'type="scale"' in svg
    # uniform scale: y component omitted
    assert 'from="1"' in svg
    assert 'to="2"' in svg


def test_unknown_kind_produces_no_animate_element():
    ps = _make_placed({
        "animate": {"kind": "shear", "from": 0.0, "to": 1.0,
                    "dur": "1s", "repeat": "1"},
    })
    assert _animate_transform(ps) == ""
    assert "<animateTransform" not in render(_make_pg(ps))


# ---------------------------------------------------------------------------
# Extraction-layer tests (text → graph → SVG)
# ---------------------------------------------------------------------------

def test_rotation_matrix_creates_transforms_edge_and_animate_meta():
    text = "The rotation matrix R rotates the vector v by 90 degrees."
    result = run_pipeline(text)
    rels = [(e.from_id, e.to_id, e.relation) for e in result.graph.edges]
    # A `transforms` edge must exist.
    transforms = [r for r in rels if r[2] == "transforms"]
    assert transforms, f"expected transforms edge, got {rels}"
    # And the target node must carry the animate meta.
    target_id = transforms[0][1]
    target = next(n for n in result.graph.nodes if n.id == target_id)
    anim = target.meta.get("animate")
    assert anim is not None, f"expected animate meta on {target_id}"
    assert anim["kind"] == "rotate"
    assert anim["to"] == 90.0
    # And the SVG carries the SMIL element.
    assert "<animateTransform" in result.svg
    assert 'type="rotate"' in result.svg


def test_self_rotation_phrasing_creates_animate_meta():
    """'X rotates by N degrees' is the agent-less variant."""
    result = run_pipeline("The square rotates by 45 degrees.")
    # The square should carry animate meta; the exact target id depends on
    # how the label normalises but at least one node must be animated.
    animated = [n for n in result.graph.nodes if n.meta.get("animate")]
    assert animated, "expected at least one animated node"
    assert animated[0].meta["animate"]["to"] == 45.0
    assert "<animateTransform" in result.svg


def test_rendering_is_deterministic_with_animation():
    """I1: identical input → identical SVG bytes, animation included."""
    text = "The rotation matrix M rotates the polygon by 60 degrees."
    a = run_pipeline(text).svg
    b = run_pipeline(text).svg
    assert a == b
    assert "<animateTransform" in a


def test_no_text_animation_no_smil_in_pipeline_output():
    """A scene with no motion verbs must not contain any SMIL element."""
    result = run_pipeline("The triangle is similar to the square.")
    assert "<animateTransform" not in result.svg
