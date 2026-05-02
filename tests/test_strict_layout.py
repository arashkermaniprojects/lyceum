"""Tests for the strict-layout post-pass.

Asserts the two post-conditions hold *after* `resolve_overlaps`:
  P1.  No two non-nested top-level groups have overlapping bounding boxes.
  P2.  Every pair of top-level groups is separated by at least min_gap on
       at least one axis.

Plus determinism, no-op behaviour for clean scenes, and integration with
the SEVIM_STRICT_LAYOUT env var.
"""
import os
import pytest

from sevim.ir import (PlacedConn, PlacedGraph, PlacedShape,
                      VisualConn, VisualShape)
from sevim.pipeline import run_pipeline
from sevim.strict_layout import resolve_overlaps, violations


def _shape(nid: str, x: float, y: float, w: float = 80, h: float = 50,
           is_container: bool = False) -> PlacedShape:
    return PlacedShape(
        shape=VisualShape(
            nid=nid, primitive="rect", label=nid,
            width=w, height=h, font_size=14, stroke_width=1.2,
            fill_index=0, is_container=is_container, meta={},
        ),
        x=x, y=y,
    )


# ---------------------------------------------------------------------------
# Direct-call tests on hand-built PlacedGraphs.
# ---------------------------------------------------------------------------

def test_clean_scene_is_no_op():
    """Scene with already-separated groups is returned unchanged."""
    pg = PlacedGraph(
        shapes=[_shape("a", 10, 10), _shape("b", 200, 10)],
        conns=[], canvas_w=400, canvas_h=200,
    )
    out = resolve_overlaps(pg, parent_of={}, min_gap=12.0)
    assert out.shapes[0].x == 10 and out.shapes[1].x == 200
    assert violations(out, {}, min_gap=12.0) == []


def test_overlapping_pair_is_separated():
    """Two overlapping rectangles get pushed apart."""
    pg = PlacedGraph(
        shapes=[_shape("a", 50, 50, w=100, h=100),
                _shape("b", 80, 80, w=100, h=100)],
        conns=[], canvas_w=400, canvas_h=400,
    )
    assert violations(pg, {}, min_gap=12.0)  # initial violation
    out = resolve_overlaps(pg, parent_of={}, min_gap=12.0)
    assert violations(out, {}, min_gap=12.0) == []  # P1 + P2 hold


def test_min_gap_enforced_even_when_touching():
    """Boxes that just touch are still pushed apart by min_gap."""
    pg = PlacedGraph(
        shapes=[_shape("a", 0, 0, w=50, h=50),
                _shape("b", 50, 0, w=50, h=50)],  # touching at x=50
        conns=[], canvas_w=200, canvas_h=100,
    )
    out = resolve_overlaps(pg, parent_of={}, min_gap=20.0)
    # Now the gap on x must be ≥ 20.
    a, b = out.shapes
    gap_x = max(a.x, b.x) - min(a.x + a.shape.width, b.x + b.shape.width)
    assert gap_x >= 20.0 - 1e-6


def test_container_moves_carry_descendants():
    """When a container is shifted, its children shift by the same delta."""
    # Group A: container 'A' with child 'a_child' at offset (20, 10).
    # Group B: bare 'b'.  A and B overlap.
    pg = PlacedGraph(
        shapes=[
            _shape("A", 50, 50, w=200, h=150, is_container=True),
            _shape("a_child", 70, 100, w=60, h=40),
            _shape("b", 100, 80, w=80, h=80),  # overlaps A
        ],
        conns=[], canvas_w=400, canvas_h=400,
    )
    parent_of = {"a_child": "A"}
    out = resolve_overlaps(pg, parent_of=parent_of, min_gap=10.0)

    # Find post-positions.
    by_id = {ps.shape.nid: ps for ps in out.shapes}
    # Compute container's delta from its starting (50, 50).
    dx_A = by_id["A"].x - 50.0
    dy_A = by_id["A"].y - 50.0
    # Child must have moved by the SAME delta.
    assert abs((by_id["a_child"].x - 70.0) - dx_A) < 1e-6
    assert abs((by_id["a_child"].y - 100.0) - dy_A) < 1e-6
    # All groups now satisfy P1+P2.
    assert violations(out, parent_of, min_gap=10.0) == []


def test_three_groups_all_separated():
    """Three mutually-overlapping groups all converge to separated."""
    pg = PlacedGraph(
        shapes=[_shape("a", 30, 30, 80, 80),
                _shape("b", 40, 40, 80, 80),
                _shape("c", 50, 50, 80, 80)],
        conns=[], canvas_w=400, canvas_h=400,
    )
    out = resolve_overlaps(pg, parent_of={}, min_gap=15.0)
    assert violations(out, {}, min_gap=15.0) == []


def test_resolve_is_deterministic():
    """Running the algorithm twice on the same input returns the same output."""
    def make() -> PlacedGraph:
        return PlacedGraph(
            shapes=[_shape("a", 20, 20, 100, 100),
                    _shape("b", 40, 40, 100, 100),
                    _shape("c", 60, 60, 100, 100)],
            conns=[], canvas_w=400, canvas_h=400,
        )
    a = resolve_overlaps(make(), {}, min_gap=12.0)
    b = resolve_overlaps(make(), {}, min_gap=12.0)
    coords_a = [(p.x, p.y) for p in a.shapes]
    coords_b = [(p.x, p.y) for p in b.shapes]
    assert coords_a == coords_b


def test_connectors_reclipped_after_move():
    """Connector endpoints get re-clipped to the new shape boundaries."""
    a = _shape("a", 50, 50, 80, 80)
    b = _shape("b", 70, 60, 80, 80)
    conn = PlacedConn(
        conn=VisualConn(eid="e_x", from_nid="a", to_nid="b",
                       pattern="arrow-directed", relation="causes"),
        points=[(90, 90), (110, 100)],  # original midpoints
    )
    pg = PlacedGraph(shapes=[a, b], conns=[conn], canvas_w=400, canvas_h=400)
    out = resolve_overlaps(pg, {}, min_gap=10.0)
    # Endpoints should now be on the new boundaries.
    new_a = next(p for p in out.shapes if p.shape.nid == "a")
    new_b = next(p for p in out.shapes if p.shape.nid == "b")
    p_start, p_end = out.conns[0].points
    # Endpoints lie on or inside their respective shape boundaries.
    assert new_a.x - 1e-6 <= p_start[0] <= new_a.x + new_a.shape.width + 1e-6
    assert new_b.x - 1e-6 <= p_end[0] <= new_b.x + new_b.shape.width + 1e-6


# ---------------------------------------------------------------------------
# Pipeline integration via SEVIM_STRICT_LAYOUT env var.
# ---------------------------------------------------------------------------

def test_env_var_off_does_not_invoke_strict_layout(monkeypatch):
    monkeypatch.delenv("SEVIM_STRICT_LAYOUT", raising=False)
    r = run_pipeline("Set A is a subset of set B. Set B is a subset of set C.")
    # No S4.6 trace event when the env var is unset.
    assert not any(t.stage == "S4.6" for t in r.trace)


def test_env_var_on_records_s46_trace_event(monkeypatch):
    monkeypatch.setenv("SEVIM_STRICT_LAYOUT", "1")
    r = run_pipeline("Set A is a subset of set B.")
    s46 = [t for t in r.trace if t.stage == "S4.6"]
    assert len(s46) == 1
    assert "before" in s46[0].refs and "after" in s46[0].refs


def test_env_var_on_resolves_violations(monkeypatch):
    """End-to-end: when the regular layout produces overlaps, strict mode
    drives them to zero."""
    monkeypatch.setenv("SEVIM_STRICT_LAYOUT", "1")
    text = (
        "The universe contains topological space X. "
        "Topological space X contains open set U. "
        "Open set U contains point p. "
        "Point p lies on line L. "
        "Line L is perpendicular to line M."
    )
    r = run_pipeline(text)
    # Build parent_of from the visual graph's containers.  The strict-layout
    # post-pass reduces violations to zero (post-condition P1+P2).
    s46 = next(t for t in r.trace if t.stage == "S4.6")
    assert s46.refs["after"] == 0


def test_strict_pipeline_is_deterministic(monkeypatch):
    monkeypatch.setenv("SEVIM_STRICT_LAYOUT", "1")
    text = ("Set A is a subset of set B. "
            "Set B is a subset of set C. "
            "Element x is a member of set A.")
    a = run_pipeline(text).svg
    b = run_pipeline(text).svg
    assert a == b
