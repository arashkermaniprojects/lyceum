"""Pin the chalkboard's no-overlap guarantee.

User reported §5.8.1 had a tall canonical-figure card (kernel-function
diagram, ~360 px tall) sitting on the same row as a short formula
card (~96 px), but the next short card placed below the formula card
ended up overlapping the bottom of the canonical figure.  The
``ReadingOrderPolicy`` only checked same-row collisions; it now AABB-
checks against every existing shape and pushes new shapes below any
collider.
"""
from __future__ import annotations

from chalkboard import Chalkboard


def _aabb_overlap(a, b) -> bool:
    return (a.x < b.x + b.w and b.x < a.x + a.w
            and a.y < b.y + b.h and b.y < a.y + a.h)


def _add(board: Chalkboard, nid: str, w: float, h: float) -> None:
    board.add(
        nid=nid, svg_body=f"<rect width='{w}' height='{h}'/>",
        primitive="formula_card", label=nid, w=w, h=h,
    )


def test_short_then_tall_then_short_no_overlap():
    """A 96-tall card next to a 360-tall card on the same row, then
    a third card placed afterwards: none of them may overlap."""
    board = Chalkboard(canvas_w=1280, canvas_h=720)
    _add(board, "short_a", w=300, h=96)   # row 0
    _add(board, "tall",    w=500, h=360)  # row 0, taller
    _add(board, "short_b", w=300, h=96)   # must not collide with 'tall'
    _add(board, "short_c", w=300, h=96)   # must not collide with anything
    shapes = list(board.shapes)
    for i, a in enumerate(shapes):
        for b in shapes[i + 1:]:
            assert not _aabb_overlap(a, b), (
                f"{a.nid}@({a.x},{a.y},{a.w},{a.h}) overlaps "
                f"{b.nid}@({b.x},{b.y},{b.w},{b.h})"
            )


def test_many_mixed_size_cards_no_overlap():
    """Stress test: dozens of cards with varying sizes."""
    board = Chalkboard(canvas_w=1280, canvas_h=720)
    sizes = [
        (300, 96), (500, 96), (480, 360), (350, 110),
        (600, 100), (280, 96), (550, 220), (320, 110),
        (480, 360), (280, 64),
    ]
    for i, (w, h) in enumerate(sizes):
        _add(board, f"s{i}", w=w, h=h)
    shapes = list(board.shapes)
    for i, a in enumerate(shapes):
        for b in shapes[i + 1:]:
            assert not _aabb_overlap(a, b), (
                f"overlap: {a.nid} ↔ {b.nid}"
            )


def test_first_shape_lands_at_padding():
    board = Chalkboard(canvas_w=1280, canvas_h=720)
    _add(board, "first", w=200, h=80)
    s = board.shapes[0]
    assert s.x > 0 and s.y > 0
    # Default padding=24
    assert s.x == 24 and s.y == 24
