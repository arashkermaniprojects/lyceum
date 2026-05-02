import re

from sevim.pipeline import run_pipeline


def _path_endpoints(svg: str) -> list[tuple[float, float, float, float]]:
    """Extract (x1,y1,x2,y2) from cubic bezier paths carrying a data-eid."""
    out = []
    for m in re.finditer(
        r'<path[^>]*data-eid="[^"]+"[^>]*'
        r'd="M([\d.]+),([\d.]+)\s+C[\d.,\s]+\s+([\d.]+),([\d.]+)"',
        svg,
    ):
        out.append(tuple(float(g) for g in m.groups()))
    return out


def test_endpoints_on_shape_boundary_not_center():
    """A→B causes arrow endpoints should lie on shape boundaries, not centres."""
    r = run_pipeline("A causes B.")
    placed = {p.shape.nid: p for p in r.placed.shapes}
    a, b = placed["n_a"], placed["n_b"]
    a_center_x = a.x + a.shape.width / 2.0
    b_center_x = b.x + b.shape.width / 2.0

    endpoints = _path_endpoints(r.svg)
    assert endpoints, "expected at least one bezier path connector"

    # Find the causes arrow (the one whose start is not at a_center and end not at b_center).
    for x1, y1, x2, y2 in endpoints:
        if abs(x1 - a_center_x) > 1 and abs(x2 - b_center_x) > 1:
            assert abs(x1 - (a.x + a.shape.width)) < 2, \
                f"start x {x1} not near A right edge {a.x + a.shape.width}"
            assert abs(x2 - b.x) < 2, \
                f"end x {x2} not near B left edge {b.x}"
            return
    raise AssertionError("did not find a causes path with clipped endpoints")
