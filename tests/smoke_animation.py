"""Smoke deliverable for SMIL animation support.

Run as a standalone script (NOT collected by pytest):

    /home/ara/Documents/Programming/sevim_math/.venv/bin/python3 \
        /home/ara/Documents/Programming/sevim_math/tests/smoke_animation.py

It writes 2-3 SVGs to /tmp/sevim_animation/ and prints each path + byte size.

Open the SVGs in a browser (Firefox / Chromium) to verify the SMIL
``<animateTransform>`` elements actually animate.
"""
from __future__ import annotations

import os

from sevim.ir import PlacedGraph, PlacedShape, VisualShape
from sevim.pipeline import run_pipeline
from sevim.s5_render import render


OUT_DIR = "/tmp/sevim_animation"


def _write(path: str, svg: str) -> None:
    """Write *svg* to *path* and print path + byte count."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    size = os.path.getsize(path)
    print(f"  wrote {path}  ({size} bytes)")


def _rotating_polygon() -> str:
    """Direct-API construction: a square rotating 0 → 360°."""
    shape = VisualShape(
        nid="n_square", primitive="polygon", label="square",
        width=120.0, height=120.0, font_size=14.0, stroke_width=1.5,
        fill_index=0,
        meta={
            "sides": 4,
            "animate": {
                "kind": "rotate", "from": 0.0, "to": 360.0,
                "dur": "4s", "repeat": "indefinite", "around": "center",
            },
        },
    )
    placed = PlacedShape(shape=shape, x=40.0, y=40.0)
    pg = PlacedGraph(shapes=[placed], conns=[], canvas_w=200.0, canvas_h=200.0)
    return render(pg)


def _vector_rotated_by_matrix() -> str:
    """End-to-end pipeline: text → SVG with SMIL animation."""
    result = run_pipeline(
        "The rotation matrix R rotates the vector v by 90 degrees."
    )
    return result.svg


def _growing_circle() -> str:
    """Direct-API construction: a circle scaling 1× → 2×."""
    shape = VisualShape(
        nid="n_circle", primitive="circle", label="C",
        width=80.0, height=80.0, font_size=14.0, stroke_width=1.5,
        fill_index=2,
        meta={
            "animate": {
                "kind": "scale", "from": 1.0, "to": 2.0,
                "dur": "2s", "repeat": "indefinite",
            },
        },
    )
    placed = PlacedShape(shape=shape, x=50.0, y=50.0)
    pg = PlacedGraph(shapes=[placed], conns=[], canvas_w=200.0, canvas_h=200.0)
    return render(pg)


def main() -> None:
    print(f"writing smoke SVGs to {OUT_DIR}/")
    _write(os.path.join(OUT_DIR, "rotating_polygon.svg"), _rotating_polygon())
    _write(os.path.join(OUT_DIR, "vector_rotated_by_matrix.svg"),
           _vector_rotated_by_matrix())
    _write(os.path.join(OUT_DIR, "growing_circle.svg"), _growing_circle())
    print("done.")


if __name__ == "__main__":
    main()
