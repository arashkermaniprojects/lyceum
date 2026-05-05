"""Low-level SVG drawing primitives shared by every canonical generator.

Hand-rolled SVG: no matplotlib, no cairo, no native deps.  Everything is
expressed as path data we synthesise in Python.  Keeps the per-figure
cost firmly under 100 ms and produces small, predictable SVG bodies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Style — keep one source of truth so figures look consistent.
# ---------------------------------------------------------------------------

PALETTE = {
    "ink":      "#212121",
    "muted":    "#9e9e9e",
    "axis":     "#37474f",
    "grid":     "#eceff1",
    "train":    "#1976d2",   # blue
    "test":     "#d81b60",   # magenta
    "bias":     "#388e3c",   # green
    "variance": "#f57c00",   # orange
    "total":    "#5e35b1",   # indigo
    "highlight":"#fdd835",
    "fold_a":   "#90caf9",
    "fold_b":   "#a5d6a7",
    "fold_c":   "#ffcc80",
}


# ---------------------------------------------------------------------------
# Tiny SVG builder — append fragments, render once.
# ---------------------------------------------------------------------------

@dataclass
class SVGCanvas:
    width: float
    height: float
    parts: list[str] = field(default_factory=list)
    title: Optional[str] = None

    def add(self, fragment: str) -> None:
        self.parts.append(fragment)

    def render_body(self) -> str:
        """Return SVG body (no outer <svg>) so the chalkboard can wrap it."""
        return "".join(self.parts)

    def render_full(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width:.0f}" height="{self.height:.0f}" '
            f'viewBox="0 0 {self.width:.0f} {self.height:.0f}">'
        )
        body = self.render_body()
        return f"{head}{body}</svg>"


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# Coordinate frames — generators draw in data coords, we map to SVG coords.
# ---------------------------------------------------------------------------

@dataclass
class Axes:
    """Linear 2-D axes.  Maps data (x, y) → SVG (px, py).

    Origin is at (margin_l, height - margin_b); +x rightward, +y upward.
    """
    canvas: SVGCanvas
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    margin_l: float = 56.0
    margin_r: float = 16.0
    margin_t: float = 28.0
    margin_b: float = 36.0
    x_label: str = ""
    y_label: str = ""
    title: str = ""

    def to_px(self, x: float) -> float:
        w = self.canvas.width - self.margin_l - self.margin_r
        return self.margin_l + (x - self.x_lo) / max(1e-9, self.x_hi - self.x_lo) * w

    def to_py(self, y: float) -> float:
        h = self.canvas.height - self.margin_t - self.margin_b
        return self.canvas.height - self.margin_b - (y - self.y_lo) / max(
            1e-9, self.y_hi - self.y_lo,
        ) * h

    def draw_frame(self, *, x_ticks: int = 5, y_ticks: int = 4) -> None:
        c = self.canvas
        w = c.width - self.margin_l - self.margin_r
        h = c.height - self.margin_t - self.margin_b
        # Axis box.
        c.add(
            f'<rect x="{self.margin_l}" y="{self.margin_t}" '
            f'width="{w}" height="{h}" fill="#fff" '
            f'stroke="{PALETTE["axis"]}" stroke-width="1"/>'
        )
        # Y grid + ticks.
        for i in range(1, y_ticks):
            t = i / y_ticks
            py = self.margin_t + (1 - t) * h
            c.add(
                f'<line x1="{self.margin_l}" y1="{py:.2f}" '
                f'x2="{self.margin_l + w}" y2="{py:.2f}" '
                f'stroke="{PALETTE["grid"]}" stroke-width="1"/>'
            )
        # Axis labels.
        if self.x_label:
            cx = self.margin_l + w / 2
            c.add(
                f'<text x="{cx:.1f}" y="{c.height - 8:.1f}" '
                f'text-anchor="middle" font-size="12" '
                f'fill="{PALETTE["axis"]}" '
                f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">{_esc(self.x_label)}</text>'
            )
        if self.y_label:
            cy = self.margin_t + h / 2
            c.add(
                f'<text x="14" y="{cy:.1f}" text-anchor="middle" '
                f'font-size="12" fill="{PALETTE["axis"]}" '
                f'font-family="DejaVu Sans, ui-sans-serif, sans-serif" '
                f'transform="rotate(-90 14 {cy:.1f})">{_esc(self.y_label)}</text>'
            )
        if self.title:
            c.add(
                f'<text x="{c.width/2:.1f}" y="18" text-anchor="middle" '
                f'font-size="13" font-weight="600" fill="{PALETTE["ink"]}" '
                f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">{_esc(self.title)}</text>'
            )

    def polyline(
        self, points: Iterable[tuple[float, float]],
        *, color: str, width: float = 2.0,
        dash: Optional[str] = None, label: Optional[str] = None,
        label_pos: Optional[tuple[float, float]] = None,
    ) -> None:
        pts = list(points)
        if not pts:
            return
        path = " ".join(
            f"{self.to_px(x):.2f},{self.to_py(y):.2f}" for x, y in pts
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.canvas.add(
            f'<polyline points="{path}" fill="none" '
            f'stroke="{color}" stroke-width="{width}"{dash_attr} '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if label and label_pos is not None:
            lx, ly = label_pos
            self.canvas.add(
                f'<text x="{self.to_px(lx):.1f}" y="{self.to_py(ly):.1f}" '
                f'fill="{color}" font-size="12" font-weight="600" '
                f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">{_esc(label)}</text>'
            )

    def scatter(
        self, points: Iterable[tuple[float, float]], *,
        color: str, r: float = 3.0,
    ) -> None:
        for x, y in points:
            self.canvas.add(
                f'<circle cx="{self.to_px(x):.2f}" cy="{self.to_py(y):.2f}" '
                f'r="{r:.1f}" fill="{color}" stroke="#fff" stroke-width="0.5"/>'
            )

    def vline(
        self, x: float, *, color: str = "#9e9e9e", dash: str = "4 3",
        label: Optional[str] = None,
    ) -> None:
        px = self.to_px(x)
        c = self.canvas
        c.add(
            f'<line x1="{px:.2f}" y1="{self.margin_t}" '
            f'x2="{px:.2f}" y2="{c.height - self.margin_b}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="{dash}"/>'
        )
        if label:
            c.add(
                f'<text x="{px:.1f}" y="{self.margin_t + 12:.1f}" '
                f'text-anchor="middle" font-size="11" '
                f'fill="{color}" font-family="DejaVu Sans, ui-sans-serif, sans-serif">'
                f'{_esc(label)}</text>'
            )
