"""Font-family constants for SVG generation.

Single source of truth for the font-family string used in every SVG
``<text>`` and inline ``style="font-family:..."`` block we emit.

Why a list with DejaVu Sans first
---------------------------------
The runtime ships SVG via two paths:

1. **The browser** in the live application — modern browsers fall back
   *per-glyph* across the font-family list, so an exotic Unicode
   character (∂, →, ←, σ²) that the user's UI font lacks is silently
   sourced from the next font in the cascade.
2. **cairosvg** in the server-side inspector / paper-figure
   rasterisation step — the bitmap rasteriser is not as graceful: it
   picks the first available font from the cascade and renders missing
   glyphs as the ``.notdef`` box (the famous ``□``).

Putting DejaVu Sans first means:

* On the dev / Linux server (DejaVu present) the rasteriser picks it,
  and DejaVu has the partial-derivative, arrow and Greek-letter
  glyphs we need.
* In the browser, modern engines do per-glyph fallback so DejaVu is
  used for the math glyphs while ``ui-sans-serif`` is still the
  effective font for prose on every platform that has a native UI
  font (most do).  On Linux without DejaVu Sans the browser drops to
  ``ui-sans-serif`` cleanly.

If you change this constant, run the SVG-fidelity tests to refresh
fixtures.
"""
from __future__ import annotations

SVG_FONT_FAMILY: str = "DejaVu Sans, ui-sans-serif, sans-serif"
"""Default ``font-family`` string for every SVG ``<text>`` we emit."""
