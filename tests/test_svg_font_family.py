"""Pin the font-family cascade for SVG ``<text>`` emit sites.

Background
----------
The bitmap rasteriser (``cairosvg``) used by the structural / VLM
inspector and by the paper-figure capture step picks the *first*
available font in the SVG's ``font-family`` cascade and renders
missing glyphs as ``.notdef`` boxes (``□``).  An earlier build of
the canonical-figure PNGs leaked the boxes into Figure~12 of the
paper: ``forward □ loss □ gradient □`` instead of
``forward → loss → gradient ←``, and ``□L/□w`` instead of
``∂L/∂w``, because the cascade was ``"ui-sans-serif,sans-serif"``
and Cairo's default sans-serif font lacks U+2202, U+2190, U+2192.

The fix is a single canonical constant ``viz.fonts.SVG_FONT_FAMILY``
whose first font is DejaVu Sans (which carries the math glyphs); every
SVG ``<text>`` emit site uses it.  This test guards the contract:

* the constant equals what we expect (DejaVu Sans first);
* no producer file contains the old buggy string anywhere; and
* every SVG-side ``font-family="…"`` whose cascade is
  ``ui-sans-serif`` -based starts with DejaVu Sans.

KaTeX and monospace cascades are intentionally different (KaTeX_Main
is itself a math font; monospace cascades render code blocks via
``ui-monospace`` whose glyph coverage is fine for our purposes).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


PRODUCER_FILES = [
    REPO_ROOT / "viz" / "render.py",
    REPO_ROOT / "viz" / "generators.py",
    REPO_ROOT / "viz" / "semantic_to_svg.py",
    REPO_ROOT / "serve" / "orchestrator.py",
    REPO_ROOT / "sevim" / "s5_render.py",
]

EXPECTED_HEAD = "DejaVu Sans"

# The historical buggy literal — must never re-appear.
BUGGY_LITERAL = "ui-sans-serif,sans-serif"

# Match SVG-attribute and inline-style font-family declarations.
_FF_ATTR    = re.compile(r'font-family\s*=\s*"([^"]+)"')
_FF_INLINE  = re.compile(r'font-family\s*:\s*([^";]+)')


def test_canonical_constant_present():
    """The single source of truth for the cascade lives in ``viz.fonts``."""
    from viz.fonts import SVG_FONT_FAMILY
    assert SVG_FONT_FAMILY == "DejaVu Sans, ui-sans-serif, sans-serif"
    assert SVG_FONT_FAMILY.split(",")[0].strip() == EXPECTED_HEAD


def test_buggy_literal_not_present_anywhere():
    """The exact ``ui-sans-serif,sans-serif`` cascade is the regression
    fingerprint; ban it everywhere in the producer surface."""
    bad: list[tuple[str, int, str]] = []
    for f in PRODUCER_FILES:
        text = f.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if BUGGY_LITERAL in line:
                bad.append((f.name, line_no, line.strip()))
    assert not bad, (
        f"the buggy font-family cascade {BUGGY_LITERAL!r} re-appeared.  "
        f"Use viz.fonts.SVG_FONT_FAMILY instead.  Offending lines:\n  "
        + "\n  ".join(f"{f}:{n}: {l}" for f, n, l in bad)
    )


def test_every_sans_serif_emit_leads_with_dejavu_sans():
    """Every emit site whose cascade is ``ui-sans-serif`` -based must
    have DejaVu Sans first.  Other cascades (KaTeX_Main, ui-monospace)
    are out of scope for this test — they target code / rendered-math
    text that does not hit the rasteriser glyph-fallback bug."""
    bad: list[tuple[str, int, str]] = []
    for f in PRODUCER_FILES:
        for line_no, line in enumerate(f.read_text(encoding="utf-8").splitlines(),
                                        start=1):
            for pat in (_FF_ATTR, _FF_INLINE):
                for m in pat.finditer(line):
                    value = m.group(1).strip()
                    if "ui-sans-serif" not in value:
                        continue
                    head = value.split(",")[0].strip().strip("'\"")
                    if head != EXPECTED_HEAD:
                        bad.append((f.name, line_no, value))
    assert not bad, (
        "every ui-sans-serif-based cascade must put DejaVu Sans first "
        "so cairosvg routes ∂, →, ←, σ etc. correctly.  Offending sites:\n  "
        + "\n  ".join(f"{f}:{n}: {v}" for f, n, v in bad)
    )
