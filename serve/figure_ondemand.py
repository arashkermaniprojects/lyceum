"""On-demand figure cropping from the source PDF.

Used when ``Book.figures`` has no matching ``FigureRef`` for a "Figure
N.M" reference the narrator just spoke.  We open the PDF directly with
pymupdf, locate the caption, and crop the rectangle above it (where the
figure image lives in the typical academic-PDF layout).

Local-only.  Results are cached to ``{stem}_figures_ondemand/{fid}.png``
so a repeated request is a disk read.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

# pymupdf is already a project dep (book/parse_pdf.py).
import fitz  # type: ignore


_CACHE_LOCK = threading.Lock()


@dataclass
class OnDemandFigure:
    """A figure cropped on-demand from the source PDF."""
    fid: str               # synthetic id, stable across runs
    label: str             # "Figure 6.14"
    page: int              # 1-indexed
    bbox: tuple[float, float, float, float]
    image_path: str        # filename within `cache_dir`
    cache_dir: str         # absolute dir
    caption: str = ""

    @property
    def full_path(self) -> str:
        return os.path.join(self.cache_dir, self.image_path)


def crop_figure_by_label(
    pdf_path: str, label: str, *,
    cache_dir: Optional[str] = None,
    pad: float = 8.0,
    max_block_height: float = 540.0,
    force: bool = False,
) -> Optional[OnDemandFigure]:
    """Find ``FIGURE {label}`` on its page and crop the figure block above
    the caption.  Returns None when the caption can't be found.

    Determinism note: the same PDF + label always yields the same fid and
    the same crop.  Cached PNGs are byte-identical given an unchanged
    pymupdf version.
    """
    if not os.path.isfile(pdf_path):
        return None

    if cache_dir is None:
        stem = os.path.splitext(pdf_path)[0]
        cache_dir = stem + "_figures_ondemand"
    with _CACHE_LOCK:
        os.makedirs(cache_dir, exist_ok=True)

    safe_label = re.sub(r"[^A-Za-z0-9.]+", "_", label)
    fid = f"od_{safe_label}"
    image_path = f"{fid}.png"
    full = os.path.join(cache_dir, image_path)

    cap_re = re.compile(
        rf"\bFIGURE\s+{re.escape(label.lstrip('Figure ').strip())}\b",
        re.I,
    )

    # Reuse cache if the rasterised crop already exists.
    if os.path.isfile(full) and not force:
        # We still need the bbox+page for the OnDemandFigure record;
        # re-locate (cheap) but serve the cached PNG.
        page_idx, bbox, caption = _locate_caption(pdf_path, cap_re)
        if page_idx < 0:
            return None
        return OnDemandFigure(
            fid=fid, label=label, page=page_idx + 1,
            bbox=bbox, image_path=image_path, cache_dir=cache_dir,
            caption=caption,
        )

    page_idx, bbox, caption = _locate_caption(pdf_path, cap_re)
    if page_idx < 0:
        return None

    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    page_h = page.rect.height
    cx0, cy0, cx1, cy1 = bbox

    # Heuristic: the figure image block sits *above* the caption text on
    # the same page in academic layouts.  Crop from the top of the page
    # (or just below the running header) down to just *above* the caption
    # so the original "FIGURE N.M" label never bleeds into the rasterised
    # crop, capped at max_block_height.
    top = max(40.0, cy0 - max_block_height)
    crop_rect = fitz.Rect(
        max(0.0, min(cx0, 40.0) - pad),
        max(0.0, top - pad),
        min(page.rect.width, max(cx1, page.rect.width - 40.0) + pad),
        cy0 - pad * 0.5,
    )
    # If the crop is degenerate, widen to full text width.
    if crop_rect.width < 80 or crop_rect.height < 80:
        crop_rect = fitz.Rect(40.0, max(40.0, cy0 - 360.0),
                               page.rect.width - 40.0, cy0)

    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, clip=crop_rect, alpha=False)
    pix.save(full)
    doc.close()
    return OnDemandFigure(
        fid=fid, label=label, page=page_idx + 1,
        bbox=tuple(crop_rect), image_path=image_path,
        cache_dir=cache_dir, caption=caption,
    )


def _locate_caption(
    pdf_path: str, cap_re: re.Pattern,
) -> tuple[int, tuple[float, float, float, float], str]:
    """Return (page_idx, line_bbox, caption_text) for the first page-line
    matching *cap_re*, or (-1, (0,0,0,0), "") if not found."""
    doc = fitz.open(pdf_path)
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text = "".join(s.get("text", "")
                                   for s in line.get("spans", []))
                    if cap_re.search(text):
                        bbox = tuple(line.get("bbox", (0, 0, 0, 0)))
                        return pno, bbox, text.strip()
        return -1, (0.0, 0.0, 0.0, 0.0), ""
    finally:
        doc.close()
