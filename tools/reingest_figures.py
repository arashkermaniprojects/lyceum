"""Re-extract figures from a book PDF and link them to BookNodes.

Local-only — uses ``pymupdf`` (already a project dep), zero API calls.
Designed to run *while the live server is up*; output is written to a
sidecar JSON so the running process is never disturbed.

Pipeline
--------
1. Open PDF with PyMuPDF.
2. For each page, find image-shaped drawing groups (raster blocks /
   vector clusters) by scanning the drawing dictionary; record bbox.
3. Look near each block for a caption matching ``Figure N.M``; that
   number becomes the figure's identity.
4. Resolve ``Figure N.M`` to a BookNode by matching the chapter
   (``b/chN``) and choosing the section/subsection that *physically*
   contains the figure's page.
5. Crop the figure region to PNG.
6. Write a sidecar at ``{book_stem}_figures_v2.json`` so the live
   server keeps using the original ingestion until you restart.

Usage
-----
    python -m tools.reingest_figures books/ESLII.json
        [--pdf books/ESLII.pdf] [--out books/ESLII_figures_v2/]

Re-run is idempotent: existing PNGs are skipped unless ``--force``.

This script is INTENTIONALLY conservative: when a caption is missing
or ambiguous it skips the figure rather than guessing — better to
under-extract than to attach the wrong figure to a passage.
"""
from __future__ import annotations

import argparse
import hashlib
import json as _json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Local-only imports — no network at import time.
import fitz  # type: ignore  (pymupdf)


_FIG_CAPTION_RE = re.compile(
    r"\bFigure\s+(\d+(?:\.\d+){0,2})[\s.:]+", re.UNICODE,
)


@dataclass
class _ExtractedFigure:
    fid: str
    label: str            # "Figure 7.1"
    chapter: str          # "ch7"
    page: int
    bbox: tuple[float, float, float, float]
    caption: str
    home_nid: str = ""
    image_path: str = ""


# ---------------------------------------------------------------------------
# BookNode index — load once, keep small in-memory.
# ---------------------------------------------------------------------------

def _load_book_index(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    nodes: list[dict] = []

    def walk(n: dict) -> None:
        nodes.append(n)
        for c in n.get("children", []):
            walk(c)

    walk(data["root"])
    by_chapter_label: dict[str, dict] = {}
    by_page: list[tuple[int, int, dict]] = []  # (start, end, node)
    for n in nodes:
        nid = n.get("nid", "")
        kind = n.get("kind", "")
        if kind == "chapter":
            num = (n.get("number") or "").strip()
            if num:
                by_chapter_label[num] = n
        if n.get("page_start") and n.get("page_end"):
            by_page.append((n["page_start"], n["page_end"], n))
    return {"all": nodes, "by_chapter_num": by_chapter_label,
            "by_page": by_page, "raw": data}


def _node_for_figure(idx: dict, label: str, page: int) -> str:
    """Return the BookNode nid that should own a figure named *label*
    located on *page*.  Strategy: pick the deepest section/subsection
    whose page range contains *page*."""
    candidates: list[tuple[int, dict]] = []
    for ps, pe, n in idx["by_page"]:
        if ps <= page <= pe:
            depth = n.get("nid", "").count("/")
            candidates.append((depth, n))
    if not candidates:
        return ""
    candidates.sort(key=lambda d_n: -d_n[0])
    return candidates[0][1].get("nid", "")


# ---------------------------------------------------------------------------
# Figure detection — image blocks + caption neighbourhood
# ---------------------------------------------------------------------------

def _scan_pdf(pdf_path: str) -> list[_ExtractedFigure]:
    doc = fitz.open(pdf_path)
    out: list[_ExtractedFigure] = []
    for pno in range(len(doc)):
        page = doc[pno]
        # Image blocks: page.get_image_info() returns each xref'd image
        # with bbox.  We use it instead of get_images() because the bbox
        # gives us the figure's actual on-page region.
        try:
            images = page.get_image_info(xrefs=True)
        except Exception:
            images = []
        if not images:
            continue
        page_text = page.get_text("text")
        # Locate all "Figure N.M" captions on the page with their y-coord.
        text_dict = page.get_text("dict")
        captions: list[tuple[str, str, tuple[float, float, float, float]]] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                line_text = "".join(s.get("text", "") for s in line.get("spans", []))
                m = _FIG_CAPTION_RE.search(line_text)
                if m:
                    captions.append((m.group(1), line_text.strip(),
                                     tuple(line.get("bbox", (0, 0, 0, 0)))))
        if not captions:
            continue
        # Pair each image block to the *nearest* caption below it.
        for img in images:
            ibbox = img.get("bbox") or (0, 0, 0, 0)
            ix0, iy0, ix1, iy1 = ibbox
            iyc = (iy0 + iy1) / 2
            best = None
            best_dy = float("inf")
            for label, cap_text, cbbox in captions:
                cyc = (cbbox[1] + cbbox[3]) / 2
                dy = cyc - iyc
                # Caption is usually below the image.
                if dy <= 0:
                    continue
                if dy < best_dy:
                    best_dy = dy
                    best = (label, cap_text, cbbox)
            if best is None:
                continue
            label, cap_text, _ = best
            chapter = "ch" + label.split(".", 1)[0]
            fid_seed = f"{pno+1}:{ibbox}:{label}"
            fid = "fig_p%d_%s" % (
                pno + 1,
                hashlib.md5(fid_seed.encode("utf-8")).hexdigest()[:12],
            )
            out.append(_ExtractedFigure(
                fid=fid, label=f"Figure {label}", chapter=chapter,
                page=pno + 1, bbox=tuple(ibbox), caption=cap_text,
            ))
    doc.close()
    return out


# ---------------------------------------------------------------------------
# Cropping & PNG output
# ---------------------------------------------------------------------------

def _crop_figure(pdf_path: str, fig: _ExtractedFigure,
                 out_dir: str, *, force: bool, pad: float = 6.0) -> str:
    rel = f"{fig.fid}.png"
    full = os.path.join(out_dir, rel)
    if os.path.exists(full) and not force:
        return rel
    doc = fitz.open(pdf_path)
    page = doc[fig.page - 1]
    x0, y0, x1, y1 = fig.bbox
    rect = fitz.Rect(
        max(0, x0 - pad), max(0, y0 - pad),
        x1 + pad, y1 + pad,
    )
    mat = fitz.Matrix(2.0, 2.0)  # 2x oversample for legibility.
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    pix.save(full)
    doc.close()
    return rel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book_json")
    ap.add_argument("--pdf", default=None,
                    help="path to source PDF (defaults to {book_stem}.pdf)")
    ap.add_argument("--out", default=None,
                    help="output figures dir (defaults to "
                         "{book_stem}_figures_v2)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap figures processed (0 = no cap)")
    args = ap.parse_args(argv)

    book_json = args.book_json
    stem = os.path.splitext(os.path.basename(book_json))[0]
    pdf_path = args.pdf or os.path.join(
        os.path.dirname(book_json) or ".", stem + ".pdf",
    )
    out_dir = args.out or os.path.join(
        os.path.dirname(book_json) or ".", stem + "_figures_v2",
    )
    os.makedirs(out_dir, exist_ok=True)
    sidecar_path = os.path.join(
        os.path.dirname(book_json) or ".", stem + "_figures_v2.json",
    )

    print(f"[reingest] pdf={pdf_path}\n[reingest] out={out_dir}", flush=True)
    if not os.path.isfile(pdf_path):
        print(f"[reingest] ERROR: pdf not found: {pdf_path}", file=sys.stderr)
        return 2

    t0 = time.time()
    idx = _load_book_index(book_json)
    print(f"[reingest] loaded {len(idx['all'])} BookNodes from {book_json}",
          flush=True)
    figs = _scan_pdf(pdf_path)
    print(f"[reingest] found {len(figs)} figure candidates "
          f"({time.time()-t0:.1f}s)", flush=True)

    if args.limit:
        figs = figs[: args.limit]

    written: list[dict] = []
    for i, fig in enumerate(figs):
        fig.home_nid = _node_for_figure(idx, fig.label, fig.page)
        try:
            fig.image_path = _crop_figure(pdf_path, fig, out_dir,
                                          force=args.force)
        except Exception as e:
            print(f"[reingest] WARN crop failed for {fig.label} "
                  f"(p.{fig.page}): {e}", flush=True)
            continue
        written.append({
            "fid": fig.fid,
            "label": fig.label,
            "home_nid": fig.home_nid,
            "page": fig.page,
            "bbox": list(fig.bbox),
            "caption": fig.caption,
            "image_path": fig.image_path,
        })
        if (i + 1) % 25 == 0:
            print(f"[reingest] {i+1}/{len(figs)} written "
                  f"({time.time()-t0:.1f}s)", flush=True)

    with open(sidecar_path, "w", encoding="utf-8") as f:
        _json.dump({
            "schema_version": 2,
            "source_pdf": pdf_path,
            "source_json": book_json,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "figures": written,
        }, f, indent=2, ensure_ascii=False)
    print(f"[reingest] wrote {len(written)} figures to "
          f"{sidecar_path} (total {time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
