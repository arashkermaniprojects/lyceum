"""Re-extract equations as real LaTeX using the local Qwen-VL endpoint.

Replaces the lossy "PDF text → heuristic LaTeX" pipeline with VLM
inference: each ``(N.M)`` marker's bounding region is rasterised and
sent to the local Qwen-VL vLLM (default ``http://127.0.0.1:8004/v1``,
model ``Qwen/Qwen2.5-VL-7B-Instruct-AWQ``).  No nougat, no pix2tex,
no Anthropic / OpenAI / external API.

This script is OFFLINE-FRIENDLY — designed to run while the live
server stays up.  It writes to a sidecar JSON and never modifies the
in-use book file.

Usage
-----
    python -m tools.reingest_equations books/ESLII.json [--pdf PDF]
        [--out books/ESLII_equations.json]
        [--engine auto|nougat|pix2tex]
        [--pages 0:50]    # restrict to a page range (0-indexed)

Output schema (sidecar)
-----------------------
    {
      "schema_version": 1,
      "engine": "nougat",
      "equations": [
        {"label": "9.8", "page": 297, "home_nid": "b/ch9/s9_1/ss9_1_2",
         "latex": "\\log\\frac{Pr(Y=1|X)}{Pr(Y=0|X)} = \\alpha + \\sum_j f_j(X_j)",
         "source_text": "log Pr(Y = 1|X) / Pr(Y = 0|X) = α + …"},
        ...
      ]
    }

Time budget: per-equation ≤3s offline; nougat batches pages so
throughput is ~1 page/s on a single AWQ-ready GPU.

Caveat: this script does NOT pull models.  Install once before
running:

    pip install nougat-ocr   # OR pip install pix2tex
"""
from __future__ import annotations

import argparse
import json as _json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import fitz  # type: ignore  (pymupdf)


_EQ_MARKER_RE = re.compile(r"\((\d+\.\d+)\)")


@dataclass
class _EqHit:
    label: str            # "9.8"
    page: int             # 1-indexed
    bbox: tuple[float, float, float, float]   # marker bbox
    eq_bbox: tuple[float, float, float, float]   # extended region (eq + marker)
    home_nid: str = ""
    pdf_text: str = ""
    latex: str = ""


# ---------------------------------------------------------------------------
# Locating equations on a page.
# ---------------------------------------------------------------------------

def _equation_hits(page) -> list[_EqHit]:
    """Find every "(N.M)" marker on *page* and return its region.

    The equation body is conservatively the rectangle on the same line
    extending leftward from the marker, plus the line above.  OCR will
    use this region.
    """
    hits: list[_EqHit] = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = "".join(s.get("text", "") for s in line.get("spans", []))
            m = _EQ_MARKER_RE.search(line_text)
            if not m:
                continue
            label = m.group(1)
            line_bbox = line.get("bbox", (0, 0, 0, 0))
            x0, y0, x1, y1 = line_bbox
            # Equation region: the line itself, plus 22pt above (≈ one
            # math line) so 2D layouts are captured.
            eq_bbox = (x0 - 6, y0 - 22, x1 + 6, y1 + 4)
            hits.append(_EqHit(
                label=label, page=page.number + 1,
                bbox=tuple(line_bbox), eq_bbox=eq_bbox,
                pdf_text=line_text.strip(),
            ))
    return hits


# ---------------------------------------------------------------------------
# Engine selection (local only)
# ---------------------------------------------------------------------------

def _make_engine(name: str):
    """Return an object with ``ocr(pdf_path, page, eq_hits) -> {label: latex}``.

    Only ``vlm`` is supported (and is the default).  Talks to the local
    Qwen-VL endpoint configured by ``VLM_BASE_URL`` / ``VLM_MODEL``
    environment variables.
    """
    if name in ("vlm", "auto"):
        from .vlm_engine import VLMEngine
        return VLMEngine()
    raise ValueError(
        f"unknown engine: {name!r} — only 'vlm' is supported"
    )


# ---------------------------------------------------------------------------
# BookNode lookup (same heuristic as figure ingestion)
# ---------------------------------------------------------------------------

def _book_index(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    by_page: list = []

    def walk(n: dict) -> None:
        if n.get("page_start") and n.get("page_end"):
            by_page.append((n["page_start"], n["page_end"], n))
        for c in n.get("children", []):
            walk(c)

    walk(data["root"])
    return {"by_page": by_page}


def _node_for_page(idx: dict, page: int) -> str:
    cands = [(n["nid"].count("/"), n["nid"])
             for ps, pe, n in idx["by_page"] if ps <= page <= pe]
    if not cands:
        return ""
    cands.sort(key=lambda d_n: -d_n[0])
    return cands[0][1]


# ---------------------------------------------------------------------------
# Page range parser
# ---------------------------------------------------------------------------

def _parse_pages(spec: str, max_page: int) -> list[int]:
    if not spec:
        return list(range(max_page))
    if ":" in spec:
        lo, _, hi = spec.partition(":")
        return list(range(int(lo or 0), int(hi or max_page)))
    return [int(spec)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book_json")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--engine", default="vlm",
                    choices=["vlm", "auto"],
                    help="VLM engine — defaults to local Qwen-VL on "
                         "$VLM_BASE_URL")
    ap.add_argument("--pages", default="",
                    help='page range, e.g. "0:50" (0-indexed)')
    args = ap.parse_args(argv)

    book_json = args.book_json
    stem = os.path.splitext(os.path.basename(book_json))[0]
    pdf_path = args.pdf or os.path.join(
        os.path.dirname(book_json) or ".", stem + ".pdf",
    )
    out_path = args.out or os.path.join(
        os.path.dirname(book_json) or ".", stem + "_equations.json",
    )
    if not os.path.isfile(pdf_path):
        print(f"[eq] pdf not found: {pdf_path}", file=sys.stderr)
        return 2

    print(f"[eq] engine={args.engine}", flush=True)
    engine = _make_engine(args.engine)
    print(f"[eq] using {type(engine).__name__}", flush=True)

    idx = _book_index(book_json)
    doc = fitz.open(pdf_path)
    pages = _parse_pages(args.pages, len(doc))

    t0 = time.time()
    all_eqs: list[_EqHit] = []
    for pno in pages:
        page = doc[pno]
        hits = _equation_hits(page)
        if not hits:
            continue
        ocr_map = {}
        try:
            ocr_map = engine.ocr(pdf_path, pno, hits)
        except Exception as e:
            print(f"[eq] OCR error on page {pno+1}: {e}", flush=True)
        for h in hits:
            h.latex = ocr_map.get(h.label, "")
            h.home_nid = _node_for_page(idx, h.page)
            all_eqs.append(h)
        print(f"[eq] page {pno+1}: {len(hits)} markers, "
              f"{sum(1 for h in hits if h.latex)} OCR-recovered "
              f"({time.time()-t0:.1f}s)", flush=True)
    doc.close()

    payload = {
        "schema_version": 1,
        "engine": args.engine,
        "source_pdf": pdf_path,
        "source_json": book_json,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "equations": [{
            "label": h.label, "page": h.page,
            "home_nid": h.home_nid,
            "bbox": list(h.eq_bbox),
            "pdf_text": h.pdf_text,
            "latex": h.latex,
        } for h in all_eqs],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[eq] wrote {len(all_eqs)} equations to {out_path} "
          f"(total {time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
