"""Extract figures, diagrams and tables from a PDF by caption-region cropping.

The default ``book.parse.parse_pdf`` only catches raster images; books
like Sipser draw their state-machine diagrams with PDF vector
primitives that ``Page.get_images()`` skips.  This tool finds every
``FIGURE N.M`` / ``TABLE N.M`` caption in the PDF, infers the figure
region above the caption, and renders that region to a PNG so vector
diagrams + tables become available alongside the embedded raster
images.

Output:
  * ``<book_stem>_figures_v2/<fid>.png`` — one PNG per detected figure.
  * ``<book_stem>.figures.json`` — sidecar with
        ``{nid: [{fid, label, caption, page, image_path}, …], …}``
    keyed by the most-specific BookNode nid that contains the page,
    so the server can serve "figures for this section" by lookup.

Usage:
    python -m tools.extract_book_figures \\
        books/sipser-introduction-to-the-theory-of-computation-3e-3a09.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Optional


# Match "Figure 1.4", "FIGURE 1.4", "Table 2.3", "TABLE 2.3", with
# optional colon / dash after the label.  Group 1 = kind, group 2 =
# label digits.
_CAPTION_RE = re.compile(
    r"\b(?P<kind>FIGURE|Figure|TABLE|Table|DIAGRAM|Diagram)\s+"
    r"(?P<label>\d+(?:\.\d+)?)\b"
)

# "Theorem 1.26" / "Definition 1.5" / "Lemma 3.7" — labeled boxes
# whose content lies *below* the label, opposite to figure/table
# captions.  Sipser typesets these inline in the text: the label is
# typeset with extra leading + capitalised "THEOREM", followed by
# 1-3 lines of statement, then a paragraph break or PROOF block.
_THEOREM_KIND_RE = re.compile(
    r"\b(?P<kind>THEOREM|Theorem|"
    r"DEFINITION|Definition|"
    r"LEMMA|Lemma|"
    r"COROLLARY|Corollary|"
    r"PROPOSITION|Proposition|"
    r"PROOF|Proof|"
    r"EXAMPLE|Example|"
    r"CLAIM|Claim|"
    r"ALGORITHM|Algorithm)\s+"
    r"(?P<label>\d+(?:\.\d+)?)\b"
)

# Aliases — labels that *narration* may use for the figure, even
# though the figure's own caption is "Figure N.M".  Sipser-style
# examples / definitions / theorems are typically printed on the
# same page as their illustrating figure, so collecting these from
# the page text and storing them on the figure means a clause
# saying "Example 1.7" can highlight the figure that lives next to
# Example 1.7 (which itself is captioned "Figure 1.8").
_ALIAS_KINDS = (
    "Example", "EXAMPLE",
    "Definition", "DEFINITION",
    "Theorem", "THEOREM",
    "Lemma", "LEMMA",
    "Corollary", "COROLLARY",
    "Proposition", "PROPOSITION",
    "Algorithm", "ALGORITHM",
    "Claim", "CLAIM",
)
_ALIAS_RE = re.compile(
    r"\b(?P<kind>" + "|".join(_ALIAS_KINDS)
    + r")\s+(?P<label>\d+(?:\.\d+)?)\b"
)


def _aliases_on_page(page) -> list[str]:
    """Pull every Example/Definition/Theorem/… label that appears
    anywhere on this page, normalised to "Kind N.M" with title-case
    so they match the spoken-label parser on the frontend."""
    text = page.get_text("text") or ""
    found: list[str] = []
    seen: set[str] = set()
    for m in _ALIAS_RE.finditer(text):
        kind_title = m.group("kind").title()
        canonical = f"{kind_title} {m.group('label')}"
        if canonical in seen:
            continue
        seen.add(canonical)
        found.append(canonical)
    return found


def _build_nid_index(root: dict) -> dict[int, str]:
    """Map each page → most-specific nid containing that page."""
    out: dict[int, str] = {}

    def _walk(n: dict):
        s = n.get("page_start") or 0
        e = n.get("page_end") or 0
        if s and e:
            for p in range(s, e + 1):
                out[p] = n.get("nid") or "b"
        for c in n.get("children") or []:
            _walk(c)

    _walk(root)
    return out


def _is_theorem_block(block_text: str) -> Optional[tuple[str, str]]:
    """Detect a labeled-box block (THEOREM N.M / DEFINITION N.M /
    LEMMA N.M / …) for which the BODY sits BELOW the label.  Same
    short-block rules as captions."""
    raw = (block_text or "").strip()
    if not raw or len(raw) > 700:
        return None
    m = _THEOREM_KIND_RE.search(raw[:120])
    if not m:
        return None
    if m.start() > 24:
        return None
    return (m.group("kind").lower(), m.group("label"))


def _is_caption_block(block_text: str) -> Optional[tuple[str, str]]:
    """Decide whether *block_text* is a figure/table CAPTION block
    (rather than a body-text mention).  Returns (kind, label) on
    hit, ``None`` otherwise.

    Heuristic: a caption block is short — usually one or two lines
    — and contains the caption pattern near the start.  PyMuPDF
    sometimes prepends a leading newline or a caption-number-only
    line before the body, so we accept the pattern anywhere in the
    first ~24 characters of the (whitespace-trimmed) block instead
    of requiring it strictly at position 0.  Body-text mentions
    ("see Figure 1.4") occur deep inside long paragraphs and so
    don't slip through this anchor.
    """
    raw = (block_text or "").strip()
    if not raw:
        return None
    # Reject obvious body paragraphs by length — caption blocks
    # are short.  Anything longer than ~700 chars is overwhelmingly
    # likely to be a paragraph body, not a caption.
    if len(raw) > 700:
        return None
    # Look for the caption pattern near the start.  We allow the
    # match to begin anywhere in the first 24 chars (covers a
    # leading blank line, page-number-only line, or column-break
    # whitespace that PyMuPDF leaves at the front of the block).
    m = _CAPTION_RE.search(raw[:120])
    if not m:
        return None
    if m.start() > 24:
        return None
    return (m.group("kind").lower(), m.group("label"))


def _theorem_region_below(
    page, label_rect, *, bottom_floor: float
) -> Optional[tuple[float, float, float, float]]:
    """Compute a clip rectangle for a labeled box (THEOREM /
    DEFINITION / LEMMA / …) whose body sits BELOW the label.

      * top    = label_rect.top - 4 px (include the label itself)
      * bottom = ``bottom_floor`` (top of the next labeled item OR
                 ~12 lines below, whichever is closer)
      * left/right = full page content width.
    """
    top = max(0.0, float(label_rect.y0) - 4.0)
    bottom = bottom_floor - 4.0
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    left = 36.0
    right = page_w - 36.0
    bottom = min(bottom, page_h - 4.0)
    # Theorem boxes are short — 1-3 line statement.  If the floor
    # gives us 200+ px of crop, cap at 200 so we don't pull in the
    # following proof.
    if bottom - top > 220.0:
        bottom = top + 220.0
    if bottom - top < 30.0 or right - left < 80.0:
        return None
    return (left, top, right, bottom)


def _figure_region_above(
    page, caption_rect, *, top_floor: float
) -> Optional[tuple[float, float, float, float]]:
    """Compute a clip rectangle for the figure that sits above
    *caption_rect*.

    Heuristic:
      * top    = ``top_floor`` (bottom of the previous text block /
                 page top, whichever is closer to the caption)
      * bottom = caption_rect.top - 2 px gap
      * left/right = the page's full content width (minus standard
                     margins) — captions are often *narrower* than
                     the figures they label, so anchoring the crop
                     to the caption x-bounds clips state-diagram
                     glyphs / labels off the right edge of wide
                     figures.  We use the page's full content
                     width and accept some surrounding whitespace.

    Filters out tiny / negative-height regions where the figure
    didn't actually live above the caption.
    """
    top = top_floor + 4.0
    bottom = float(caption_rect.y0) - 2.0
    if bottom - top < 60.0:
        return None
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    # Standard scientific layout uses ~72 pt (1 inch) margins; we
    # snap a little tighter so light grid backgrounds don't take
    # over the rendered card.
    left = 36.0
    right = page_w - 36.0
    bottom = min(bottom, page_h - 4.0)
    if right - left < 80.0 or bottom - top < 60.0:
        return None
    return (left, top, right, bottom)


def _render_clip(page, clip_rect, *, scale: float = 2.0) -> bytes:
    """Render *clip_rect* of *page* to PNG bytes at the given scale."""
    import fitz
    mat = fitz.Matrix(scale, scale)
    clip = fitz.Rect(*clip_rect)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return pix.tobytes("png")


def extract(corpus_json: str, *, force: bool = False) -> int:
    if not os.path.isfile(corpus_json):
        print(f"error: {corpus_json} not found", file=sys.stderr)
        return 1
    stem = (corpus_json[: -len(".json")]
            if corpus_json.endswith(".json") else corpus_json)
    out_dir = stem + "_figures_v2"
    sidecar_path = stem + ".figures.json"
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(sidecar_path) and not force:
        print(f"[figures] sidecar exists at {sidecar_path}; "
              f"pass --force to rebuild")
        return 0

    payload = json.load(open(corpus_json))
    root = payload.get("root") or {}
    nid_by_page = _build_nid_index(root)

    # Find the original PDF — convention: same stem with .pdf.
    pdf_candidates = [stem + ".pdf"]
    pdf_path = next((p for p in pdf_candidates if os.path.isfile(p)), "")
    if not pdf_path:
        print(f"[figures] no PDF found at {pdf_candidates}", file=sys.stderr)
        return 1

    import fitz
    doc = fitz.open(pdf_path)
    figures_by_nid: dict[str, list[dict]] = {}
    written = 0
    # Body-paragraph threshold: only blocks with this many characters
    # are treated as a "ceiling" above which a figure cannot extend.
    # Short blocks (table cells, state-machine state labels, axis
    # tick labels) are PART of the figure, so they shouldn't shrink
    # the crop region down to nothing.
    _BODY_BLOCK_MIN_CHARS = 110
    for pi in range(len(doc)):
        page = doc[pi]
        page_num = pi + 1
        page_aliases = _aliases_on_page(page)
        # Pull text blocks with bboxes, sorted top-to-bottom.
        blocks = sorted(
            (b for b in page.get_text("blocks") if (b[4] or "").strip()),
            key=lambda b: b[1],
        )
        # First pass: collect every "label start" (theorem-like
        # blocks AND captions) so theorem regions can be bounded
        # by the next label.  We then iterate again to emit cards.
        theorem_starts: list[tuple[float, float, float, float, str, str]] = []
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            t = (text or "").strip()
            if not t:
                continue
            for re_obj in (_CAPTION_RE, _THEOREM_KIND_RE):
                m = re_obj.search(t[:120])
                if m and m.start() <= 24:
                    theorem_starts.append((y0, x0, x1, y1, t, "any"))
                    break
        theorem_starts.sort(key=lambda t: t[0])

        def _next_label_top(after_y: float) -> float:
            """Y of the next labeled item below ``after_y`` on this
            page — used as the bottom_floor for theorem crops."""
            for y0_, _x0, _x1, _y1, _t, _ in theorem_starts:
                if y0_ > after_y + 1.0:
                    return float(y0_)
            return float(page.rect.height) - 24.0

        # Track separately: bottom of the last *body-paragraph* block
        # (a true ceiling for figure crops) vs. bottom of any text
        # block (which would include table rows etc.).
        prev_body_bottom = float(page.rect.y0)
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            cap = _is_caption_block(text)
            thm = _is_theorem_block(text) if cap is None else None
            if cap is None and thm is None:
                t = (text or "").strip()
                if len(t) >= _BODY_BLOCK_MIN_CHARS:
                    prev_body_bottom = max(prev_body_bottom, y1)
                continue
            import fitz as _fitz
            label_rect = _fitz.Rect(x0, y0, x1, y1)
            if cap is not None:
                kind, label = cap
                clip = _figure_region_above(
                    page, label_rect, top_floor=prev_body_bottom,
                )
            else:
                kind, label = thm
                # PROOF blocks are technical body, not their own
                # cards — skip them.
                if kind in ("proof",):
                    continue
                clip = _theorem_region_below(
                    page, label_rect,
                    bottom_floor=_next_label_top(y0),
                )
            if clip is None:
                continue
            try:
                png = _render_clip(page, clip)
            except Exception as e:
                print(f"  page {page_num}: render failed: {e}",
                      file=sys.stderr)
                continue
            digest = hashlib.sha256(png).hexdigest()[:12]
            safe_kind = kind.replace(" ", "_")
            safe_label = label.replace(".", "_")
            fid = f"{safe_kind}{safe_label}_p{page_num}_{digest}"
            rel_path = f"{fid}.png"
            with open(os.path.join(out_dir, rel_path), "wb") as f:
                f.write(png)
            home_nid = nid_by_page.get(page_num, "b")
            entry = {
                "fid": fid,
                "label": f"{kind.title()} {label}",
                "kind": kind,
                "label_number": label,
                "caption": text.strip()[:240],
                "page": page_num,
                "home_nid": home_nid,
                "image_path": rel_path,
                "bbox": list(clip),
                # Every Example/Definition/Theorem… label on the
                # same page; lets the frontend highlight this card
                # when narration says "Example 1.7" even though the
                # caption itself reads "Figure 1.8".
                "aliases": list(page_aliases),
            }
            figures_by_nid.setdefault(home_nid, []).append(entry)
            written += 1
            # Label region itself becomes a ceiling for the next
            # figure crop on the same page (regions don't overlap).
            prev_body_bottom = max(prev_body_bottom, y1)

    if not written:
        print(f"[figures] found no captioned figures in {pdf_path}")
    sidecar = {
        "version": 1,
        "image_dir": os.path.basename(out_dir),
        "by_nid": figures_by_nid,
    }
    tmp = sidecar_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    os.replace(tmp, sidecar_path)
    print(f"[figures] wrote {written} figure(s) for "
          f"{sum(len(v) for v in figures_by_nid.values())} entries "
          f"across {len(figures_by_nid)} sections; sidecar: {sidecar_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", help="path to the book's <stem>.json")
    p.add_argument("--force", action="store_true",
                   help="rebuild even when the sidecar exists")
    args = p.parse_args(argv)
    return extract(args.corpus, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
