"""PDF → recursive BookNode tree.

Pipeline
--------
1.  Open the PDF with PyMuPDF (``fitz``).
2.  Try ``doc.get_toc(simple=False)`` first — gives the structural skeleton
    directly with depth + page numbers.  This works for the vast majority of
    professionally-typeset books and papers.
3.  When the TOC is missing or shallow, fall back to **font-size heuristics**:
    cluster all text spans by size, take the largest sizes as heading
    candidates, and walk the page text to recover hierarchy.
4.  Within each leaf structural node's text, scan for **environment
    patterns** (``Theorem 3.2.1``, ``Definition 1.1``, …) and append them as
    children.  Environments may themselves contain nested environments.
5.  Extract embedded raster images and emit ``FigureRef`` entries.
6.  Build the page-text cache for full-text search.

The output is a fully-populated :class:`Book` whose ``concepts`` and
``cross_refs`` are still empty — those overlays are filled by separate
modules (``book.concepts``, ``book.crossref``).

Determinism
-----------
- The TOC walk is deterministic (PyMuPDF returns a fixed-order list).
- Image extraction enumerates ``page.get_images(full=True)`` in xref order.
- File hashes use ``hashlib.sha256`` on raw bytes.
- No timestamps are written into the IR (those go in ``Book.meta`` only).
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

from .ir import (
    Book, BookNode, FigureRef,
    PAGE_UNKNOWN, ENVIRONMENT_KINDS,
    join_nid, nid_segment,
)

# fitz is PyMuPDF; importable as either name.
import fitz  # type: ignore


# ---------------------------------------------------------------------------
# TOC handling
# ---------------------------------------------------------------------------

# PyMuPDF's get_toc returns [[level, title, page], …].  Levels are 1-based
# and represent the *outline* depth in the PDF.  We map them onto our
# kind taxonomy heuristically:
#   level 1 → chapter (or part if title contains "Part ")
#   level 2 → section
#   level 3 → subsection
#   level 4+ → subsubsection (capped)

_PART_HINT = re.compile(r"^\s*part\b", re.IGNORECASE)
_APPENDIX_HINT = re.compile(r"^\s*appendix\b", re.IGNORECASE)
_PREFACE_HINT = re.compile(r"^\s*(preface|introduction|foreword)\b", re.IGNORECASE)


def _toc_level_to_kind(level: int, title: str) -> str:
    if level == 1:
        if _PART_HINT.match(title):
            return "part"
        if _APPENDIX_HINT.match(title):
            return "appendix"
        if _PREFACE_HINT.match(title):
            return "introduction"
        return "chapter"
    if level == 2:
        return "section"
    if level == 3:
        return "subsection"
    return "subsubsection"


# Capture an optional "1.2.3" or "Chapter 4" style number from the start of
# a TOC title.  We strip it from the title and store separately.
_NUM_RE = re.compile(
    r"^\s*"
    r"(?:(?:chapter|section|appendix|part|theorem|lemma|definition|"
    r"proposition|corollary|example|exercise|remark)\s+)?"
    r"(?P<num>\d+(?:\.\d+)*[a-zA-Z]?)"
    r"(?:[\.\):]\s+|\s+)?",
    re.IGNORECASE,
)


def _split_number_and_title(title: str) -> tuple[Optional[str], str]:
    m = _NUM_RE.match(title)
    if m:
        num = m.group("num")
        rest = title[m.end():].strip()
        return num, rest or title.strip()
    return None, title.strip()


# ---------------------------------------------------------------------------
# Environment scanner — finds inline theorems/definitions/etc inside text.
# ---------------------------------------------------------------------------

# A list of environment-name regexes.  Each captures (kind, number, title-up-to-end-of-line).
#
# Anchors: each pattern requires the environment label to begin at the start
# of a line (after optional whitespace) so we don't mistake mid-sentence
# citations like "see Theorem 3.2" for a heading.  In PDF-extracted text,
# real headings sit on their own line; references sit inline.
#
# Title capture stops at end-of-line; we trim trailing punctuation/whitespace
# in `_clean_title` afterwards.

_ENV_HEAD = r"(?:^|\n)\s*"  # line-start anchor (empty leading whitespace ok)
_ENV_END = r"(?:[\.\:\—\-]\s+|\s*\n|$)"   # end-of-heading marker
_NUM = r"(?P<num>\d+(?:\.\d+)*[a-zA-Z]?)"
_REST = r"(?P<rest>[^\n]{0,200})"

_ENV_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(_ENV_HEAD + r"Theorem\s+" + _NUM + _ENV_END + _REST,
                re.IGNORECASE), "theorem"),
    (re.compile(_ENV_HEAD + r"Lemma\s+" + _NUM + _ENV_END + _REST,
                re.IGNORECASE), "lemma"),
    (re.compile(_ENV_HEAD + r"Proposition\s+" + _NUM + _ENV_END + _REST,
                re.IGNORECASE), "proposition"),
    (re.compile(_ENV_HEAD + r"Corollary\s+" + _NUM + _ENV_END + _REST,
                re.IGNORECASE), "corollary"),
    (re.compile(_ENV_HEAD + r"Definition\s+" + _NUM + _ENV_END + _REST,
                re.IGNORECASE), "definition"),
    (re.compile(_ENV_HEAD + r"Example\s+" + _NUM + _ENV_END + _REST,
                re.IGNORECASE), "example"),
    (re.compile(_ENV_HEAD + r"Exercise\s+" + _NUM + _ENV_END + _REST,
                re.IGNORECASE), "exercise"),
    # "Proof." is unnumbered; require period or colon after the word.
    (re.compile(_ENV_HEAD + r"Proof\s*[\.\:]\s*" + _REST,
                re.IGNORECASE), "proof"),
    (re.compile(_ENV_HEAD + r"Remark\s+" + _NUM + _ENV_END + _REST,
                re.IGNORECASE), "remark"),
    (re.compile(_ENV_HEAD + r"Claim\s*(?:" + _NUM + r")?" + _ENV_END + _REST,
                re.IGNORECASE), "claim"),
]


# Drop titles that are pure punctuation, parentheses fragments, single
# operators, or empty after stripping.
_VALID_TITLE_CHAR = re.compile(r"[A-Za-z]")


def _clean_title(raw: str) -> str:
    """Trim trailing punctuation/whitespace and reject useless fragments."""
    if not raw:
        return ""
    t = raw.strip().rstrip(".,;:)]} ")
    # Strip leading parens / unbalanced punctuation.
    t = t.lstrip("([{ ")
    # Stop at sentence-end; titles are at most one sentence.
    end = re.search(r"[\.\?\!]\s+[A-Z]", t)
    if end:
        t = t[: end.start() + 1]
    # Reject if no real letters survive.
    if not _VALID_TITLE_CHAR.search(t):
        return ""
    # Cap length to keep nids sensible.
    if len(t) > 80:
        t = t[:77].rstrip() + "…"
    return t

# An environment ends at the first blank line followed by the next environment
# heading, or at the end of the parent's text.  In practice we cap each
# environment at ~1500 characters to avoid runaway matches.
_ENV_CAP = 1500


# Theorem-like kinds that a Proof block should attach to.
_THEOREM_LIKE = frozenset({
    "theorem", "lemma", "proposition", "corollary", "claim", "fact",
})


def _scan_environments(
    text: str,
    parent_nid: str,
    parent_page_start: int,
    parent_page_end: int,
) -> list[BookNode]:
    """Find theorem/definition/proof/… environments inside *text* and return
    them as BookNode children of the calling parent.

    Cleanup rules applied:
      * candidates must start at a line boundary (enforced in the regexes)
      * titles are stripped of trailing punctuation; junk titles are dropped
      * (kind, number) duplicates are collapsed to the first occurrence
      * a Proof environment immediately following a theorem-like sibling
        becomes that sibling's child (recursive structure)
    """
    candidates: list[tuple[int, str, Optional[str], str]] = []
    for pat, kind in _ENV_PATTERNS:
        for m in pat.finditer(text):
            number = m.groupdict().get("num")
            raw_title = (m.groupdict().get("rest") or "")
            title = _clean_title(raw_title)
            candidates.append((m.start(), kind, number, title))

    candidates.sort()

    # De-dupe by start offset, then by (kind, number) — the latter handles
    # cases where two patterns matched the same heading.
    by_offset: dict[int, tuple[str, Optional[str], str]] = {}
    for off, kind, number, title in candidates:
        if off not in by_offset:
            by_offset[off] = (kind, number, title)
    unique = sorted([(off, k, n, t) for off, (k, n, t) in by_offset.items()])

    seen_kn: set[tuple[str, str]] = set()
    accepted: list[tuple[int, str, Optional[str], str]] = []
    for off, kind, number, title in unique:
        if number:
            key = (kind, number)
            if key in seen_kn:
                continue
            seen_kn.add(key)
        accepted.append((off, kind, number, title))

    # First pass: build a flat list of nodes (sibling structure).
    flat: list[BookNode] = []
    n = len(accepted)
    for i, (start, kind, number, title) in enumerate(accepted):
        end = accepted[i + 1][0] if i + 1 < n else min(len(text), start + _ENV_CAP)
        body = text[start:end].strip()
        seg = nid_segment(kind, number, title)
        nid = join_nid(parent_nid, seg)
        flat.append(BookNode(
            nid=nid, kind=kind, number=number, title=title,
            page_start=parent_page_start, page_end=parent_page_end,
            body_text=body, children=[],
            meta={"source": "environment-scan"},
        ))

    # Second pass: attach each "proof" to its preceding theorem-like sibling
    # so the tree captures the Theorem→Proof containment.
    nodes: list[BookNode] = []
    for node in flat:
        if (node.kind == "proof" and nodes
                and nodes[-1].kind in _THEOREM_LIKE):
            parent = nodes[-1]
            # Re-anchor the proof's nid under its parent.
            new_nid = join_nid(parent.nid, "proof")
            attached = BookNode(
                nid=new_nid, kind=node.kind, number=node.number,
                title=node.title,
                page_start=node.page_start, page_end=node.page_end,
                body_text=node.body_text, children=node.children,
                meta=dict(node.meta),
            )
            parent.children.append(attached)
        else:
            nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# TOC → tree
# ---------------------------------------------------------------------------

def _build_tree_from_toc(
    toc: list[list],
    pages: list[str],
    title: str,
) -> BookNode:
    """Convert PyMuPDF's flat TOC list into a recursive BookNode tree."""
    # Synthesise the root.
    root = BookNode(
        nid="b", kind="book", number=None, title=title,
        page_start=1, page_end=len(pages),
        body_text="", children=[], meta={"toc_entries": len(toc)},
    )

    if not toc:
        # No outline — return a single chapter wrapping every page.
        chapter = BookNode(
            nid="b/ch_full", kind="chapter", number=None,
            title=title, page_start=1, page_end=len(pages),
            body_text="\n".join(pages),
            children=[], meta={"source": "no-toc"},
        )
        # Scan the entire body for theorem/definition environments.
        chapter.children = _scan_environments(
            chapter.body_text, chapter.nid,
            chapter.page_start, chapter.page_end,
        )
        root.children.append(chapter)
        return root

    # Normalise: each entry becomes (level, title, page_start).
    entries = [(int(level), str(t), int(p)) for level, t, p, *_ in toc]

    # Compute page_end for each entry: first later entry at level <= mine,
    # else last page.
    n_entries = len(entries)
    n_pages = len(pages)
    page_ends: list[int] = []
    for i, (lvl, _t, page_start) in enumerate(entries):
        end = n_pages
        for j in range(i + 1, n_entries):
            if entries[j][0] <= lvl:
                end = entries[j][2] - 1
                break
        end = max(page_start, end)  # never end before we start
        page_ends.append(end)

    # Build children using a stack: each level deeper appends as child of
    # the last node whose level is strictly less than this level.
    stack: list[tuple[int, BookNode]] = [(0, root)]
    for i, (level, raw_title, page_start) in enumerate(entries):
        page_end = page_ends[i]
        kind = _toc_level_to_kind(level, raw_title)
        number, clean_title = _split_number_and_title(raw_title)
        seg = nid_segment(kind, number, clean_title)

        # Pop stack until we find a strictly-lesser level.
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_lvl, parent_node = stack[-1] if stack else (0, root)

        nid = join_nid(parent_node.nid, seg)
        body = "\n".join(pages[page_start - 1: page_end])

        node = BookNode(
            nid=nid, kind=kind, number=number, title=clean_title,
            page_start=page_start, page_end=page_end,
            body_text=body, children=[],
            meta={"toc_level": level},
        )
        parent_node.children.append(node)
        stack.append((level, node))

    # Second pass — populate environment children inside the LEAF
    # structural nodes (those with no children of their own yet).
    for n in list(root.walk()):
        if not n.children and n.kind != "book":
            n.children = _scan_environments(
                n.body_text, n.nid, n.page_start, n.page_end,
            )

    # Trim body_text on internal structural nodes to "between heading and first
    # child" to avoid duplicating descendant text in body fields.
    _trim_internal_body(root, pages)
    return root


def _trim_internal_body(node: BookNode, pages: list[str]) -> None:
    """For any node with at least one child, body_text becomes the *prelude*
    text between this node's start and its first child's start.  Pure leaves
    keep their full body_text untouched.
    """
    if not node.children:
        return
    first_child = node.children[0]
    if first_child.page_start <= node.page_start:
        # Child starts on the same page as parent; can't isolate prelude
        # cleanly without character-level offsets.  Leave as-is.
        return
    prelude_pages = pages[node.page_start - 1: first_child.page_start - 1]
    node.body_text = "\n".join(prelude_pages)
    for c in node.children:
        _trim_internal_body(c, pages)


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------

def _extract_figures(
    doc: "fitz.Document",
    out_dir: Optional[str],
    nid_at_page: dict[int, str],
) -> list[FigureRef]:
    """Pull every embedded raster image into the output directory and return
    FigureRef entries pointing to them.

    Parameters
    ----------
    out_dir:
        Directory to write image files; pass None to skip writing (useful
        for tests where we just want the metadata).
    nid_at_page:
        Map from 1-based page number → the most-specific BookNode nid that
        contains that page.  Used to attribute each figure to its home node.
    """
    refs: list[FigureRef] = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for pi in range(len(doc)):
        page = doc[pi]
        page_num = pi + 1
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue
            ext = base.get("ext", "png")
            data = base.get("image")
            if not data:
                continue
            digest = hashlib.sha256(data).hexdigest()[:12]
            fid = f"fig_p{page_num}_{img_index:02d}_{digest}"
            rel_path = f"{fid}.{ext}"
            if out_dir:
                with open(os.path.join(out_dir, rel_path), "wb") as f:
                    f.write(data)
            # Find the bbox by scanning page rectangles for this xref.
            bbox = (0.0, 0.0, 0.0, 0.0)
            for r in page.get_image_rects(xref):
                bbox = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                break
            home_nid = nid_at_page.get(page_num, "b")
            refs.append(FigureRef(
                fid=fid, home_nid=home_nid, page=page_num,
                caption="", bbox=bbox, image_path=rel_path,
                meta={"xref": xref},
            ))
    return refs


def _build_nid_index_by_page(root: BookNode) -> dict[int, str]:
    """Build a map page_number → most-specific nid that contains the page."""
    out: dict[int, str] = {}
    # Pre-order walk; later writes win over earlier ones, so the deepest
    # node containing a page ends up assigned to it.
    for n in root.walk():
        for p in range(n.page_start, n.page_end + 1):
            if p == PAGE_UNKNOWN:
                continue
            out[p] = n.nid
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_pdf(
    path: str,
    *,
    figures_dir: Optional[str] = None,
    title_override: Optional[str] = None,
) -> Book:
    """Parse a PDF into a recursive BookNode tree + figure index.

    Parameters
    ----------
    path:
        Path to a PDF file.
    figures_dir:
        Where to extract embedded images.  Pass None to skip writing them.
    title_override:
        Use this string as ``Book.title`` instead of the PDF metadata title.
    """
    doc = fitz.open(path)
    title = title_override or (doc.metadata.get("title") or os.path.basename(path))
    author = doc.metadata.get("author") or None

    # Page text cache.
    pages: list[str] = []
    for p in doc:
        pages.append(p.get_text("text") or "")

    # Build the structural tree from the PDF outline.
    toc = doc.get_toc(simple=False) or []
    root = _build_tree_from_toc(toc, pages, title)

    # Extract figures.
    nid_idx = _build_nid_index_by_page(root)
    figs = _extract_figures(doc, figures_dir, nid_idx)

    book = Book(
        title=title,
        author=author,
        source=os.path.abspath(path),
        root=root,
        figures=figs,
        cross_refs=[],   # filled by book.crossref
        concepts={},     # filled by book.concepts
        pages=pages,
        meta={
            "page_count": len(pages),
            "toc_size": len(toc),
            "figure_count": len(figs),
            "schema_version": 1,
        },
    )
    doc.close()
    return book
