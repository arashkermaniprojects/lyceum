"""Resolve a citation (kind, ref_label) → actual content from the book.

Given a reference like ``("Equation", "9.8")``, ``("Algorithm", "9.2")``,
or ``("Figure", "7.9")``, return either a body of text to render in a
card, or a :class:`FigureRef` whose image we can embed.

Heuristics
----------
- **Equation**: scan the chapter's body_text for ``(N.M)``; the equation
  body sits immediately before the marker, terminated by a paragraph
  break or a sentence-final period followed by capitalised prose.
- **Algorithm**: scan for ``Algorithm N.M`` and grab the title + numbered
  steps up to the next prose paragraph or another ``Algorithm`` / blank-
  line boundary.
- **Theorem / Lemma / Proposition / Corollary / Definition / Example /
  Exercise**: prefer the matching environment BookNode (these are
  ``is_environment(kind)`` nodes in the IR); fall back to substring
  scanning of the chapter body_text.
- **Figure**: try to match a :class:`FigureRef` by ``home_nid``; for
  ESLII most figures don't carry a numeric label, so we also expose
  the cross_ref's ``to_nid`` so the caller can fall back to a card.
- **Table**: substring scan ``Table N.M`` similar to algorithms.
- **Section / Chapter**: walk the BookNode tree.

All functions are pure and deterministic — same book, same query, same
output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from book.ir import Book, BookNode, FigureRef, is_environment


@dataclass
class RefContent:
    """Resolved content for one reference."""
    kind: str                     # the matched kind (Equation, Algorithm, …)
    ref_label: str                # "9.8", "9.2", "7.9", …
    text: str = ""                # extracted body text (may be multi-line)
    figure: Optional[FigureRef] = None   # set for Figure refs we could resolve
    target_nid: str = ""          # BookNode this reference points at
    latex: str = ""               # OCR-recovered LaTeX (when available)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def resolve_reference(
    book: Book, kind: str, ref_label: str, *, hint_nid: str = "",
) -> RefContent:
    """Return the best-effort content for ``(kind, ref_label)``."""
    if kind == "Equation":
        # Prefer OCR-recovered LaTeX from the sidecar if ingestion has
        # populated it.  Falls back to the heuristic text-extraction.
        ocr_latex = _equation_latex_from_sidecar(book, ref_label)
        text = _find_equation(book, ref_label)
        return RefContent(kind=kind, ref_label=ref_label,
                          text=text, latex=ocr_latex)
    if kind == "Algorithm":
        text = _find_algorithm(book, ref_label)
        return RefContent(kind=kind, ref_label=ref_label, text=text)
    if kind == "Table":
        text = _find_table(book, ref_label)
        return RefContent(kind=kind, ref_label=ref_label, text=text)
    if kind == "Figure":
        fig, target = _find_figure(book, ref_label, hint_nid)
        return RefContent(kind=kind, ref_label=ref_label,
                          figure=fig, target_nid=target)
    if kind in ("Theorem", "Lemma", "Proposition", "Corollary",
                "Definition", "Example", "Exercise"):
        text, target = _find_environment(book, kind, ref_label)
        return RefContent(kind=kind, ref_label=ref_label,
                          text=text, target_nid=target)
    if kind in ("Section", "Chapter"):
        text, target = _find_section_summary(book, kind, ref_label)
        return RefContent(kind=kind, ref_label=ref_label,
                          text=text, target_nid=target)
    return RefContent(kind=kind, ref_label=ref_label)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equation_latex_from_sidecar(book: Book, label: str) -> str:
    """Look up real LaTeX for ``label`` in the equations sidecar, when
    ``tools/reingest_equations.py`` has populated ``book.meta['equations']``.
    """
    eqs = book.meta.get("equations") if hasattr(book, "meta") else None
    if not eqs:
        return ""
    for e in eqs:
        if e.get("label") == label:
            lat = (e.get("latex") or "").strip()
            if lat:
                return lat
    return ""


def _chapter_for(label: str) -> str:
    """Return ``ch{N}`` for a label like ``"9.8"`` or ``"9.2"``."""
    head = label.split(".", 1)[0]
    return f"ch{head}"


def _walk_chapter(book: Book, label: str):
    """Yield BookNodes within the chapter implied by *label*."""
    chap = _chapter_for(label)
    target_prefix = f"b/{chap}"
    for n in book.root.walk():
        if n.nid == target_prefix or n.nid.startswith(target_prefix + "/"):
            yield n


# ---- Equation --------------------------------------------------------------

_EQ_BODY_LINE = re.compile(
    r"(?:\d|[A-Za-z]|=|[+\-*/^]|\(|\)|\[|\]|\\|"
    r"[Ͱ-Ͽ∀-⋿⨀-⫿])"
)


_MATH_CHARS = set(
    "=+-*/^<>≤≥≈≠≡≃∼∝→←⇒⇔∈∉⊂⊆⊃⊇∪∩∧∨"
    "∑∏∫∮∂∇√∞±∓·×÷|‖"
    "αβγδεζηθικλμνξοπρστυφχψω"
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    "ˆ¯˜"
)


# ---------------------------------------------------------------------------
# Plain-text → LaTeX heuristic
# ---------------------------------------------------------------------------
# PDF text extraction yields Unicode-laden math (α, ε, ·, ˆω, σ², …) split
# across multiple lines.  We don't have the original LaTeX; the best we can
# do is convert the common Unicode atoms back into LaTeX commands so KaTeX
# can render the result.  Imperfect but readable.

_GREEK_LOWER = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\varphi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
}
_GREEK_UPPER = {
    "Α": "A", "Β": "B", "Γ": r"\Gamma", "Δ": r"\Delta", "Ε": "E",
    "Ζ": "Z", "Η": "H", "Θ": r"\Theta", "Ι": "I", "Κ": "K",
    "Λ": r"\Lambda", "Μ": "M", "Ν": "N", "Ξ": r"\Xi", "Ο": "O",
    "Π": r"\Pi", "Ρ": "P", "Σ": r"\Sigma", "Τ": "T", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Χ": "X", "Ψ": r"\Psi", "Ω": r"\Omega",
}
_OPERATORS = {
    "·": r"\cdot", "×": r"\times", "÷": r"\div",
    "≤": r"\le", "≥": r"\ge", "≠": r"\ne", "≈": r"\approx",
    "≡": r"\equiv", "∼": r"\sim", "∝": r"\propto",
    "→": r"\to", "←": r"\leftarrow", "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq",
    "⊃": r"\supset", "⊇": r"\supseteq", "∪": r"\cup", "∩": r"\cap",
    "∧": r"\wedge", "∨": r"\vee",
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint",
    "∂": r"\partial", "∇": r"\nabla", "√": r"\sqrt",
    "∞": r"\infty", "±": r"\pm", "∓": r"\mp",
    "‖": r"\|", "·": r"\cdot",
    "−": "-", "—": "-",
}
_SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "ⁿ": "n",
}
_SUBSCRIPTS = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}


def to_latex(text: str) -> str:
    """Best-effort conversion of PDF-extracted math text to LaTeX source.

    Handles Greek letters, common operators, hats/bars (ˆx → \\hat{x}),
    sub/superscript Unicode, and `^N` / `_i` patterns.  Assumes the result
    will be wrapped in display-math delimiters by the caller.

    Lossy: we cannot reconstruct 2D layout (fractions stacked across two
    lines), so vertical splits produce ``\\\\`` line breaks within the
    math block — readable, even if not a true ``\\frac``.
    """
    # First, handle ˆ¯˜ over Greek letters (Unicode) BEFORE we replace the
    # Greek with backslash-commands, so ˆω → \hat{\omega} cleanly.
    greek_class = "".join(_GREEK_LOWER) + "".join(_GREEK_UPPER)
    text = re.sub(rf"ˆ([{greek_class}])",
                  lambda m: r"\hat{" + _GREEK_LOWER.get(
                      m.group(1), _GREEK_UPPER.get(m.group(1), m.group(1))
                  ) + "}", text)
    text = re.sub(rf"¯([{greek_class}])",
                  lambda m: r"\bar{" + _GREEK_LOWER.get(
                      m.group(1), _GREEK_UPPER.get(m.group(1), m.group(1))
                  ) + "}", text)
    text = re.sub(rf"˜([{greek_class}])",
                  lambda m: r"\tilde{" + _GREEK_LOWER.get(
                      m.group(1), _GREEK_UPPER.get(m.group(1), m.group(1))
                  ) + "}", text)
    # Then handle ASCII letters under hats/bars.
    text = re.sub(r"ˆ([A-Za-z]\w?)", lambda m: r"\hat{" + m.group(1) + "}", text)
    text = re.sub(r"¯([A-Za-z]\w?)", lambda m: r"\bar{" + m.group(1) + "}", text)
    text = re.sub(r"˜([A-Za-z]\w?)", lambda m: r"\tilde{" + m.group(1) + "}", text)

    # Substitute Greek + operators + sub/superscript glyphs.
    sub_runs: list[str] = []
    sup_runs: list[str] = []

    out_chars: list[str] = []
    pending_sup = ""
    pending_sub = ""

    def _flush_pending():
        nonlocal pending_sup, pending_sub
        if pending_sup:
            out_chars.append("^{" + pending_sup + "}")
            pending_sup = ""
        if pending_sub:
            out_chars.append("_{" + pending_sub + "}")
            pending_sub = ""

    for ch in text:
        if ch in _SUPERSCRIPTS:
            _flush_pending()  # only one run at a time
            pending_sup += _SUPERSCRIPTS[ch]
            continue
        if ch in _SUBSCRIPTS:
            _flush_pending()
            pending_sub += _SUBSCRIPTS[ch]
            continue
        _flush_pending()
        if ch in _GREEK_LOWER:
            out_chars.append(_GREEK_LOWER[ch] + " ")
        elif ch in _GREEK_UPPER:
            out_chars.append(_GREEK_UPPER[ch] + " ")
        elif ch in _OPERATORS:
            out_chars.append(_OPERATORS[ch] + " ")
        else:
            out_chars.append(ch)
    _flush_pending()

    s = "".join(out_chars)
    # Escape characters KaTeX treats as parameter / control chars.  These
    # come in via the PDF text and otherwise crash the parser.
    s = s.replace("#", r"\#").replace("&", r"\&").replace("%", r"\%")
    # Tidy multiple spaces.
    s = re.sub(r" {2,}", " ", s)
    # Per-line: detect a single uppercase letter line right after a value
    # — that's often the denominator of a stacked fraction (PDF flattening
    # ``d / N``).  We can't reliably reconstruct \frac, so leave it as is
    # but emit hard line breaks (\\) so KaTeX renders multi-line clearly.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return " \\\\ ".join(lines)


def _is_math_line(s: str) -> bool:
    """Heuristic: does *s* look like a piece of an equation rather than prose?"""
    if not s:
        return False
    if any(c in _MATH_CHARS for c in s):
        return True
    # Short symbol-only lines (e.g. "d", "N", "ε.").
    if len(s) <= 6 and not s[0].islower():
        # Single letter or short symbol with optional punctuation.
        return True
    if len(s) <= 8 and s.endswith((".", ",")) and not s[0].islower():
        return True
    return False


def _is_prose_line(s: str) -> bool:
    """Sentence-like prose that should NOT be glued onto the equation card."""
    if not s:
        return False
    if any(c in _MATH_CHARS for c in s):
        return False
    # Long line ending with sentence punctuation, no math characters.
    if len(s) > 40 and s.endswith((".", "?", "!")):
        return True
    # Short title-cased headings ("Estimates of In-Sample Prediction Error").
    if len(s) < 70 and not s.endswith((".", ",", ":", ";")):
        words = s.split()
        if 2 <= len(words) <= 10 and sum(
            1 for w in words if w[:1].isupper()
        ) >= max(2, len(words) - 2):
            return True
    return False


def _find_equation(book: Book, label: str) -> str:
    """Locate the body of equation *label* (e.g. "9.8") within its chapter.

    Walks back from the ``(N.M)`` marker and accepts only lines that look
    like equation content (math operators, Greek letters, short symbols).
    Stops at any prose paragraph or section heading.
    """
    marker = f"({label})"
    for n in _walk_chapter(book, label):
        body = n.body_text or ""
        idx = body.find(marker)
        if idx < 0:
            continue
        chunk = body[max(0, idx - 800):idx].rstrip()
        lines = chunk.split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        eq_lines: list[str] = []
        prior_marker = re.compile(r"\(\d+\.\d+\)")
        for line in reversed(lines):
            s = line.strip()
            if not s:
                if eq_lines:
                    break
                continue
            # Stop at the next equation marker above us — that belongs to
            # a different equation.
            if prior_marker.search(s):
                break
            if _is_prose_line(s):
                break
            if not _is_math_line(s) and eq_lines:
                break
            if _is_math_line(s) or not eq_lines:
                eq_lines.append(line)
            if len(eq_lines) >= 6:
                break
        eq_lines.reverse()
        text = "\n".join(eq_lines).strip()
        if text:
            return text
    return ""


# ---- Algorithm -------------------------------------------------------------

def _find_algorithm(book: Book, label: str) -> str:
    """Capture the heading + numbered body of ``Algorithm {label}``.

    PDF body_text typically contains the bare phrase ``Algorithm 9.2``
    several times (running headers, captions, prose mentions) before the
    real heading line, e.g. ``Algorithm 9.2 Local Scoring Algorithm for
    the Additive Logistic Regression Model.``  We pick the occurrence
    whose match is immediately followed (on the same line) by extended
    title text AND has a numbered step ``1.`` within the next ~12 lines.
    """
    # Match "Algorithm 9.2" followed by at least 4 chars of title text on
    # the same line — distinguishes the real heading from bare references.
    head_re = re.compile(
        rf"\bAlgorithm\s+{re.escape(label)}\s+[A-Za-z][^\n]{{3,}}"
    )
    other_alg_re = re.compile(r"^\s*Algorithm\s+\d+\.\d+\b")
    for n in _walk_chapter(book, label):
        body = n.body_text or ""
        for m in head_re.finditer(body):
            start = m.start()
            preview_lines = body[start:start + 1200].splitlines()[:14]
            has_step1 = any(re.match(r"^\s*1\.\s", ln)
                            for ln in preview_lines[1:])
            if not has_step1:
                continue
            window = body[start:start + 2400]
            lines = window.split("\n")
            captured: list[str] = []
            seen_step = False
            for line in lines:
                s = line.strip()
                if not s:
                    if seen_step:
                        break
                    continue
                # Stop at the next algorithm heading (different number).
                if seen_step and other_alg_re.match(line) and \
                        not s.startswith(f"Algorithm {label}"):
                    break
                # Stop when we hit a long prose paragraph after the steps.
                if seen_step and len(s) > 70 and s[0].isupper() \
                        and not re.match(r"\([a-z]\)", s) \
                        and not re.match(r"\d+\.", s) \
                        and "=" not in s and "ˆ" not in s:
                    break
                if re.match(r"\d+\.", s) or re.match(r"\([a-z]\)", s) \
                        or s.lower().startswith("iterate"):
                    seen_step = True
                captured.append(line)
                if len(captured) > 32:
                    break
            text = "\n".join(captured).strip()
            if seen_step:
                return text
    return ""


# ---- Table -----------------------------------------------------------------

def _find_table(book: Book, label: str) -> str:
    head_re = re.compile(rf"\bTable\s+{re.escape(label)}\b")
    for n in _walk_chapter(book, label):
        body = n.body_text or ""
        m = head_re.search(body)
        if not m:
            continue
        start = m.start()
        window = body[start:start + 800]
        # Take up to a blank-line break.
        parts = window.split("\n\n", 1)
        return parts[0].strip()
    return ""


# ---- Figure ----------------------------------------------------------------

def _find_figure(
    book: Book, label: str, hint_nid: str,
) -> tuple[Optional[FigureRef], str]:
    """Best-effort: resolve a "Figure N.M" to a :class:`FigureRef`.

    Strategy (cheap first):

      1. Cross-ref labelled exactly "Figure {label}" → to_nid → matching
         FigureRef.
      2. FigureRef whose ``meta['label']`` matches (post-v2 ingestion).
      3. **On-demand crop from the source PDF** — opens the PDF,
         locates the caption "FIGURE {label}", crops the block above
         it, caches the PNG.  This is the path that fixes references
         the original ingestion missed (e.g. ESLII Figure 6.14, which
         has no FigureRef and no cross_ref entry).
    """
    target_label = f"Figure {label}"
    target_nid = ""
    for cr in book.cross_refs:
        if cr.label == target_label:
            target_nid = cr.to_nid
            break
    if target_nid:
        for f in book.figures:
            if f.home_nid == target_nid:
                return f, target_nid
            if f.home_nid.startswith(target_nid + "/"):
                return f, target_nid

    # Match by v2 ingestion label.
    for f in book.figures:
        if f.meta.get("label") == target_label:
            return f, target_nid

    # On-demand crop from PDF.
    pdf_path = _book_pdf_path(book)
    if pdf_path:
        try:
            from .figure_ondemand import crop_figure_by_label
            od = crop_figure_by_label(pdf_path, label)
            if od is not None:
                synthetic = FigureRef(
                    fid=od.fid, home_nid=target_nid or hint_nid or "b",
                    page=od.page, caption=od.caption,
                    bbox=tuple(od.bbox), image_path=od.image_path,
                    meta={"source": "ondemand",
                          "label": target_label,
                          "cache_dir": od.cache_dir},
                )
                return synthetic, target_nid
        except Exception as e:
            # Never let on-demand crop break the reference card.
            print(f"[refcontent] on-demand crop failed for {target_label}: {e}")

    return None, target_nid


def _book_pdf_path(book: Book) -> str:
    """Best-effort path to the source PDF.  Looks at ``book.source`` and
    falls back to a sibling ``{book_stem}.pdf`` of the corpus JSON.
    """
    import os
    src = (book.source or "").strip()
    if src.lower().endswith(".pdf") and os.path.isfile(src):
        return src
    # If source field doesn't carry the PDF, infer from the meta.
    json_path = book.meta.get("source_json", "") or src
    if json_path:
        stem = os.path.splitext(json_path)[0]
        cand = stem + ".pdf"
        if os.path.isfile(cand):
            return cand
    return ""


# ---- Environments ---------------------------------------------------------

def _find_environment(
    book: Book, kind: str, label: str,
) -> tuple[str, str]:
    """Find the BookNode of kind *kind* whose ``number`` matches *label*.

    Returns ``(text, target_nid)``.
    """
    kind_lc = kind.lower()
    candidates: list[BookNode] = []
    for n in book.root.walk():
        if n.kind != kind_lc and not (
            kind_lc in ("theorem", "lemma", "proposition", "corollary",
                        "definition", "example", "exercise")
            and is_environment(n.kind) and n.kind == kind_lc
        ):
            continue
        if (n.number or "").strip() == label:
            candidates.append(n)
    if candidates:
        node = candidates[0]
        text = node.body_text.strip() or node.title
        return text, node.nid
    # Substring fallback — chapter body_text scan.
    head_re = re.compile(rf"\b{re.escape(kind)}\s+{re.escape(label)}\b")
    for n in _walk_chapter(book, label):
        body = n.body_text or ""
        m = head_re.search(body)
        if not m:
            continue
        start = m.start()
        # Capture until the next environment heading or a long prose
        # break (heuristic: 2 blank lines).
        window = body[start:start + 800]
        parts = re.split(r"\n\n+", window, maxsplit=2)
        return parts[0].strip(), n.nid
    return "", ""


# ---- Section / Chapter ----------------------------------------------------

def _find_section_summary(
    book: Book, kind: str, label: str,
) -> tuple[str, str]:
    """Return ``(summary_text, nid)`` for a Section/Chapter ref."""
    target = None
    for n in book.root.walk():
        num = (n.number or "").strip()
        if num != label:
            continue
        if kind == "Chapter" and n.kind == "chapter":
            target = n
            break
        if kind == "Section" and n.kind in ("section", "subsection"):
            target = n
            break
    if target is None:
        return "", ""
    title = target.title or ""
    # First non-empty paragraph of body_text — but skip PDF-extraction
    # artefacts (running headers like "Printer: Opaq", solitary digits
    # from page-numbers, repeated chapter titles) so the reference card
    # shows real prose, not OCR header debris.
    body = (target.body_text or "").strip()
    paragraph = ""
    for raw in body.split("\n\n"):
        cand = _clean_section_paragraph(raw, title=title, label=label)
        if cand:
            paragraph = cand
            break
    summary = (title + "\n" + paragraph).strip() if title else paragraph
    return summary[:600], target.nid


# Patterns that mark a body line as PDF extraction noise rather than
# real prose: running-header artefacts, solitary digits (page numbers
# leaking into the body), or the chapter title repeated.
_PDF_NOISE_RE = re.compile(
    r"^(?:Printer\s*:\s*\S+|Page\s*\d+|\d+|"
    r"Chapter\s*\d+|Section\s*[\d\.]+)\s*$",
    flags=re.IGNORECASE,
)


def _clean_section_paragraph(raw: str, *, title: str, label: str) -> str:
    """Strip PDF-extraction noise from *raw* and return the residue.

    Drops standalone page numbers, ``Printer: Opaq`` running headers,
    blank lines, and the chapter / section title when it appears
    duplicated above the body.  Joins the remaining lines with spaces
    so the card displays clean prose instead of an OCR ladder.
    """
    if not raw:
        return ""
    title_norm = (title or "").strip().lower()
    keep: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _PDF_NOISE_RE.match(s):
            continue
        if title_norm and s.lower() == title_norm:
            continue
        if label and s == label:
            continue
        keep.append(s)
    if not keep:
        return ""
    return " ".join(keep)
