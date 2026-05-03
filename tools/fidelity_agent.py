"""Math-aware formula fidelity agent (Section A17 of QUALITY_INSPECTOR_DESIGN.md).

For every formula in ``<book>.math_graph.json`` whose ``cite_label``
matches an ``Equation N.M`` pattern, this agent:

  1. resolves the formula's host page from its ``home_nid``,
  2. crops the PDF region containing the cite marker (e.g. ``(11.9)``),
  3. asks the local Qwen2.5-VL endpoint at :8004 whether the
     extracted ``latex`` faithfully represents the math in the crop,
  4. on disagreement, asks the local Qwen text LLM at :8000 to repair
     the latex given the VLM's reason,
  5. re-checks against the same crop with the repaired latex, up to
     ``max_iters`` times,
  6. records ``match`` / ``repaired`` / ``degraded`` per formula.

This is the second agentic surface point (the first is
``tools/ingest_agent.py``).  It runs once per math-graph rebuild,
NOT in the user-facing narration path.

Output: ``books/<stem>.fidelity.json``  + optional patch file with
the repaired latex strings (apply with ``--apply``).

Local-only: VLM at :8004, text LLM at :8000.  No external API.

Usage:
    .venv/bin/python3 -m tools.fidelity_agent books/ESLII.json
    .venv/bin/python3 -m tools.fidelity_agent books/ESLII.json --max 30
    .venv/bin/python3 -m tools.fidelity_agent books/ESLII.json --apply
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

PROJECT = Path(__file__).resolve().parent.parent

VLM_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8004/v1") \
          + "/chat/completions"
VLM_MODEL = os.environ.get("VLM_MODEL",
                            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"

# How many "ask Qwen to repair, re-check" cycles per formula before
# we mark it degraded.
DEFAULT_MAX_ITERS = 2

# Crop around the cite marker.  Typical equation height is 30-60 px,
# width 200-500 px.  We grab a big enough rectangle ABOVE the marker
# and to its left to cover the formula body.
CROP_HEIGHT_PT = 90.0
CROP_WIDTH_PT = 360.0
CROP_LEFT_PAD_PT = 60.0
CROP_BELOW_PT = 16.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _build_nid_to_page(corpus: dict) -> dict[str, int]:
    """Walk the BookNode tree, return ``{nid: page_index}`` for every
    node that carries a ``page_start``."""
    out: dict[str, int] = {}
    def walk(n: dict):
        if not isinstance(n, dict):
            return
        nid = n.get("nid")
        if nid:
            ps = n.get("page_start")
            if isinstance(ps, int):
                out[nid] = max(0, ps - 1)  # PyMuPDF uses 0-based
        for c in n.get("children", []) or []:
            walk(c)
    walk(corpus.get("root", corpus) or {})
    return out


# ---------------------------------------------------------------------------
# VLM + LLM calls
# ---------------------------------------------------------------------------

_COMPARE_PROMPT_TEMPLATE = (
    "I'm verifying an extracted LaTeX representation of an equation "
    "from a textbook.\n\n"
    "EXTRACTED LATEX (what we have):\n  {latex}\n\n"
    "Look at the image.  Does the LaTeX above represent the same "
    "mathematical statement as the equation in the image (modulo "
    "harmless formatting differences like ``\\\\theta`` vs ``θ``)?\n\n"
    "Reply with ONE LINE of JSON only, no other text:\n"
    '  {{"match": true,  "reason": "..."}}\n'
    "or\n"
    '  {{"match": false, "reason": "<one line: what is wrong>", '
    '   "expected_latex": "<your best LaTeX of what the image actually '
    'shows>"}}'
)

_REPAIR_PROMPT_TEMPLATE = (
    "An OCR pass extracted this LaTeX from a textbook equation, but a "
    "vision check disagreed.\n\n"
    "ORIGINAL LATEX (broken):\n  {original}\n\n"
    "VISION-MODEL DIAGNOSIS:\n  {reason}\n\n"
    "VISION-MODEL'S BEST GUESS OF THE TRUE LATEX:\n  {expected}\n\n"
    "Produce a corrected LaTeX string that addresses the diagnosis.  "
    "Prefer the smallest fix that resolves the disagreement; do not "
    "rewrite parts the diagnosis didn't flag.  Reply with one line of "
    "JSON only:\n"
    '  {{"latex": "<corrected LaTeX>"}}'
)


def _vlm_compare(png_bytes: bytes, latex: str, *,
                  timeout: float = 25.0) -> dict:
    """Returns ``{"match": bool, "reason": str, "expected_latex": str}``
    or empty dict on failure."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    body = json.dumps({
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": _COMPARE_PROMPT_TEMPLATE.format(latex=latex)},
            ],
        }],
        "max_tokens": 280,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }).encode("utf-8")
    req = urllib.request.Request(
        VLM_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local-vlm"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"_error": repr(e)}
    txt = ((payload.get("choices") or [{}])[0]
           .get("message") or {}).get("content", "").strip()
    if not txt:
        return {"_error": "empty content"}
    # The VLM sometimes wraps its JSON in ``` fences.  Strip them.
    txt = re.sub(r"^```(?:json)?\s*", "", txt).rstrip("`").rstrip()
    try:
        d = json.loads(txt)
    except Exception:
        # Last-ditch: regex out the first {...} payload.
        m = re.search(r"\{[^{}]*\}", txt, re.DOTALL)
        if not m:
            return {"_error": "no JSON in response", "_raw": txt[:200]}
        try:
            d = json.loads(m.group(0))
        except Exception:
            return {"_error": "bad JSON", "_raw": txt[:200]}
    if "match" not in d:
        return {"_error": "no match key", "_raw": txt[:200]}
    return d


def _llm_repair(original_latex: str, reason: str,
                  expected: str, *, timeout: float = 30.0) -> str:
    """Ask the text LLM to repair ``original_latex``.  Returns the
    repaired string or ``""`` on failure."""
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{
            "role": "user",
            "content": _REPAIR_PROMPT_TEMPLATE.format(
                original=original_latex, reason=reason, expected=expected,
            ),
        }],
        "max_tokens": 240,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        LLM_URL, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""
    txt = ((payload.get("choices") or [{}])[0]
           .get("message") or {}).get("content", "").strip()
    if not txt:
        return ""
    try:
        d = json.loads(txt)
    except Exception:
        return ""
    return (d.get("latex") or "").strip()


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def _find_cite_bbox(page: "fitz.Page", cite_text: str
                      ) -> Optional[tuple[float, float, float, float]]:
    """Return the bbox of the cite marker (e.g. ``(11.9)``) on this
    page, or None if not found.  PyMuPDF's ``search_for`` returns a
    list of rects; we take the LAST occurrence (equation citations
    usually sit at the right margin AFTER the equation body)."""
    hits = page.search_for(cite_text)
    if not hits:
        return None
    r = hits[-1]
    return (r.x0, r.y0, r.x1, r.y1)


def _crop_around_cite(page: "fitz.Page", cite_bbox: tuple[float, float, float, float]
                        ) -> bytes:
    """Render a region of the page above/around the cite marker as PNG."""
    import fitz
    x0, y0, x1, y1 = cite_bbox
    w = page.rect.width
    h = page.rect.height
    # The equation body sits ABOVE and to the LEFT of the cite marker,
    # which is right-aligned in the typical style.
    crop = fitz.Rect(
        max(0.0, x0 - CROP_WIDTH_PT - CROP_LEFT_PAD_PT),
        max(0.0, y0 - CROP_HEIGHT_PT),
        min(w, x1 + 30.0),
        min(h, y1 + CROP_BELOW_PT),
    )
    pix = page.get_pixmap(clip=crop, dpi=180)
    return pix.tobytes("png")


# ---------------------------------------------------------------------------
# Per-formula loop
# ---------------------------------------------------------------------------

@dataclass
class _Outcome:
    formula_id: str
    cite_label: str
    home_nid: str
    page: int
    original_latex: str
    final_latex: str
    iterations: int
    status: str                # match | repaired | degraded | unverifiable
    reasons: list[str] = field(default_factory=list)


def _verify_one(*, formula_id: str, formula: dict,
                 nid_page: dict[str, int], pdf,
                 max_iters: int) -> _Outcome:
    cite = (formula.get("cite_label") or "").strip()
    home = formula.get("home_nid") or ""
    original = (formula.get("latex") or "").strip()
    out = _Outcome(
        formula_id=formula_id, cite_label=cite, home_nid=home,
        page=-1, original_latex=original, final_latex=original,
        iterations=0, status="unverifiable",
    )
    if not cite or not original or not home:
        out.reasons.append("missing cite_label / latex / home_nid")
        return out
    # Resolve page.  Try the formula's home_nid, then walk up.
    page_idx = nid_page.get(home)
    if page_idx is None:
        parent = home
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            if parent in nid_page:
                page_idx = nid_page[parent]
                break
    if page_idx is None:
        out.reasons.append(f"no page resolved for home {home}")
        return out
    out.page = page_idx
    # Prefer "(11.9)" over "Equation 11.9" as the search target
    # because PDFs almost always print the parenthesised form.
    m = re.search(r"(\d+\.\d+)", cite)
    if not m:
        out.reasons.append(f"no numeric label inside {cite!r}")
        return out
    label_num = m.group(1)
    # The home_nid often resolves only to the chapter root, while the
    # equation can sit dozens of pages into the chapter.  Sweep
    # forward up to ``LOOKAHEAD`` pages from the home page until we
    # find the cite marker.  ``(N.M)`` strings are unique in a
    # textbook so the first match is unambiguous.
    LOOKAHEAD = 60
    bbox = None
    page = None
    found_page_idx = -1
    end = min(pdf.page_count, page_idx + LOOKAHEAD)
    for p_idx in range(page_idx, end):
        p = pdf.load_page(p_idx)
        b = _find_cite_bbox(p, f"({label_num})")
        if b is not None:
            page = p
            bbox = b
            found_page_idx = p_idx
            break
    if bbox is None:
        out.reasons.append(f"cite ({label_num}) not found in pages "
                           f"{page_idx}..{end-1}")
        return out
    out.page = found_page_idx
    png = _crop_around_cite(page, bbox)

    current_latex = original
    for _ in range(max_iters + 1):
        verdict = _vlm_compare(png, current_latex)
        out.iterations += 1
        if verdict.get("_error"):
            out.reasons.append(f"vlm error: {verdict['_error']}")
            out.status = "unverifiable"
            return out
        if verdict.get("match") is True:
            out.final_latex = current_latex
            out.status = "match" if out.iterations == 1 else "repaired"
            return out
        reason = (verdict.get("reason") or "").strip()
        expected = (verdict.get("expected_latex") or "").strip()
        out.reasons.append(reason)
        if out.iterations > max_iters:
            break
        repaired = _llm_repair(current_latex, reason, expected)
        if not repaired or repaired == current_latex:
            break
        current_latex = repaired
    out.final_latex = current_latex
    out.status = "degraded"
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(book_json: str, *, max_formulas: int = 0,
        max_iters: int = DEFAULT_MAX_ITERS,
        verbose: bool = True,
        apply_repairs: bool = False) -> dict:
    import fitz
    stem = book_json[: -len(".json")] if book_json.endswith(".json") \
           else book_json
    corpus = _read_json(book_json) or {}
    mg = _read_json(stem + ".math_graph.json") or {}
    pdf_path = stem + ".pdf"
    if not os.path.isfile(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")
    formulas = mg.get("formulas") or {}
    if not formulas:
        raise SystemExit(f"math_graph has no formulas: {stem}.math_graph.json")
    nid_page = _build_nid_to_page(corpus)
    pdf = fitz.open(pdf_path)

    # Eligible: those with a cite_label like "Equation N.M".
    eligible = [
        (fid, f) for fid, f in formulas.items()
        if isinstance(f, dict)
        and any(re.search(r"\d+\.\d+", c or "")
                for c in (f.get("cite_labels") or []))
    ]
    # Each formula carries a list of cite_labels; collapse to one
    # per (fid, label) so we don't double-check.
    work: list[tuple[str, dict, str]] = []
    seen: set[tuple[str, str]] = set()
    for fid, f in eligible:
        labels = f.get("cite_labels") or []
        for lab in labels:
            num = re.search(r"\d+\.\d+", lab or "")
            if not num:
                continue
            key = (fid, num.group(0))
            if key in seen:
                continue
            seen.add(key)
            work.append((fid, dict(f, cite_label=lab), num.group(0)))
            break  # one citation per fid is enough for the verdict
    if max_formulas:
        work = work[:max_formulas]

    print(f"[fidelity] {len(work)} formulas to verify", flush=True)
    outcomes: list[_Outcome] = []
    counts = {"match": 0, "repaired": 0, "degraded": 0,
              "unverifiable": 0}
    t0 = time.time()
    for i, (fid, f, _num) in enumerate(work, 1):
        try:
            out = _verify_one(
                formula_id=fid, formula=f,
                nid_page=nid_page, pdf=pdf, max_iters=max_iters,
            )
        except Exception as e:
            out = _Outcome(
                formula_id=fid, cite_label=f.get("cite_label", ""),
                home_nid=f.get("home_nid", ""),
                page=-1, original_latex=f.get("latex", ""),
                final_latex=f.get("latex", ""),
                iterations=0, status="unverifiable",
                reasons=[f"exception: {e!r}"],
            )
        counts[out.status] = counts.get(out.status, 0) + 1
        outcomes.append(out)
        if verbose and (i % 25 == 0 or i == len(work)):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.01)
            eta = (len(work) - i) / max(rate, 0.01)
            print(f"[fidelity] {i}/{len(work)}  match={counts['match']} "
                  f"repaired={counts['repaired']} "
                  f"degraded={counts['degraded']} "
                  f"unverif={counts['unverifiable']}  "
                  f"~{eta:.0f}s remaining", flush=True)

    report = {
        "book":     book_json,
        "ts":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed":  round(time.time() - t0, 1),
        "summary":  counts,
        "outcomes": [asdict(o) for o in outcomes],
    }
    out_path = stem + ".fidelity.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[fidelity] wrote {out_path}", flush=True)
    print(f"[fidelity] summary: {counts}", flush=True)

    if apply_repairs:
        n = 0
        for o in outcomes:
            if o.status == "repaired" and o.final_latex \
                    and o.final_latex != o.original_latex:
                f = formulas.get(o.formula_id)
                if isinstance(f, dict):
                    f["latex"] = o.final_latex
                    n += 1
        if n:
            with open(stem + ".math_graph.json", "w") as out_f:
                json.dump(mg, out_f, indent=2, default=str)
            print(f"[fidelity] applied {n} repairs to math_graph.json",
                  flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("book_json", help="path to books/<stem>.json")
    ap.add_argument("--max", type=int, default=0,
                    help="check at most this many formulas (0 = all)")
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS,
                    help="repair attempts per formula (default 2)")
    ap.add_argument("--apply", action="store_true",
                    help="patch math_graph.json with repaired latex")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.book_json):
        print(f"book corpus not found: {args.book_json}", file=sys.stderr)
        return 1
    rep = run(args.book_json, max_formulas=args.max,
              max_iters=args.max_iters, apply_repairs=args.apply)
    return 0 if rep["summary"]["unverifiable"] == 0 \
                 and rep["summary"]["degraded"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
