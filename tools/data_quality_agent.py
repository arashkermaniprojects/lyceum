"""Tier-1 data-quality agent (Section A of QUALITY_INSPECTOR_DESIGN.md).

Reads every sidecar a book needs, runs the full battery of data-layer
checks (A1-A10 in the design doc), and writes a structured health
report to ``books/<stem>.health.json``.

Output schema:

    {
      "book": "books/ESLII.json",
      "ts":   "2026-05-03T08:01:00Z",
      "summary": {"pass": 9, "fail": 1, "skip": 0},
      "checks": {
        "A1_sidecars_present":        {"status": "pass", "detail": {...}},
        "A2_math_graph_integrity":    {"status": "fail", "detail": {...}},
        ...
      }
    }

Entirely local — reads JSON files on disk + makes one optional HTTP
call to KaTeX (server-side render via the running Lyceum process or
a node-side helper if available).  No outbound HTTPS.

Usage:
    .venv/bin/python3 -m tools.data_quality_agent books/ESLII.json
    # exit code 0 if every check passed, 2 if any failed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _walk_chapters(corpus: dict) -> list[str]:
    """Same heuristic as serve.ingest_pipeline._is_real_chapter, abridged."""
    out: list[str] = []
    front = {
        "cover", "title page", "title", "copyright", "dedication",
        "contents", "table of contents", "index", "indices",
        "preface", "preface to the first edition",
        "preface to the second edition", "preface to the third edition",
        "front matter", "back matter", "bibliography", "references",
        "selected bibliography", "acknowledgments", "acknowledgements",
        "errata", "colophon",
        "name index", "subject index", "author index", "topic index",
        "list of figures", "list of tables", "list of symbols",
        "list of theorems", "list of definitions",
        "notation", "abbreviations", "glossary", "statement",
    }
    def walk(n: dict):
        kind = (n.get("kind") or "").lower()
        title = (n.get("title") or "").strip().lower()
        if kind == "chapter" and title not in front:
            out.append(n.get("nid", ""))
            return
        for c in n.get("children", []) or []:
            walk(c)
    walk(corpus.get("root", corpus) or {})
    return [n for n in out if n]


def _all_nids(corpus: dict) -> set[str]:
    out: set[str] = set()
    def walk(n: dict):
        nid = n.get("nid")
        if nid: out.add(nid)
        for c in n.get("children", []) or []:
            walk(c)
    walk(corpus.get("root", corpus) or {})
    return out


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_A1_sidecars_present(stem: str, corpus: Optional[dict]
                                ) -> dict:
    """Every required sidecar exists on disk."""
    required = {
        "corpus_json":   stem + ".json",
        "pdf":           stem + ".pdf",
        "concepts":      stem + ".concepts.json",
        "formulas":      stem + ".formulas.json",
        "math_graph":    stem + ".math_graph.json",
        "figures":       stem + ".figures.json",
    }
    missing = [k for k, p in required.items() if not os.path.isfile(p)]
    if not corpus:
        return {"status": "fail",
                "detail": {"missing": missing,
                           "note": "corpus JSON unreadable"}}
    chapters = _walk_chapters(corpus)
    missing_maps: list[str] = []
    for ch in chapters:
        flat = ch.replace("/", "_")
        if not os.path.isfile(f"{stem}.chapter_map.{flat}.json"):
            missing_maps.append(ch)
    return {
        "status": "pass" if (not missing and not missing_maps) else "fail",
        "detail": {
            "missing":          missing,
            "missing_maps":     missing_maps[:8],
            "n_chapters":       len(chapters),
            "n_chapter_maps":   len(chapters) - len(missing_maps),
        },
    }


_TRUNC_RE = re.compile(r"=\s*-\s*$|^\s*X\s+\w+\s*=\s*\d+\s*$")
_OCR_ARTIFACT_RE = re.compile(
    r"\bX\s+[a-zA-Zα-ω](?:,[a-zA-Zα-ω])+\b|"        # multi-index sum unrepaired
    r"\b[A-Za-z]T\s+[a-z]\s*[A-Z]\b|"               # transpose-subscript
    r"\b[A-Za-z][a-z]{1,2}(?![a-zA-Z])(?=[ ,)\]}=+\-*/^_\\\(]|$)" # glued sub
)


def check_A2_math_graph_integrity(stem: str, corpus: Optional[dict]
                                    ) -> dict:
    mg = _read_json(stem + ".math_graph.json")
    if not isinstance(mg, dict):
        return {"status": "fail", "detail": {"reason": "math_graph missing"}}
    formulas = mg.get("formulas") or {}
    if not formulas:
        return {"status": "fail",
                "detail": {"reason": "math_graph has no formulas"}}
    all_nids = _all_nids(corpus or {})
    n_total = 0
    n_homeless = n_truncated = n_empty = 0
    n_orphan_home = 0
    n_unrepaired = 0
    samples_truncated: list[str] = []
    samples_unrepaired: list[str] = []
    for fid, f in formulas.items():
        if not isinstance(f, dict):
            continue
        n_total += 1
        latex = (f.get("latex") or "").strip()
        home = f.get("home_nid") or ""
        if not latex:
            n_empty += 1
            continue
        if not home:
            n_homeless += 1
        elif all_nids and home not in all_nids:
            n_orphan_home += 1
        if _TRUNC_RE.search(latex):
            n_truncated += 1
            if len(samples_truncated) < 4:
                cite = (f.get("cite_labels") or [None])[0] or fid
                samples_truncated.append(f"{cite}: {latex[:80]}")
        # Heuristic for "unrepaired OCR" — fragments still containing
        # the X-glyph or transpose-subscript or glued single-letter
        # tails OUTSIDE of an existing subscript brace.
        if "_{" not in latex and "X " in latex \
                and re.search(r"\bX\s+[a-zA-Z]", latex):
            n_unrepaired += 1
            if len(samples_unrepaired) < 4:
                samples_unrepaired.append(latex[:80])
    bad = (n_truncated + n_homeless + n_orphan_home + n_unrepaired) > 0
    return {
        "status": "fail" if bad else "pass",
        "detail": {
            "n_formulas":         n_total,
            "n_truncated":        n_truncated,
            "n_empty":            n_empty,
            "n_homeless":         n_homeless,
            "n_orphan_home_nid":  n_orphan_home,
            "n_unrepaired_OCR":   n_unrepaired,
            "samples_truncated":  samples_truncated,
            "samples_unrepaired": samples_unrepaired,
        },
    }


def check_A3_figures_sidecar(stem: str, corpus: Optional[dict]) -> dict:
    figures = _read_json(stem + ".figures.json")
    if not isinstance(figures, dict):
        return {"status": "fail", "detail": {"reason": "figures sidecar missing"}}
    by_nid = figures.get("by_nid") or {}
    image_dir = figures.get("image_dir") or ""
    img_root = os.path.join(os.path.dirname(stem + ".json"), image_dir) \
               if image_dir else ""
    all_nids = _all_nids(corpus or {})
    n_total = n_orphan = n_missing_png = 0
    for home, entries in by_nid.items():
        if all_nids and home not in all_nids:
            n_orphan += 1
        for e in (entries or []):
            n_total += 1
            ip = e.get("image_path") or ""
            if img_root and ip and not os.path.isfile(
                    os.path.join(img_root, ip)):
                n_missing_png += 1
    bad = n_total < 5 or n_orphan > 0 or n_missing_png > 0
    return {
        "status": "fail" if bad else "pass",
        "detail": {
            "n_figures":      n_total,
            "n_orphan_nids":  n_orphan,
            "n_missing_pngs": n_missing_png,
            "image_dir":      image_dir,
        },
    }


def check_A4_chapter_map_coverage(stem: str, corpus: Optional[dict],
                                    threshold: float = 0.75) -> dict:
    if not corpus:
        return {"status": "skip", "detail": {"reason": "no corpus"}}
    chapters = _walk_chapters(corpus)
    rows: list[dict] = []
    n_below = 0
    for ch in chapters:
        flat = ch.replace("/", "_")
        cm = _read_json(f"{stem}.chapter_map.{flat}.json")
        if not isinstance(cm, dict):
            rows.append({"chapter": ch, "coverage": None,
                         "note": "missing"})
            n_below += 1
            continue
        total = pop = 0
        def walk(n: dict):
            nonlocal total, pop
            total += 1
            if (n.get("story_paragraph") or "").strip():
                pop += 1
            for c in n.get("children", []) or []:
                walk(c)
        walk(cm.get("root", cm) or {})
        cov = pop / max(total, 1)
        if cov < threshold:
            n_below += 1
        rows.append({"chapter": ch, "coverage": round(cov, 2),
                     "populated": pop, "total": total})
    return {
        "status": "fail" if n_below else "pass",
        "detail": {"threshold": threshold,
                   "n_below_threshold": n_below,
                   "per_chapter": rows},
    }


def check_A5_concept_coverage(stem: str, corpus: Optional[dict]) -> dict:
    cc = _read_json(stem + ".concepts.json")
    if not isinstance(cc, dict):
        return {"status": "fail", "detail": {"reason": "concepts missing"}}
    by_home = cc.get("by_home_nid") or {}
    if len(by_home) < 5:
        return {"status": "fail",
                "detail": {"n_concepts": len(by_home),
                           "reason": "fewer than 5 concept entries"}}
    return {"status": "pass", "detail": {"n_concepts": len(by_home)}}


def check_A6_formula_explanations(stem: str) -> dict:
    fl = _read_json(stem + ".formulas.json")
    if not isinstance(fl, dict):
        return {"status": "fail", "detail": {"reason": "formulas.json missing"}}
    by_id = fl.get("by_formula_id") or {}
    if not by_id:
        return {"status": "fail",
                "detail": {"reason": "formulas.json has no by_formula_id"}}
    n_total = len(by_id)
    n_full = sum(
        1 for v in by_id.values()
        if isinstance(v, dict)
        and (v.get("F0_role") or "").strip()
        and (v.get("F1_meaning") or "").strip()
        and (v.get("F2_walk") or "").strip()
    )
    cov = n_full / max(n_total, 1)
    return {
        "status": "pass" if cov >= 0.5 else "fail",
        "detail": {"n_formulas_in_layer": n_total,
                   "n_with_full_F0F1F2": n_full,
                   "coverage": round(cov, 2)},
    }


def check_A7_theorem_coverage(stem: str, corpus: Optional[dict]) -> dict:
    """Scan body_text for THEOREM/DEFINITION/ALGORITHM markers; verify
    the figures sidecar (which holds these) has an entry per marker."""
    if not corpus:
        return {"status": "skip", "detail": {"reason": "no corpus"}}
    figures = _read_json(stem + ".figures.json") or {}
    by_nid = figures.get("by_nid") or {}
    have: set[str] = set()
    for entries in by_nid.values():
        for e in entries or []:
            lab = (e.get("label") or "").strip()
            if lab:
                have.add(lab)
    marker = re.compile(
        r"\b(?:THEOREM|Theorem|DEFINITION|Definition|"
        r"LEMMA|Lemma|COROLLARY|Corollary|"
        r"ALGORITHM|Algorithm)\s+(\d+\.\d+)"
    )
    missing: list[str] = []
    n_seen = 0
    def walk(n: dict):
        nonlocal n_seen
        body = n.get("body_text", "") or ""
        for m in marker.finditer(body):
            kind = m.group(0).split()[0].title()
            label = f"{kind} {m.group(1)}"
            n_seen += 1
            # Also accept the same number under the "Theorem" / "Lemma"
            # umbrella regardless of original kind, since the figures
            # extractor sometimes consolidates.
            label_alt = f"Theorem {m.group(1)}"
            if label not in have and label_alt not in have:
                if label not in missing:
                    missing.append(label)
        for c in n.get("children", []) or []:
            walk(c)
    walk(corpus.get("root", corpus) or {})
    cov = (n_seen - len(missing)) / max(n_seen, 1)
    return {
        "status": "pass" if cov >= 0.7 else "fail",
        "detail": {"n_markers_in_corpus": n_seen,
                   "n_missing_in_sidecar": len(missing),
                   "coverage": round(cov, 2),
                   "samples_missing": missing[:8]},
    }


def check_A8_cross_references(stem: str, corpus: Optional[dict]) -> dict:
    """Every (N.M) citation in the corpus should resolve to either an
    equation (in math_graph), a figure / theorem (in figures sidecar),
    or have a known label."""
    if not corpus:
        return {"status": "skip", "detail": {"reason": "no corpus"}}
    mg = _read_json(stem + ".math_graph.json") or {}
    figures = _read_json(stem + ".figures.json") or {}
    eq_labels: set[str] = set()
    for f in (mg.get("formulas") or {}).values():
        if isinstance(f, dict):
            for c in (f.get("cite_labels") or []):
                eq_labels.add(c.strip())
    fig_labels: set[str] = set()
    for entries in (figures.get("by_nid") or {}).values():
        for e in entries or []:
            lab = (e.get("label") or "").strip()
            if lab:
                fig_labels.add(lab)
    cite_re = re.compile(r"\((\d+\.\d+)\)")
    n_seen = 0
    unresolved: list[str] = []
    seen_unresolved: set[str] = set()
    def walk(n: dict):
        nonlocal n_seen
        body = n.get("body_text", "") or ""
        for m in cite_re.finditer(body):
            n_seen += 1
            num = m.group(1)
            ok = (
                f"Equation {num}" in eq_labels
                or f"Figure {num}" in fig_labels
                or f"Table {num}" in fig_labels
                or f"Theorem {num}" in fig_labels
                or f"Algorithm {num}" in fig_labels
                or f"Definition {num}" in fig_labels
            )
            if not ok and num not in seen_unresolved:
                seen_unresolved.add(num)
                unresolved.append(num)
        for c in n.get("children", []) or []:
            walk(c)
    walk(corpus.get("root", corpus) or {})
    cov = (n_seen - len(unresolved)) / max(n_seen, 1)
    return {
        "status": "pass" if cov >= 0.7 else "fail",
        "detail": {"n_citations": n_seen,
                   "n_unresolved": len(unresolved),
                   "coverage": round(cov, 2),
                   "samples_unresolved": unresolved[:10]},
    }


def check_A9_katex_validity(stem: str, sample: int = 30) -> dict:
    """Sample math_graph latex strings and verify they parse via KaTeX.

    KaTeX server-side requires Node; we approximate with a shell-out
    that is best-effort.  When ``katex`` isn't installed we SKIP this
    check rather than fail (KaTeX validity is then guarded by the
    runtime when the page renders)."""
    try:
        import shutil
        if not shutil.which("katex"):
            return {"status": "skip",
                    "detail": {"reason": "katex CLI not installed"}}
    except Exception:
        return {"status": "skip", "detail": {"reason": "shutil unavailable"}}
    mg = _read_json(stem + ".math_graph.json") or {}
    formulas = mg.get("formulas") or {}
    if not formulas:
        return {"status": "skip", "detail": {"reason": "no formulas"}}
    import random, subprocess
    keys = list(formulas.keys())
    random.shuffle(keys)
    fail: list[str] = []
    n_tried = 0
    for fid in keys[:sample]:
        f = formulas[fid]
        if not isinstance(f, dict):
            continue
        latex = (f.get("latex") or "").strip()
        if not latex:
            continue
        n_tried += 1
        try:
            proc = subprocess.run(
                ["katex"], input=latex, text=True,
                capture_output=True, timeout=5,
            )
            if proc.returncode != 0:
                fail.append(f"{fid}: {latex[:60]}")
        except Exception as e:
            fail.append(f"{fid}: <katex error: {e}>")
            break
    return {
        "status": "fail" if fail else "pass",
        "detail": {"n_sampled": n_tried,
                   "n_failed": len(fail),
                   "samples_failed": fail[:5]},
    }


def check_A10_local_only_invariant() -> dict:
    """Grep the entire serve/ + tools/ + sevim/ tree for outbound URLs
    that would imply a non-local API call."""
    bad_hosts = [
        "api.openai.com", "api.anthropic.com", "claude.ai",
        "openai.azure.com", "googleapis.com/generativelanguage",
        "api.together.xyz", "api.mistral.ai",
    ]
    hits: list[str] = []
    # Skip files that legitimately *list* these hostnames as part of
    # their detection logic (this agent itself, plus any other
    # health/security checker).
    self_path = Path(__file__).resolve()
    for sub in ("serve", "tools", "sevim", "narrator", "chalkboard"):
        root = PROJECT / sub
        if not root.is_dir():
            continue
        for fp in root.rglob("*.py"):
            if fp.resolve() == self_path:
                continue
            try:
                content = fp.read_text(errors="ignore")
            except Exception:
                continue
            for h in bad_hosts:
                if h in content:
                    hits.append(f"{fp.relative_to(PROJECT)}: {h}")
    # Same scan over the static JS.
    js = PROJECT / "serve" / "static" / "index.html"
    if js.is_file():
        content = js.read_text(errors="ignore")
        for h in bad_hosts:
            if h in content:
                hits.append(f"serve/static/index.html: {h}")
    return {
        "status": "fail" if hits else "pass",
        "detail": {"n_violations": len(hits), "hits": hits[:10]},
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CHECKS = [
    ("A1_sidecars_present",       check_A1_sidecars_present),
    ("A2_math_graph_integrity",   check_A2_math_graph_integrity),
    ("A3_figures_sidecar",        check_A3_figures_sidecar),
    ("A4_chapter_map_coverage",   check_A4_chapter_map_coverage),
    ("A5_concept_coverage",       lambda s, c: check_A5_concept_coverage(s, c)),
    ("A6_formula_explanations",   lambda s, c: check_A6_formula_explanations(s)),
    ("A7_theorem_coverage",       check_A7_theorem_coverage),
    ("A8_cross_references",       check_A8_cross_references),
    ("A9_katex_validity",         lambda s, c: check_A9_katex_validity(s)),
    ("A10_local_only_invariant",  lambda s, c: check_A10_local_only_invariant()),
]


def run(book_json: str) -> dict:
    stem = book_json[: -len(".json")] if book_json.endswith(".json") \
           else book_json
    corpus = _read_json(book_json)
    results: dict = {}
    counts = {"pass": 0, "fail": 0, "skip": 0}
    for cid, fn in CHECKS:
        try:
            r = fn(stem, corpus)
        except Exception as e:
            r = {"status": "fail", "detail": {"exception": repr(e)}}
        if not isinstance(r, dict) or "status" not in r:
            r = {"status": "fail", "detail": {"reason": "malformed result"}}
        results[cid] = r
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    report = {
        "book":    book_json,
        "ts":      _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": counts,
        "checks":  results,
    }
    out_path = stem + ".health.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[health] wrote {out_path}")
    print(f"[health] summary: pass={counts['pass']} "
          f"fail={counts['fail']} skip={counts['skip']}")
    for cid, r in results.items():
        flag = {"pass": "✓", "fail": "✗", "skip": "·"}.get(r["status"], "?")
        print(f"  {flag} {cid}")
        if r["status"] == "fail":
            d = r.get("detail") or {}
            for k, v in list(d.items())[:5]:
                if isinstance(v, list) and v:
                    print(f"      {k}: {v[:3]}")
                else:
                    print(f"      {k}: {v}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("book_json", help="path to books/<stem>.json")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.book_json):
        print(f"book corpus not found: {args.book_json}", file=sys.stderr)
        return 1
    rep = run(args.book_json)
    return 0 if rep["summary"]["fail"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
