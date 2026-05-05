"""M3 — Visual-primitive emission profile.

For every per-chapter sidecar in ``books/ESLII.chapter_map.b_chN.json``,
walk each ``story_paragraph`` and split it into clauses, then run the
runtime primitive detectors against each clause:

* ``operation_card``    — ``viz.operations.find_operations``
* ``reference_card``    — ``serve/orchestrator._REF_PATTERNS``
* ``math_note``         — math-token clustering identical to the runtime
  detector in ``serve/orchestrator`` (``_MATH_CHARS_RE``).

Reports, per chapter and overall:

* ops emitted per clause  (mean, p50, p95, max)
* per-primitive histogram
* dedup rate (operations whose ``(label, latex)`` is suppressed because
  it has already been emitted earlier in the same narration — mirrors
  the orchestrator's per-session ``seen_ops`` set)

This is purely offline; the orchestrator is not invoked.  Output:
``bench/eval/results/m3_viz_ops.json``.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))                 # repo root
sys.path.insert(0, str(THIS.parents[1]))                 # bench/eval/

from bench.eval._common import RESULTS, write_json, now_iso  # noqa: E402

from viz.operations import find_operations                       # noqa: E402

CHAPTER_GLOB = "ESLII.chapter_map.b_ch*.json"
BOOKS = THIS.parents[3] / "books"


# Mirror narrator/serve regexes verbatim so the harness measures the
# real production detectors.
_REF_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Algorithm",   re.compile(r"\bAlgorithm\s+(\d+(?:\.\d+){0,2})\b")),
    ("Figure",      re.compile(r"\bFigure\s+(\d+(?:\.\d+){0,2})\b")),
    ("Table",       re.compile(r"\bTable\s+(\d+(?:\.\d+){0,2})\b")),
    ("Theorem",     re.compile(r"\bTheorem\s+(\d+(?:\.\d+){0,2})\b")),
    ("Lemma",       re.compile(r"\bLemma\s+(\d+(?:\.\d+){0,2})\b")),
    ("Proposition", re.compile(r"\bProposition\s+(\d+(?:\.\d+){0,2})\b")),
    ("Corollary",   re.compile(r"\bCorollary\s+(\d+(?:\.\d+){0,2})\b")),
    ("Definition",  re.compile(r"\bDefinition\s+(\d+(?:\.\d+){0,2})\b")),
    ("Example",     re.compile(r"\bExample\s+(\d+(?:\.\d+){0,2})\b")),
    ("Exercise",    re.compile(r"\bExercise\s+(\d+(?:\.\d+){0,2})\b")),
    ("Section",     re.compile(r"\bSection\s+(\d+(?:\.\d+){0,2})\b")),
    ("Chapter",     re.compile(r"\bChapter\s+(\d+)\b")),
    ("Equation",    re.compile(
        r"\b(?:[Ee]quations?|Eqs?\.?)\s*\(?(\d+\.\d+)\)?")),
    ("Equation",    re.compile(r"\((\d+\.\d+)\)")),
]

_MATH_CHARS_RE = re.compile("[α-ωΑ-Ω=+·×÷≈≠≤≥∑∏∫∂∇√∞±^]")
_GREEK_RE = re.compile("[α-ωΑ-Ω]")
_TOKEN_RE = re.compile(r"\S+")
_PROSE_STOP = frozenset((
    "the of is applied where a an and or for with from to that this "
    "we it as in on by be is are was were such has have had given "
    "thus then so but if then otherwise also when may can will use uses "
    "let not no over under without"
).split())


def _split_clauses(prose: str) -> list[str]:
    """Sentence-grain split mirroring narrator's clause boundary."""
    if not prose:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])|\n\s*\n+", prose)
    return [p.strip() for p in parts if p and p.strip()]


def _detect_inline_math(text: str) -> bool:
    """Replicate the inline-math acceptance test in the paper.

    Cluster math-character-bearing tokens within a 2-token gap; accept a
    cluster iff [contains '='] OR [|cluster| >= 3] OR [hasGreek AND |cluster| >= 2].
    """
    spans = [(m.start(), m.end(), m.group(0))
             for m in _TOKEN_RE.finditer(text)]
    if not spans:
        return False
    flagged = [bool(_MATH_CHARS_RE.search(t)) for _s, _e, t in spans]
    n = len(spans)
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        # Grow a run including any flagged token within 2-token gap.
        j = i
        run = [spans[i][2]]
        while j + 1 < n:
            # Search ahead up to 2 unflagged tokens for the next flagged.
            k = j + 1
            while k < n and k - j <= 3:
                if flagged[k]:
                    break
                k += 1
            if k < n and flagged[k] and k - j <= 3:
                # Trim filler tokens from the prose stop list.
                run.append(spans[k][2])
                j = k
            else:
                break
        text_run = " ".join(run)
        size = len(run)
        contains_eq = "=" in text_run
        has_greek = bool(_GREEK_RE.search(text_run))
        accept = contains_eq or size >= 3 or (has_greek and size >= 2)
        if accept:
            return True
        i = j + 1
    return False


def _detect_references(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, pat in _REF_PATTERNS:
        for m in pat.finditer(text):
            label = m.group(1)
            key = (kind, label)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _walk_narration(root: dict):
    yield root
    for c in root.get("children", []) or []:
        yield from _walk_narration(c)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _process_chapter(chapter_json: Path) -> dict:
    d = json.load(chapter_json.open())
    seen_ops: set[tuple[str, str]] = set()
    seen_refs: set[tuple[str, str]] = set()
    n_clauses = 0
    n_inline_math = 0
    ops_per_clause: list[int] = []
    refs_per_clause: list[int] = []
    op_hist: Counter[str] = Counter()
    ref_hist: Counter[str] = Counter()
    op_dedup_hits = 0
    ref_dedup_hits = 0

    for node in _walk_narration(d.get("root", {})):
        prose = (node.get("story_paragraph") or "").strip()
        for clause in _split_clauses(prose):
            n_clauses += 1
            ops = find_operations(clause)
            unique_ops = []
            for label, latex, _s, _e in ops:
                key = (label, latex)
                if key in seen_ops:
                    op_dedup_hits += 1
                    continue
                seen_ops.add(key)
                unique_ops.append(key)
                op_hist[label] += 1
            ops_per_clause.append(len(unique_ops))

            refs = _detect_references(clause)
            unique_refs = []
            for kind, label in refs:
                if (kind, label) in seen_refs:
                    ref_dedup_hits += 1
                    continue
                seen_refs.add((kind, label))
                unique_refs.append((kind, label))
                ref_hist[kind] += 1
            refs_per_clause.append(len(unique_refs))

            if _detect_inline_math(clause):
                n_inline_math += 1

    chapter_label = d.get("root", {}).get("title") or chapter_json.stem
    return {
        "file": chapter_json.name,
        "chapter": chapter_label,
        "n_clauses": n_clauses,
        "n_clauses_with_inline_math": n_inline_math,
        "inline_math_rate": (n_inline_math / n_clauses) if n_clauses else 0.0,
        "ops_per_clause_mean":  (sum(ops_per_clause) / n_clauses)  if n_clauses else 0.0,
        "ops_per_clause_p50":   median(ops_per_clause) if ops_per_clause else 0.0,
        "ops_per_clause_p95":   _quantile(ops_per_clause, 0.95),
        "ops_per_clause_max":   max(ops_per_clause) if ops_per_clause else 0,
        "refs_per_clause_mean": (sum(refs_per_clause) / n_clauses) if n_clauses else 0.0,
        "refs_per_clause_max":  max(refs_per_clause) if refs_per_clause else 0,
        "op_dedup_hits":  op_dedup_hits,
        "ref_dedup_hits": ref_dedup_hits,
        "op_dedup_rate":  (op_dedup_hits  / max(1, op_dedup_hits  + len(seen_ops))),
        "ref_dedup_rate": (ref_dedup_hits / max(1, ref_dedup_hits + len(seen_refs))),
        "op_histogram":  dict(op_hist.most_common()),
        "ref_histogram": dict(ref_hist.most_common()),
        "n_unique_ops":  len(seen_ops),
        "n_unique_refs": len(seen_refs),
    }


def main() -> None:
    chapters = sorted(BOOKS.glob(CHAPTER_GLOB))
    if not chapters:
        raise SystemExit(f"no chapter sidecars at {BOOKS / CHAPTER_GLOB}")

    t0 = time.perf_counter()
    per_chapter = [_process_chapter(p) for p in chapters]
    elapsed_s = time.perf_counter() - t0

    # Aggregate.
    all_ops: list[int] = []
    all_refs: list[int] = []
    all_op_hist: Counter[str] = Counter()
    all_ref_hist: Counter[str] = Counter()
    n_clauses_total = 0
    n_inline_math_total = 0
    op_dedup = 0
    ref_dedup = 0
    seen_ops_global: set[str] = set()
    seen_refs_global: set[str] = set()
    for c in per_chapter:
        n_clauses_total += c["n_clauses"]
        n_inline_math_total += c["n_clauses_with_inline_math"]
        op_dedup += c["op_dedup_hits"]
        ref_dedup += c["ref_dedup_hits"]
        for k, v in c["op_histogram"].items():
            all_op_hist[k] += v
            seen_ops_global.add(k)
        for k, v in c["ref_histogram"].items():
            all_ref_hist[k] += v
            seen_refs_global.add(k)
        # Reconstruct distribution from chapter's mean × n_clauses isn't
        # exact — but we already retain the per-chapter p50/p95.  For an
        # overall distribution we re-tabulate from the mean × clause counts:
        # not statistically rigorous, so we publish per-chapter dists too.

    overall = {
        "n_chapters": len(per_chapter),
        "n_clauses_total": n_clauses_total,
        "n_clauses_with_inline_math": n_inline_math_total,
        "inline_math_rate": (n_inline_math_total / n_clauses_total)
                            if n_clauses_total else 0.0,
        "n_unique_ops_global":  sum(all_op_hist.values()),
        "n_unique_refs_global": sum(all_ref_hist.values()),
        "op_dedup_total":  op_dedup,
        "ref_dedup_total": ref_dedup,
        "ops_per_clause_mean_overall":  (sum(c["ops_per_clause_mean"]  * c["n_clauses"] for c in per_chapter) / n_clauses_total)  if n_clauses_total else 0.0,
        "refs_per_clause_mean_overall": (sum(c["refs_per_clause_mean"] * c["n_clauses"] for c in per_chapter) / n_clauses_total) if n_clauses_total else 0.0,
        "op_histogram_overall":  dict(all_op_hist.most_common()),
        "ref_histogram_overall": dict(all_ref_hist.most_common()),
    }

    payload = {
        "metric": "M3_visual_op_profile",
        "ts": now_iso(),
        "wall_seconds": elapsed_s,
        "overall": overall,
        "per_chapter": per_chapter,
        "service_calls": 0,
        "api_cost_usd": 0.0,
    }
    out = RESULTS / "m3_viz_ops.json"
    write_json(out, payload)
    o = overall
    print(f"[m3] wrote {out.relative_to(THIS.parents[3])}: "
          f"chapters={o['n_chapters']}  clauses={o['n_clauses_total']}  "
          f"ops/clause={o['ops_per_clause_mean_overall']:.2f}  "
          f"refs/clause={o['refs_per_clause_mean_overall']:.2f}  "
          f"inline-math={o['inline_math_rate']*100:.1f}%  "
          f"op-dedup={o['op_dedup_total']}  ref-dedup={o['ref_dedup_total']}")


if __name__ == "__main__":
    main()
