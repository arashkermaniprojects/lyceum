"""M4 — Math semantic-graph anchor coverage.

For each passage in ``books/ESLII.math_graph.json``, count the number of
graph anchors (formulas via ``defines``/``contains``, vars via ``uses``,
concepts via ``about``, derivation links via ``derived_from``) that
target it.  Reports overall coverage, per-chapter breakdown, and the
edge-type histogram.

Output: ``bench/eval/results/m4_graph_coverage.json``.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))                 # repo root
sys.path.insert(0, str(THIS.parents[1]))                 # bench/eval/

from bench.eval._common import RESULTS, write_json, now_iso  # noqa: E402

GRAPH = THIS.parents[3] / "books" / "ESLII.math_graph.json"

# Edge types whose presence indicates the passage carries math content
# anchored in the graph.  These match the 11 edge types listed in the
# paper's contributions section (sec:graph).
PASSAGE_ANCHOR_TYPES = {
    "defines", "contains", "uses", "about", "references",
    "derived_from", "paired_in_clause", "binds",
}


def _chapter_of_nid(home_nid: str) -> str:
    """``b/ch5/...`` → ``ch5``; non-chapter homes return ``other``."""
    if not home_nid:
        return "other"
    m = re.match(r"^b/(ch\d+)\b", home_nid)
    return m.group(1) if m else "other"


def main() -> None:
    t0 = time.perf_counter()
    g = json.load(GRAPH.open())

    passages = g["passages"]
    edges = g["edges"]

    # Index: passage id -> set of anchor edge types pointing at it.
    anchored_by: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for e in edges:
        et = e.get("type")
        if et not in PASSAGE_ANCHOR_TYPES:
            continue
        # Either src or dst can be a passage id.  Math-graph convention:
        # `paired_in_clause` is bidirectional, `defines` is concept->passage,
        # `contains` is passage->formula, etc.  Count whichever endpoint
        # is in the passages dict.
        for endpoint in (e.get("src"), e.get("dst")):
            if endpoint and endpoint in passages:
                anchored_by[endpoint][et] += 1

    # Coverage
    n_passages_total = len(passages)
    n_passages_anchored = sum(1 for pid in passages if anchored_by[pid])
    coverage = (n_passages_anchored / n_passages_total
                if n_passages_total else 0.0)

    # Per-chapter coverage.
    per_chapter_total: Counter[str] = Counter()
    per_chapter_anchored: Counter[str] = Counter()
    for pid, p in passages.items():
        ch = _chapter_of_nid(p.get("home_nid", ""))
        per_chapter_total[ch] += 1
        if anchored_by[pid]:
            per_chapter_anchored[ch] += 1

    per_chapter_rows = []
    for ch in sorted(per_chapter_total.keys(), key=lambda s: (
        # numeric chapter order; 'other' / 'intro_*' to the bottom
        9999 if not s.startswith("ch") else int(s[2:])
    )):
        n_tot = per_chapter_total[ch]
        n_anc = per_chapter_anchored[ch]
        per_chapter_rows.append({
            "chapter": ch,
            "n_passages": n_tot,
            "n_anchored": n_anc,
            "coverage": (n_anc / n_tot) if n_tot else 0.0,
        })

    # Per-edge-type counts.
    edge_type_hist = Counter(e["type"] for e in edges)

    # Anchor-density distribution (anchors per anchored passage).
    densities = [sum(anchored_by[pid].values()) for pid in passages
                 if anchored_by[pid]]
    densities.sort()
    if densities:
        mean_density = sum(densities) / len(densities)
        p50 = densities[len(densities) // 2]
        p95 = densities[max(0, int(len(densities) * 0.95) - 1)]
    else:
        mean_density = p50 = p95 = 0.0

    payload = {
        "metric": "M4_math_graph_coverage",
        "ts": now_iso(),
        "graph_file": "books/ESLII.math_graph.json",
        "wall_seconds": time.perf_counter() - t0,
        "totals": {
            "n_passages":  n_passages_total,
            "n_formulas":  len(g["formulas"]),
            "n_vars":      len(g["vars"]),
            "n_concepts":  len(g["concepts"]),
            "n_edges":     len(edges),
        },
        "coverage": {
            "n_passages_anchored":     n_passages_anchored,
            "fraction_passages_anchored": coverage,
            "anchor_density_mean": mean_density,
            "anchor_density_p50":  p50,
            "anchor_density_p95":  p95,
        },
        "per_chapter": per_chapter_rows,
        "edge_type_histogram": dict(edge_type_hist.most_common()),
        "service_calls": 0,
        "api_cost_usd": 0.0,
    }
    out = RESULTS / "m4_graph_coverage.json"
    write_json(out, payload)
    print(f"[m4] wrote {out.relative_to(THIS.parents[3])}: "
          f"passages={n_passages_total}  anchored={n_passages_anchored} "
          f"({100*coverage:.1f}%)  edges={len(edges)}  "
          f"density mean={mean_density:.2f} p95={p95}")


if __name__ == "__main__":
    main()
