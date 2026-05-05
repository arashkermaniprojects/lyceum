"""M5 — Inline-math detector vs graph-anchor labels.

The math semantic graph carries an ``about`` edge from each passage to
the formulas / variables it mentions.  We treat:

* **positive** = passage has ≥1 ``about`` edge to a formula or variable;
* **negative** = passage has zero ``about`` edges to formula / variable
  / concept *and* zero ``contains`` mentions in the formula sub-graph.

We then run the runtime inline-math detector on each passage's clauses
and report agreement with the graph-anchor labels.

This is **not** ground-truth precision/recall — the graph anchor itself
has finite recall (e.g.\\ over PDF-OCR ligatures).  We frame the metric
as agreement vs an automated weak label, with hand-spot-check on a
random sample of the disagreement set.

Output: ``bench/eval/results/m5_inline_math.json``.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))                 # repo root
sys.path.insert(0, str(THIS.parents[1]))                 # bench/eval/

from bench.eval._common import RESULTS, write_json, now_iso  # noqa: E402

# Reuse the harness's exact clone of the runtime detector.
from bench.eval.drivers.m3_viz_ops import (                       # noqa: E402
    _detect_inline_math, _split_clauses,
)

GRAPH = THIS.parents[3] / "books" / "ESLII.math_graph.json"

SAMPLE_PER_CLASS = 500
RANDOM_SEED = 42


def _detect_passage(text: str) -> bool:
    """A passage flags math iff any of its clauses do."""
    for c in _split_clauses(text):
        if _detect_inline_math(c):
            return True
    return False


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    g = json.load(GRAPH.open())
    passages = g["passages"]
    formulas = set(g["formulas"].keys())
    vars_set = set(g["vars"].keys())

    # Build per-passage anchor counts.
    anchor_count: defaultdict[str, int] = defaultdict(int)
    for e in g["edges"]:
        if e.get("type") != "about":
            continue
        src = e.get("src", "")
        dst = e.get("dst", "")
        if not src.startswith("p:"):
            continue
        if dst in formulas or dst in vars_set:
            anchor_count[src] += 1

    # Split passages by anchor count.
    pos = [pid for pid, p in passages.items()
           if anchor_count[pid] > 0 and len(p.get("text", "")) >= 30]
    neg = [pid for pid, p in passages.items()
           if anchor_count[pid] == 0 and len(p.get("text", "")) >= 30]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_sample = pos[:SAMPLE_PER_CLASS]
    neg_sample = neg[:SAMPLE_PER_CLASS]

    t0 = time.perf_counter()
    pos_hits = []
    pos_miss = []
    for pid in pos_sample:
        text = passages[pid]["text"]
        flagged = _detect_passage(text)
        (pos_hits if flagged else pos_miss).append(pid)

    neg_hits = []   # detector says math, anchor says no math
    neg_correct = []
    for pid in neg_sample:
        text = passages[pid]["text"]
        flagged = _detect_passage(text)
        (neg_hits if flagged else neg_correct).append(pid)
    elapsed_s = time.perf_counter() - t0

    n_pos = len(pos_sample)
    n_neg = len(neg_sample)
    tp = len(pos_hits)
    fn = len(pos_miss)
    fp = len(neg_hits)
    tn = len(neg_correct)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # Sample disagreements for the appendix / hand-spot-check.
    rng2 = random.Random(RANDOM_SEED + 1)
    fp_sample = rng2.sample(neg_hits,  min(10, len(neg_hits)))
    fn_sample = rng2.sample(pos_miss,  min(10, len(pos_miss)))

    def _row(pid: str) -> dict:
        return {"id": pid,
                "home_nid": passages[pid].get("home_nid", ""),
                "anchors": anchor_count[pid],
                "text": (passages[pid]["text"] or "")[:200]}

    payload = {
        "metric": "M5_inline_math_vs_anchor",
        "ts": now_iso(),
        "graph_file": "books/ESLII.math_graph.json",
        "n_positives_total": len(pos),
        "n_negatives_total": len(neg),
        "sample_per_class": SAMPLE_PER_CLASS,
        "wall_seconds": elapsed_s,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "false_positive_sample": [_row(pid) for pid in fp_sample],
        "false_negative_sample": [_row(pid) for pid in fn_sample],
        "service_calls": 0,
        "api_cost_usd": 0.0,
    }
    out = RESULTS / "m5_inline_math.json"
    write_json(out, payload)
    print(f"[m5] wrote {out.relative_to(THIS.parents[3])}: "
          f"sample={SAMPLE_PER_CLASS}/class  "
          f"precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  acc={acc:.3f}")


if __name__ == "__main__":
    main()
