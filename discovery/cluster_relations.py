"""Cluster relation phrases from triples.jsonl by Qwen embedding similarity.

Output: discovery/ontology_candidate.md — Markdown table of clusters
(representative phrase, occurrence count, members) sorted by frequency.
Ready for human review and visual-pattern assignment.

Usage (via venv python with torch + transformers):
    "$VENV_PY" discovery/cluster_relations.py --tau 0.82
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from svggpt.embed import encode, is_available  # noqa: E402

TRIPLES_PATH = Path(__file__).parent / "triples.jsonl"
OUT_PATH = Path(__file__).parent / "ontology_candidate.md"


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return 0.0 if na == 0 or nb == 0 else dot / math.sqrt(na * nb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.82,
                    help="cosine threshold for merging a phrase into a cluster")
    ap.add_argument("--triples", type=Path, default=TRIPLES_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ns = ap.parse_args()

    if not is_available():
        print("Qwen encoder unavailable — run this with the venv python.",
              file=sys.stderr)
        return 1

    counts: collections.Counter[str] = collections.Counter()
    with open(ns.triples) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            r = t["r"].lower().strip()
            if r:
                counts[r] += 1

    unique = [r for r, _ in counts.most_common()]
    print(f"embedding {len(unique)} unique relation phrases ...", flush=True)

    embs: dict[str, tuple[float, ...]] = {}
    for i, r in enumerate(unique):
        if i % 25 == 0:
            print(f"  {i}/{len(unique)}", flush=True)
        embs[r] = encode(r)

    # Greedy single-link by best-cluster-mean similarity (deterministic: iterate
    # unique in descending count order, then alphabetic).
    clusters: list[dict] = []
    for r in unique:
        e = embs[r]
        if not e:
            continue
        best: dict | None = None
        best_sim = ns.tau
        for c in clusters:
            sim = cosine(e, c["centroid"])
            if sim > best_sim:
                best_sim = sim
                best = c
        if best is not None:
            best["members"].append(r)
            best["member_counts"].append(counts[r])
            # Recompute centroid as mean of member embeddings.
            d = len(e)
            mean = [0.0] * d
            for m in best["members"]:
                mv = embs[m]
                for j in range(d):
                    mean[j] += mv[j]
            n = len(best["members"])
            best["centroid"] = tuple(v / n for v in mean)
        else:
            clusters.append({
                "representative": r,
                "members": [r],
                "member_counts": [counts[r]],
                "centroid": e,
            })

    # Rank clusters by total occurrence.
    ranked = sorted(
        clusters,
        key=lambda c: (-sum(c["member_counts"]), c["representative"]),
    )

    with open(ns.out, "w") as f:
        f.write("# Data-driven candidate ontology\n\n")
        f.write(f"Source: `{ns.triples.name}` ({sum(counts.values())} triples, "
                f"{len(unique)} unique relation phrases).\n\n")
        f.write(f"Clustering: greedy single-link, cosine τ = {ns.tau}, "
                f"embeddings from Qwen2.5-7B mean-pool.\n\n")
        f.write("| # | total | representative | members |\n")
        f.write("|---|---|---|---|\n")
        for i, c in enumerate(ranked, 1):
            total = sum(c["member_counts"])
            members = ", ".join(
                f"{m} ({n})" for m, n in zip(c["members"], c["member_counts"])
            )
            f.write(f"| {i} | {total} | **{c['representative']}** | {members} |\n")

    print(f"wrote {ns.out} with {len(ranked)} clusters", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
