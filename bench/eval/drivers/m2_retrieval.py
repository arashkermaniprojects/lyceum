"""M2 — Retrieval ranking.

For every section/subsection in ``books/ESLII.json`` whose title is
distinctive (≥2 alphabetic tokens after stop-word removal, unique
within ESLII), construct the natural query ``"explain <title>"`` with
gold = that node's nid.  Score:

* **Lexical**: pure BM25 (``narrator.planner._bm25_scores``).
* **Lexical+title-boost**: BM25 with the multiplicative title-overlap
  bonus (Eq.~\\ref{eq:title} in the paper), which is the system's
  runtime default before RRF fusion.
* **Hybrid (BM25 + dense, RRF, K=60)**: requires a live Qwen3 embedding
  endpoint to embed the *query* (book-side embeddings are pre-stored).
  When the endpoint is unreachable the hybrid columns are written as
  ``null`` and the table generator renders ``--``.

Reports MRR, Recall@1, Recall@5, Recall@10 over the section-targeted
query set.  Output: ``bench/eval/results/m2_retrieval.json``.

The implementation pre-tokenises the corpus once and caches
per-document term frequencies, so a query against ~350 candidates with
O(|q|) terms runs in microseconds.
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))                 # repo root
sys.path.insert(0, str(THIS.parents[1]))                 # bench/eval/

from bench.eval._common import RESULTS, write_json, now_iso  # noqa: E402

from book.corpus import load_corpus                              # noqa: E402
from book import embeddings as _emb                              # noqa: E402
from narrator.planner import _tokenize, _STOP                     # noqa: E402

CORPUS = THIS.parents[3] / "books" / "ESLII.json"

K1 = 1.5
B  = 0.75
RRF_K = 60.0


def _candidate_section_nodes(book) -> list:
    section_kinds = {"section", "subsection", "subsubsection"}
    return [n for n in book.root.walk()
            if n.kind in section_kinds and (n.body_text or "").strip()]


def _meaningful_title(title: str) -> bool:
    toks = [t for t in _tokenize(title) if t not in _STOP]
    if len(toks) < 2:
        return False
    if {"introduction", "summary", "discussion", "appendix",
        "preface", "preliminaries", "background"} >= set(toks):
        return False
    return True


def _build_queries(book) -> list[dict]:
    sections = _candidate_section_nodes(book)
    title_count: Counter[str] = Counter()
    for n in sections:
        title_count[n.title.strip().lower()] += 1
    out = []
    for n in sections:
        if not _meaningful_title(n.title):
            continue
        if title_count[n.title.strip().lower()] > 1:
            continue
        out.append({
            "query": f"explain {n.title}",
            "gold_nid": n.nid,
            "gold_title": n.title,
            "gold_kind": n.kind,
        })
    return out


# ---------------------------------------------------------------------------
# Pre-computed BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """Pre-computed BM25 over a fixed corpus of token lists.

    Mirrors ``narrator.planner._bm25_scores`` mathematically but caches
    document length, per-doc term frequencies, and document frequency.
    """
    def __init__(self, docs: list[list[str]]):
        self.n = len(docs)
        self.dls = [len(d) for d in docs]
        self.avg_dl = (sum(self.dls) / self.n) if self.n else 0.0
        self.tfs: list[Counter[str]] = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.df = df
        self.idf: dict[str, float] = {
            t: math.log((self.n - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
            for t, n_qi in df.items()
        }

    def score_all(self, q_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for term in q_tokens:
            if term in _STOP:
                continue
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if not f:
                    continue
                dl = self.dls[i]
                denom = f + K1 * (1 - B + B * dl / max(self.avg_dl, 1.0))
                scores[i] += idf * (f * (K1 + 1)) / max(denom, 1e-9)
        return scores


def _ranked(scores: list[float], cands) -> list[tuple[float, int]]:
    return sorted(
        ((sc, i) for i, sc in enumerate(scores)),
        key=lambda sc_i: (-sc_i[0], cands[sc_i[1]].page_start,
                           cands[sc_i[1]].nid),
    )


def _rrf(rank_lists: list[dict[int, int]]) -> list[tuple[float, int]]:
    fused: dict[int, float] = {}
    keys: set[int] = set()
    for r in rank_lists:
        keys |= set(r.keys())
    for idx in keys:
        s = 0.0
        for r in rank_lists:
            if idx in r:
                s += 1.0 / (RRF_K + r[idx])
        fused[idx] = s
    return sorted(((s, i) for i, s in fused.items()),
                  key=lambda s_i: -s_i[0])


def _score(rank_of_gold: int) -> dict:
    return {
        "rank": rank_of_gold,
        "rr": (1.0 / (rank_of_gold + 1)) if rank_of_gold >= 0 else 0.0,
        "hit1":  1 if rank_of_gold == 0 else 0,
        "hit5":  1 if 0 <= rank_of_gold < 5 else 0,
        "hit10": 1 if 0 <= rank_of_gold < 10 else 0,
    }


def _summarise(per_query: list[dict]) -> dict:
    n = len(per_query)
    if not n:
        return {"n": 0, "mrr": 0.0, "recall_at_1": 0.0,
                "recall_at_5": 0.0, "recall_at_10": 0.0}
    return {
        "n": n,
        "mrr":          sum(s["rr"]    for s in per_query) / n,
        "recall_at_1":  sum(s["hit1"]  for s in per_query) / n,
        "recall_at_5":  sum(s["hit5"]  for s in per_query) / n,
        "recall_at_10": sum(s["hit10"] for s in per_query) / n,
    }


def main() -> None:
    print("[m2] loading corpus…", flush=True)
    book = load_corpus(str(CORPUS))
    candidates = _candidate_section_nodes(book)
    nid_to_idx = {n.nid: i for i, n in enumerate(candidates)}
    queries = _build_queries(book)
    print(f"[m2] {len(candidates)} candidates, {len(queries)} queries",
          flush=True)

    print("[m2] building BM25 index over corpus…", flush=True)
    t0 = time.perf_counter()
    docs = [_tokenize(n.body_text) for n in candidates]
    bm = BM25Index(docs)
    titles = [set(t for t in _tokenize(n.title) if t not in _STOP)
              for n in candidates]
    cand_vecs = [tuple(n.meta.get("embedding", ())) for n in candidates]
    print(f"[m2] index built in {time.perf_counter() - t0:.1f} s", flush=True)

    base_url = os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8003/v1")
    dense_available = _emb.is_available(base_url=base_url)
    print(f"[m2] embedder @ {base_url}: "
          f"{'available' if dense_available else 'unavailable'}", flush=True)

    bm25_scores: list[dict] = []
    boost_scores: list[dict] = []
    hybrid_scores: list[dict] = []

    t0 = time.perf_counter()
    for q in queries:
        gold_idx = nid_to_idx.get(q["gold_nid"], -1)
        if gold_idx < 0:
            continue
        q_tokens = _tokenize(q["query"])

        # --- Pure BM25
        sc = bm.score_all(q_tokens)
        ranked = _ranked(sc, candidates)
        rank_g = next(
            (r for r, (s, i) in enumerate(ranked) if i == gold_idx and s > 0),
            -1,
        )
        bm25_scores.append({**q, **_score(rank_g)})

        # --- BM25 + title boost
        qset = set(t for t in q_tokens if t not in _STOP)
        sc_b = list(sc)
        for i, ttoks in enumerate(titles):
            if not ttoks:
                continue
            o = len(qset & ttoks)
            if o:
                sc_b[i] += 4.0 * o + 8.0 * (o / len(ttoks))
        ranked_b = _ranked(sc_b, candidates)
        rank_g_b = next(
            (r for r, (s, i) in enumerate(ranked_b) if i == gold_idx and s > 0),
            -1,
        )
        boost_scores.append({**q, **_score(rank_g_b)})

        # --- Hybrid RRF (BM25-with-boost + dense)
        if dense_available:
            qv = _emb.embed_text(q["query"], base_url=base_url)
            if qv:
                bm25_ranks = {i: r for r, (s, i) in enumerate(ranked_b) if s > 0}
                dpaired = _emb.ranked_by_cosine(qv, cand_vecs)
                dense_ranks = {i: r for r, (s, i) in enumerate(dpaired) if s > 0}
                fused = _rrf([bm25_ranks, dense_ranks])
                rank_g_h = next(
                    (r for r, (s, i) in enumerate(fused) if i == gold_idx),
                    -1,
                )
                hybrid_scores.append({**q, **_score(rank_g_h)})
    elapsed_s = time.perf_counter() - t0
    print(f"[m2] scored {len(queries)} queries in {elapsed_s:.2f} s",
          flush=True)

    payload = {
        "metric": "M2_retrieval_ranking",
        "ts": now_iso(),
        "corpus": "books/ESLII.json",
        "n_candidates": len(candidates),
        "n_queries": len(queries),
        "wall_seconds": elapsed_s,
        "embedder_endpoint": base_url,
        "embedder_available": dense_available,
        "results": {
            "bm25_only":       _summarise(bm25_scores),
            "bm25_titleboost": _summarise(boost_scores),
            "hybrid_rrf":      _summarise(hybrid_scores) if hybrid_scores
                                else {"skipped": "embedder_unavailable"},
        },
        "rows": {
            "bm25_only":       bm25_scores,
            "bm25_titleboost": boost_scores,
            "hybrid_rrf":      hybrid_scores,
        },
        "service_calls": len(queries) if dense_available else 0,
        "api_cost_usd": 0.0,
    }
    out = RESULTS / "m2_retrieval.json"
    write_json(out, payload)
    s = payload["results"]
    print(f"[m2] wrote {out.relative_to(THIS.parents[3])}: "
          f"queries={len(queries)}, "
          f"bm25 MRR={s['bm25_only']['mrr']:.3f}  R@1={s['bm25_only']['recall_at_1']:.3f}  "
          f"R@5={s['bm25_only']['recall_at_5']:.3f}; "
          f"+title MRR={s['bm25_titleboost']['mrr']:.3f}  "
          f"R@1={s['bm25_titleboost']['recall_at_1']:.3f}; "
          f"hybrid={'on' if dense_available else 'skipped'}")


if __name__ == "__main__":
    main()
