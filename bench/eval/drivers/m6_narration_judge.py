"""M6 — Narration-quality LLM judge + objective ROUGE-L faithfulness proxy.

For up to ``N_SAMPLES`` (source passage, narration paragraph) pairs
sampled from the chapter sidecars + the corpus, we score:

* **subjective**: Claude Sonnet 4.6 as a single judge, fixed rubric
  (``rubrics/narration_rubric.txt``), temperature 0, JSON output, four
  1-5 scales (faithfulness / clarity / pedagogical_flow / tts_safety).
* **objective**: ROUGE-L F1 between narration and source as a
  lexical-faithfulness proxy.  Always reported, no service needed.

When ``ANTHROPIC_API_KEY`` is absent or the ``anthropic`` SDK isn't
installed we still write the objective ROUGE-L numbers and note the
subjective half as ``skipped``; this is the fallback the paper's
Methodology subsection cites.

Cost ledger: every Sonnet 4.6 call records prompt+completion tokens and
USD spend in ``bench/eval/cost_ledger.json``; the cap is 20 USD.

Output: ``bench/eval/results/m6_narration_judge.json``.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))                 # repo root
sys.path.insert(0, str(THIS.parents[1]))                 # bench/eval/

from bench.eval._common import (                              # noqa: E402
    RESULTS, RUBRICS, write_json, now_iso,
    have_anthropic_key, ledger_record,
)

from book.corpus import load_corpus                              # noqa: E402

CORPUS = THIS.parents[3] / "books" / "ESLII.json"
CHAPTER_GLOB = "ESLII.chapter_map.b_ch*.json"
BOOKS = THIS.parents[3] / "books"
RUBRIC = RUBRICS / "narration_rubric.txt"

N_SAMPLES = 30
RANDOM_SEED = 42

JUDGE_MODEL = "claude-sonnet-4-6"
# 2025 Sonnet 4.6 pricing: input USD 3 / 1M tok, output USD 15 / 1M tok.
USD_IN_PER_TOKEN  = 3.0  / 1_000_000
USD_OUT_PER_TOKEN = 15.0 / 1_000_000


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def _walk(node: dict):
    yield node
    for c in node.get("children", []) or []:
        yield from _walk(c)


def build_pairs() -> list[dict]:
    """Pair each chapter-map narration node with the corresponding section's
    source body_text (truncated)."""
    book = load_corpus(str(CORPUS))
    nid_to_node = {n.nid: n for n in book.root.walk()}

    pairs: list[dict] = []
    for cm in sorted(BOOKS.glob(CHAPTER_GLOB)):
        d = json.load(cm.open())
        for node in _walk(d.get("root", {})):
            prose = (node.get("story_paragraph") or "").strip()
            if len(prose) < 40:
                continue
            nid = node.get("nid", "")
            src_node = nid_to_node.get(nid)
            if src_node is None:
                continue
            src_text = (src_node.body_text or "").strip()
            if len(src_text) < 80:
                continue
            pairs.append({
                "nid": nid,
                "title": node.get("title", ""),
                "source_excerpt": src_text[:1400],
                "narration": prose,
                "chapter_file": cm.name,
            })
    return pairs


# ---------------------------------------------------------------------------
# ROUGE-L (objective faithfulness proxy)
# ---------------------------------------------------------------------------

_TOK = re.compile(r"[A-Za-z][A-Za-z\-]{1,}")


def _tok(s: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(s or "")]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    # Hirschberg-style row reduction — O(|a||b|) time, O(min(|a|,|b|)) memory.
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    for y in b:
        cur = [0] * (len(a) + 1)
        for i, x in enumerate(a, start=1):
            if x == y:
                cur[i] = prev[i - 1] + 1
            else:
                cur[i] = max(cur[i - 1], prev[i])
        prev = cur
    return prev[-1]


def rouge_l_f1(reference: str, hypothesis: str, beta: float = 1.0) -> float:
    a = _tok(reference)
    b = _tok(hypothesis)
    if not a or not b:
        return 0.0
    L = _lcs_len(a, b)
    if L == 0:
        return 0.0
    p = L / len(b)
    r = L / len(a)
    if p + r == 0:
        return 0.0
    f = ((1 + beta * beta) * p * r) / (beta * beta * p + r)
    return f


# ---------------------------------------------------------------------------
# Claude Sonnet 4.6 judge
# ---------------------------------------------------------------------------

def _judge(rubric_text: str, source: str, narration: str) -> dict:
    """Single Sonnet-4.6 judgement.  Records cost in the ledger."""
    import anthropic                          # imported lazily

    client = anthropic.Anthropic()
    user = (f"SOURCE:\n{source}\n\n"
            f"NARRATION:\n{narration}")
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        temperature=0.0,
        system=rubric_text,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    in_tok  = msg.usage.input_tokens
    out_tok = msg.usage.output_tokens
    usd = in_tok * USD_IN_PER_TOKEN + out_tok * USD_OUT_PER_TOKEN
    ledger_record({
        "metric": "M6_narration_judge",
        "model": JUDGE_MODEL,
        "in_tokens": in_tok, "out_tokens": out_tok,
        "usd": usd,
    })
    # Strict JSON parse with fallback regex extraction.
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^}]+\}", text, re.S)
        if not m:
            raise
        scores = json.loads(m.group(0))
    return {
        "faithfulness":     int(scores.get("faithfulness", 0)),
        "clarity":          int(scores.get("clarity", 0)),
        "pedagogical_flow": int(scores.get("pedagogical_flow", 0)),
        "tts_safety":       int(scores.get("tts_safety", 0)),
        "raw":              text,
        "in_tokens":        in_tok,
        "out_tokens":       out_tok,
        "usd":              usd,
    }


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    rubric_text = RUBRIC.read_text(encoding="utf-8")
    pairs = build_pairs()
    rng.shuffle(pairs)
    sample = pairs[:N_SAMPLES]

    have_key = have_anthropic_key()
    have_sdk = False
    try:
        import anthropic                                   # noqa: F401
        have_sdk = True
    except ImportError:
        pass
    judge_runnable = have_key and have_sdk

    t0 = time.perf_counter()
    rows: list[dict] = []
    judge_total_usd = 0.0
    for p in sample:
        rouge = rouge_l_f1(p["source_excerpt"], p["narration"])
        row = {
            "nid": p["nid"], "title": p["title"],
            "rouge_l_f1": rouge,
            "judge": None,
        }
        if judge_runnable:
            try:
                row["judge"] = _judge(rubric_text,
                                      p["source_excerpt"],
                                      p["narration"])
                judge_total_usd += row["judge"]["usd"]
            except Exception as e:
                row["judge"] = {"error": repr(e)}
        rows.append(row)
    elapsed_s = time.perf_counter() - t0

    # Aggregate
    rouge_vals = [r["rouge_l_f1"] for r in rows]
    rouge_mean = (sum(rouge_vals) / len(rouge_vals)) if rouge_vals else 0.0
    rouge_min  = min(rouge_vals) if rouge_vals else 0.0
    rouge_max  = max(rouge_vals) if rouge_vals else 0.0

    judge_means: dict[str, float] = {}
    judge_status = "ok" if judge_runnable else (
        "skipped_api_key_absent" if not have_key
        else "skipped_sdk_absent"
    )
    if judge_runnable:
        keys = ("faithfulness", "clarity", "pedagogical_flow", "tts_safety")
        valid = [r["judge"] for r in rows if isinstance(r.get("judge"), dict)
                 and "error" not in r["judge"]]
        for k in keys:
            judge_means[k] = (sum(j[k] for j in valid) / len(valid)
                              if valid else 0.0)
        judge_means["overall_mean"] = (sum(judge_means[k] for k in keys) / 4
                                        if valid else 0.0)
        judge_means["n_valid"] = len(valid)

    payload = {
        "metric": "M6_narration_judge",
        "ts": now_iso(),
        "n_pairs_total": len(pairs),
        "n_sampled": len(sample),
        "judge": {
            "model": JUDGE_MODEL,
            "rubric_path": "bench/eval/rubrics/narration_rubric.txt",
            "status": judge_status,
            "scores_mean": judge_means,
            "spend_usd": judge_total_usd,
        },
        "objective_proxy": {
            "metric": "rouge_l_f1",
            "n": len(rouge_vals),
            "mean": rouge_mean,
            "min":  rouge_min,
            "max":  rouge_max,
        },
        "rows": rows,
        "wall_seconds": elapsed_s,
    }
    out = RESULTS / "m6_narration_judge.json"
    write_json(out, payload)
    if judge_runnable:
        print(f"[m6] wrote {out.relative_to(THIS.parents[3])}: "
              f"n={len(sample)}  judge_overall_mean="
              f"{judge_means.get('overall_mean', 0.0):.2f}/5  "
              f"rouge-L F1 mean={rouge_mean:.3f}  "
              f"spend=${judge_total_usd:.4f}")
    else:
        print(f"[m6] wrote {out.relative_to(THIS.parents[3])}: "
              f"n={len(sample)}  rouge-L F1 mean={rouge_mean:.3f}  "
              f"judge status={judge_status} (set ANTHROPIC_API_KEY + "
              f"`pip install anthropic` to enable)")


if __name__ == "__main__":
    main()
