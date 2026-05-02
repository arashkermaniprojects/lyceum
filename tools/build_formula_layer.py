"""Pre-compute multi-level explanations for every Formula in a book's
math graph, persisted to ``<book>.formulas.json``.

When a formula card lands on the chalkboard at runtime, the orchestrator
looks up its description here and speaks one of three levels:

    F0_role     "this is the regularization functional"
                — one-clause role label.  Used in default narration to
                  give the formula a name as it appears.

    F1_meaning  "minimize the sum of prediction errors plus a
                 smoothness penalty controlled by lambda"
                — single-sentence plain-English meaning.  No symbol-by-
                  symbol enumeration.  Reads like a teacher pointing
                  at the board.

    F2_walk     "the left side `min_f ∑ L(yᵢ, f(xᵢ)) + λ J(f)` says:
                 over all candidate functions f, minimize this sum;
                 the first term is the data-fit; the second term is
                 a smoothness penalty; lambda balances the two."
                — multi-sentence walk for users who tap "+" to dive in.

Mirrors ``tools/build_concept_layer.py`` exactly — same
local-Qwen-via-vLLM pipeline, same resumable on-disk format, same
prompt-engineering rules ("math is a language, not grammar").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional


LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"


@dataclass
class FormulaExplanation:
    formula_id: str
    cite_label: str = ""
    latex: str = ""
    home_nid: str = ""
    F0_role: str = ""
    F1_meaning: str = ""
    F2_walk: str = ""


_SYSTEM_PROMPT = """You are an expert teacher of mathematics writing for an
adult learner who is curious but not a mathematician.  You receive ONE
formula at a time, with a small amount of surrounding context.  Your job
is to produce three nested explanations of that formula at increasing
depth.

Hard rules:
  * Math is a LANGUAGE.  Speak it like one.  Do NOT enumerate variables
    one-by-one.  Do NOT recite indices.  Tell the reader what the
    formula DOES, not how it is parsed.
  * F0_role is a LABEL.  One short clause naming what kind of object
    this formula is ("regularization functional", "kernel
    eigen-expansion", "representer of evaluation").  No verb.
  * F1_meaning is one English sentence describing what the formula
    SAYS, in the language the rest of the textbook uses.  Mention the
    role of lambda, sums, integrals, etc. by their function, not their
    notation.
  * F2_walk is 2–4 short sentences expanding F1.  May reference the
    pieces of the formula but each piece must come with its meaning,
    not just its name.

Output STRICT JSON only:

{
  "F0_role":    "single short noun phrase, no period",
  "F1_meaning": "one full sentence, plain English",
  "F2_walk":    "2–4 sentences, plain English with pieces named-and-explained"
}
"""


def _build_user_prompt(*, latex: str, cite_label: str,
                       section_title: str,
                       section_gist: str) -> str:
    """Assemble the per-formula user prompt.  Includes section title +
    L0 gist so the LLM has enough context to choose the right role.
    """
    cite = f"({cite_label})" if cite_label else ""
    return (
        f"FORMULA{(' ' + cite) if cite else ''}:\n"
        f"  {latex}\n\n"
        f"FROM SECTION: {section_title}\n"
        f"SECTION GIST: {section_gist or '(unavailable)'}\n\n"
        f"Produce the JSON now."
    )


def _call_llm(system: str, user: str, *,
              max_tokens: int = 600,
              temperature: float = 0.4,
              retries: int = 2) -> Optional[dict]:
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode()
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                LLM_URL, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read())
            content = (raw["choices"][0]["message"]
                       .get("content") or "").strip()
            if not content:
                return None
            return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    print(f"  [llm] failed after {retries + 1} tries: {last_err}",
          file=sys.stderr)
    return None


def build(book_path: str, out_path: str, *,
          section_concepts_path: str = "",
          home_nid: str = "",
          max_formulas: int = 0,
          force: bool = False) -> dict:
    """Generate F0/F1/F2 for every formula in the book's math graph.

    Resumable: existing entries skipped unless ``force``.
    ``--home-nid`` restricts to one section's formulas (use for chapter-
    by-chapter builds).
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from sevim.math_graph import MathGraph

    mg_path = book_path.replace(".json", ".math_graph.json")
    if not os.path.exists(mg_path):
        print(f"[error] no math graph at {mg_path}", file=sys.stderr)
        return {}
    mg = MathGraph.load(mg_path)
    print(f"[formula-layer] loaded {len(mg.formulas)} formulas "
          f"from {mg_path}")

    # Section concepts give us a per-section gist that grounds each
    # formula's role.  Optional but improves quality a lot.
    section_concepts: dict[str, dict] = {}
    if not section_concepts_path:
        section_concepts_path = book_path.replace(".json", ".concepts.json")
    if os.path.exists(section_concepts_path):
        try:
            with open(section_concepts_path) as f:
                section_concepts = (json.load(f) or {}).get(
                    "by_home_nid", {})
            print(f"[formula-layer] loaded {len(section_concepts)} "
                  f"section concepts for grounding")
        except Exception as e:
            print(f"[warn] could not load section concepts: {e}",
                  file=sys.stderr)

    existing: dict[str, dict] = {}
    if os.path.exists(out_path) and not force:
        try:
            with open(out_path) as f:
                existing = (json.load(f) or {}).get("by_formula_id", {})
        except Exception:
            existing = {}

    out: dict[str, dict] = dict(existing)
    n_new = 0
    formulas = list(mg.formulas.values())
    if home_nid:
        formulas = [f for f in formulas if f.home_nid == home_nid]
    for i, f in enumerate(formulas):
        if max_formulas and n_new >= max_formulas:
            break
        if not f.latex or not f.latex.strip():
            continue
        if f.id in out:
            continue
        cite = f.cite_labels[0] if f.cite_labels else ""
        sect = section_concepts.get(f.home_nid, {})
        prompt = _build_user_prompt(
            latex=f.latex,
            cite_label=cite,
            section_title=sect.get("title", f.home_nid),
            section_gist=sect.get("L0_gist", ""),
        )
        result = _call_llm(_SYSTEM_PROMPT, prompt)
        if result is None:
            continue
        # Qwen sometimes emits the multi-sentence F2_walk field as a
        # JSON array of strings.  Tolerate both shapes.
        def _coerce(v) -> str:
            if isinstance(v, list):
                return " ".join(str(s) for s in v).strip()
            return (str(v) if v is not None else "").strip()
        fe = FormulaExplanation(
            formula_id=f.id,
            cite_label=cite,
            latex=f.latex,
            home_nid=f.home_nid,
            F0_role=_coerce(result.get("F0_role")),
            F1_meaning=_coerce(result.get("F1_meaning")),
            F2_walk=_coerce(result.get("F2_walk")),
        )
        out[f.id] = asdict(fe)
        n_new += 1
        if n_new % 10 == 0:
            print(f"  [{n_new}] {f.id} {cite}: "
                  f"{fe.F0_role[:60]}")
        # Persist after every 5 to keep interrupt loss small.
        if n_new % 5 == 0:
            with open(out_path, "w") as fh:
                json.dump({"by_formula_id": out,
                           "model": LLM_MODEL}, fh, indent=2)
    with open(out_path, "w") as fh:
        json.dump({"by_formula_id": out,
                   "model": LLM_MODEL}, fh, indent=2)
    print(f"[formula-layer] wrote {n_new} new explanations "
          f"({len(out)} total) to {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("book_json")
    ap.add_argument("--out", default="")
    ap.add_argument("--concepts", default="",
                    help="path to <book>.concepts.json (auto-detected)")
    ap.add_argument("--home-nid", default="")
    ap.add_argument("--max-formulas", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.out or args.book_json.replace(".json", ".formulas.json")
    build(args.book_json, out,
          section_concepts_path=args.concepts,
          home_nid=args.home_nid,
          max_formulas=args.max_formulas,
          force=args.force)


if __name__ == "__main__":
    main()
