"""Pre-compute multi-level narrative explanations for each section of a
book, persisted to ``<book>.concepts.json``.

The runtime narrator picks one level (L0..L4) per session based on
user preference and prepends it to the first clause of each section,
so the learner hears the *story* before the formula appears.

Levels (Bruner's spiral curriculum × Lakoff conceptual metaphor):

    L0  one-line gist                ("regularization balances fit vs smoothness")
    L1  story + motivation           (5–8 sentences; opens with a phenomenon
                                      or question that the section answers)
    L2  story with formulas embedded (the same story, with the central
                                      equation woven in by name — NOT a
                                      symbol-by-symbol enumeration)
    L3  connections                  (what this generalizes / specializes /
                                      depends on, in prose)
    L4  worked numerical or geometric anchor

We generate ALL FIVE for every section, so the runtime can switch
levels on demand without any further LLM calls.

Usage::

    python -m tools.build_concept_layer books/ESLII.json \
        --out books/ESLII.concepts.json \
        --root b/ch5 --max-sections 10

The ``--root`` and ``--max-sections`` flags are for incremental builds —
running the full ESLII book produces ~250 sections × 5 levels = a long
LLM job; in practice you'll want to build chapter-by-chapter.
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

# Local-only path — Qwen via vLLM's OpenAI-compatible endpoint.  The
# user's locked rule is that no Anthropic / OpenAI calls go to the
# production path; this offline tool runs against the local Qwen.
LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"


@dataclass
class SectionConcept:
    """Five-tier narrative explanation of a single book section.

    Each level is a self-contained string; the runtime picks one level
    by user preference and prepends it to the section's first clause.
    """
    home_nid: str
    title: str
    L0_gist: str = ""
    L1_story: str = ""
    L2_with_formulas: str = ""
    L3_connections: str = ""
    L4_anchor: str = ""
    metaphor: str = ""
    prerequisites: list[str] = field(default_factory=list)
    key_formula_labels: list[str] = field(default_factory=list)


_SYSTEM_PROMPT = """You are an expert teacher of mathematics writing for an
adult learner who is curious but NOT a mathematician.  Your job is to
turn one section of a textbook into a layered narrative.

Hard rules:
  * Math is a LANGUAGE.  Teach it like one — vocabulary in context, not
    grammar drills.  Never enumerate variables one-by-one.  Weave the
    formula into the story.
  * Speak in plain English.  No raw LaTeX in L0–L3.  Use a formula's
    citation label ("Equation 5.42") when you need to refer to it.
  * Open with the WHY before the WHAT.  L0 is the punchline.  L1 sets
    up the question the section answers.  L2 lets the formula appear
    as the natural answer.

L2_with_formulas is special — it is a CONNECTED NARRATIVE, not bullet
points.  Treat it as a teacher walking the learner from the section's
opening question to its main result, naming each formula AT THE MOMENT
the narrative arrives at it.  Format:

  * 6–10 short sentences forming one continuous thread.
  * For every formula you reference, place an explicit anchor token
    of the form ``[FORMULA:N.M]`` where N.M is the citation number
    listed in the formulas list.  The anchor goes IMMEDIATELY after
    the sentence that introduces the formula's role.
  * Every formula anchor MUST be preceded by a "this-is-why" sentence
    naming the role of that formula in the unfolding story.
  * Every formula anchor SHOULD be followed by a sentence either
    interpreting the formula or transitioning to the next idea.
  * No bullet lists, no ASCII math; pure prose with anchors.

Example (do NOT echo this; produce one for the actual section):

  "Suppose we want to fit a smooth curve through noisy points.  We
  need a way to balance fit against wiggliness — that is the role of
  the regularization functional. [FORMULA:5.42] The lambda inside it
  is the dial we turn.  This raises a question: what shape of f
  actually minimises this?  The answer is the representer formula,
  which says f is finite-dimensional. [FORMULA:5.50] So even though
  the search space is infinite, the solution lives in a small,
  manageable subspace."

Output STRICT JSON only, in this exact shape:

{
  "L0_gist":           "single-sentence punchline",
  "L1_story":          "4–6 sentences, no formulas, opens with a question or phenomenon",
  "L2_with_formulas":  "6–10 sentences with [FORMULA:N.M] anchors, each preceded by a role sentence",
  "L3_connections":    "2–3 sentences relating to other equations in the book",
  "L4_anchor":         "one concrete worked example (numerical or geometric)",
  "metaphor":          "one everyday-life analogy",
  "prerequisites":     ["concept-name-1", "concept-name-2"]
}
"""


def _build_user_prompt(*, title: str, body: str,
                       formulas: list[dict]) -> str:
    """Assemble the per-section user prompt for Qwen.

    Includes the section's body text and a compact list of its
    formulas (so the LLM can refer to them by citation label without
    needing to re-extract).
    """
    formulas_repr = "\n".join(
        f"  ({f.get('cite_label','?')}) {f.get('latex','')[:160]}"
        for f in formulas[:8]
    )
    body_clip = body[:3000]   # vLLM context budget; sections beyond
                              # that are rare and we only need motivation.
    return (
        f"SECTION TITLE: {title}\n\n"
        f"FORMULAS IN THIS SECTION:\n{formulas_repr or '  (none)'}\n\n"
        f"BODY TEXT (truncated):\n{body_clip}\n\n"
        f"Produce the JSON now."
    )


def _call_llm(system: str, user: str, *,
              max_tokens: int = 1200,
              temperature: float = 0.4,
              retries: int = 2) -> Optional[dict]:
    """POST to the local Qwen endpoint, parse the assistant message as
    JSON.  Returns None on hard failure (caller skips the section).
    """
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read())
            content = (raw["choices"][0]["message"]
                       .get("content") or "").strip()
            if not content:
                return None
            # vLLM with response_format=json_object returns clean JSON.
            return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    print(f"  [llm] failed after {retries + 1} tries: {last_err}",
          file=sys.stderr)
    return None


def _section_formulas(book, math_graph, home_nid: str) -> list[dict]:
    """Return the formulas associated with *home_nid* in the math
    graph, as plain dicts: ``[{"cite_label": "5.42", "latex": "…"}, …]``.
    """
    out: list[dict] = []
    if math_graph is None:
        return out
    for f in math_graph.formulas.values():
        if f.home_nid == home_nid and f.latex:
            out.append({
                "cite_label": (f.cite_labels[0] if f.cite_labels else ""),
                "latex": f.latex,
            })
    return out


def _walk_sections(node, depth: int = 0,
                   max_depth: int = 4) -> list:
    """Yield (depth, BookNode) for every node that has body text or
    children, up to ``max_depth`` levels deep."""
    out = [(depth, node)]
    if depth >= max_depth:
        return out
    for c in node.children or []:
        out.extend(_walk_sections(c, depth + 1, max_depth))
    return out


def build(book_path: str, out_path: str, *,
          root_nid: str = "",
          max_sections: int = 0,
          force: bool = False) -> dict:
    """Extract per-section concepts and write them to *out_path*.

    Resumable: if *out_path* exists and ``force`` is False, sections
    already present are skipped.  This makes it safe to interrupt and
    resume long builds.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from book.corpus import load_corpus
    from sevim.math_graph import MathGraph

    book = load_corpus(book_path)
    mg = None
    mg_path = book_path.replace(".json", ".math_graph.json")
    if os.path.exists(mg_path):
        try:
            mg = MathGraph.load(mg_path)
        except Exception as e:
            print(f"[warn] could not load math graph: {e}",
                  file=sys.stderr)

    existing: dict[str, dict] = {}
    if os.path.exists(out_path) and not force:
        try:
            with open(out_path, "r") as f:
                existing = (json.load(f) or {}).get("by_home_nid", {})
        except Exception:
            existing = {}

    target = book.find(root_nid) if root_nid else book.root
    if target is None:
        print(f"[error] root_nid {root_nid!r} not found", file=sys.stderr)
        return {}

    sections = [
        (d, n) for d, n in _walk_sections(target)
        if (n.body_text or "").strip()
    ]
    print(f"[concept-layer] {len(sections)} sections under "
          f"{target.nid!r}; existing={len(existing)}")
    out_concepts: dict[str, dict] = dict(existing)
    n_new = 0
    for i, (depth, node) in enumerate(sections):
        if max_sections and n_new >= max_sections:
            break
        if node.nid in out_concepts:
            continue
        formulas = _section_formulas(book, mg, node.nid)
        title = node.title or node.nid
        prompt = _build_user_prompt(
            title=title,
            body=node.body_text,
            formulas=formulas,
        )
        print(f"  [{i+1}/{len(sections)}] {node.nid} :: "
              f"{title[:50]}  (formulas={len(formulas)})")
        result = _call_llm(_SYSTEM_PROMPT, prompt)
        if result is None:
            continue
        sc = SectionConcept(
            home_nid=node.nid,
            title=title,
            L0_gist=(result.get("L0_gist") or "").strip(),
            L1_story=(result.get("L1_story") or "").strip(),
            L2_with_formulas=(result.get("L2_with_formulas") or "").strip(),
            L3_connections=(result.get("L3_connections") or "").strip(),
            L4_anchor=(result.get("L4_anchor") or "").strip(),
            metaphor=(result.get("metaphor") or "").strip(),
            prerequisites=list(result.get("prerequisites") or []),
            key_formula_labels=[
                f["cite_label"] for f in formulas if f.get("cite_label")
            ],
        )
        out_concepts[node.nid] = asdict(sc)
        n_new += 1
        # Persist after every section so an interrupt doesn't lose work.
        with open(out_path, "w") as f:
            json.dump({"by_home_nid": out_concepts,
                       "book": book.title,
                       "model": LLM_MODEL}, f, indent=2)
    print(f"[concept-layer] wrote {n_new} new section explanations to {out_path}")
    return out_concepts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("book_json", help="path to <book>.json")
    ap.add_argument("--out", default="",
                    help="output path; default <book>.concepts.json")
    ap.add_argument("--root", default="",
                    help="restrict to this nid (e.g. b/ch5)")
    ap.add_argument("--max-sections", type=int, default=0,
                    help="cap # sections to process this run")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing entries")
    args = ap.parse_args()
    out = args.out or args.book_json.replace(".json", ".concepts.json")
    build(args.book_json, out,
          root_nid=args.root,
          max_sections=args.max_sections,
          force=args.force)


if __name__ == "__main__":
    main()
