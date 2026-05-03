"""Generate per-node ``formula_explanation`` strings for a chapter-map sidecar.

The chapter-zoom narrator names each section's canonical equation
("this is the idea written down as equation five point one") but stops
short of explaining what the symbols mean.  This script fills that gap:
for every node whose narration emits the formula pin (own formula or
first inheritor of a label, mirroring ``plan_chapter_zoom``'s rule),
ask the local Qwen to write a 3-5 sentence plain-English walk-through
that names each parameter in the equation and ties it back to the
section's story.  Result is stored in the chapter-map sidecar JSON
under each node's ``formula_explanation`` key; the planner picks it
up at narration time.

Usage:

    python -m tools.build_formula_explanations \\
        books/ESLII.chapter_map.b_ch5.json

Idempotent: skips nodes that already have a non-empty
``formula_explanation``, so re-running fills only the gaps.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from typing import Optional


LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"


_SYSTEM_PROMPT_EQUATION = """You are explaining one mathematical equation to a
curious adult who has not done college math in years.  Output ONE plain-
English paragraph (3-5 sentences) tied to the section's story.

HARD RULES:

1. NAME EVERY SYMBOL.  Walk through the equation symbol by symbol.
   Each variable, coefficient, sum, or operator gets one short clause
   saying what it stands for in this context.  If the equation is
   "f(X) = sum from m=1 to M of beta_m * h_m(X)", you must name f(X),
   the sum, the index m, the upper limit M, the coefficient beta_m,
   and the basis function h_m(X) — every one of them.

2. PLAIN ENGLISH.  Spell out math: "the sum from m equals 1 to capital
   M" not "Σ".  "f of X" not "f(X)".  No raw LaTeX, no symbol glyphs.

3. STORY-FIRST.  Open with the role the equation plays in the
   section's narrative — what concept it captures — before any symbol
   is named.  The section's gist tells you what the story is about;
   the equation is the story written down precisely.

4. NO PADDING.  Don't say "this equation is important" or "let us
   consider".  Every sentence carries content.

OUTPUT FORMAT: JSON ``{"explanation": "..."}`` — one paragraph as a
single string."""


_SYSTEM_PROMPT_DEFINITION = """You are explaining one DEFINITION from
a math/CS textbook to a curious adult.  The definition introduces a
new mathematical object (often as an n-tuple or a relation).  Output
a thorough plain-English paragraph (5-8 sentences) that makes the
definition click for someone hearing it for the first time.

HARD RULES:

1. INTUITION FIRST.  Before any formula, give one sentence of
   plain-English intuition for the object — what kind of thing is
   it, what does it model.  ("A finite automaton is a tiny machine
   that reads symbols one at a time and decides whether to accept
   the input.")

2. EVERY COMPONENT EXPLAINED.  If the definition is "M is a 5-tuple
   (Q, Σ, δ, q0, F)", walk through ALL FIVE parts in order, giving
   each its own clause: what Q is (set of states), what Σ is
   (alphabet of input symbols), what δ does (the rule that says
   "if I'm in this state and read this symbol, here is the next
   state"), what q0 marks (where the machine starts), what F
   selects (which states make the input accepted).  Don't skip a
   part because "it's obvious".

3. CONTRAST WITH NEAR NEIGHBOURS.  When the definition has a
   well-known sibling (DFA vs NFA, finite vs infinite, total vs
   partial, deterministic vs nondeterministic), end with one short
   clause naming the difference, so the listener slots this object
   in their mental taxonomy.

4. PLAIN ENGLISH.  Spell math: "the set Q", "the alphabet sigma",
   "the transition function delta from Q-cross-sigma to Q".  No raw
   LaTeX, no glyphs.

OUTPUT FORMAT: JSON ``{"explanation": "..."}`` — one paragraph."""


_SYSTEM_PROMPT_THEOREM = """You are explaining one THEOREM (or
LEMMA / COROLLARY / PROPOSITION) from a math/CS textbook to a
curious adult.  Output a thorough plain-English paragraph (4-6
sentences).

HARD RULES:

1. SAY WHAT IT CLAIMS in one sentence.  Strip the formal language;
   say it the way a colleague would explain it at a whiteboard.

2. WHY DOES IT MATTER.  One sentence on what the theorem buys
   you — what kinds of arguments / constructions / closure
   properties it lets you make.  ("Closure under union is what
   lets us combine two pattern-matchers into one machine without
   blowing up the state space.")

3. EXPLAIN ANY NAMED OBJECT in the statement.  If the theorem
   references "regular languages", say in one clause what those
   are.  If it talks about "nondeterministic finite automata",
   the listener already heard that defined; you don't have to
   re-define, just remind ("the more flexible kind of machine we
   defined earlier").

4. INTUITION.  Close with one sentence of why the theorem is
   plausible — a sketch of the idea ("you build the union machine
   by running both originals in parallel and accepting if either
   one accepts"), without giving the full proof.

5. PLAIN ENGLISH.  No raw LaTeX, no glyphs.

OUTPUT FORMAT: JSON ``{"explanation": "..."}`` — one paragraph."""


_SYSTEM_PROMPT_ALGORITHM = """You are explaining one ALGORITHM
from a math/CS textbook to a curious adult.  Output a thorough
plain-English paragraph (5-8 sentences).

HARD RULES:

1. WHAT IS IT FOR.  Open with one sentence on the problem the
   algorithm solves and the input it expects.

2. STEP-BY-STEP.  Walk the algorithm's stages in order, each as
   one short clause.  ("First, you initialise an empty queue and
   put the start state in it.  Then you repeatedly pull a state
   off the queue …").  Skip nothing material.

3. EXPLAIN ANY NAMED VARIABLES.  If the algorithm introduces a
   counter "i" or a working set "S", name it briefly when first
   used.

4. CORRECTNESS / TERMINATION HINT.  One sentence on why the
   algorithm halts and returns the right answer — usually a loop
   invariant or a monotonic measure.

5. PLAIN ENGLISH.  No raw LaTeX, no pseudocode glyphs.

OUTPUT FORMAT: JSON ``{"explanation": "..."}`` — one paragraph."""


_SYSTEM_PROMPT_BY_KIND: dict[str, str] = {
    "definition":  _SYSTEM_PROMPT_DEFINITION,
    "theorem":     _SYSTEM_PROMPT_THEOREM,
    "lemma":       _SYSTEM_PROMPT_THEOREM,
    "corollary":   _SYSTEM_PROMPT_THEOREM,
    "proposition": _SYSTEM_PROMPT_THEOREM,
    "claim":       _SYSTEM_PROMPT_THEOREM,
    "algorithm":   _SYSTEM_PROMPT_ALGORITHM,
}


def _system_prompt_for(node: dict) -> str:
    """Pick the kind-tuned system prompt based on the chapter-map
    node's ``kind`` field — definitions / theorems / algorithms get
    structurally-richer paragraphs than equations."""
    kind = (node.get("kind") or "").strip().lower()
    return _SYSTEM_PROMPT_BY_KIND.get(kind, _SYSTEM_PROMPT_EQUATION)


# Backwards-compat alias for the old constant name some callers
# may have imported.
_SYSTEM_PROMPT = _SYSTEM_PROMPT_EQUATION


def _call_llm(system: str, user: str, *,
              max_tokens: int = 360,
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
            with urllib.request.urlopen(req, timeout=120) as resp:
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


_STRUCTURAL_KINDS = {
    "theorem", "definition", "lemma", "corollary",
    "proposition", "claim", "algorithm", "example",
}


def _which_nodes_need_pin(root: dict) -> list[dict]:
    """Pick every node that should get a plain-English ``formula_explanation``.

    Two routes in:
      * Equation-bearing nodes — the chapter root (always) plus any
        cell that introduces a *new* equation label compared to its
        DFS predecessor.  Mirrors ``plan_chapter_zoom``'s
        ``last_pin_label`` rule so we explain the equation exactly
        when the narrator first names it.
      * Structural nodes — theorems, definitions, lemmas, algorithms,
        etc.  These deserve a thorough explanation regardless of
        whether the math-graph attached an equation to them: a
        definition without a 5-tuple still needs its parts walked,
        a theorem statement needs the claim restated in plain
        English.
    """
    out: list[dict] = []
    last_label = ""

    def _walk(n: dict, *, is_root: bool) -> None:
        nonlocal last_label
        cf_label = (n.get("canonical_formula_label") or "").strip()
        cf_latex = (n.get("canonical_formula_latex") or "").strip()
        kind = (n.get("kind") or "").strip().lower()
        # Equation-bearing nodes follow the first-occurrence rule
        # so the narrator's pin-and-explanation chain doesn't repeat.
        if cf_label and cf_latex:
            introduces = (is_root or cf_label != last_label)
            if introduces:
                out.append(n)
                if not is_root:
                    last_label = cf_label
        # Structural nodes (theorem / definition / algorithm / …)
        # get explained even without a formula attached, AND on top
        # of the equation-pin if both apply.
        elif kind in _STRUCTURAL_KINDS:
            text = ((n.get("story_paragraph") or "").strip()
                    or (n.get("gist") or "").strip())
            if text:
                out.append(n)
        for c in n.get("children", []) or []:
            _walk(c, is_root=False)

    _walk(root, is_root=True)
    return out


def _user_prompt(node: dict) -> str:
    title = (node.get("title") or "").strip()
    number = (node.get("number") or "").strip()
    kind = (node.get("kind") or "section").replace("_", " ")
    label = (f"{kind} {number}: {title}" if number and title
             else (title or kind))
    gist = (node.get("gist") or "").strip()
    story = (node.get("story_paragraph") or "").strip()
    cf_label = (node.get("canonical_formula_label") or "").strip()
    cf_latex = (node.get("canonical_formula_latex") or "").strip()
    body_text = (node.get("body_text") or "").strip()
    parts = [
        f"SECTION KIND: {kind}",
        f"SECTION LABEL: {label}",
        f"GIST: {gist}" if gist else "",
        f"NARRATIVE PARAGRAPH: {story}" if story else "",
        f"BODY TEXT: {body_text[:1500]}" if body_text else "",
    ]
    if cf_label and cf_latex:
        parts.append(f"EQUATION TO EXPLAIN — label: {cf_label}")
        parts.append(f"LaTeX: {cf_latex}")
        parts.append(
            "Write the plain-English explanation as JSON "
            "``{\"explanation\": \"...\"}``.  Follow the system "
            "prompt's rules for this kind of section."
        )
    else:
        parts.append(
            "Write the plain-English explanation of this "
            "section's central idea as JSON "
            "``{\"explanation\": \"...\"}``.  Follow the system "
            "prompt's rules for this kind of section."
        )
    return "\n\n".join(p for p in parts if p)


def enrich(sidecar_path: str, *, force: bool = False) -> int:
    if not os.path.isfile(sidecar_path):
        print(f"[explanations] no sidecar at {sidecar_path}", file=sys.stderr)
        return 1
    payload = json.load(open(sidecar_path))
    root = payload.get("root") or {}
    if not root:
        print(f"[explanations] sidecar has no root", file=sys.stderr)
        return 1

    # The renderer's _mark_inherited_formulas is needed to know which
    # nodes own vs inherit; replicate the minimal logic locally so we
    # don't have to import viz.* (keeps this tool self-contained).
    def _mark(n: dict, parent_id: str) -> None:
        own = (n.get("canonical_formula_id") or "").strip()
        n["_inherits_parent_formula"] = bool(parent_id and own == parent_id)
        propagated = own or parent_id
        for c in n.get("children", []) or []:
            _mark(c, propagated)
    _mark(root, parent_id="")

    targets = _which_nodes_need_pin(root)
    print(f"[explanations] {len(targets)} node(s) need a walk-through",
          flush=True)
    written = 0
    for i, node in enumerate(targets, 1):
        existing = (node.get("formula_explanation") or "").strip()
        if existing and not force:
            continue
        nid = node.get("nid") or "?"
        cf_label = node.get("canonical_formula_label") or "?"
        print(f"  [{i}/{len(targets)}] {nid} :: {cf_label}", flush=True)
        out = _call_llm(_system_prompt_for(node), _user_prompt(node))
        if not isinstance(out, dict):
            print(f"    skipped — LLM gave no JSON", flush=True)
            continue
        text = (out.get("explanation") or "").strip()
        if not text:
            print(f"    skipped — empty explanation", flush=True)
            continue
        node["formula_explanation"] = text
        written += 1

    if not written:
        print("[explanations] nothing to write", flush=True)
        return 0
    # Strip the transient marker before saving so we don't pollute the
    # JSON with an internal flag the renderer recomputes anyway.
    def _strip(n: dict) -> None:
        n.pop("_inherits_parent_formula", None)
        for c in n.get("children", []) or []:
            _strip(c)
    _strip(root)
    tmp = sidecar_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, sidecar_path)
    print(f"[explanations] wrote {written} explanation(s) to {sidecar_path}",
          flush=True)
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sidecar", help="path to chapter_map.<root>.json")
    ap.add_argument("--force", action="store_true",
                    help="regenerate explanations even if present")
    args = ap.parse_args()
    return enrich(args.sidecar, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
