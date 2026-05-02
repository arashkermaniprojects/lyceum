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


_SYSTEM_PROMPT = """You are explaining one mathematical equation to a
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


def _which_nodes_need_pin(root: dict) -> list[dict]:
    """Mirror plan_chapter_zoom + _mark_visible_formulas: every node
    whose narration emits a formula_pin clause is a candidate for an
    explanation.  That's the chapter root (always) plus any cell that
    introduces a new equation label compared to its DFS predecessor.
    """
    out: list[dict] = []
    last_label = ""

    def _walk(n: dict, *, is_root: bool) -> None:
        nonlocal last_label
        cf_label = (n.get("canonical_formula_label") or "").strip()
        cf_latex = (n.get("canonical_formula_latex") or "").strip()
        own_id = (n.get("canonical_formula_id") or "").strip()
        if not cf_label or not cf_latex:
            for c in n.get("children", []) or []:
                _walk(c, is_root=False)
            return
        # First-occurrence rule, same as the renderer's
        # _mark_visible_formulas + the planner's last_pin_label.  The
        # chapter root paints its formula but the planner does NOT
        # call _emit_node on it, so its label does not advance the
        # planner's last_pin_label tracker.  Mirror that here so the
        # next L1 child (e.g. §5.1) is still treated as the *first*
        # introducer of its label and gets its own explanation.
        introduces = (is_root or cf_label != last_label)
        if introduces:
            out.append(n)
            if not is_root:
                last_label = cf_label
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
    return (
        f"SECTION: {label}\n"
        f"GIST: {gist}\n"
        f"NARRATIVE PARAGRAPH: {story}\n\n"
        f"EQUATION TO EXPLAIN — label: {cf_label}\n"
        f"LaTeX: {cf_latex}\n\n"
        f"Write the 3-5 sentence plain-English walk-through of this "
        f"equation as JSON: {{\"explanation\": \"...\"}}.  Name every "
        f"symbol; tie the meaning back to the section's narrative."
    )


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
        out = _call_llm(_SYSTEM_PROMPT, _user_prompt(node))
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
