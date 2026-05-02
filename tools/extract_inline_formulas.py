"""Lift inline mathematical formulas out of section text into
``canonical_formula_latex``.

Books like Sipser write definitions in prose ("a finite automaton is
a 5-tuple (Q, Σ, δ, q0, F), where …") rather than as set-piece
display equations.  The LaTeX-targeting math-graph misses those, so
the chapter-zoom cells render text-only and the side panel can't
key formulas to them.  This tool walks the chapter-map sidecar,
finds every node where ``canonical_formula_latex`` is empty *and*
the body text contains math-shaped content, asks the local Qwen for
a LaTeX rendering of the central formula, and writes it back into
the sidecar.

Output: in-place update of the chapter-map JSON.

Idempotent: re-running skips nodes whose ``canonical_formula_latex``
is already populated unless ``--force`` is passed.

Usage:
    python -m tools.extract_inline_formulas \\
        books/sipser-…chapter_map.b_p1_s_ch_1_regular_languages.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from typing import Optional


LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"


_SYSTEM_PROMPT = """You extract the *central* mathematical formula
from one section of a textbook into LaTeX.  You are given the
section's narration text; if that text contains a definition,
theorem, equation, or any other piece of math worth rendering,
return it as KaTeX-renderable LaTeX along with a short label.  If
the section is purely prose with no central formula, return an
empty string for ``latex``.

OUTPUT FORMAT — JSON:

  {
    "latex": "M = (Q, \\\\Sigma, \\\\delta, q_0, F)",
    "label": "Definition 1.5"
  }

HARD RULES:

1. ONE central formula per section.  Pick the most defining one —
   usually the section's principal definition / theorem statement /
   recurrence.  Don't list multiple equations.

2. Use proper LaTeX with single backslashes (``\\sum``, ``\\Sigma``,
   ``\\delta``, ``\\hat{x}``, ``\\frac{a}{b}``).  Subscripts and
   superscripts use ``_{...}`` / ``^{...}`` with braces.  No
   Unicode math glyphs in the LaTeX (write ``\\Sigma`` not ``Σ``).

3. The ``label`` field uses the section's existing identifier when
   one is mentioned ("Definition 1.5", "Theorem 1.39", "Equation
   2.4", "Lemma 3.7", "Example 5.2").  When no label is implied,
   use ``"Definition"``, ``"Theorem"``, ``"Equation"``, ``"Recurrence"``
   etc., as appropriate.

4. SKIP cases where the section is description-only.  Cases that
   warrant skipping: a worked-example narrative, a generic
   discussion paragraph with no symbol, a chapter or section
   intro that names other things but doesn't itself state a
   formula.  In those cases, return ``{"latex": "", "label": ""}``.

5. NO commentary outside the JSON.  Whole response = the JSON
   object."""


def _call_llm(system: str, user: str, *,
              max_tokens: int = 320,
              temperature: float = 0.2,
              retries: int = 1) -> Optional[dict]:
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
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = json.loads(resp.read())
            content = (raw["choices"][0]["message"]
                       .get("content") or "").strip()
            if content:
                return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    print(f"  [llm] failed: {last_err}", file=sys.stderr)
    return None


# Heuristic: "math-shaped" text — has at least one of the typical
# inline-math indicators.  Keeps us from spending LLM calls on
# pure-prose sections that obviously have no formula.
_MATH_HINT_RE = re.compile(
    r"(?:[A-Z][_\^]\w|"
    r"\b\d+\s*[-+*/=<>]\s*\d+|"
    r"\([\sA-Za-zΑ-Ωα-ω,_\d^.]*[A-Z][\sA-Za-zΑ-Ωα-ω,_\d^.]*\)|"
    r"[Σ∑Π∏∫∂∇λμνπρστθφψωΩΔα-ω]|"
    r"\b\d+-tuple\b|"
    r"\bdefin\w+\b|\btheorem\b|\bequation\b|\blemma\b|"
    r"\brecurr\w+\b|\bformula\b|\binequal\w+\b|\bidentity\b)",
    re.IGNORECASE,
)


def _looks_like_math(text: str) -> bool:
    return bool(_MATH_HINT_RE.search(text or ""))


def _normalise_latex(s: str) -> str:
    """Collapse double-backslash command names (``\\\\hat`` → ``\\hat``)
    that some LLMs emit because they're confused about JSON
    escaping levels."""
    if not s:
        return s
    return re.sub(r"\\\\([a-zA-Z])", r"\\\1", s)


def _gather_text(node: dict) -> str:
    parts = []
    for key in ("gist", "story_paragraph", "formula_explanation"):
        v = (node.get(key) or "").strip()
        if v:
            parts.append(v)
    return "  ".join(parts)


def _user_prompt(node: dict, body: str) -> str:
    title = (node.get("title") or "").strip()
    num = (node.get("number") or "").strip()
    kind = (node.get("kind") or "").strip()
    label_hint = ""
    if num and kind:
        label_hint = f"{kind.title()} {num}"
    return (
        f"SECTION TITLE: {title}\n"
        f"SECTION KIND: {kind}\n"
        + (f"SUGGESTED LABEL: {label_hint}\n" if label_hint else "")
        + f"\nNARRATION TEXT:\n{body}\n\n"
        f"Return ONE JSON object with the central formula as "
        f"described in the system prompt."
    )


def repair(sidecar_path: str, *, force: bool = False) -> int:
    if not os.path.isfile(sidecar_path):
        print(f"error: {sidecar_path} not found", file=sys.stderr)
        return 1
    payload = json.load(open(sidecar_path))
    root = payload.get("root") or {}

    nodes_with_changes = 0
    skipped_no_text = 0
    skipped_existing = 0
    skipped_no_math = 0
    llm_calls = 0

    def _walk(n: dict):
        nonlocal nodes_with_changes, skipped_no_text
        nonlocal skipped_existing, skipped_no_math, llm_calls
        existing = (n.get("canonical_formula_latex") or "").strip()
        if existing and not force:
            skipped_existing += 1
        else:
            body = _gather_text(n)
            if not body:
                skipped_no_text += 1
            elif not _looks_like_math(body):
                skipped_no_math += 1
            else:
                llm_calls += 1
                nid = n.get("nid") or "?"
                print(f"  [{llm_calls}] {nid}", flush=True)
                out = _call_llm(_SYSTEM_PROMPT, _user_prompt(n, body))
                if isinstance(out, dict):
                    latex = _normalise_latex(
                        (out.get("latex") or "").strip()
                    )
                    label = (out.get("label") or "").strip()
                    if latex:
                        n["canonical_formula_latex"] = latex
                        if label and not (n.get("canonical_formula_label")
                                           or "").strip():
                            n["canonical_formula_label"] = label
                        print(f"      → {label}: {latex[:80]}",
                              flush=True)
                        nodes_with_changes += 1
                    else:
                        print(f"      → no formula in section",
                              flush=True)
        for c in n.get("children") or []:
            _walk(c)

    _walk(root)

    print(f"\n[extract-inline-formulas] LLM calls: {llm_calls}, "
          f"updated: {nodes_with_changes}, "
          f"skipped (had latex): {skipped_existing}, "
          f"skipped (no text): {skipped_no_text}, "
          f"skipped (no math hint): {skipped_no_math}")
    if not nodes_with_changes:
        return 0
    tmp = sidecar_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, sidecar_path)
    print(f"[extract-inline-formulas] wrote {sidecar_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sidecar", help="path to chapter_map.<root>.json")
    p.add_argument("--force", action="store_true",
                   help="re-query even when canonical_formula_latex "
                        "is already populated")
    args = p.parse_args(argv)
    return repair(args.sidecar, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
