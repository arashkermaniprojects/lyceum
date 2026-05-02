"""Reconstruct OCR-damaged ``canonical_formula_latex`` strings.

PyMuPDF's text extraction sometimes truncates the right-hand side of
an equation to a placeholder letter — e.g. Equation 5.9 lands as
``J[f] = Z`` even though the real equation is

    J[f] = sum_{i=1}^N (y_i - f(x_i))^2 + lambda * int (f''(t))^2 dt.

The chapter-zoom narrator reads ``formula_explanation`` ("the first
part is the sum of squared differences …, the second part is lambda
times the integral of the squared second derivative …") which carries
all the information needed to rebuild the LaTeX.  This tool walks the
chapter-map sidecar, finds nodes whose OCR'd LaTeX is suspiciously
short or otherwise broken, and asks the local Qwen to reconstruct
the correct LaTeX from the explanation.  Result is written back into
``canonical_formula_latex`` on the same node.

Usage:

    python -m tools.repair_formula_latex \\
        books/ESLII.chapter_map.b_ch5.json

Idempotent: only nodes whose LaTeX matches the broken-OCR heuristics
are re-queried.  Pass ``--all`` to re-query every node with both an
explanation and a label, even when the OCR'd LaTeX looks ok.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from typing import Optional


LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"


_SYSTEM_PROMPT = """You repair OCR-damaged LaTeX equations.  PyMuPDF
truncates the right-hand side of equations and drops subscripts.  You
are given the broken LaTeX and a plain-English explanation of the same
equation; output the corrected LaTeX.

HARD RULES:

1. OUTPUT FORMAT: JSON {"latex": "..."} — exactly one LaTeX equation,
   inline form (no ``\\begin{equation}``, no ``\\[`` ``\\]``).

2. Use standard LaTeX: ``\\sum_{i=1}^{N}``, ``\\int``, ``\\lambda``,
   ``\\theta``, ``\\beta_m``, ``f''(t)``, ``\\hat{\\alpha}_i``, etc.
   Subscripts and superscripts use ``_{...}`` and ``^{...}`` with
   braces around multi-character bodies.

3. Reconstruct from the explanation when the OCR'd LaTeX is obviously
   incomplete — e.g. when the right-hand side is one or two capital
   letters with no math operators, when the equation is just a
   variable name, or when sums/integrals/operators are missing.

4. When the OCR'd LaTeX is already a complete equation that matches
   the explanation, return it UNCHANGED.  Do not invent new equations.

5. The label fragment at the end (``(5.9)`` or similar) must be
   stripped — the renderer adds the label separately.

6. NO commentary, just the JSON object.  Your entire response is one
   line of JSON."""


def _is_broken(latex: str) -> bool:
    """Heuristic: is this LaTeX an OCR artefact rather than real math?

    Triggers when the right-hand side is a single capital letter or
    short fragment with no operators / subscripts / superscripts /
    Greek symbols, or when the whole expression has no operators.
    """
    s = (latex or "").strip()
    if not s:
        return True
    # Count "real math" indicators.
    indicators = (
        s.count("\\")
        + s.count("_") + s.count("^")
        + s.count("\\sum") * 5
        + s.count("\\int") * 5
        + s.count("\\frac") * 3
        + len(re.findall(r"[+\-*/=<>]", s))
    )
    if "=" in s:
        rhs = s.split("=", 1)[1].strip()
        # Strip trailing label ``(5.9)``.
        rhs = re.sub(r",?\s*\(\d+\.\d+\)\s*$", "", rhs).strip()
        # The right-hand side reduced to one capital letter or a
        # bare placeholder is a clear OCR artefact.
        if re.fullmatch(r"[A-Z]", rhs):
            return True
        if re.fullmatch(r"[A-Z](\s*[+\-]\s*[A-Z])?", rhs):
            return True
        # No structural math symbols anywhere on the RHS.
        if not re.search(r"[\\_^+\-*/]", rhs) and len(rhs) <= 3:
            return True
    elif indicators < 2 and len(s) <= 8:
        return True
    return False


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
            if not content:
                return None
            return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    print(f"  [llm] failed: {last_err}", file=sys.stderr)
    return None


def _user_prompt(node: dict) -> str:
    cf_label = (node.get("canonical_formula_label") or "").strip()
    cf_latex = (node.get("canonical_formula_latex") or "").strip()
    cf_expl = (node.get("formula_explanation") or "").strip()
    title = (node.get("title") or "").strip()
    return (
        f"SECTION: {title}\n"
        f"EQUATION LABEL: {cf_label}\n"
        f"OCR'd LATEX (possibly broken): {cf_latex}\n"
        f"EXPLANATION: {cf_expl}\n\n"
        f"Reconstruct the correct LaTeX equation as "
        f"JSON {{\"latex\": \"...\"}}."
    )


def repair(sidecar_path: str, *, repair_all: bool = False) -> int:
    if not os.path.isfile(sidecar_path):
        print(f"[repair] no sidecar at {sidecar_path}", file=sys.stderr)
        return 1
    payload = json.load(open(sidecar_path))
    root = payload.get("root") or {}

    def _walk(n: dict):
        yield n
        for c in n.get("children", []) or []:
            yield from _walk(c)

    nodes = list(_walk(root))
    candidates = []
    for n in nodes:
        cf_latex = (n.get("canonical_formula_latex") or "").strip()
        cf_expl = (n.get("formula_explanation") or "").strip()
        if not cf_latex or not cf_expl:
            continue
        if repair_all or _is_broken(cf_latex):
            candidates.append(n)
    if not candidates:
        print(f"[repair] nothing to repair", flush=True)
        return 0

    print(f"[repair] {len(candidates)} formula(s) need repair",
          flush=True)
    written = 0
    for i, node in enumerate(candidates, 1):
        nid = node.get("nid") or "?"
        cf_label = node.get("canonical_formula_label") or "?"
        cf_old = node.get("canonical_formula_latex") or ""
        print(f"  [{i}/{len(candidates)}] {nid} :: {cf_label}", flush=True)
        print(f"    old: {cf_old}", flush=True)
        out = _call_llm(_SYSTEM_PROMPT, _user_prompt(node))
        if not isinstance(out, dict):
            print(f"    skipped — LLM gave no JSON", flush=True)
            continue
        new_latex = (out.get("latex") or "").strip()
        if not new_latex:
            print(f"    skipped — empty latex", flush=True)
            continue
        # Qwen sometimes double-escapes: ``\\\\hat`` instead of
        # ``\\hat`` in JSON, which decodes to a literal ``\\hat`` —
        # KaTeX renders that as a backslash followed by the word.
        # Collapse the doubled backslashes whenever they prefix an
        # alphabetic command name (real ``\\\\`` line-breaks survive
        # because LaTeX line-breaks aren't followed by a letter).
        new_latex = re.sub(r"\\\\([a-zA-Z])", r"\\\1", new_latex)
        if new_latex == cf_old:
            print(f"    unchanged", flush=True)
            continue
        node["canonical_formula_latex"] = new_latex
        print(f"    new: {new_latex}", flush=True)
        written += 1

    if not written:
        print("[repair] nothing changed", flush=True)
        return 0
    tmp = sidecar_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, sidecar_path)
    print(f"[repair] wrote {written} repaired equation(s) to {sidecar_path}",
          flush=True)
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sidecar", help="path to chapter_map.<root>.json")
    ap.add_argument("--all", action="store_true", dest="repair_all",
                    help="re-query every node, even when the OCR'd "
                         "LaTeX looks plausible")
    args = ap.parse_args()
    return repair(args.sidecar, repair_all=args.repair_all)


if __name__ == "__main__":
    sys.exit(main())
