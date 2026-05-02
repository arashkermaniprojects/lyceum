"""Recover clean LaTeX for an OCR-fragmented book equation via the
local text LLM (Qwen2.5-14B-Instruct-AWQ on ``localhost:8000``).

ESLII's PDF text extraction flattens 2-D math layout into ragged
single-token lines (``f∈H`` / ``" N`` / ``X`` / ``i=1`` / ``L(yi,
f(xi)) + λJ(f)``).  KaTeX can't render these.  Sending the
fragments through Qwen with a focused prompt recovers a coherent
LaTeX expression that *can* render — keeping the user inside the
local-only constraint of this codebase.

Pure helper: no caching, no threading.  Callers (the orchestrator)
own the cache and concurrency policy.
"""
from __future__ import annotations

import json as _json
import os
import re
import urllib.request


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.environ.get(
    "LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ",
)


_PROMPT_SYSTEM = (
    "You convert PDF-extracted equation OCR fragments into clean "
    "LaTeX source.  The input is a multi-line fragment of one "
    "equation whose 2-D layout was flattened by the PDF extractor.  "
    "Reconstruct the original equation.  Output ONLY the LaTeX "
    "source — no $, no \\[ \\] delimiters, no code fences, no "
    "commentary.  Hints: a lone capital `X` line is usually "
    "\\sum (sigma); a stray `\"` before a variable is usually a "
    "superscript marker; `ε` is \\varepsilon; `β` is \\beta; `λ` "
    "is \\lambda; subscripts under a sum like ``i=1`` and a bound "
    "like ``N`` go in \\sum_{i=1}^{N}; a bare ``f∈H`` under a "
    "summation/optimization is the index of \\min_{f\\in H} or "
    "\\sum_{f\\in H}.  When uncertain, prefer the most common "
    "convention from machine-learning textbooks."
)


def _strip_wrappers(text: str) -> str:
    """Remove code fences, math delimiters, and surrounding $."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    s = s.strip()
    # Strip outer math delimiters.
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2]
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2]
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2]
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s.strip()


def clean_via_llm(
    ocr_text: str,
    *,
    ref_label: str = "",
    base_url: str = LLM_BASE_URL,
    model: str = LLM_MODEL,
    timeout: float = 6.0,
    context: str = "",
) -> str:
    """Send *ocr_text* to the local LLM and return clean LaTeX.

    ``ref_label`` (e.g. ``"5.42"``) is included in the prompt for
    tracing only.  ``context`` lets the caller pass a few prose
    sentences from around the equation so the model can pick the
    right operator (``\\min`` vs ``\\max`` vs bare ``\\sum``).

    Returns ``""`` on any failure (timeout, connection refused,
    malformed reply).  Callers fall back to their own rendering.
    """
    if not ocr_text or not ocr_text.strip():
        return ""
    user_prompt = (
        f"PDF-extracted text for Equation {ref_label}:\n"
        f"```\n{ocr_text.strip()}\n```\n"
    )
    if context.strip():
        user_prompt += (
            "\nSurrounding prose for context (do not include in output):\n"
            f"```\n{context.strip()[:600]}\n```\n"
        )
    user_prompt += "\nReturn ONLY the clean LaTeX source."
    body = _json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.1,
        "top_p": 1.0,
        "seed": 7,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local-llm"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""
    choice = (payload.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    cleaned = _strip_wrappers(content)
    # Sanity check: if the model just echoed a prose comment instead
    # of LaTeX, reject it.  A real LaTeX equation typically has at
    # least one of: ``\``, ``=``, ``+``, ``_``, ``^``, ``(``, ``{``.
    if not cleaned:
        return ""
    if not re.search(r"[\\=+_^({]", cleaned):
        return ""
    return cleaned


# ---------------------------------------------------------------------------
# Body-cleanup: prose + math → KaTeX-aware HTML
# ---------------------------------------------------------------------------

_BODY_PROMPT_SYSTEM = (
    "You are reformatting PDF-extracted reference-card text from a "
    "machine-learning textbook so it can be rendered by KaTeX in a "
    "browser.  The input mixes prose with math expressions whose 2-D "
    "layout was flattened by the PDF extractor into ragged lines.  "
    "Reconstruct the math expressions and wrap each one in delimiters: "
    "use \\(...\\) for inline math (within a sentence) and \\[...\\] for "
    "display equations (on their own line).  Keep the prose as plain "
    "text.  Drop OCR artefacts: stray quotation marks, bare ``#`` "
    "marks, fragmented page-header lines (``This is page 4`` / "
    "``Printer: Opaq``), and the equation-number labels that the OCR "
    "pulled to the wrong line.  Output ONLY the reformatted body — no "
    "code fences, no ``Here is...`` preamble, no commentary.  Hints: a "
    "lone capital ``X`` line is usually \\sum; ``ε`` is \\varepsilon; "
    "``λ`` is \\lambda; ``∈`` is \\in; ``⟨`` and ``⟩`` are \\langle / "
    "\\rangle; subscripts under a sum like ``i=1`` and a bound ``N`` "
    "go in \\sum_{i=1}^{N}; ``α_iK(x, x_i)`` is `\\alpha_i K(x, x_i)`."
)


_PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:'s|\s+is|\s+are)\s+[^.\n]*[.:]\s*"
    r"|reformatted[^.\n]*[.:]\s*"
    r"|the\s+reformatted\s+[^.\n]*[.:]\s*)",
    re.I,
)


def clean_body_via_llm(
    body_text: str,
    *,
    kind: str = "Reference",
    ref_label: str = "",
    base_url: str = LLM_BASE_URL,
    model: str = LLM_MODEL,
    timeout: float = 7.0,
) -> str:
    """Send a reference-card body through the local LLM and return
    a KaTeX-aware reformatted version.

    Returns ``""`` on any failure (timeout, malformed reply, model
    refusing).  The caller falls back to monospace prose when empty.
    """
    if not body_text or not body_text.strip():
        return ""
    user_prompt = (
        f"PDF-extracted body of {kind} {ref_label}:\n"
        f"```\n{body_text.strip()[:1800]}\n```\n"
        "Return ONLY the reformatted body."
    )
    body = _json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _BODY_PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 700,
        "temperature": 0.1,
        "top_p": 1.0,
        "seed": 7,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local-llm"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""
    choice = (payload.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    cleaned = _strip_wrappers(content)
    cleaned = _PREAMBLE_RE.sub("", cleaned).strip()
    if not cleaned:
        return ""
    return cleaned
