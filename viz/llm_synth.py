"""Tier-3 generator: ask the local text Qwen for an SVG illustrating *any* topic.

Used as a fallback when no curated topic in :mod:`viz.registry` matches.
Calls only the local vLLM endpoint (``LLM_BASE_URL``, default
``http://127.0.0.1:8000/v1``); never Anthropic / OpenAI.

Hard time budget: ``LLM_VIZ_BUDGET_S`` (default 1.6 s).  We pass it as
``timeout`` to urlopen so we cap the wait.

Quality is lower than the hand-rolled generators — but it covers any
topic the user asks about, which is exactly the gap they pointed at.
The structural inspector still rejects degenerate output.
"""
from __future__ import annotations

import json as _json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
LLM_VIZ_BUDGET_S = float(os.environ.get("LLM_VIZ_BUDGET_S", "1.6"))


_PROMPT_SYSTEM = (
    "You are a visualization synthesiser for a math/stats teaching whiteboard. "
    "Given a single topic, you respond with ONLY an SVG body (no preamble, no "
    "explanation, no markdown code fences) inside <svg ...>...</svg> tags.\n"
    "Pick ONE of two modes based on the topic:\n"
    "  CHART mode (for quantitative topics like overfitting, ROC, "
    "gradient descent, sigmoid, distributions): include an axis frame "
    "as a <rect> with stroke=\"#37474f\" plus AT LEAST ONE <polyline> "
    "with real numeric coordinates, plus <text> labels for x-axis, "
    "y-axis, and a title.\n"
    "  DIAGRAM mode (for architectures and conceptual layouts like "
    "autoencoder, transformer, CNN, perceptron, k-means clustering): "
    "use <rect>/<circle> for nodes/blocks, <line>/<polyline> for "
    "connections/arrows, <text> for labels and a title.  Include AT "
    "LEAST 3 lines or paths and AT LEAST 2 labelled nodes.  No axis "
    "frame is required in this mode.\n"
    "STRICT structural requirements:\n"
    "- viewBox=\"0 0 480 300\" and the outer <svg ...> attributes set "
    "width=\"480\" height=\"300\".\n"
    "- White background.\n"
    "- AT LEAST 2 <text> elements: a title plus one or more labels.\n"
    "- Use these colours when sensible: training=#1976d2, test=#d81b60, "
    "bias=#388e3c, variance=#f57c00, total=#5e35b1, encoder=#1976d2, "
    "decoder=#5e35b1, attention=#f57c00.\n"
    "- All x in [10, 470] and y in [20, 290] (interior of viewBox).\n"
    "- No <foreignObject>, no <script>, no external href.\n"
    "LAYOUT rules — ESSENTIAL, the diagram is rejected if any of these "
    "fail:\n"
    "- Compute each label's width as len(text) * font_size * 0.6 and "
    "treat it as a bounding rectangle.  No two <text> rectangles may "
    "overlap.\n"
    "- Use font-size=\"12\" or \"13\" for node labels, font-size=\"14\" "
    "for the title.  Add explicit font-size to every <text> element.\n"
    "- Place node labels BELOW their nodes, not on top of them: a "
    "circle at (cx, cy) with radius r should have its label at "
    "(cx, cy + r + 14) with text-anchor=\"middle\".  A rect at "
    "(x, y, w, h) should have its label at (x + w/2, y + h + 14).\n"
    "- When multiple labelled nodes sit on the same row, leave at "
    "least 90 px of horizontal centre-to-centre spacing so labels "
    "don't collide.\n"
    "- Keep the title at y=20 with text-anchor=\"middle\" and x=240; "
    "do not place any other text within 12 px of it.\n"
    "- Output the SVG only.  Nothing before <svg, nothing after </svg>."
)


def _user_prompt(topic: str, *, hint: str = "") -> str:
    body = (
        f"Topic: {topic}\n"
        f"Draw a labelled diagram or chart that explains this concept "
        f"to a student.  If the concept involves curves vs. an "
        f"x-axis variable, draw at least two contrasting series "
        f"(e.g. two regimes, two methods, training vs. test).  "
        f"Include axis titles."
    )
    if hint:
        body += f"\nAdditional guidance: {hint}"
    return body


@dataclass
class LLMVizResult:
    svg_body: str
    full_svg: str
    width: float
    height: float
    elapsed_ms: float
    raw: str = ""


def _call(prompt: str, *, deadline: float) -> str:
    """Single chat-completion against the local text Qwen, bounded by *deadline*
    (perf_counter timestamp)."""
    body = _json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _PROMPT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 900,
        "temperature": 0.1,
        "top_p": 1.0,
        "seed": 7,
    }).encode("utf-8")
    timeout = max(0.2, deadline - time.perf_counter())
    req = urllib.request.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
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
    return ((choice.get("message") or {}).get("content") or "").strip()


_SVG_RE = re.compile(r"<svg\b[^>]*>(.*?)</svg>", re.S | re.I)


def _extract_svg(text: str) -> tuple[str, str]:
    """Return (full_svg, body_only) — strips code fences and any prose
    the model snuck in despite the prompt."""
    if not text:
        return "", ""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    m = _SVG_RE.search(s)
    if not m:
        return "", ""
    body = m.group(1).strip()
    full = m.group(0)
    return full, body


def synthesise(topic: str, *,
               budget_s: Optional[float] = None,
               feedback: str = "") -> Optional[LLMVizResult]:
    """Ask the local text LLM for an SVG illustrating *topic*.

    Returns None on failure (network error, no SVG in reply, budget
    exhausted, …).  The caller is expected to inspect the result and
    decide whether to accept / regenerate.

    *feedback* is fed back to the model on retries, e.g. the previous
    inspector verdict (\"text overlap: 'Function Space' ↔ 'Scalar Value'\")
    so the next attempt can correct that specific failure.
    """
    deadline = time.perf_counter() + (budget_s or LLM_VIZ_BUDGET_S)
    t0 = time.perf_counter()
    raw = _call(_user_prompt(topic, hint=feedback), deadline=deadline)
    full, body = _extract_svg(raw)
    if not body:
        return None
    return LLMVizResult(
        svg_body=body,
        full_svg=full,
        width=480.0,
        height=300.0,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        raw=raw,
    )
