"""VLM-driven extraction — equations, figure captions, anything else.

A single class, ``VLMEngine``, that talks to a *local* Qwen-VL vLLM
endpoint (default ``http://127.0.0.1:8004/v1``, model
``Qwen/Qwen2.5-VL-7B-Instruct-AWQ``).  The same engine handles:

  * **Equation extraction** — crop the page region around an
    ``(N.M)`` marker, ask the VLM to return *only* the LaTeX source
    for the equation in that image.  No nougat, no pix2tex.

  * **Figure description** — crop a figure bbox, ask the VLM what the
    figure is and (when present) which "Figure N.M" label it carries.
    Replaces the brittle caption-regex used by the old re-ingester.

  * **Inspection (re-export)** — the canonical-figure inspector
    in ``viz.inspector`` already calls the same endpoint; we keep
    that wiring unchanged so all VLM calls share one model.

Local-only by policy: no Anthropic / OpenAI / external API.  See
``feedback_local_only.md`` in user memory.
"""
from __future__ import annotations

import base64
import io
import json as _json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional


VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8004/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
VLM_TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "20.0"))


_EQ_PROMPT = (
    "Below is an image of a single mathematical equation taken from a "
    "statistics textbook page.  Output ONLY the LaTeX source for the "
    "equation — no \\begin{{equation}} wrapper, no \\tag, no commentary, "
    "no English text, just the LaTeX body that would render the math.  "
    "If the image does not contain a clear equation, output exactly the "
    "single token NONE."
)


_FIG_PROMPT = (
    "This is a region cropped from a textbook page.  In one short line, "
    "say what kind of figure this is (plot type and topic).  Then on a "
    "second line, output the figure label exactly as printed (e.g. "
    "\"Figure 7.1\") if visible, or NONE if no label is visible.  Do "
    "not add any other text."
)


@dataclass
class _Probe:
    last_check: float = 0.0
    ok: bool = False


_probe = _Probe()


def vlm_reachable(*, force: bool = False) -> bool:
    """Cheap probe of the local VLM endpoint with a short cache."""
    now = time.time()
    if not force and now - _probe.last_check < 30.0:
        return _probe.ok
    try:
        req = urllib.request.Request(
            VLM_BASE_URL.rstrip("/") + "/models",
            headers={"Authorization": "Bearer local-vlm"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            served = _json.loads(resp.read().decode("utf-8"))
            ids = {m.get("id") for m in served.get("data", [])}
            _probe.ok = bool(ids)
    except Exception:
        _probe.ok = False
    _probe.last_check = now
    return _probe.ok


def _png_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def _vlm_call(prompt: str, *, png_bytes: bytes,
              max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Single chat-completion call against the VLM. Returns the assistant
    string, or '' on failure."""
    body = _json.dumps({
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {
                     "url": f"data:image/png;base64,{_png_b64(png_bytes)}"
                 }},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": 42,
    }).encode("utf-8")
    req = urllib.request.Request(
        VLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local-vlm"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=VLM_TIMEOUT) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""
    choice = (payload.get("choices") or [{}])[0]
    return ((choice.get("message") or {}).get("content") or "").strip()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VLMEngine:
    """Same interface as the old OCR engines.  Calls the local VLM for
    equation extraction and figure description.  Lazy-imports pymupdf
    when actually invoked so import is cheap.
    """

    def __init__(self) -> None:
        if not vlm_reachable():
            raise RuntimeError(
                f"VLM not reachable at {VLM_BASE_URL}; start vLLM with "
                f"Qwen2.5-VL on that port (see tools/start_vlm.sh)"
            )

    # ----- Equation extraction --------------------------------------------

    def ocr(self, pdf_path: str, page_index: int, eq_hits) -> dict:
        """Return ``{label: latex}`` for each equation hit on the page.

        Same signature as the old OCR engines so the existing
        ``reingest_equations.py`` flow keeps working unmodified.
        """
        import fitz  # type: ignore
        out: dict = {}
        if not eq_hits:
            return out
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        for h in eq_hits:
            x0, y0, x1, y1 = h.eq_bbox
            rect = fitz.Rect(x0, y0, x1, y1)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2.5, 2.5), clip=rect, alpha=False,
            )
            png = pix.tobytes("png")
            response = _vlm_call(_EQ_PROMPT, png_bytes=png, max_tokens=256)
            cleaned = _clean_latex_response(response)
            out[h.label] = cleaned
        doc.close()
        return out

    # ----- Figure caption / label identification --------------------------

    def describe_figure(self, png_bytes: bytes) -> tuple[str, str]:
        """Return ``(short_description, figure_label_or_empty)``.

        figure_label is "Figure 7.1" style if the VLM saw it printed,
        otherwise ''.
        """
        text = _vlm_call(_FIG_PROMPT, png_bytes=png_bytes, max_tokens=64)
        if not text:
            return "", ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        desc = lines[0] if lines else ""
        label = ""
        for ln in lines[1:] + [desc]:
            m = re.search(r"Figure\s+(\d+(?:\.\d+){0,2})", ln)
            if m:
                label = "Figure " + m.group(1)
                break
        return desc, label


def _clean_latex_response(text: str) -> str:
    """Strip code fences and commentary the VLM sometimes prepends."""
    if not text:
        return ""
    s = text.strip()
    if s == "NONE":
        return ""
    # Strip ```latex / ``` fences if present.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    # Strip $...$ or $$...$$ wrappers if present.
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$") and len(s) > 2:
        s = s[1:-1].strip()
    return s
