"""Clause text → :class:`SemanticGraph`.

Fast-path: regex + keyword matching.  Aim is ≤ 20 ms per clause; in
practice this module finishes in ~1 ms for the recognised cases.

Slow-path: an optional :func:`llm_parse` hook that queries the local
Qwen and parses its JSON reply.  We only use it when the regex pass
returns an empty graph *and* the caller passes ``allow_llm=True``.
The orchestrator's Tier-3 fallback path can opt in; the realtime stream
keeps it off so the budget stays bounded.

Coverage of the regex pass (deterministic, sub-ms):

  * Function-form keywords: linear, quadratic, cubic, polynomial, exp,
    log, sin, cos, sigmoid, ReLU, tanh, softmax.
  * Equations: ``x = y * 2``, ``f(x) = …``, etc.  We don't try to parse
    them mathematically — we capture lhs/rhs as LaTeX-ready strings.
  * Vectors and scalars by symbol (``v``, ``\\vec v``, ``α``, ``θ``).
  * Matrix mentions (``A``, ``the matrix W``, ``2x3 matrix``).
  * Geometric shapes (circle, line, parabola, ellipse, triangle).
  * Relationships: increase / decrease / minimize / maximize / depends on.
  * Operations: gradient, derivative, dot product, transpose, …
    (delegated to :mod:`viz.operations`).

The renderers (`semantic_to_svg`, `semantic_to_latex`) are pure
functions of the graph, so deterministic input ⇒ deterministic output.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Optional

from .semantic_ir import SemanticEdge, SemanticGraph, SemanticNode


# ---------------------------------------------------------------------------
# Function-form recognition
# ---------------------------------------------------------------------------

# Order matters: longer / more specific phrases first.  Each entry is
# (form_key, regex).  We deliberately key on noun-phrase patterns
# ("linear function", "quadratic loss") rather than bare adjectives so
# we don't over-fire on "linear regression" inside arbitrary prose.
_FUNC_FORMS: list[tuple[str, re.Pattern]] = [
    ("quadratic", re.compile(
        r"\bquadratic\s+(?:function|loss|polynomial|form|cost|equation)\b",
        re.I)),
    ("cubic", re.compile(r"\bcubic\s+(?:function|polynomial|curve)\b", re.I)),
    ("linear", re.compile(
        r"\blinear\s+(?:function|map|model|equation|fit|loss)\b", re.I)),
    ("polynomial", re.compile(
        r"\bpolynomial\s+(?:function|of\s+degree|fit|regression)\b", re.I)),
    ("exp", re.compile(
        r"\bexponential\s+(?:function|growth|decay|curve)\b", re.I)),
    ("log", re.compile(
        r"\b(?:logarithm(?:ic)?|log)\s+(?:function|curve)\b", re.I)),
    ("sin", re.compile(r"\bsine\s+(?:function|wave|curve)\b", re.I)),
    ("cos", re.compile(r"\bcosine\s+(?:function|wave|curve)\b", re.I)),
    ("sigmoid", re.compile(r"\b(?:sigmoid|logistic)\s+(?:function|curve)\b", re.I)),
    ("relu", re.compile(r"\b(?:ReLU|rectified\s+linear)\b", re.I)),
    ("tanh", re.compile(r"\btanh\s+(?:function|activation)\b", re.I)),
    ("softmax", re.compile(r"\bsoftmax\s+(?:function|activation|layer)\b", re.I)),
    # Bare-noun fallbacks (no qualifier).  Only fire when the word stands
    # alone enough that it's clearly the math object, not a loose adjective.
    ("parabola", re.compile(r"\bparabola\b", re.I)),
    ("sigmoid", re.compile(r"\bsigmoid\b(?!\s+regression)", re.I)),
]

_FORM_LABELS = {
    "linear":      "linear function",
    "quadratic":   "quadratic function",
    "cubic":       "cubic function",
    "polynomial":  "polynomial",
    "exp":         "exponential",
    "log":         "logarithm",
    "sin":         "sine",
    "cos":         "cosine",
    "sigmoid":     "sigmoid",
    "relu":        "ReLU",
    "tanh":        "tanh",
    "softmax":     "softmax",
    "parabola":    "parabola",
}


# ---------------------------------------------------------------------------
# Geometric shape recognition (no axes / data — purely shapes)
# ---------------------------------------------------------------------------

_SHAPES: list[tuple[str, re.Pattern]] = [
    ("circle",   re.compile(r"\b(?:unit\s+)?circle\b", re.I)),
    ("ellipse",  re.compile(r"\bellipse\b", re.I)),
    ("triangle", re.compile(r"\btriangle\b", re.I)),
    ("rectangle",re.compile(r"\brectangle\b", re.I)),
    ("square",   re.compile(r"\bsquare\b(?!\s+root)", re.I)),
    ("polygon",  re.compile(r"\bpolygon\b", re.I)),
    ("line",     re.compile(r"\b(?:straight\s+)?line\b(?!ar)", re.I)),
]


# ---------------------------------------------------------------------------
# Relationship cues
# ---------------------------------------------------------------------------

_REL_INCREASES = re.compile(
    r"\b(?:increases?|grows?|rises?)\s+with\b", re.I)
_REL_DECREASES = re.compile(
    r"\b(?:decreases?|shrinks?|drops?|falls?)\s+with\b", re.I)
_REL_MINIMIZE = re.compile(
    r"\bminimi[sz]e(?:s|d)?\s+(?:the\s+)?(?P<obj>[a-zA-Z][\w\s]{0,30}?)\b", re.I)
_REL_MAXIMIZE = re.compile(
    r"\bmaximi[sz]e(?:s|d)?\s+(?:the\s+)?(?P<obj>[a-zA-Z][\w\s]{0,30}?)\b", re.I)
_REL_DEPENDS = re.compile(
    r"\b(?P<a>[a-zA-Z]\w*)\s+depends\s+on\s+(?P<b>[a-zA-Z]\w*)", re.I)


# English stop-words / question-fragments that the depends-on regex
# may match in non-mathematical prose ("we know that depends on an
# existing rule" → "that depends on an").  When either side of the
# match is a stop-word, the relation isn't a real dependency — skip it.
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "as", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "and", "or",
    "but", "if", "so", "we", "you", "they", "he", "she", "i",
    "what", "which", "who", "how", "why", "when", "where",
})


# ---------------------------------------------------------------------------
# Equation recognition
# ---------------------------------------------------------------------------

# Match "<lhs> = <rhs>" where lhs is an identifier or function call.  We
# keep the original strings; the LaTeX renderer handles polishing.
_EQ_PATTERN = re.compile(
    r"(?P<lhs>(?:\\?[a-zA-Z_][\w]*\s*\([^()]{0,40}\))|(?:\\?[a-zA-Z_][\w]*))"
    r"\s*=\s*"
    r"(?P<rhs>[^.,;\n]{1,80})"
)


# ---------------------------------------------------------------------------
# Matrix recognition
# ---------------------------------------------------------------------------

_MATRIX_NAMED = re.compile(
    r"\b(?:matrix\s+(?P<name1>[A-Z]\w?)|(?P<name2>[A-Z]\w?)\s*(?:\^[T\\top]+|transpose))\b"
)
# Matches both "2x3 matrix" and "matrix 2x3" / "matrix with 3 rows".
_MATRIX_PHRASE = re.compile(
    r"\b(?:"
    r"(?P<dims_pre>\d+\s*[x×]\s*\d+)\s+matrix"
    r"|matrix\s+(?P<dims_post>\d+\s*[x×]\s*\d+|with\s+\d+\s+rows?(?:\s+and\s+\d+\s+columns?)?)"
    r")\b",
    re.I,
)
# Inline bracket matrix: [[1, 2], [3, 4]]
_MATRIX_INLINE = re.compile(
    r"\[\s*\[\s*[\-\d\.,\s]+\]\s*(?:,\s*\[\s*[\-\d\.,\s]+\]\s*)+\]"
)


# ---------------------------------------------------------------------------
# Vector recognition
# ---------------------------------------------------------------------------

# Match "vector v" / "vector x_1" / "\vec{v}" — names must end at a word
# boundary so we don't grab two letters out of "vectors".
_VECTOR_NAMED = re.compile(
    r"(?:\bvector\s+(?P<name1>[a-zA-Z]\w?)\b"
    r"|\\vec\s*\{\s*(?P<name2>[a-zA-Z]\w?)\s*\})"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(text: str, *, allow_llm: bool = False,
          llm_budget_s: float = 0.8) -> SemanticGraph:
    """Convert *text* to a :class:`SemanticGraph`.

    Fast path is regex-only; the LLM path is opt-in and bounded by
    *llm_budget_s*.  An empty graph is returned when nothing matches —
    callers (e.g. the orchestrator) decide whether to fall back further.
    """
    t0 = time.perf_counter()
    g = SemanticGraph()
    if not text or not text.strip():
        g.meta["parse_ms"] = 0.0
        g.meta["source"] = "regex"
        return g

    _extract_functions(text, g)
    _extract_shapes(text, g)
    _extract_equations(text, g)
    _extract_matrices(text, g)
    _extract_vectors(text, g)
    _extract_relationships(text, g)
    _extract_operations(text, g)

    g.meta["parse_ms"] = (time.perf_counter() - t0) * 1000
    g.meta["source"] = "regex"
    g.meta["clause"] = text

    if g.is_empty() and allow_llm:
        llm_g = llm_parse(text, budget_s=llm_budget_s)
        if llm_g is not None and not llm_g.is_empty():
            llm_g.meta.setdefault("clause", text)
            llm_g.meta["parse_ms"] = (time.perf_counter() - t0) * 1000
            llm_g.meta["source"] = "llm"
            return llm_g

    return g


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_functions(text: str, g: SemanticGraph) -> None:
    """Detect function-form keywords ("quadratic loss", "sigmoid", …)."""
    seen: set[str] = set()
    for form, pat in _FUNC_FORMS:
        if form in seen:
            continue
        if pat.search(text):
            seen.add(form)
            g.add_node(SemanticNode(
                id=f"func_{form}",
                type="function",
                label=_FORM_LABELS.get(form, form),
                params={"form": form},
            ))


def _extract_shapes(text: str, g: SemanticGraph) -> None:
    seen: set[str] = set()
    for kind, pat in _SHAPES:
        if kind in seen:
            continue
        if pat.search(text):
            seen.add(kind)
            g.add_node(SemanticNode(
                id=f"shape_{kind}",
                type="shape",
                label=kind,
                params={"kind": kind},
            ))


_PSEUDOCODE_STOPWORDS = frozenset({
    # Words >= 3 chars that mark prose, not math.
    "draw", "drawn", "iterate", "compute", "computed", "sample", "samples",
    "size", "training", "test", "data", "set", "step", "from", "the",
    "input", "output", "where", "let", "such", "that", "this", "these",
    "those", "for", "each", "all", "any", "with", "via", "using",
    "defined", "denote", "denoted", "given", "obtain", "obtained",
    "return", "yields", "produces", "consider", "perform", "until",
    "while", "loop", "repeat",
    # Two-char prose connectors that often appear in algorithm steps
    # ("for i = 1 to N do") but never in real math expressions.
    "to", "do", "of", "is", "be", "as", "if", "in", "on", "at", "by",
    "we", "an", "or",
})


def _is_prose_rhs(rhs: str) -> bool:
    """Reject rhs values that look like pseudocode / English prose.

    Real equations have at most one word-like token (e.g. ``max(0, x)``
    has 'max'); pseudocode steps like ``b = 1 to B: (a) Draw a
    bootstrap sample ...`` have many.

    Triggers on:
      * any colon (algorithm step boundary).
      * a parenthesised single letter that is *not* a function
        argument — i.e. ``(a)`` standalone but not ``(x)`` in
        ``p(x)``.  Caught by requiring no letter immediately before
        the opening paren.
      * ≥ 2 word-like tokens (length ≥ 2 alphabetic).
      * any token in :data:`_PSEUDOCODE_STOPWORDS`.
      * unbalanced parens / brackets / braces — fragments like
        ``1 L(yi`` from PDF text extraction.
    """
    if ":" in rhs:
        return True
    if re.search(r"(?<![A-Za-z])\([a-z]\)", rhs):
        return True
    if not _balanced_brackets(rhs):
        return True
    tokens = re.findall(r"[A-Za-z]{2,}", rhs)
    if len(tokens) >= 2:
        return True
    if any(t.lower() in _PSEUDOCODE_STOPWORDS for t in tokens):
        return True
    return False


def _balanced_brackets(s: str) -> bool:
    """True iff every ``(``, ``[``, ``{`` has a matching closer."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _extract_equations(text: str, g: SemanticGraph) -> None:
    """Capture ``lhs = rhs`` fragments without trying to parse the math.

    The match must look like math, not pseudocode: rhs needs at least
    one math character (digit / operator / Greek / backslash command),
    and must NOT look like an algorithm step (colon, parenthesised
    step labels, or ≥ 2 word-like tokens).  Rejecting these stops the
    Bagging-pseudocode-as-equation misfire seen in ESLII §8.7.
    """
    for m in _EQ_PATTERN.finditer(text):
        lhs = m.group("lhs").strip()
        rhs = m.group("rhs").strip().rstrip(",;:.")
        if not lhs or not rhs:
            continue
        # Skip prose-y matches: rhs must contain a digit, operator, Greek
        # letter, or known math token to qualify as an equation.
        if not re.search(r"[\d\+\-\*\/\^=αβγδθπσμλΣ]|\\\w", rhs):
            continue
        if _is_prose_rhs(rhs):
            continue
        nid = f"eq_{_slug(lhs + '_' + rhs)}"
        g.add_node(SemanticNode(
            id=nid, type="equation",
            label=f"{lhs} = {rhs}",
            params={"lhs": lhs, "rhs": rhs},
        ))


def _extract_matrices(text: str, g: SemanticGraph) -> None:
    for m in _MATRIX_NAMED.finditer(text):
        name = m.group("name1") or m.group("name2") or ""
        if not name:
            continue
        nid = f"mat_{name}"
        g.add_node(SemanticNode(
            id=nid, type="matrix",
            label=f"matrix {name}",
            params={"name": name},
        ))
    m = _MATRIX_PHRASE.search(text)
    if m:
        dims_text = m.group("dims_pre") or m.group("dims_post")
        rows, cols = _parse_dims(dims_text)
        g.add_node(SemanticNode(
            id="mat_generic",
            type="matrix",
            label=f"{rows}×{cols} matrix" if rows else "matrix",
            params={"nrows": rows, "ncols": cols},
        ))
    inline = _MATRIX_INLINE.search(text)
    if inline:
        rows = _parse_inline_matrix(inline.group(0))
        if rows:
            g.add_node(SemanticNode(
                id=f"mat_inline_{_slug(inline.group(0))}",
                type="matrix",
                label="matrix",
                params={"rows": rows,
                        "nrows": len(rows),
                        "ncols": len(rows[0]) if rows else 0},
            ))


def _extract_vectors(text: str, g: SemanticGraph) -> None:
    for m in _VECTOR_NAMED.finditer(text):
        name = m.group("name1") or m.group("name2") or ""
        if not name:
            continue
        nid = f"vec_{name}"
        g.add_node(SemanticNode(
            id=nid, type="vector",
            label=f"vector {name}",
            params={"name": name},
        ))


def _extract_relationships(text: str, g: SemanticGraph) -> None:
    """Lightweight relationship detection — one edge per obvious cue."""
    if _REL_INCREASES.search(text):
        g.add_edge(SemanticEdge(
            source="_y", target="_x", relation="increases_with",
        ))
    if _REL_DECREASES.search(text):
        g.add_edge(SemanticEdge(
            source="_y", target="_x", relation="decreases_with",
        ))
    m = _REL_MINIMIZE.search(text)
    if m:
        obj = m.group("obj").strip().rstrip(".,;")
        target = f"obj_{_slug(obj)[:24]}"
        g.add_node(SemanticNode(
            id=target, type="label", label=obj, params={"role": "minimized"},
        ))
        g.add_edge(SemanticEdge(
            source="_optimizer", target=target, relation="minimizes",
        ))
    m = _REL_MAXIMIZE.search(text)
    if m:
        obj = m.group("obj").strip().rstrip(".,;")
        target = f"obj_{_slug(obj)[:24]}"
        g.add_node(SemanticNode(
            id=target, type="label", label=obj, params={"role": "maximized"},
        ))
        g.add_edge(SemanticEdge(
            source="_optimizer", target=target, relation="maximizes",
        ))
    m = _REL_DEPENDS.search(text)
    if m:
        a_word = m.group("a")
        b_word = m.group("b")
        # Skip stop-word matches — a clause like "we know that depends
        # on an existing model" would otherwise spawn a "that → an"
        # diagram.  Real math depends-on phrases use single-letter or
        # symbolic identifiers (``y depends on x``).
        if (a_word.lower() not in _STOP_WORDS
                and b_word.lower() not in _STOP_WORDS):
            a = f"sym_{a_word.lower()}"
            b = f"sym_{b_word.lower()}"
            g.add_node(SemanticNode(id=a, type="scalar", label=a_word,
                                    params={"name": a_word}))
            g.add_node(SemanticNode(id=b, type="scalar", label=b_word,
                                    params={"name": b_word}))
            g.add_edge(SemanticEdge(source=a, target=b, relation="depends_on"))


def _extract_operations(text: str, g: SemanticGraph) -> None:
    """Delegate to :mod:`viz.operations` for canonical phrases."""
    try:
        from .operations import find_operations
    except Exception:
        return
    seen: set[str] = set()
    for label, latex, _start, _end in find_operations(text):
        key = label.lower().replace(" ", "_")
        if key in seen:
            continue
        seen.add(key)
        g.add_node(SemanticNode(
            id=f"op_{key}",
            type="operation",
            label=label,
            params={"key": key, "latex": latex},
        ))


# ---------------------------------------------------------------------------
# LLM fallback (opt-in only, bounded budget)
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You convert one short math sentence into a JSON semantic graph. "
    "Reply ONLY with JSON of the shape "
    "{\"nodes\":[{\"id\":..,\"type\":..,\"label\":..,\"params\":{}}],"
    "\"edges\":[{\"source\":..,\"target\":..,\"relation\":..}]}. "
    "Allowed types: function, graph, equation, vector, matrix, scalar, "
    "shape, point, operation, set, label, axis. "
    "Allowed relations: depends_on, maps_to, part_of, equals, applied_to, "
    "increases_with, decreases_with, minimizes, maximizes, orthogonal_to. "
    "Use stable snake_case ids. No prose, no markdown."
)


def llm_parse(text: str, *, budget_s: float = 0.8) -> Optional[SemanticGraph]:
    """Slow-path: query the local Qwen and parse its JSON reply.

    Returns None on any failure (no LLM, timeout, malformed reply).
    Pure local; never calls Anthropic / OpenAI.
    """
    try:
        import json
        import urllib.error
        import urllib.request
        from .llm_synth import LLM_BASE_URL, LLM_MODEL  # type: ignore
    except Exception:
        return None

    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": text.strip()},
        ],
        "max_tokens": 500,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 7,
    }
    try:
        req = urllib.request.Request(
            LLM_BASE_URL.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer local-llm"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=budget_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    choice = (payload.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    if not content:
        return None
    # Strip code fences if the model added them despite the system prompt.
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
    try:
        data = json.loads(content)
    except Exception:
        return None
    g = SemanticGraph()
    for raw in data.get("nodes", []) or []:
        nid = str(raw.get("id") or "").strip()
        ntype = str(raw.get("type") or "").strip()
        if not nid or not ntype:
            continue
        g.add_node(SemanticNode(
            id=nid, type=ntype,
            label=str(raw.get("label") or ""),
            params=dict(raw.get("params") or {}),
        ))
    for raw in data.get("edges", []) or []:
        src = str(raw.get("source") or "").strip()
        dst = str(raw.get("target") or "").strip()
        rel = str(raw.get("relation") or "").strip()
        if not (src and dst and rel):
            continue
        g.add_edge(SemanticEdge(source=src, target=dst, relation=rel))
    return g


# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    if not safe:
        safe = "x"
    if len(safe) > 16:
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
        safe = safe[:16] + "_" + h
    return safe


def _parse_dims(dims_text: str) -> tuple[int, int]:
    if not dims_text:
        return 0, 0
    m = re.search(r"(\d+)\s*[x×]\s*(\d+)", dims_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\s+rows?(?:\s+and\s+(\d+)\s+columns?)?", dims_text, re.I)
    if m:
        rows = int(m.group(1))
        cols = int(m.group(2)) if m.group(2) else rows
        return rows, cols
    return 0, 0


def _parse_inline_matrix(s: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_match in re.finditer(r"\[([^\[\]]+)\]", s):
        cells = [c.strip() for c in row_match.group(1).split(",")]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append(cells)
    # Drop the outer wrapper that re-matched the inner spans? No — the
    # outer brackets contain the row brackets, so finditer on the inner
    # `[...]` already gives us one entry per row.
    if len(rows) >= 1 and len(rows[0]) > 0:
        # Normalise: pad to max width with empty strings so the renderer
        # can iterate uniformly.
        ncols = max(len(r) for r in rows)
        for r in rows:
            while len(r) < ncols:
                r.append("")
    return rows


__all__ = ["parse", "llm_parse"]
