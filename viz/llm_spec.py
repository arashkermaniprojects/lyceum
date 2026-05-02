"""LLM-driven semantic-graph spec — produced by the model, rendered by us.

The point of this module is the inverse of :mod:`viz.llm_synth`: the
local LLM does NOT hand us SVG markup.  Instead, it fills in a small
JSON spec (a :class:`SemanticGraph`) that names the topic's core math
entities — a function shape, a vector, a matrix, an operation, etc.
The system's own renderer (:mod:`viz.semantic_to_svg`) then turns that
spec into SVG using primitives that live in this repo.

This keeps the visualisation produced by *our* code, not downloaded —
the LLM can't sneak in arbitrary SVG, and every diagram remains
inspectable, restyle-able, and stylistically uniform with the rest of
the chalkboard.

Local-only by policy: never calls Anthropic / OpenAI.
"""
from __future__ import annotations

import json as _json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

from .semantic_ir import (
    EDGE_RELATIONS,
    NODE_TYPES,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
)


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
LLM_SPEC_BUDGET_S = float(os.environ.get("LLM_SPEC_BUDGET_S", "6.0"))


# Schema documentation included in the system prompt so the LLM
# knows exactly which fields to fill in and what values are valid.
# Keep this small — Qwen pays attention to short, concrete schemas.
_SCHEMA_DOC = (
    'Output JSON of the form:\n'
    '{\n'
    '  "nodes": [\n'
    '    {"id": "<short stable id>", "type": "<node type>",\n'
    '     "label": "<one-line caption>", "params": { ... }},\n'
    '    ...\n'
    '  ],\n'
    '  "edges": [\n'
    '    {"source": "<id>", "target": "<id>", "relation": "<rel>"}\n'
    '  ]\n'
    '}\n\n'
    'Node types and their required params:\n'
    '  - "function" — params: form (one of: linear, quadratic, cubic,\n'
    '    polynomial, exp, log, sin, cos, sigmoid, relu, tanh, softmax,\n'
    '    gaussian, parabola).  label is the formula in plain text.\n'
    '  - "vector" — params: components (list of strings, e.g.\n'
    '    ["v1", "v2", "v3"]) OR symbol (string).\n'
    '  - "matrix" — params: rows (list of list of strings),\n'
    '    name (string).\n'
    '  - "operation" — params: key (e.g. "dot_product", "transpose",\n'
    '    "gradient", "integral", "derivative").\n'
    '  - "equation" — params: lhs (string), rhs (string), formula\n'
    '    (full LaTeX-ready text).\n'
    '  - "scalar" — params: name (string), value (string, optional).\n'
    '  - "shape" — params: kind (one of: circle, line, rect, triangle,\n'
    '    polygon).\n'
    '  - "point" — params: x, y (numbers), label.\n'
    '  - "set" — params: name, members (list of strings).\n'
    '  - "label" — purely textual; params: text.\n'
    '  - "axis" — params: name (x|y), range ([min, max]).\n\n'
    'Edge relations: depends_on, maps_to, part_of, equals, applied_to,\n'
    '  increases_with, decreases_with, minimizes, maximizes,\n'
    '  orthogonal_to.\n'
)


_PROMPT_SYSTEM = (
    'You are a math/stats visualisation spec generator.  Given a topic '
    'or question, you decide which structured math entities best '
    'visualise it, and emit a small JSON graph that names them.  A '
    'separate renderer turns your graph into an SVG diagram.\n'
    '\n'
    + _SCHEMA_DOC +
    '\n'
    'Pick types that match the topic\'s natural visualisation:\n'
    '  - For "gradient": include a "function" (the underlying f, e.g.\n'
    '    quadratic), a "vector" (the gradient ∇f with concrete\n'
    '    components like ["∂f/∂x", "∂f/∂y"]), and an "operation" with\n'
    '    key "gradient", connected by "applied_to" / "depends_on" edges.\n'
    '  - For "dot product": "vector" + "vector" + "operation" with\n'
    '    key "dot_product", plus an "equation" giving the formula.\n'
    '  - For "sigmoid": "function" with form "sigmoid".\n'
    '  - For "softmax": "function" with form "softmax".\n'
    '  - For "matrix multiplication": "matrix" + "matrix" + "operation"\n'
    '    key "matrix_multiplication".\n'
    '  - For "regularization": "function" with form "quadratic" (the\n'
    '    penalty), plus an "equation" giving the regularised loss.\n'
    '  - For "Bayes\' rule": "equation" with formula =\n'
    '    "P(A|B) = P(B|A) P(A) / P(B)".\n'
    '\n'
    'Rules:\n'
    '  - Output ONLY the JSON object, no prose, no markdown fences.\n'
    '  - Use 2 to 5 nodes — enough to be informative, not overwhelming.\n'
    '  - Every "edge" must reference node ids that exist in "nodes".\n'
    '  - Pick concrete params (real components, real form names) so\n'
    '    the renderer can draw something — never leave params empty.\n'
    '  - Labels must be standard math notation (e.g. "v", "f(x)", "A",\n'
    '    "∇f"), NEVER fragments of the question text ("is", "what",\n'
    '    "the", "a"). For a vector use a single-letter symbol like\n'
    '    "v" or "x"; for a function use "f(x)" or "g(x)"; for a matrix\n'
    '    use "A" or "M".\n'
    '  - For "vector" nodes, set params.name to a single short symbol\n'
    '    (e.g. "v") AND fill params.components with concrete component\n'
    '    names (e.g. ["v_1", "v_2"] or ["x", "y", "z"]).\n'
)


def _user_prompt(question: str) -> str:
    return f'Topic / question: {question}'


def fetch_semantic_spec(
    question: str,
    *,
    base_url: str = LLM_BASE_URL,
    model: str = LLM_MODEL,
    budget_s: float = LLM_SPEC_BUDGET_S,
) -> Optional[SemanticGraph]:
    """Call the local LLM and return the parsed :class:`SemanticGraph`.

    Returns ``None`` when the endpoint is unreachable, the LLM returns
    invalid JSON, or no node passes validation against
    :data:`NODE_TYPES`.  On any failure path we degrade silently — the
    caller can fall back to other tiers (curated registry / semantic
    parser) without the user noticing.
    """
    if not question or not question.strip():
        return None

    body = _json.dumps({
        'model': model,
        'messages': [
            {'role': 'system', 'content': _PROMPT_SYSTEM},
            {'role': 'user', 'content': _user_prompt(question)},
        ],
        'max_tokens': 600,
        'temperature': 0.0,
        'top_p': 1.0,
        'seed': 42,
    }).encode('utf-8')
    try:
        req = urllib.request.Request(
            base_url.rstrip('/') + '/chat/completions',
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer local-vllm',
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=budget_s) as resp:
            payload = _json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, ValueError) as e:
        print(f'[viz.llm_spec] vLLM call failed: {e}')
        return None
    choice = (payload.get('choices') or [{}])[0]
    content = ((choice.get('message') or {}).get('content') or '').strip()
    if not content:
        return None
    spec = _extract_json(content)
    if spec is None:
        return None
    return _spec_to_graph(spec)


def _extract_json(text: str) -> Optional[dict]:
    """Find the first JSON object in *text*.  Tolerates code-fence
    wrappers (``​​​``json … ``​​​``) the
    LLM sometimes adds despite the system prompt.
    """
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    # Greedy: take the outermost balanced { ... }.  Cheap heuristic —
    # find the first '{' and the matching '}' by depth-counting.
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return _json.loads(blob)
                except _json.JSONDecodeError:
                    return None
    return None


def _spec_to_graph(spec: dict) -> Optional[SemanticGraph]:
    """Convert a JSON spec dict into a :class:`SemanticGraph`, dropping
    nodes/edges with unknown types or relations.  Returns ``None`` if
    nothing valid survives.
    """
    raw_nodes = spec.get('nodes') or []
    raw_edges = spec.get('edges') or []
    graph = SemanticGraph()
    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get('id') or '').strip()
        ntype = str(n.get('type') or '').strip()
        if not nid or ntype not in NODE_TYPES:
            continue
        label = str(n.get('label') or '').strip()
        params = n.get('params') or {}
        if not isinstance(params, dict):
            params = {}
        graph.add_node(SemanticNode(
            id=nid, type=ntype, label=label, params=params,
        ))
    if graph.is_empty():
        return None
    valid_ids = {n.id for n in graph.nodes}
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        src = str(e.get('source') or '').strip()
        dst = str(e.get('target') or '').strip()
        rel = str(e.get('relation') or '').strip()
        if src not in valid_ids or dst not in valid_ids:
            continue
        if rel not in EDGE_RELATIONS:
            continue
        graph.add_edge(SemanticEdge(
            source=src, target=dst, relation=rel,
            params=e.get('params') or {},
        ))
    graph.meta = {'source': 'llm_spec'}
    return graph


__all__ = ['fetch_semantic_spec']
