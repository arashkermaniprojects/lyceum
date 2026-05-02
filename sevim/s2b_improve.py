"""S2b — Optional LLM-based graph inspector and improver.

After the rule-based S2 extraction, this module optionally calls
Claude Haiku (claude-haiku-4-5-20251001) via the Anthropic Messages API
to review and rewrite the extracted scene graph in JSON.  The LLM can:

  - Merge redundant or near-identical nodes
  - Remove spurious edges that do not reflect the source text
  - Add obvious missing edges (e.g. causes / requires / instance_of)
  - Clean verbose or artefact-ridden node labels
  - Correct reversed relation directions

This path is DISABLED by default and must be explicitly opted into.
Set both environment variables to enable it:

    SEVIM_IMPROVE=1          — enable the LLM path
    ANTHROPIC_API_KEY=<key>  — Anthropic API credentials

When disabled (the default), improve() is a no-op that returns the graph
unchanged.  The full S1–S5 pipeline operates entirely without any external
API calls when this path is off.

Important distinction from the Qwen encoder
-------------------------------------------
Qwen2.5-7B (embed.py) produces floating-point embedding vectors for node
labels — it never reads or generates text.  Claude Haiku (this module)
reads the graph JSON and rewrites it — it never produces embeddings.
These are two completely separate components.

Note on invariant I1 (determinism)
-----------------------------------
When S2b is active, I1 is broken: Claude Haiku is a stochastic model and
its JSON response can differ across calls.  The trace log records this.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir import SceneGraph

# The exact model ID to call.  Pin this explicitly so that future
# Claude releases do not silently change behaviour.
_MODEL = "claude-haiku-4-5-20251001"

# Relations the LLM is allowed to use.  Any relation outside this set
# is silently dropped so the LLM cannot introduce schema violations.
_VALID_RELATIONS = frozenset({
    'causes', 'used_for', 'requires', 'reduces_to', 'measures',
    'contains', 'part_of', 'instance_of', 'similar_to', 'opposes',
    'attribute_of', 'sequence',
})

_SYSTEM_PROMPT = """\
You are a knowledge-graph quality inspector for educational technical diagrams.

Given:
  • source_text — the original sentence(s) being visualised
  • graph — the automatically extracted nodes and edges (JSON)

Your job: return an improved graph that accurately represents the key concepts
and relationships in the source text. Rules:

1. Keep only nodes that represent meaningful concepts from the text (1–4 words max).
2. Merge nodes whose labels are near-duplicates (e.g. "changing data distribution"
   vs "data distribution" — keep the shorter/cleaner one).
3. Remove edges that are factually wrong or do not appear in the text.
4. Add edges for obvious relationships stated in the text that were missed.
5. Fix relation types that are wrong (e.g. use "causes" not "requires" when A
   produces / leads to B).
6. Node IDs must follow the pattern  n_<words_joined_by_underscore>  .
7. Allowed relation values: causes, used_for, requires, reduces_to, measures,
   contains, part_of, instance_of, similar_to, opposes, attribute_of, sequence.

Return ONLY valid JSON — no prose, no markdown fences — in this exact format:
{
  "nodes": [{"id": "n_...", "label": "..."}],
  "edges": [{"from": "n_...", "to": "n_...", "relation": "..."}]
}
"""


def _nid(label: str) -> str:
    """Convert a human-readable label to a canonical node ID."""
    return "n_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def improve(g: "SceneGraph", source_text: str) -> "SceneGraph":
    """Optionally call Claude Haiku to inspect and improve *g*.

    Returns *g* unchanged on any error (API failure, bad JSON, etc.) so
    the pipeline always continues to S3 regardless of LLM availability.

    Parameters
    ----------
    g:
        The SceneGraph produced by S2 Extract.
    source_text:
        The original input text, sent to the LLM for context.

    Returns
    -------
    SceneGraph
        The (potentially rewritten) graph.  Same object as *g*, mutated
        in place via _apply_improvements.
    """
    # Guard: only run when explicitly enabled.
    if os.environ.get("SEVIM_IMPROVE") != "1":
        return g

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return g

    # Nothing to improve on an empty graph.
    if not g.nodes:
        return g

    # Serialise the current graph for the LLM.
    graph_payload = {
        "nodes": [{"id": n.id, "label": n.label} for n in g.nodes],
        "edges": [
            {"from": e.from_id, "to": e.to_id, "relation": e.relation}
            for e in g.edges
        ],
    }
    user_msg = (
        f"source_text:\n{source_text.strip()}\n\n"
        f"graph:\n{json.dumps(graph_payload, indent=2)}"
    )

    payload = json.dumps({
        "model": _MODEL,
        "max_tokens": 1024,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
        raw = result["content"][0]["text"]
    except Exception as exc:
        print(f"[s2b_improve] API call failed: {exc}")
        return g

    # Extract the JSON object from the response.
    # The LLM sometimes wraps its answer in markdown fences or adds prose.
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= 0:
        print("[s2b_improve] no JSON found in LLM response")
        return g

    try:
        improved = json.loads(raw[start:end])
    except json.JSONDecodeError as exc:
        print(f"[s2b_improve] JSON parse error: {exc}")
        return g

    _apply_improvements(g, improved)
    return g


def _apply_improvements(g: "SceneGraph", improved: dict) -> None:
    """Mutate *g* in place to reflect the LLM's suggestions.

    Performs three safety checks before accepting any change:
      1. Relation must be in _VALID_RELATIONS (schema guard).
      2. Both endpoints of every edge must exist in the new node set.
      3. Self-loops are dropped.

    Node IDs from the LLM are re-canonicalised from the label so that
    the LLM cannot inject arbitrary ID strings.  A prefix-lookup handles
    the common case where the LLM abbreviates an ID (e.g. "n_fixed" for
    "n_fixed_distribution_assumption").
    """
    from .ir import SceneNode, SceneEdge, SpanRef

    new_nodes_data = improved.get("nodes", [])
    new_edges_data = improved.get("edges", [])

    if not new_nodes_data:
        return  # refuse to blank the graph

    old_by_id = {n.id: n for n in g.nodes}

    # Build an ID remap: LLM-supplied ID → canonical ID derived from label.
    # We trust the label and re-derive the ID so the LLM cannot inject arbitrary strings.
    id_remap: dict[str, str] = {}   # llm_id → canonical_id

    kept_nodes: list = []
    for nd in new_nodes_data:
        llm_id: str = nd.get("id", "")
        label: str = nd.get("label", "").strip()
        if not label:
            continue
        canonical = _nid(label)
        id_remap[llm_id] = canonical
        id_remap[canonical] = canonical  # identity mapping for self-referential IDs

        # Reuse the old node (preserving embedding, salience, spans) if possible.
        old = old_by_id.get(llm_id) or old_by_id.get(canonical)
        if old:
            kept_nodes.append(SceneNode(
                id=canonical, label=label,
                node_type=old.node_type,
                embedding=old.embedding,
                salience=old.salience,
                src_spans=old.src_spans,
            ))
        else:
            # New node introduced by the LLM — use neutral defaults.
            kept_nodes.append(SceneNode(
                id=canonical, label=label,
                node_type="entity", embedding=(), salience=0.5, src_spans=[],
            ))

    kept_node_ids = {n.id for n in kept_nodes}

    def _resolve(llm_id: str) -> str:
        """Resolve an LLM-supplied ID to the canonical ID, with prefix fallback."""
        if llm_id in id_remap:
            return id_remap[llm_id]
        # Prefix match: "n_fixed" → "n_fixed_distribution_assumption"
        for canonical in kept_node_ids:
            if canonical.startswith(llm_id) or llm_id.startswith(canonical):
                return canonical
        return llm_id  # will be rejected below if not in kept_node_ids

    kept_edges: list = []
    for ed in new_edges_data:
        relation = ed.get("relation", "")
        if relation not in _VALID_RELATIONS:
            continue  # schema guard: unknown relation → drop

        from_id = _resolve(ed.get("from", ""))
        to_id   = _resolve(ed.get("to", ""))

        if from_id not in kept_node_ids or to_id not in kept_node_ids:
            continue  # dangling edge → drop

        if from_id == to_id:
            continue  # self-loop → drop

        eid = f"e_{relation}_{from_id}_{to_id}"
        kept_edges.append(SceneEdge(
            id=eid, from_id=from_id, to_id=to_id,
            relation=relation,  # type: ignore[arg-type]
            src_spans=[],       # LLM does not emit provenance references
        ))

    g.nodes[:] = kept_nodes
    g.edges[:] = kept_edges
    g.revision += 1
