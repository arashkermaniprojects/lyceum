"""Semantic intermediate representation for the Tier-3 deterministic renderer.

A :class:`SemanticGraph` is a tiny, structured description of what a
narrated clause is *about* — function shapes, equations, vectors,
matrices, geometric objects, and the relationships between them.

It sits between :mod:`viz.semantic_parser` (clause text → graph) and
:mod:`viz.semantic_to_svg` / :mod:`viz.semantic_to_latex` (graph → SVG /
LaTeX).  Both renderers are pure functions of the graph, so for any
given graph the output is byte-identical run-to-run.

Stable node IDs
---------------
Every parser path assigns a deterministic ID to each node — either an
operation/topic key (``func_quadratic``, ``mat_A``) or a hash-derived
slug.  The orchestrator reuses these IDs as DOM ``nid`` values so the
frontend can diff between successive renders without re-creating
elements (Step 7: incremental updates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Node / edge types — kept as string literals (not enums) to keep the IR
# trivially serialisable to JSON for SSE transport.
# ---------------------------------------------------------------------------

# Recognised node kinds.  Renderers decide what to do with each; unknown
# kinds are ignored gracefully so the IR can grow without breaking
# downstream code.
NODE_TYPES = frozenset({
    "function",        # params: form (linear|quadratic|cubic|exp|log|sin|cos|sigmoid|relu|tanh)
    "graph",           # plot frame; may anchor multiple function nodes
    "equation",        # params: lhs, rhs (LaTeX-ready strings)
    "vector",          # params: components or symbol
    "matrix",          # params: rows (list[list[str]]), name
    "scalar",          # params: name, value
    "shape",           # params: kind (circle|line|rect|triangle|polygon)
    "point",           # params: x, y, label
    "operation",       # params: key (e.g. dot_product, transpose, gradient)
    "set",             # params: name, members
    "label",           # purely textual annotation
    "axis",            # params: name (x|y), range
})

EDGE_RELATIONS = frozenset({
    "depends_on",      # source's value depends on target
    "maps_to",         # source maps to target (function application, transform)
    "part_of",         # source is component of target
    "equals",          # source equals target (identity / equation)
    "applied_to",      # operation applied to operand
    "increases_with",  # monotonic relationship
    "decreases_with",
    "minimizes",       # we want to minimise target via source
    "maximizes",
    "orthogonal_to",   # geometric perpendicularity
})


@dataclass
class SemanticNode:
    """One typed entity in a semantic graph.

    ``id`` must be stable for a given input — renderers and the frontend
    both rely on it for incremental updates.  ``params`` is a free-form
    dict; specific renderers know which keys to read for their type.
    """
    id: str
    type: str
    label: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticEdge:
    source: str
    target: str
    relation: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticGraph:
    nodes: list[SemanticNode] = field(default_factory=list)
    edges: list[SemanticEdge] = field(default_factory=list)
    # Optional metadata — provenance, parse cost, source clause text.
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def add_node(self, node: SemanticNode) -> SemanticNode:
        """Append *node*, deduplicating by ``id`` (later wins on params)."""
        for i, existing in enumerate(self.nodes):
            if existing.id == node.id:
                # Merge params; last write wins.
                merged = {**existing.params, **node.params}
                self.nodes[i] = SemanticNode(
                    id=existing.id, type=node.type or existing.type,
                    label=node.label or existing.label, params=merged,
                )
                return self.nodes[i]
        self.nodes.append(node)
        return node

    def add_edge(self, edge: SemanticEdge) -> SemanticEdge:
        """Append *edge*, deduplicating by (source, target, relation)."""
        for existing in self.edges:
            if (existing.source == edge.source
                    and existing.target == edge.target
                    and existing.relation == edge.relation):
                return existing
        self.edges.append(edge)
        return edge

    def node(self, nid: str) -> Optional[SemanticNode]:
        for n in self.nodes:
            if n.id == nid:
                return n
        return None

    def by_type(self, type_: str) -> list[SemanticNode]:
        return [n for n in self.nodes if n.type == type_]

    def is_empty(self) -> bool:
        return not self.nodes

    def to_dict(self) -> dict:
        """JSON-serialisable view (used by SSE transport + tests)."""
        return {
            "nodes": [
                {"id": n.id, "type": n.type, "label": n.label,
                 "params": n.params}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target,
                 "relation": e.relation, "params": e.params}
                for e in self.edges
            ],
            "meta": self.meta,
        }


__all__ = [
    "SemanticNode",
    "SemanticEdge",
    "SemanticGraph",
    "NODE_TYPES",
    "EDGE_RELATIONS",
]
