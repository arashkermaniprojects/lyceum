"""Verifies S2 cosine-similarity merge when embeddings are present."""
import math

from sevim.s2_extract import MERGE_TAU, _ensure_node
from sevim.ir import SceneGraph, SpanRef


def _orthogonal_pair(d: int = 8) -> tuple[tuple[float, ...], tuple[float, ...]]:
    a = tuple(1.0 if i == 0 else 0.0 for i in range(d))
    b = tuple(1.0 if i == 1 else 0.0 for i in range(d))
    return a, b


def _similar_pair(d: int = 8) -> tuple[tuple[float, ...], tuple[float, ...]]:
    a = tuple(1.0 / math.sqrt(d) for _ in range(d))
    # Near-identical direction, small perturbation kept well under 1-τ.
    b = tuple((1.0 + (0.001 if i == 0 else 0.0)) / math.sqrt(d) for i in range(d))
    return a, b


def test_orthogonal_embeddings_do_not_merge():
    g = SceneGraph()
    ea, eb = _orthogonal_pair()
    _ensure_node(g, "Alpha", SpanRef(0, 5), ea)
    _ensure_node(g, "Beta", SpanRef(6, 10), eb)
    assert len(g.nodes) == 2


def test_similar_embeddings_merge():
    g = SceneGraph()
    ea, eb = _similar_pair()
    first = _ensure_node(g, "Alpha", SpanRef(0, 5), ea)
    second = _ensure_node(g, "Alfa", SpanRef(6, 10), eb)
    # Labels differ; embeddings nearly identical → should merge into one node.
    assert first == second, f"{first} != {second}"
    assert len(g.nodes) == 1
    assert len(g.nodes[0].src_spans) == 2


def test_no_embedding_falls_back_to_label_only():
    g = SceneGraph()
    a = _ensure_node(g, "Alpha", SpanRef(0, 5))
    b = _ensure_node(g, "Beta", SpanRef(6, 10))
    assert a != b
    assert len(g.nodes) == 2


def test_tau_frozen_value():
    assert MERGE_TAU == 0.85
