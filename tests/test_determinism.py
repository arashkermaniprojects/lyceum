import re

from sevim.pipeline import run_pipeline

SAMPLE = (
    "Gravity causes objects to fall. "
    "The arm is part of the body. "
    "The body has a heart."
)


def test_i1_same_input_same_svg():
    a = run_pipeline(SAMPLE)
    b = run_pipeline(SAMPLE)
    assert a.svg == b.svg


def test_i1_bytewise():
    a = run_pipeline(SAMPLE)
    b = run_pipeline(SAMPLE)
    assert a.svg.encode("utf-8") == b.svg.encode("utf-8")


def test_i3_forward_provenance():
    r = run_pipeline(SAMPLE)
    for n in r.graph.nodes:
        assert n.src_spans, f"node {n.id} has no provenance"
    for e in r.graph.edges:
        assert e.src_spans, f"edge {e.id} has no provenance"


def test_i3_reverse_provenance():
    r = run_pipeline(SAMPLE)
    node_ids = {n.id for n in r.graph.nodes}
    edge_ids = {e.id for e in r.graph.edges}
    nids = set(re.findall(r'data-nid="([^"]+)"', r.svg))
    eids = set(re.findall(r'data-eid="([^"]+)"', r.svg))
    assert nids.issubset(node_ids), f"orphan nids: {nids - node_ids}"
    assert eids.issubset(edge_ids), f"orphan eids: {eids - edge_ids}"


def test_extraction_finds_phase1_relations():
    r = run_pipeline(SAMPLE)
    relations = {e.relation for e in r.graph.edges}
    assert "causes" in relations
    assert "part_of" in relations
    assert "attribute_of" in relations
