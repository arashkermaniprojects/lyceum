from sevim.embed import is_available
from sevim.pipeline import run_pipeline


def test_fallback_embedding_empty_when_encoder_unavailable():
    r = run_pipeline("Gravity causes objects to fall.")
    if not is_available():
        for n in r.graph.nodes:
            assert n.embedding == ()


def test_determinism_with_embeddings():
    sample = "Gravity causes objects to fall. The arm is part of the body."
    a = run_pipeline(sample)
    b = run_pipeline(sample)
    assert a.svg == b.svg
    assert [n.embedding for n in a.graph.nodes] == [n.embedding for n in b.graph.nodes]


def test_phi_distinct_params_for_distinct_embeddings():
    if not is_available():
        return  # skipped when encoder isn't installed
    r = run_pipeline("A causes B. C causes D.")
    placed = {p.shape.nid: p for p in r.placed.shapes}
    a = placed.get("n_a")
    c = placed.get("n_c")
    if a and c:
        assert (a.shape.width, a.shape.height) != (c.shape.width, c.shape.height)


def test_phi_fallback_identical_to_phase1_when_no_embeddings():
    r = run_pipeline("Gravity causes objects to fall.")
    if is_available():
        return
    # Phase 1 salience-only params: w=130, h=60 at salience=0.5
    for p in r.placed.shapes:
        assert p.shape.width == 130.0
        assert p.shape.height == 60.0
