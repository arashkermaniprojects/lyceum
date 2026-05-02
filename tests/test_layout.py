import re

from sevim.pipeline import run_pipeline


def test_container_encloses_child():
    r = run_pipeline("The arm is part of the body.")
    placed = {p.shape.nid: p for p in r.placed.shapes}
    assert "n_arm" in placed and "n_body" in placed
    body, arm = placed["n_body"], placed["n_arm"]
    assert body.x <= arm.x
    assert body.y <= arm.y
    assert body.x + body.shape.width >= arm.x + arm.shape.width
    assert body.y + body.shape.height >= arm.y + arm.shape.height


def test_sugiyama_layers_causes_chain():
    r = run_pipeline("A causes B. B causes C.")
    placed = {p.shape.nid: p for p in r.placed.shapes}
    a, b, c = placed.get("n_a"), placed.get("n_b"), placed.get("n_c")
    assert a and b and c, f"missing nodes in {list(placed)}"
    assert a.x < b.x < c.x, f"expected left→right, got {a.x} {b.x} {c.x}"


def test_redundant_containment_edge_not_rendered():
    r = run_pipeline("The arm is part of the body.")
    eids = set(re.findall(r'data-eid="([^"]+)"', r.svg))
    assert "e_part_of_n_arm_n_body" not in eids


def test_determinism_survives_layout_extension():
    sample = (
        "A causes B. B causes C. "
        "The arm is part of the body. "
        "The body has a heart."
    )
    a = run_pipeline(sample)
    b = run_pipeline(sample)
    assert a.svg == b.svg
