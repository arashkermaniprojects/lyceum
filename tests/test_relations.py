from sevim.pipeline import run_pipeline


def _relations(text: str) -> set[str]:
    r = run_pipeline(text)
    return {e.relation for e in r.graph.edges}


def test_part_of_detected():
    assert "part_of" in _relations("The arm is part of the body.")


def test_contains_detected():
    assert "contains" in _relations("The box contains a toy.")


def test_similar_to_detected():
    assert "similar_to" in _relations("A dolphin is similar to a whale.")


def test_opposes_detected():
    assert "opposes" in _relations("Heat opposes cold.")


def test_causes_detected():
    assert "causes" in _relations("Gravity causes objects to fall.")


def test_sequence_detected():
    assert "sequence" in _relations("Wake up then drink coffee.")


def test_attribute_of_detected():
    assert "attribute_of" in _relations("The body has a heart.")


def test_instance_of_example_detected():
    assert "instance_of" in _relations("A dolphin is an example of a mammal.")


def test_instance_of_isa_detected():
    assert "instance_of" in _relations("A dolphin is a mammal.")


def test_all_eight_relations_covered():
    text = (
        "Heat opposes cold. "
        "A dolphin is similar to a whale. "
        "The arm is part of the body. "
        "The box contains a toy. "
        "Gravity causes acceleration. "
        "Wake up then drink coffee. "
        "The body has a heart. "
        "A dolphin is a mammal."
    )
    got = _relations(text)
    expected = {
        "opposes", "similar_to", "part_of", "contains",
        "causes", "sequence", "attribute_of", "instance_of",
    }
    assert expected.issubset(got), f"missing: {expected - got}"
