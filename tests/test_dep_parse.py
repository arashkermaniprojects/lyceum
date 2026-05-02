"""Tests for the dep-parse S2 path + the 4 new relations.

Skipped silently when spaCy is unavailable — the regex fallback is covered
by test_relations.py in that environment.
"""
from sevim.s2_extract import dep_parse_available
from sevim.pipeline import run_pipeline


def _rels(text: str) -> set[str]:
    return {e.relation for e in run_pipeline(text).graph.edges}


def test_used_for_detected():
    if not dep_parse_available():
        return
    got = _rels("The algorithm uses a binary heap.")
    assert "used_for" in got, got


def test_requires_detected():
    if not dep_parse_available():
        return
    got = _rels("Merge sort requires additional memory.")
    assert "requires" in got, got


def test_reduces_to_regex_variant():
    # Covered by regex rule _REDUCES_TO (dep-parse lemma "reduce" also routes there).
    got = _rels("The problem reduces to sorting.")
    assert "reduces_to" in got, got


def test_measures_detected():
    if not dep_parse_available():
        return
    got = _rels("The perimeter equals four times the side length.")
    assert "measures" in got, got


def test_dep_parse_beats_regex_on_variant_phrasing():
    """Phrasings the original regex cascade misses but dep-parse catches."""
    if not dep_parse_available():
        return
    # "will produce" — regex _CAUSE only matches "causes|leads to|results in|triggers".
    got = _rels("The algorithm produces a sorted array.")
    assert "causes" in got, got


def test_passive_still_extracts_something():
    """Passive constructions — dep-parse finds nsubjpass + agent or prep object."""
    if not dep_parse_available():
        return
    # "is used as" routes through used_for (either via copular attr=used + prep
    # or via regex backstop).
    got = _rels("The stack is used as an auxiliary structure.")
    assert "used_for" in got, got


def test_verb_map_covers_all_four_new_relations():
    from sevim.s2_extract import VERB_RELATION_MAP
    targets = {"used_for", "requires", "reduces_to", "measures"}
    assert targets.issubset(set(VERB_RELATION_MAP.values())), \
        f"missing relations in verb map: {targets - set(VERB_RELATION_MAP.values())}"


def test_12_relations_in_ontology():
    """The original 12 concept-diagram relations must remain registered.

    The math extension is additive: 18 math relations are added on top, but
    every concept-diagram relation must still have a connector pattern.
    """
    from sevim.s3_map import _RELATION_PATTERN
    expected = {
        "contains", "part_of", "causes", "sequence",
        "attribute_of", "similar_to", "opposes", "instance_of",
        "used_for", "requires", "reduces_to", "measures",
    }
    assert expected <= set(_RELATION_PATTERN.keys())
