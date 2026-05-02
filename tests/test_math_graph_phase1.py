"""Phase-1 rule-based enrichment tests.

Each test seeds a tiny MathGraph and calls ``enrich_clause`` on a
realistic clause to verify the right edges/concepts land.
"""
from __future__ import annotations

from sevim.math_graph import MathGraph
from sevim.math_graph_phase1 import enrich_clause


# ---------------------------------------------------------------------------
# Definitions: "where L is the loss function"
# ---------------------------------------------------------------------------

def test_define_attaches_concept_and_instance_of_edge():
    g = MathGraph(book_id="t")
    # Phase-0 first: a formula that uses L.
    g.ingest_formula(nid="eq42",
                     latex=r"\sum L(y_i, f(x_i)) + \lambda J(f)")
    g.ingest_passage(seq=9, text="(placeholder)", home_nid="ch5",
                     formula_ids_in_clause=["eq42"])
    # Now Phase-1 enrichment on the *same* passage's text:
    rep = enrich_clause(
        g, seq=9,
        text="where L is a loss function and J is the penalty functional.",
        home_nid="ch5",
        formula_ids_in_clause=["eq42"],
    )
    # Concept nodes created.
    assert "c:loss_function" in g.concepts
    assert "c:penalty_functional" in g.concepts
    # instance_of edges attached.
    inst = {(e.src, e.dst) for e in g.edges if e.type == "instance_of"}
    assert ("v:L", "c:loss_function") in inst
    assert ("v:J", "c:penalty_functional") in inst
    # ``defines`` edge attached the formula to the concept too —
    # because eq42 uses both L and J.
    defs = {(e.src, e.dst) for e in g.edges if e.type == "defines"}
    assert ("eq42", "c:loss_function") in defs
    assert ("eq42", "c:penalty_functional") in defs
    # Telemetry counts non-zero.
    assert rep["new_concepts"] >= 2
    assert rep["new_definitions"] >= 2


# ---------------------------------------------------------------------------
# Derivation: "rewriting (5.42) we have …"
# ---------------------------------------------------------------------------

def test_rewriting_emits_derived_from():
    g = MathGraph(book_id="t")
    # Eq 5.42 lands first.
    g.ingest_formula(nid="eq42", latex=r"\sum L(y_i, f(x_i)) + \lambda J(f)",
                     cite_labels=["5.42"])
    g.ingest_passage(seq=9, text="(prior)", home_nid="ch5",
                     formula_ids_in_clause=["eq42"])
    # Eq 5.48 lands later, in a "rewriting (5.42)" clause.
    g.ingest_formula(nid="eq48",
                     latex=r"\min_{f \in HK} \dots + \lambda \|f\|^2",
                     cite_labels=["5.48"])
    rep = enrich_clause(
        g, seq=22,
        text="Rewriting (5.42) we have min f in HK …",
        home_nid="ch5",
        formula_ids_in_clause=["eq48"],
    )
    derived = {(e.src, e.dst) for e in g.edges if e.type == "derived_from"}
    assert ("eq48", "eq42") in derived
    assert rep["new_derivations"] >= 1


# ---------------------------------------------------------------------------
# Equivalence: "or equivalently"
# ---------------------------------------------------------------------------

def test_or_equivalently_links_pair():
    g = MathGraph(book_id="t")
    g.ingest_formula(nid="A", latex=r"y = m x + b")
    g.ingest_formula(nid="B", latex=r"y - m x = b")
    rep = enrich_clause(
        g, seq=3,
        text="we have y = m x + b or equivalently y - m x = b.",
        home_nid="ch3",
        formula_ids_in_clause=["A", "B"],
    )
    related = {(e.src, e.dst) for e in g.edges if e.type == "related_to"}
    assert ("A", "B") in related
    assert ("B", "A") in related
    assert rep["new_equivalences"] >= 1


# ---------------------------------------------------------------------------
# Specialization
# ---------------------------------------------------------------------------

def test_specializing_to_links_to_prior_formula():
    g = MathGraph(book_id="t")
    g.ingest_formula(nid="general",
                     latex=r"K(f, g) = \int f(x) g(x) k(x, x) dx")
    g.ingest_formula(nid="special",
                     latex=r"K(f, f) = \int f(x)^2 k(x, x) dx")
    rep = enrich_clause(
        g, seq=10,
        text="In the special case where g = f we obtain K of f, f.",
        home_nid="ch5",
        formula_ids_in_clause=["special"],
    )
    spec = {(e.src, e.dst) for e in g.edges if e.type == "specializes"}
    assert ("special", "general") in spec
    assert rep["new_specializations"] >= 1


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

def test_enrich_is_idempotent():
    g = MathGraph(book_id="t")
    g.ingest_formula(nid="eq42", latex=r"\sum L(y_i, f(x_i))",
                     cite_labels=["5.42"])
    g.ingest_formula(nid="eq48",
                     latex=r"\min_{f \in HK} \dots + \lambda \|f\|^2",
                     cite_labels=["5.48"])
    text = "Rewriting (5.42) we have …"
    enrich_clause(g, seq=22, text=text, home_nid="ch5",
                  formula_ids_in_clause=["eq48"])
    n1 = sum(1 for e in g.edges if e.type == "derived_from")
    enrich_clause(g, seq=22, text=text, home_nid="ch5",
                  formula_ids_in_clause=["eq48"])
    n2 = sum(1 for e in g.edges if e.type == "derived_from")
    assert n1 == n2  # no duplicates on re-run
