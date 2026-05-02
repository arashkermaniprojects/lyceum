"""Phase-0 math semantic graph smoke tests.

Just enough to lock the contract the orchestrator + frontend will
plug into: nodes round-trip through JSON, edges are extracted from
LaTeX correctly, and the layout helper finds the right anchor.
"""
from __future__ import annotations

import os
import tempfile

from sevim.math_graph import (
    MathGraph,
    canonical_var_name,
    parse_formula_vars,
    _name_mentions,
    graph_path_for_book,
)


# ---------------------------------------------------------------------------
# Variable-token canonicalisation
# ---------------------------------------------------------------------------

def test_canonical_var_name_basics():
    assert canonical_var_name("f") == "f"
    assert canonical_var_name("\\lambda") == "lambda"
    assert canonical_var_name("λ") == "λ"
    assert canonical_var_name("x_i") == "x"
    assert canonical_var_name("f^{T}") == "f"
    assert canonical_var_name("123") == ""
    assert canonical_var_name("\\sum") == ""    # bigop, not a var
    assert canonical_var_name("\\frac") == ""
    assert canonical_var_name("HK") == "HK"     # ALL-CAPS multi-letter ok


# ---------------------------------------------------------------------------
# Formula → (used, bound, defined)
# ---------------------------------------------------------------------------

def test_parse_formula_vars_function_def():
    used, bound, defined = parse_formula_vars(r"f(x) = \int k(x,y) \phi(y) dy")
    # ``f`` is defined (LHS function head).
    assert "f" in defined
    # Used variables include f, x, k, y, phi.
    assert {"f", "x", "k", "y", "phi"} <= used
    # ``y`` is the differential variable → bound.
    assert "y" in bound


def test_parse_formula_vars_sum_binder():
    used, bound, defined = parse_formula_vars(
        r"k(x,y) = \sum_{i=1}^n \alpha_i k(x_i, y_i)"
    )
    assert "k" in defined
    assert "i" in bound       # \sum_{i=...}
    # alpha + k + x + y all used.
    assert {"k", "x", "y", "alpha"} <= used


def test_parse_formula_vars_norm():
    # Eq 5.48 surface form
    used, _, _ = parse_formula_vars(r"\min_{f \in HK} L(y_i, f(x_i)) + \lambda \|f\|^2_{HK}")
    assert {"f", "L", "y", "x", "lambda", "HK"} <= used


# ---------------------------------------------------------------------------
# Graph: ingest formula + edges
# ---------------------------------------------------------------------------

def test_ingest_formula_creates_uses_edges():
    g = MathGraph(book_id="t")
    g.ingest_formula(nid="n_f", latex=r"f(x) = \int k(x,y) \phi(y) dy")
    # The Formula node lands.
    assert "n_f" in g.formulas
    # ``uses`` edges to f, x, k, y, phi.
    used_var_ids = {e.dst for e in g.out_edges("n_f", "uses")}
    expected = {g.get_or_make_var(v).id for v in ("f", "x", "k", "y", "phi")}
    assert expected <= used_var_ids
    # ``defines(f)``.
    defines = {e.dst for e in g.out_edges("n_f", "defines")}
    assert g.vars["v:f"].id in defines


def test_shared_vars_anchor():
    """When two formulas share variables, ``best_anchor_for`` returns
    the one with the largest overlap — that's what the layout uses."""
    g = MathGraph(book_id="t")
    g.ingest_formula(nid="A", latex=r"f(x) = a + b")
    g.ingest_formula(nid="B", latex=r"g(x) = c + d")     # shares x
    g.ingest_formula(nid="C", latex=r"f(y) + g(y)")      # shares f,g
    # C's best anchor should be B (or A) — anything sharing > 0 vars
    # with C.  Strongly-shared neighbours preferred over weakly-shared.
    anchor = g.best_anchor_for("C")
    assert anchor in {"A", "B"}
    # Symmetric: A's best anchor is B (shares x), not C (shares only f).
    anchor_a = g.best_anchor_for("A")
    assert anchor_a == "B"


# ---------------------------------------------------------------------------
# Citation cross-link
# ---------------------------------------------------------------------------

def test_citation_reference_edge():
    g = MathGraph(book_id="t")
    # Eq 5.42 is added first.
    g.ingest_formula(nid="eq42", latex=r"\sum L(y_i, f(x_i)) + \lambda J(f)",
                     cite_labels=["5.42"])
    # Eq 5.48 cites 5.42 → references edge points eq48 → eq42.
    g.ingest_formula(nid="eq48", latex=r"\min_{f \in HK} \dots + \lambda \|f\|^2",
                     cite_labels=["5.48", "5.42"])
    refs = {(e.src, e.dst) for e in g.edges if e.type == "references"}
    assert ("eq48", "eq42") in refs


# ---------------------------------------------------------------------------
# Passage ingestion
# ---------------------------------------------------------------------------

def test_passage_about_formula_and_var():
    g = MathGraph(book_id="t")
    g.ingest_formula(nid="eq42",
                     latex=r"\sum L(y_i, f(x_i)) + \lambda J(f)")
    p = g.ingest_passage(
        seq=9,
        text="we have min f in H N X i=1 L of yi, f of xi + lambda J of f",
        home_nid="b/ch5/s5_8",
        formula_ids_in_clause=["eq42"],
    )
    abouts = [(e.src, e.dst) for e in g.edges if e.type == "about"]
    # Passage → Formula
    assert (p.id, "eq42") in abouts
    # Passage → Vars (J, L, f, lambda are mentioned via "X of Y").
    var_dsts = {e.dst for e in g.out_edges(p.id, "about")
                if e.dst.startswith("v:")}
    assert {"v:J", "v:L", "v:f", "v:lambda"} <= var_dsts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_round_trip_json():
    g = MathGraph(book_id="ESLII")
    g.ingest_formula(nid="A", latex="y = m x + b", cite_labels=["3.1"])
    g.ingest_passage(seq=1, text="y of x", home_nid="ch3",
                     formula_ids_in_clause=["A"])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.json")
        g.save(path)
        g2 = MathGraph.load(path)
    assert g2.book_id == "ESLII"
    assert "A" in g2.formulas
    assert g2.formulas["A"].cite_labels == ["3.1"]
    # Edges round-trip.
    edge_types = {e.type for e in g2.edges}
    assert {"uses", "defines", "about"} <= edge_types


def test_graph_path_for_book():
    p = graph_path_for_book("books/ESLII.json")
    assert p == "books/ESLII.math_graph.json"


# ---------------------------------------------------------------------------
# Mention scanner
# ---------------------------------------------------------------------------

def test_name_mentions_picks_up_function_calls_and_greek():
    text = "we have f of x = K of f g + lambda J of f"
    mentions = set(_name_mentions(text))
    assert {"f", "K", "lambda", "J"} <= mentions


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

def test_find_subexpression_parent_strict_substring():
    """``J(f)`` lives inside ``\\sum L(yi, f(xi)) + λ J(f)``.  Adding
    ``J(f)`` later should detect the parent and let the orchestrator
    fold it into the existing card instead of duplicating."""
    from sevim.math_graph import find_subexpression_parent
    g = MathGraph(book_id="t")
    g.ingest_formula(nid="big",
                     latex=r"\sum L(y_i, f(x_i)) + \lambda J(f)")
    parent = find_subexpression_parent(g, r"J(f)")
    assert parent == "big"
    # Same-sized formulas don't subsume each other.
    g.ingest_formula(nid="exact", latex=r"J(f)")
    parent2 = find_subexpression_parent(g, r"J(f)")
    # ``big`` still longer; ``exact`` same length so doesn't subsume.
    assert parent2 == "big"
    # Truly unrelated formula yields None.
    assert find_subexpression_parent(g, r"y = m x + b") is None


def test_coverage_report_flags_math_passages_without_formula():
    g = MathGraph(book_id="t")
    # Passage 1: math content, has a formula attached → covered.
    g.ingest_formula(nid="A", latex="y = m x + b")
    g.ingest_passage(seq=1, text="we let y of x = m x + b",
                     home_nid="ch3",
                     formula_ids_in_clause=["A"])
    # Passage 2: math content, NO formula attached → uncovered.
    g.ingest_passage(seq=2, text="now compute lambda + alpha squared",
                     home_nid="ch3",
                     formula_ids_in_clause=[])
    # Passage 3: pure prose → ignored, not flagged uncovered.
    g.ingest_passage(seq=3, text="we begin with a brief outline",
                     home_nid="ch3",
                     formula_ids_in_clause=[])
    rep = g.coverage_report()
    assert rep["covered_passages"] == 1
    uncovered_texts = {u["text"] for u in rep["uncovered_passages"]}
    assert any("compute lambda" in t for t in uncovered_texts)
    # Pure prose passage should NOT be flagged.
    assert not any("brief outline" in t for t in uncovered_texts)
