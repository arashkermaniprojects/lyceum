"""Pin the cross-clause annotation attachment.

User explicitly asked: every math notation in the narration must
appear on the panel; ``where x is …`` declarations and equation
IDs must land *on the same box as the function*, even when the
formula was emitted in an earlier clause.

These tests run the orchestrator's ``_apply_orphan_annotations``
directly with synthetic formula state so we don't need a real book
or TTS pipeline.
"""
from __future__ import annotations

from book.ir import Book, BookNode
from chalkboard import Chalkboard
from narrator import NarrationClause, NarrationPlan
from narrator.tts import NullTTS
from serve.orchestrator import (
    Orchestrator,
    _formula_card_size,
    _formula_card_svg,
    _is_math_symbol,
    _variable_definitions_in,
)


def _orch() -> Orchestrator:
    root = BookNode(nid="b", kind="book", number=None, title="t",
                    page_start=1, page_end=10)
    book = Book(title="t", author=None, source="", root=root,
                concepts={}, pages=[], figures=[])
    plan = NarrationPlan(
        topic="<t>", book_title="t",
        clauses=[NarrationClause(text="x", home_nid="b",
                                  concepts=[], suggested_dur=1.0)],
        visited_nids=["b"], meta={"mode": "full"},
    )
    return Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())


def _seed_formula(orch: Orchestrator,
                  *, nid: str, frag: str, latex: str,
                  cites=(), var_defs=()) -> None:
    """Drop a formula card directly onto the chalkboard and register
    its state, mimicking what ``_inline_formula_visual_ops`` does."""
    cites = list(cites); var_defs = [tuple(t) for t in var_defs]
    w, h = _formula_card_size(frag, cites, var_defs)
    svg = _formula_card_svg(frag, latex, w, h,
                            cite_labels=cites, var_defs=var_defs)
    orch.chalkboard.add(
        nid=nid, svg_body=svg,
        primitive="formula_card", label=frag[:40],
        w=w, h=h,
        meta={"home_nid": "b", "fragment": frag, "latex": latex,
              "via": "inline", "cites": list(cites),
              "var_defs": [list(t) for t in var_defs]},
    )
    orch.seen_nids.add(nid)
    orch._last_function_nid = nid
    orch._function_card_state[nid] = {
        "kind": "formula", "fragment": frag, "latex": latex,
        "cites": list(cites), "var_defs": list(var_defs),
    }


# ---------------------------------------------------------------------------
# Update-op path: prior formula card gets new annotations
# ---------------------------------------------------------------------------

def test_var_def_in_later_clause_updates_prior_formula():
    orch = _orch()
    _seed_formula(orch, nid="f1", frag="f(x) = a x + b", latex="f(x) = a x + b")

    # Later clause introduces ``where b is the bias term``.
    ops = orch._apply_orphan_annotations(
        clause_text="and where b is the bias term, we minimize the error.",
        home_nid="b", seq=1, audio_dur=2.0, word_timestamps=[],
        current_ops=[],
    )
    assert len(ops) == 1
    op = ops[0]
    assert op["kind"] == "update"
    assert op["nid"] == "f1"
    # The new SVG must carry the variable-definition line.
    assert "b: the bias term" in op["svg"]
    # Stored state has the new var_def appended.
    state = orch._function_card_state["f1"]
    assert ("b", "the bias term") in state["var_defs"]


def test_citation_in_later_clause_updates_prior_formula():
    orch = _orch()
    _seed_formula(orch, nid="f1", frag="f(x) = a x + b", latex="f(x) = a x + b")

    ops = orch._apply_orphan_annotations(
        clause_text="this result follows from Equation 5.42 above.",
        home_nid="b", seq=2, audio_dur=2.0, word_timestamps=[],
        current_ops=[],
    )
    assert len(ops) == 1
    op = ops[0]
    assert op["kind"] == "update"
    # The card's header now includes the citation chip.
    assert "Equation 5.42" in op["svg"]
    # And the dedup-key was registered so the reference scanner
    # won't emit a duplicate reference card.
    assert "Equation::5.42" in orch.seen_refs


def test_no_double_attach_for_same_clause():
    """When the inline-formula path already emitted a card *this*
    clause that carries the annotations, the orphan-attacher must
    leave it alone (no duplicate update op)."""
    orch = _orch()
    # Simulate a same-clause emission: state recorded, op emitted.
    _seed_formula(orch, nid="f2", frag="y = m x", latex="y = m x",
                  var_defs=[("m", "the slope")])
    same_clause_op = {"kind": "add", "nid": "f2",
                      "primitive": "formula_card",
                      "svg": "<rect/>", "t": 0.0, "w": 100, "h": 50}

    ops = orch._apply_orphan_annotations(
        clause_text="y = m x, where m is the slope of the line.",
        home_nid="b", seq=0, audio_dur=2.0, word_timestamps=[],
        current_ops=[same_clause_op],
    )
    assert ops == []


# ---------------------------------------------------------------------------
# Math-note fallback: nothing on the board yet
# ---------------------------------------------------------------------------

def test_math_note_emitted_when_no_prior_function():
    orch = _orch()
    ops = orch._apply_orphan_annotations(
        clause_text="we begin where x is the input vector.",
        home_nid="b", seq=0, audio_dur=2.0, word_timestamps=[],
        current_ops=[],
    )
    assert len(ops) == 1
    op = ops[0]
    assert op["kind"] == "add"
    assert op["primitive"] == "math_note"
    assert "x: the input vector" in op["svg"]
    # The note becomes the new "last function nid" so future
    # annotations attach to it instead of orphaning again.
    assert orch._last_function_nid == op["nid"]


def test_no_orphan_emission_when_clause_has_no_annotations():
    orch = _orch()
    _seed_formula(orch, nid="f1", frag="y = m x", latex="y = m x")
    ops = orch._apply_orphan_annotations(
        clause_text="now consider a different approach entirely.",
        home_nid="b", seq=1, audio_dur=2.0, word_timestamps=[],
        current_ops=[],
    )
    assert ops == []


# ---------------------------------------------------------------------------
# Symbol regex: function-call notation
# ---------------------------------------------------------------------------

def test_var_def_captures_function_call_symbol():
    """``where L(y, f(x)) is the loss function`` should now bring
    back the full ``L(y, f(x))`` symbol, not just ``L``."""
    text = "where L(y, f(x)) is the loss function for the model."
    defs = _variable_definitions_in(text)
    assert defs == [("L(y, f(x))", "the loss function for the model")]


def test_var_def_captures_simple_call_symbol():
    text = "where J(f) is a penalty functional on the space."
    defs = _variable_definitions_in(text)
    assert defs == [("J(f)", "a penalty functional on the space")]


def test_is_math_symbol_accepts_function_calls():
    assert _is_math_symbol("f(x)")
    assert _is_math_symbol("J(f)")
    assert _is_math_symbol("L(y, f(x))")
    # Long English head before parens still rejected.
    assert not _is_math_symbol("parameters(weights)")


# ---------------------------------------------------------------------------
# Comma-separated where continuations + function-call symbols
# ---------------------------------------------------------------------------

def test_three_comma_separated_where_clauses_all_captured():
    """User-reported regression: ``where L(y, f(x)) is a loss function,
    J(f) is a penalty functional, and H is a space of functions on
    which J(f) is defined`` should yield all three definitions.
    Earlier the comma-only continuation for ``J(f)`` was dropped."""
    text = ("we minimize the criterion in (5.42) "
            "where L(y, f(x)) is a loss function, "
            "J(f) is a penalty functional, "
            "and H is a space of functions on which J(f) is defined.")
    defs = _variable_definitions_in(text)
    syms = [s for s, _ in defs]
    assert "L(y, f(x))" in syms
    assert "J(f)" in syms
    assert "H" in syms


def test_function_arg_subscripts_dont_block_math_detection():
    """Bug: ``L(yi, f(xi)) + λJ(f)`` was rejected as pseudocode
    because ``yi`` and ``xi`` were counted as English word tokens.
    Function-arg subscripts inside parens should not block math
    detection."""
    from serve.orchestrator import _detect_math_fragments
    text = ("we have the form min f∈H N X i=1 L(yi, f(xi)) + λJ(f) "
            "where L(y, f(x)) is a loss function.")
    fragments = _detect_math_fragments(text)
    assert fragments, "expected at least one math fragment"
    frag_text = " ".join(f for f, _ in fragments)
    assert "λJ(f)" in frag_text
    assert "L(yi" in frag_text or "L(yi," in frag_text
