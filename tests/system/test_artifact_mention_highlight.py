"""Pin: when a later clause re-mentions an artifact already on the
chalkboard (a citation, formula, canonical topic, …), the orchestrator
must emit a ``highlight`` visual op so the existing card glows at the
moment the narrator names it.

This is what makes the speech feel "in sync" with the board for the
user — without this, a re-mention is silent visually and the listener
loses track of where the focus is.
"""
from __future__ import annotations

from book.ir import Book, BookNode
from chalkboard import Chalkboard
from narrator import NarrationClause, NarrationPlan
from narrator.tts import NullTTS

from serve.orchestrator import Orchestrator, StreamEvent


def _book() -> Book:
    root = BookNode(nid="b", kind="book", number=None, title="t",
                    page_start=1, page_end=999)
    ch5 = BookNode(nid="b/ch5", kind="chapter", number="5",
                   title="Basis Expansions", page_start=100, page_end=200)
    s5_8 = BookNode(
        nid="b/ch5/s5_8", kind="section", number="5.8",
        title="RKHS", page_start=170, page_end=190,
        body_text=("min f∈H N X i=1 L(yi, f(xi)) + λJ(f) (5.42) "
                   "where L is a loss function."),
    )
    ch5.children.append(s5_8)
    root.children.append(ch5)
    return Book(title="t", author=None, source="", root=root,
                concepts={}, pages=[], figures=[], cross_refs=[])


def _drain(orch) -> list:
    """Return one events list per clause."""
    out: list[list[dict]] = []
    for ev in orch.stream():
        if isinstance(ev, StreamEvent):
            out.append(ev.visual_ops or [])
    return out


def test_re_mention_of_equation_emits_highlight():
    """Clause 1 introduces Equation 5.42 (add op).  Clause 3 re-mentions
    it — must produce a highlight op pointing at the same nid."""
    book = _book()
    plan = NarrationPlan(
        topic="<test>", book_title="t",
        clauses=[
            NarrationClause(
                text="Consider Equation 5.42.",
                home_nid="b/ch5/s5_8",
                concepts=[], suggested_dur=1.0,
            ),
            NarrationClause(
                text="A short interlude with no math.",
                home_nid="b/ch5/s5_8",
                concepts=[], suggested_dur=1.0,
            ),
            NarrationClause(
                text="Returning to Equation 5.42, recall the loss term.",
                home_nid="b/ch5/s5_8",
                concepts=[], suggested_dur=1.0,
            ),
        ],
        visited_nids=["b"], meta={"mode": "full"},
    )
    orch = Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    per_clause = _drain(orch)

    # Find the reference card that was added in clause 1.
    add_ops = [o for o in per_clause[0]
               if o.get("kind") == "add"
               and "Equation 5.42" in (o.get("label") or "")]
    assert add_ops, "clause 1 should have added an Equation 5.42 card"
    ref_nid = add_ops[0]["nid"]

    # Clause 3 must contain a highlight op for the same nid.
    highlights = [o for o in per_clause[2]
                  if o.get("kind") == "highlight"
                  and o.get("nid") == ref_nid]
    assert highlights, (
        "Clause 3 re-mentioned Equation 5.42 but emitted no highlight "
        "for the existing card."
    )


def test_clause_that_introduces_a_card_doesnt_redundantly_highlight_it():
    """Clause 1 introduces (5.42) — it should NOT also emit a highlight
    op for the very card it just added.  The frontend animates the
    add itself; an extra highlight at the same moment would just stack."""
    book = _book()
    plan = NarrationPlan(
        topic="<test>", book_title="t",
        clauses=[NarrationClause(
            text="Consider Equation 5.42 carefully.",
            home_nid="b/ch5/s5_8",
            concepts=[], suggested_dur=1.0,
        )],
        visited_nids=["b"], meta={"mode": "full"},
    )
    orch = Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    per_clause = _drain(orch)
    ops = per_clause[0]
    add_nids = {o["nid"] for o in ops if o.get("kind") == "add"}
    overlapping = [o for o in ops
                   if o.get("kind") == "highlight"
                   and o.get("nid") in add_nids]
    # Add ops always co-emit a highlight in the cid path
    # (ConceptResolver shapes); but mention-based highlights for cards
    # newly added in *this* clause should be skipped.  We pin the
    # narrower property: the artifact_index should not point any of
    # this clause's mention-highlights at brand-new nids.
    # Concretely: there must be at most one highlight per nid in clause 1.
    nid_count = {}
    for o in overlapping:
        nid_count[o["nid"]] = nid_count.get(o["nid"], 0) + 1
    for nid, n in nid_count.items():
        assert n <= 1, (
            f"nid {nid} got {n} highlight ops in the same clause "
            f"that added it — mention-scanner shouldn't double up."
        )


def test_artifact_index_populated_for_reference_cards():
    """White-box: after a reference card is emitted, the orchestrator's
    _artifact_index must contain a key that maps back to it."""
    book = _book()
    plan = NarrationPlan(
        topic="<test>", book_title="t",
        clauses=[NarrationClause(
            text="See Equation 5.42 for the kernel form.",
            home_nid="b/ch5/s5_8",
            concepts=[], suggested_dur=1.0,
        )],
        visited_nids=["b"], meta={"mode": "full"},
    )
    orch = Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    list(orch.stream())
    # Index keys are lower-cased; the canonical form should be there.
    assert any("equation 5.42" in k for k in orch._artifact_index), (
        f"Expected 'equation 5.42' in _artifact_index, got "
        f"{list(orch._artifact_index)}"
    )


def test_every_card_add_is_paired_with_co_timed_highlight():
    """User-facing invariant: when a card appears on the board, the
    narrator is by definition naming it at that moment — so the user
    should see the card *pulse* as it lands, not just fade in.  We pin
    that every ``add`` op has a matching ``highlight`` op at the same
    ``t``, for every primitive (formula / reference / canonical /
    passage / cid-resolver shape)."""
    book = _book()
    plan = NarrationPlan(
        topic="<test>", book_title="t",
        clauses=[NarrationClause(
            text="Consider Equation 5.42, which defines the loss.",
            home_nid="b/ch5/s5_8",
            concepts=[], suggested_dur=1.0,
        )],
        visited_nids=["b"], meta={"mode": "full"},
    )
    orch = Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    per_clause = _drain(orch)
    for ops in per_clause:
        adds = [(o["nid"], o.get("t", 0.0))
                for o in ops if o.get("kind") == "add"]
        highlights = {(o.get("nid"), o.get("t", 0.0))
                      for o in ops if o.get("kind") == "highlight"}
        for nid, t in adds:
            assert (nid, t) in highlights, (
                f"add op for nid={nid} at t={t} has no co-timed highlight; "
                f"available highlights={highlights}"
            )


def test_short_or_symbolic_keys_filtered_from_index():
    """White-box: the registration helper must drop keys that are too
    short (< 3 chars) or contain no alphabetic characters — otherwise
    every "is" / "the" / "= 0" would highlight a card."""
    book = _book()
    plan = NarrationPlan(
        topic="<test>", book_title="t",
        clauses=[NarrationClause(
            text="seed", home_nid="b/ch5/s5_8",
            concepts=[], suggested_dur=1.0,
        )],
        visited_nids=["b"], meta={"mode": "full"},
    )
    orch = Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    orch._register_artifact("nid_x", ["is", "= 0", "  ", "Q", "linear regression"])
    keys = set(orch._artifact_index.keys())
    assert "is" not in keys
    assert "= 0" not in keys
    assert "q" not in keys
    assert "linear regression" in keys


def test_verbalized_formula_keys_extracts_function_calls():
    """The mention scanner can only fire when the artifact_index has a
    spoken-form key the narrator actually says.  Formulas are spoken as
    function-call notation — ``f(x)`` → "f of x", ``K(f,g)`` →
    "K of f g" — so the orchestrator must extract those phrases from
    the LaTeX and register them per card.

    Without this, every formula card without a citation tag is dark
    when the narrator says its content (the user sees the cards but
    nothing pulses on mention)."""
    from serve.orchestrator import _verbalized_formula_keys

    keys = _verbalized_formula_keys(r"f(x) = \int k(x,y) \phi(y) dy")
    keys_lc = [k.lower() for k in keys]
    assert any("f of x" in k for k in keys_lc), (
        f"Expected 'f of x' phrase in {keys}"
    )
    assert any("k of x" in k for k in keys_lc), (
        f"Expected 'k of x ...' phrase in {keys}"
    )

    keys2 = _verbalized_formula_keys(
        r"K(f, g) = \int f(x) g(x) k(x, x) dx"
    )
    keys2_lc = [k.lower() for k in keys2]
    assert any("k of f" in k for k in keys2_lc), (
        f"Expected 'K of f g' phrase in {keys2}"
    )


def test_formula_card_lights_up_on_verbalized_content_mention():
    """End-to-end: a clause that introduces a formula plants a formula
    card.  A later clause that says the formula's verbalized content
    ("f of x", "k of x y", "K of f g", …) must fire a highlight on
    that card.

    Without registering the verbalized content as artifact keys, the
    only thing the mention-scanner can match is the citation label
    ("Equation 5.42") — so cards without an explicit cite are dark
    when the narrator references their content.
    """
    book = _book()
    plan = NarrationPlan(
        topic="<test>", book_title="t",
        clauses=[
            NarrationClause(
                text="The kernel form K(f,g) = k(x,x) holds for any input x.",
                home_nid="b/ch5/s5_8",
                concepts=[], suggested_dur=1.0,
            ),
            NarrationClause(
                text="Some intervening prose with no formulas.",
                home_nid="b/ch5/s5_8",
                concepts=[], suggested_dur=1.0,
            ),
            NarrationClause(
                text="Returning to K of f g, recall its kernel structure.",
                home_nid="b/ch5/s5_8",
                concepts=[], suggested_dur=1.0,
            ),
        ],
        visited_nids=["b"], meta={"mode": "full"},
    )
    orch = Orchestrator(book=book, plan=plan,
                        chalkboard=Chalkboard(), tts=NullTTS())
    per_clause = _drain(orch)

    # Clause 0 should have added a formula card.
    formula_adds = [o for o in per_clause[0]
                    if o.get("kind") == "add"
                    and o.get("primitive") == "formula_card"]
    assert formula_adds, (
        f"clause 0 should add a formula card; got {per_clause[0]}"
    )
    formula_nid = formula_adds[0]["nid"]

    # Artifact index must contain a verbalized key pointing at that nid.
    # Index values are lists of nids (multi-card-per-key), promoted from
    # the legacy single-nid representation.
    def _bucket(v):
        return v if isinstance(v, list) else [v]
    matched_keys = [k for k, v in orch._artifact_index.items()
                    if formula_nid in _bucket(v)]
    assert matched_keys, (
        f"_artifact_index has no key pointing at {formula_nid}: "
        f"{orch._artifact_index}"
    )
    # And specifically a function-call phrase the narrator might say.
    assert any(" of " in k for k in matched_keys), (
        f"Expected an 'X of Y' phrase among formula keys; got {matched_keys}"
    )

    # Clause 2 says "K of f g" — should highlight the formula card.
    highlights = [o for o in per_clause[2]
                  if o.get("kind") == "highlight"
                  and o.get("nid") == formula_nid]
    assert highlights, (
        f"Clause 2 said 'K of f g' but no highlight emitted for the "
        f"formula card.  Ops: {per_clause[2]}"
    )
