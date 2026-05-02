"""Phase 1 — rule-based math-graph enrichers.

Phase 0 (`sevim.math_graph`) gives us a deterministic skeleton of
``uses / binds / defines / about / references`` from raw LaTeX and
clause text.  Phase 1 layers semantic relationships on top — the
relations that need to be *inferred* from prose patterns rather than
parsed from a single formula:

  defines(Formula, Concept)        "where L is the loss function"
  instance_of(Var, Concept)         …same: Var L is an instance of
                                     Concept "loss function"
  derived_from(Formula, Formula)    "rewriting (5.42), we have …"
  specializes(Formula, Formula)     "in the special case where g = f"
  related_to(Formula, Formula)      "or equivalently", "the same as"

Why a separate module:
  * Keeps the deterministic Phase-0 extractor pure and tiny.
  * Lets us toggle Phase 1 off for tests / regressions.
  * Phase 2 (OntoMathPRO grounding) attaches an IRI to the same
    Concept nodes Phase 1 creates — so this layer establishes the
    Concept node identity that later phases enrich.

Enrichers are *idempotent* — re-running them on the same clause
neither duplicates edges nor downgrades earlier rule strength.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from .math_graph import MathGraph, Concept


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Definitions are detected with a small DFA-ish two-stage scan
# (see ``_iter_definitions``):
#  1. A "primary" hit *must* be introduced by ``where|let|denote``
#     (otherwise plain prose like "it is a function" would match).
#  2. After a primary hit, conjunctions ``and / , / ;`` carry the
#     "where" semantics — so we run a *secondary* pattern (no prefix
#     required) on the rest of the sentence to pick up
#     "and J is the penalty functional".
_DEFINE_PRIMARY_RE = re.compile(
    r"(?:where|with|let|denote\s+by|denoting)\s+"
    r"([A-Za-z\\]+(?:\s+of\s+[a-z]+)?)"
    r"\s+(?:is|denotes?|denote|represents?)\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    r"([a-z][a-z\s\-]{2,40}?)"
    r"(?=[.,;]|\s+(?:and|where|with|so|on|but|in)\b|$)",
    re.IGNORECASE,
)
_DEFINE_SECONDARY_RE = re.compile(
    r"(?:^|[,;]|\band\b)\s*"
    r"([A-Za-z\\]+(?:\s+of\s+[a-z]+)?)"
    r"\s+(?:is|denotes?|denote|represents?)\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    r"([a-z][a-z\s\-]{2,40}?)"
    r"(?=[.,;]|\s+(?:and|where|with|so|on|but|in)\b|$)",
    re.IGNORECASE,
)


def _iter_definitions(text: str):
    """Yield ``(var_phrase, concept_phrase)`` pairs for every
    define-style match in *text*.  Requires a primary "where/let"
    hit first; subsequent conjunctions then admit the secondary
    pattern."""
    primary_hits = list(_DEFINE_PRIMARY_RE.finditer(text))
    if not primary_hits:
        return
    for m in primary_hits:
        yield m.group(1), m.group(2)
    # Run the secondary scan on the tail of the FIRST primary hit so
    # ``and J is the penalty functional`` inside the same sentence is
    # caught.  We bound by the next ``where|let|`` keyword to avoid
    # bleeding into a fresh definition cluster.
    tail_start = primary_hits[0].end()
    tail = text[tail_start:]
    for m in _DEFINE_SECONDARY_RE.finditer(tail):
        yield m.group(1), m.group(2)

# "rewriting (5.42)" / "from (5.42)" / "starting from Equation 5.42"
# / "in (5.42)" / "applying (5.42)"
_DERIVATION_PATTERNS = [
    re.compile(
        r"(?:rewriting|substituting\s+into|from|using|applying|"
        r"starting\s+from|deriving\s+from|combining)\s+"
        r"(?:Eq\.?\s*|Equation\s*)?"
        r"\(?(\d+\.\d+[a-z]?)\)?",
        re.IGNORECASE,
    ),
]

# "or equivalently", "equivalently", "the same as"
_EQUIVALENCE_PATTERNS = [
    re.compile(r"\bor\s+equivalently\b", re.IGNORECASE),
    re.compile(r"\bequivalently\b", re.IGNORECASE),
    re.compile(r"\bsame\s+as\b\s*(?:Eq\.?\s*|Equation\s*)?\(?(\d+\.\d+)\)?",
               re.IGNORECASE),
]

# "in the special case where x = 0" / "specializing to f = g"
# The connector word ("where", "to", "with") is optional so we can
# catch both phrasings.
_SPECIALIZE_PATTERNS = [
    re.compile(
        r"(?:special\s+case|specializing|specialization|setting|"
        r"taking|when)\s+"
        r"(?:where|to|with)?\s*"
        r"([A-Za-z\\]+\s*=\s*[A-Za-z0-9\\]+)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

# Stop-words inside the captured concept phrase that signal we
# overshot (the prose continues into another clause).
_PHRASE_STOP_HEAD = frozenset({
    "and", "or", "but", "since", "because", "while", "as", "to",
})


def _clean_concept(name: str) -> str:
    """Normalise a concept phrase: strip articles, trailing punctuation,
    and stop after the first conjunction so we don't capture half a
    sentence."""
    if not name:
        return ""
    s = name.strip().rstrip(".,;:")
    s = re.sub(r"^(?:a|an|the)\s+", "", s, flags=re.IGNORECASE)
    # Truncate at the first stop-word.
    parts = s.split()
    out = []
    for w in parts:
        if w.lower() in _PHRASE_STOP_HEAD and out:
            break
        out.append(w)
    return " ".join(out).lower().strip()


def _canonical_var_token(raw: str) -> str:
    """Best-effort extraction of a Var name from a phrase like
    ``L of y`` / ``\\lambda`` / ``HK``."""
    from .math_graph import canonical_var_name
    if not raw:
        return ""
    s = raw.strip()
    # If the phrase is "X of …", keep only the head.
    head = re.match(r"\\?[A-Za-z]+", s)
    if not head:
        return ""
    return canonical_var_name(head.group(0))


# ---------------------------------------------------------------------------
# Enricher entry point
# ---------------------------------------------------------------------------

def enrich_clause(
    g: MathGraph, *,
    seq: int,
    text: str,
    home_nid: str,
    formula_ids_in_clause: Iterable[str],
    citation_to_formula_id: Optional[dict[str, str]] = None,
) -> dict:
    """Run all Phase-1 rule enrichers on *text*.

    Args:
      g: the persistent math graph; mutated in place.
      seq: clause sequence (used to look up the Passage node).
      text: the **spoken** form of the clause (post-sanitization);
        Phase-0 stored this in the Passage node already.
      home_nid: passage anchor.
      formula_ids_in_clause: nids emitted while processing this
        clause; used as the "source" of newly-inferred derivations
        when the rule names a citation that points to an *earlier*
        formula on the chalkboard.
      citation_to_formula_id: optional mapping ``"5.42"`` →
        chalkboard nid for fast lookups; built from ``g.formulas``
        when omitted.

    Returns a small report dict for telemetry.
    """
    if citation_to_formula_id is None:
        citation_to_formula_id = _build_citation_lookup(g)

    pid = f"p:{seq}@{home_nid}" if home_nid else f"p:{seq}"
    new_concepts = 0
    new_definitions = 0
    new_derivations = 0
    new_specializations = 0
    new_equivalences = 0

    formula_ids = list(formula_ids_in_clause)

    # ------ definitions ----------------------------------------------------
    for var_phrase, concept_phrase in _iter_definitions(text):
        concept_phrase = _clean_concept(concept_phrase)
        if not concept_phrase or len(concept_phrase) < 3:
            continue
        var_token = _canonical_var_token(var_phrase)
        if not var_token:
            continue
        v = g.get_or_make_var(var_token)
        c = g.get_or_make_concept(concept_phrase,
                                  aliases=[var_phrase.strip().lower()])
        new_concepts += 1
        g.add_edge(v.id, "instance_of", c.id,
                   meta={"clause": pid, "source": "rule:define"})
        new_definitions += 1
        # If there's a Formula in this clause, attach the concept
        # to that formula too (e.g. the "loss function" concept
        # belongs to the formula card that contains L).
        for fid in formula_ids:
            # Only link when the Formula actually uses the var.
            if any(e.dst == v.id and e.type == "uses"
                   for e in g.out_edges(fid, "uses")):
                g.add_edge(fid, "defines", c.id,
                           meta={"clause": pid,
                                 "source": "rule:define"})

    # ------ derivations ----------------------------------------------------
    for pat in _DERIVATION_PATTERNS:
        for m in pat.finditer(text):
            cite = m.group(1)
            src_fid = citation_to_formula_id.get(cite)
            if not src_fid:
                continue
            for tgt_fid in formula_ids:
                if tgt_fid == src_fid:
                    continue
                g.add_edge(tgt_fid, "derived_from", src_fid,
                           meta={"clause": pid, "label": cite,
                                 "source": "rule:derived"})
                new_derivations += 1

    # ------ equivalence ----------------------------------------------------
    # When the clause contains "or equivalently" / "equivalently",
    # link this clause's formulas pairwise as related_to with a
    # stronger weight than the ambient paired_in_clause.
    for pat in _EQUIVALENCE_PATTERNS:
        if pat.search(text):
            for i, a in enumerate(formula_ids):
                for b in formula_ids[i + 1:]:
                    g.add_edge(a, "related_to", b, weight=2.0,
                               meta={"clause": pid,
                                     "source": "rule:equivalent"})
                    g.add_edge(b, "related_to", a, weight=2.0,
                               meta={"clause": pid,
                                     "source": "rule:equivalent"})
                    new_equivalences += 1
            break

    # ------ specialization -------------------------------------------------
    for pat in _SPECIALIZE_PATTERNS:
        for m in pat.finditer(text):
            # We can't always identify which formulas are the parent
            # vs the specialization, but if the clause has at least
            # one formula AND the immediately-preceding clause's
            # formulas exist on the graph, we can call the new ones a
            # specialization of the previous.  Use the most-recent
            # Formula on the graph as the parent.
            parent = _most_recent_formula_other_than(
                g, formula_ids,
            )
            if not parent:
                continue
            for tgt_fid in formula_ids:
                if tgt_fid == parent:
                    continue
                g.add_edge(tgt_fid, "specializes", parent,
                           meta={"clause": pid,
                                 "expr": m.group(1),
                                 "source": "rule:specialize"})
                new_specializations += 1

    return {
        "passage": pid,
        "new_concepts": new_concepts,
        "new_definitions": new_definitions,
        "new_derivations": new_derivations,
        "new_equivalences": new_equivalences,
        "new_specializations": new_specializations,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_citation_lookup(g: MathGraph) -> dict[str, str]:
    """Map ``"5.42"`` → most-recent Formula nid that cites it."""
    out: dict[str, str] = {}
    for fid, f in g.formulas.items():
        for lab in f.cite_labels:
            out[lab] = fid    # last-write wins; OK for derivation
    return out


def _most_recent_formula_other_than(
    g: MathGraph, exclude: Iterable[str],
) -> Optional[str]:
    excl = set(exclude)
    # Iterate insertion order in reverse — Python's dict preserves it.
    for fid in reversed(list(g.formulas.keys())):
        if fid in excl:
            continue
        return fid
    return None
