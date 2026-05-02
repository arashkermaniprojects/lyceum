"""Phase-0 math semantic graph.

Nodes
-----
  Var       - a single mathematical symbol (``f``, ``x``, ``K``, ``λ``).
  Formula   - one LaTeX expression as it appears on the chalkboard.
  Passage   - a narration clause / passage that *talks about* math.
  Concept   - free-form named concept (e.g. "kernel function") with
              our own internal ID.  Phase 2 may later attach an
              OntoMathPRO IRI alongside.

Edges (deterministic in this phase)
-----------------------------------
  uses(Formula, Var)            free variables of the formula
  binds(Formula, Var)            variables introduced by Σ / ∫ / ∀ / λ
  defines(Formula, Var)          variable that appears as the LHS
  references(Formula, Formula)   citation-label match (Eq 5.48 → 5.42)
  about(Passage, Formula)        a passage clause mentions a formula
  about(Passage, Var)            a passage clause names a single variable
  paired_in_clause(Formula, F2)  formulas emitted in the same clause
  shared_vars(Formula, F2)       *derived* — count of overlapping uses

Persistence
-----------
  Per-book JSON next to the corpus file:
      books/<book>.math_graph.json
  The graph is **enriched over time**: each session loads the existing
  graph, adds new nodes / edges, and writes it back.  Existing edges
  are never silently dropped — only superseded when the same source
  edge re-emerges with a stricter type.

Deliberately *not* in this phase
--------------------------------
  Phase 1 — rule-based ``derived_from`` / ``specializes``.
  Phase 2 — OntoMathPRO IRI grounding (Concept.iri).
  Phase 3 — Tangent-CFT embeddings on Formula nodes.
  Phase 4 — local-Qwen-extracted triples.

The data layer here makes those phases additive: each future phase
appends edge types or fills in optional Concept fields without
needing to re-architect.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@dataclass
class Var:
    """A mathematical symbol.

    The name is normalised to a single canonical form so ``f``, ``f_i``,
    and ``f_{ij}`` all collapse to the same Var.  Subscripts /
    superscripts go on the *uses* / *binds* edges as decoration, not
    on the Var itself — otherwise a textbook with thousands of
    indexed forms (``x_1``, ``x_2``, …) would explode the graph.
    """
    id: str
    name: str
    # Cheap heuristic role tag — "function", "scalar", "vector",
    # "operator" — refined by the rule pass in Phase 1.
    role: str = "scalar"


@dataclass
class Formula:
    """One LaTeX expression.

    ``id`` is the chalkboard ``nid`` so the graph can be cross-
    referenced from visual ops.  Multiple Passage clauses can refer
    to the same Formula via separate ``about`` edges.
    """
    id: str
    latex: str
    surface: str = ""        # short label for UX (first ~40 chars)
    home_nid: str = ""        # passage anchor when known
    cite_labels: list[str] = field(default_factory=list)


@dataclass
class Passage:
    """A narration clause / passage."""
    id: str                   # e.g. f"clause_{seq}@{home_nid}"
    text: str
    home_nid: str = ""


@dataclass
class Concept:
    """Free-form mathematical concept (Phase-0).

    ``iri`` is left blank for now; Phase 2 fills it from OntoMathPRO.
    The user's locked decision is to keep our own internal IDs as the
    primary key and treat OntoMathPRO IRIs as an optional secondary
    grounding so we can compare coverage later.
    """
    id: str
    name: str
    iri: str = ""             # OntoMathPRO IRI (Phase 2)
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

# Edge types are plain strings in Phase 0.  Each edge is
# (src_id, type, dst_id, weight, meta).
_EDGE_TYPES = frozenset({
    "uses", "binds", "defines",
    "references", "about", "paired_in_clause",
    "shared_vars", "instance_of",
    # ``contains`` — a Formula whose LaTeX is structurally a
    # sub-expression of another (e.g. ``J(f)`` lives inside
    # ``L(yi, f(xi)) + λJ(f)``).  Used to fold the smaller formula
    # into the parent's card instead of duplicating it on the board.
    "contains",
    # Reserved for later phases — declared here so the persisted JSON
    # already accepts them without a schema change later.
    "derived_from", "specializes", "similar_to", "related_to",
})


@dataclass
class Edge:
    src: str
    type: str
    dst: str
    weight: float = 1.0
    meta: dict = field(default_factory=dict)


@dataclass
class MathGraph:
    """In-memory graph + JSON-on-disk persistence.

    The structure is intentionally flat (dicts of nodes + a list of
    edges) so it round-trips through JSON with no custom encoders.
    """
    book_id: str = ""
    vars: dict[str, Var] = field(default_factory=dict)
    formulas: dict[str, Formula] = field(default_factory=dict)
    passages: dict[str, Passage] = field(default_factory=dict)
    concepts: dict[str, Concept] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def get_or_make_var(self, name: str, role: str = "scalar") -> Var:
        canon = canonical_var_name(name)
        if not canon:
            return None  # type: ignore[return-value]
        vid = f"v:{canon}"
        v = self.vars.get(vid)
        if v is None:
            v = Var(id=vid, name=canon, role=role)
            self.vars[vid] = v
        elif role != "scalar" and v.role == "scalar":
            # Upgrade role when later evidence is more specific.
            v.role = role
        return v

    def get_or_make_concept(self, name: str,
                            aliases: Optional[list[str]] = None) -> Concept:
        cid = f"c:{name.strip().lower().replace(' ', '_')}"
        c = self.concepts.get(cid)
        if c is None:
            c = Concept(id=cid, name=name, aliases=list(aliases or []))
            self.concepts[cid] = c
        elif aliases:
            for a in aliases:
                if a not in c.aliases:
                    c.aliases.append(a)
        return c

    def add_formula(self, f: Formula) -> Formula:
        # Treat the chalkboard nid as canonical id; if the same nid
        # re-arrives, merge metadata rather than duplicate.
        existing = self.formulas.get(f.id)
        if existing is None:
            self.formulas[f.id] = f
            return f
        # Merge cite labels (don't lose accumulated annotations).
        for lab in f.cite_labels:
            if lab not in existing.cite_labels:
                existing.cite_labels.append(lab)
        if f.surface and not existing.surface:
            existing.surface = f.surface
        return existing

    def add_passage(self, p: Passage) -> Passage:
        self.passages.setdefault(p.id, p)
        return self.passages[p.id]

    # ------------------------------------------------------------------
    # Edge helpers
    # ------------------------------------------------------------------

    def add_edge(self, src: str, edge_type: str, dst: str, *,
                 weight: float = 1.0,
                 meta: Optional[dict] = None) -> None:
        if edge_type not in _EDGE_TYPES:
            raise ValueError(f"unknown edge type {edge_type!r}; "
                             f"add it to _EDGE_TYPES first")
        # Idempotent: don't accumulate identical (src, type, dst) edges
        # — bump the weight instead.
        for e in self.edges:
            if e.src == src and e.type == edge_type and e.dst == dst:
                e.weight += weight
                if meta:
                    e.meta.update(meta)
                return
        self.edges.append(Edge(src=src, type=edge_type, dst=dst,
                               weight=weight, meta=dict(meta or {})))

    def out_edges(self, src: str,
                  edge_type: Optional[str] = None) -> Iterator[Edge]:
        for e in self.edges:
            if e.src != src:
                continue
            if edge_type is None or e.type == edge_type:
                yield e

    def in_edges(self, dst: str,
                 edge_type: Optional[str] = None) -> Iterator[Edge]:
        for e in self.edges:
            if e.dst != dst:
                continue
            if edge_type is None or e.type == edge_type:
                yield e

    def shared_vars_count(self, fid_a: str, fid_b: str) -> int:
        a = {e.dst for e in self.out_edges(fid_a, "uses")}
        b = {e.dst for e in self.out_edges(fid_b, "uses")}
        return len(a & b)

    def formulas_using(self, var_id: str) -> list[str]:
        return [e.src for e in self.in_edges(var_id, "uses")]

    # ------------------------------------------------------------------
    # Population — given a freshly-emitted Formula card
    # ------------------------------------------------------------------

    def ingest_formula(self, *, nid: str, latex: str,
                       cite_labels: Optional[list[str]] = None,
                       home_nid: str = "") -> Formula:
        """Add a Formula node + its uses/binds/defines edges.

        This is the deterministic Phase-0 extractor: no LaTeXML, no
        embeddings — pure regex over the LaTeX surface.  It catches
        ~80 % of textbook formulas, which is enough to ground the
        Phase-0 UX features (clustering, var-level highlighting,
        coverage audit).
        """
        surface = (latex.strip()[:40]).strip()
        f = self.add_formula(Formula(
            id=nid, latex=latex, surface=surface,
            home_nid=home_nid,
            cite_labels=list(cite_labels or []),
        ))
        used, bound, defined = parse_formula_vars(latex)
        for name in used:
            v = self.get_or_make_var(name)
            if v is not None:
                self.add_edge(f.id, "uses", v.id)
        for name in bound:
            v = self.get_or_make_var(name)
            if v is not None:
                self.add_edge(f.id, "binds", v.id)
        for name in defined:
            v = self.get_or_make_var(name, role="function")
            if v is not None:
                self.add_edge(f.id, "defines", v.id)
        # Citation references: ``Eq 5.48`` was rewritten from 5.42 →
        # add a ``references`` edge to whatever earlier Formula
        # already cited 5.42.  Phase 1 sharpens this with derivation
        # patterns ("rewriting (5.42) we have").
        for lab in f.cite_labels:
            for other in self.formulas.values():
                if other.id == f.id:
                    continue
                if lab in other.cite_labels:
                    self.add_edge(f.id, "references", other.id,
                                  meta={"label": lab})
        return f

    def ingest_passage(self, *, seq: int, text: str,
                       home_nid: str = "",
                       formula_ids_in_clause: Iterable[str] = ()
                       ) -> Passage:
        """Record a narration clause + its ``about`` edges.

        ``formula_ids_in_clause`` is the list of Formula nodes the
        orchestrator emitted while processing this clause; we wire
        ``about(Passage, Formula)`` for each.  Phase 1 adds
        ``about(Passage, Var)`` based on which variable names appear
        in the (verbalized) text.
        """
        pid = f"p:{seq}@{home_nid}" if home_nid else f"p:{seq}"
        p = self.add_passage(Passage(id=pid, text=text, home_nid=home_nid))
        ids = list(formula_ids_in_clause)
        for fid in ids:
            self.add_edge(p.id, "about", fid)
        # ``paired_in_clause`` between every pair of formulas in the
        # same clause — symmetric, so add both directions.
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if a == b:
                    continue
                self.add_edge(a, "paired_in_clause", b)
                self.add_edge(b, "paired_in_clause", a)
        # Variable-name mentions in the text — the verbalizer emits
        # ``f of x``, ``K of f g`` so we look for "X of " and "X(".
        for var_name in _name_mentions(text):
            v = self.get_or_make_var(var_name)
            if v is not None:
                self.add_edge(p.id, "about", v.id)
        return p

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "vars": {k: v.__dict__ for k, v in self.vars.items()},
            "formulas": {k: v.__dict__ for k, v in self.formulas.items()},
            "passages": {k: v.__dict__ for k, v in self.passages.items()},
            "concepts": {k: v.__dict__ for k, v in self.concepts.items()},
            "edges": [e.__dict__ for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MathGraph":
        g = cls(book_id=data.get("book_id", ""))
        for k, v in (data.get("vars") or {}).items():
            g.vars[k] = Var(**v)
        for k, v in (data.get("formulas") or {}).items():
            g.formulas[k] = Formula(**v)
        for k, v in (data.get("passages") or {}).items():
            g.passages[k] = Passage(**v)
        for k, v in (data.get("concepts") or {}).items():
            g.concepts[k] = Concept(**v)
        for e in data.get("edges") or []:
            g.edges.append(Edge(
                src=e["src"], type=e["type"], dst=e["dst"],
                weight=e.get("weight", 1.0),
                meta=dict(e.get("meta") or {}),
            ))
        return g

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, *, book_id: str = "") -> "MathGraph":
        if not os.path.isfile(path):
            return cls(book_id=book_id)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        g = cls.from_dict(data)
        if book_id and not g.book_id:
            g.book_id = book_id
        return g

    # ------------------------------------------------------------------
    # Query helpers — used by orchestrator / frontend layout / mention
    # scanner.
    # ------------------------------------------------------------------

    def coverage_report(self) -> dict:
        """Return a dict summarising math coverage for the graph.

        Phase-0 definition of "covered": a Passage is covered when it
        has at least one ``about(Passage, Formula)`` edge.  Phase-1
        will sharpen this with named-formula heuristics, but this is
        already enough to surface the user's "every math notation in
        the text should be on the panel" complaint.
        """
        # Snapshot before iterating: the orchestrator runs main + tangent
        # generators in separate request threads (ThreadingHTTPServer) and
        # both can call ingest_passage / ingest_formula / add_edge while
        # this report is being built.  Without the snapshot, Python raises
        # ``dictionary changed size during iteration`` mid-traversal and
        # the per-clause graph_stats payload for that clause is silently
        # dropped (the orchestrator catches the error so the clause itself
        # still emits, but the coverage badge stops updating).
        edges_snap = list(self.edges)
        passages_snap = list(self.passages.items())
        about_formula: dict[str, list[str]] = {}
        for e in edges_snap:
            if e.type != "about":
                continue
            if e.dst.startswith("v:"):
                continue   # var mentions, not formula coverage
            about_formula.setdefault(e.src, []).append(e.dst)
        covered = []
        uncovered = []
        for pid, p in passages_snap:
            if about_formula.get(pid):
                covered.append(pid)
            else:
                # Only flag as uncovered when the passage *looks* math-y.
                # Otherwise plain prose paragraphs would show up as a
                # complaint.
                if _passage_has_math_signal(p.text):
                    uncovered.append(pid)
        return {
            "covered_passages": len(covered),
            "uncovered_passages": [
                {"id": pid,
                 "home_nid": self.passages[pid].home_nid,
                 "text": self.passages[pid].text[:160]}
                for pid in uncovered
            ],
            "n_formulas": len(self.formulas),
            "n_vars": len(self.vars),
        }

    # Lookup helpers — used by the runtime to bind a session clause
    # back to the offline-prebuilt Passage.
    def _norm_passage_text(self, text: str) -> str:
        # Mirror what `_split_sentences` would produce — collapse
        # whitespace + lowercase.  Good enough for text-equality
        # lookup since both sides come through the same pipeline.
        return re.sub(r"\s+", " ", (text or "").strip()).lower()

    def passage_by_home_text(self, home_nid: str,
                             text: str) -> Optional["Passage"]:
        """Return the Passage with this ``home_nid`` whose stored
        text matches *text* (after whitespace / case normalisation).

        Used by the orchestrator to map a runtime clause back to its
        offline-prebuilt Passage so we can emit cards from the graph
        instead of re-extracting at runtime.
        """
        if not home_nid:
            return None
        target = self._norm_passage_text(text)
        if not target:
            return None
        for p in self.passages.values():
            if p.home_nid != home_nid:
                continue
            if self._norm_passage_text(p.text) == target:
                return p
        return None

    def formulas_for_passage(self, pid: str) -> list["Formula"]:
        out: list[Formula] = []
        for e in self.out_edges(pid, "about"):
            f = self.formulas.get(e.dst)
            if f is not None:
                out.append(f)
        return out

    def best_anchor_for(self, fid: str) -> Optional[str]:
        """Return the existing-Formula id that shares the most
        variables with *fid*, or None if nothing overlaps.

        Layout uses this to place new cards next to their best match.
        """
        if fid not in self.formulas:
            return None
        my_vars = {e.dst for e in self.out_edges(fid, "uses")}
        if not my_vars:
            return None
        best, best_score = None, 0
        for other_id in self.formulas:
            if other_id == fid:
                continue
            other_vars = {e.dst for e in self.out_edges(other_id, "uses")}
            score = len(my_vars & other_vars)
            if score > best_score:
                best, best_score = other_id, score
        return best


# ---------------------------------------------------------------------------
# LaTeX → variable extractor (deterministic, regex-driven)
# ---------------------------------------------------------------------------

# Heuristic LaTeX command names we should NOT treat as variables.
_LATEX_NON_VARS = frozenset({
    # operators / spacing / decoration
    "frac", "sqrt", "left", "right", "begin", "end", "cdots", "ldots",
    "vdots", "ddots", "to", "rightarrow", "leftarrow", "Rightarrow",
    "Leftarrow", "iff", "implies",
    # bigops
    "sum", "prod", "int", "oint", "iint", "iiint", "lim", "max", "min",
    "sup", "inf", "argmax", "argmin",
    # logic / sets
    "in", "notin", "subset", "supset", "subseteq", "supseteq", "cup",
    "cap", "emptyset", "forall", "exists", "neg",
    # text
    "text", "mbox", "mathrm", "mathbf", "mathit", "mathcal", "mathbb",
    "mathfrak", "operatorname", "displaystyle", "textstyle",
    # decorations
    "hat", "tilde", "bar", "dot", "ddot", "vec", "overline", "underline",
    # spacing
    "quad", "qquad",
})

# Greek letters we DO treat as variables.
_LATEX_GREEK_VARS = frozenset({
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon",
    "zeta", "eta", "theta", "vartheta", "iota", "kappa", "lambda",
    "mu", "nu", "xi", "omicron", "pi", "varpi", "rho", "varrho",
    "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi",
    "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
    "Upsilon", "Phi", "Psi", "Omega",
})

# Unicode Greek-letter codepoints we accept as variable names.
_UNICODE_GREEK_VARS = frozenset(
    chr(c) for c in
    list(range(0x0391, 0x03A2)) + list(range(0x03A3, 0x03AA))
    + list(range(0x03B1, 0x03CA))
)


def canonical_var_name(raw: str) -> str:
    """Normalise a variable token to its canonical form."""
    if not raw:
        return ""
    s = raw.strip()
    # Strip leading backslash for LaTeX commands.
    if s.startswith("\\"):
        s = s[1:]
    # Strip subscripts / superscripts: x_i / x_{ij} / x^2 → x
    s = re.split(r"[_^]", s, maxsplit=1)[0]
    # Strip surrounding braces / parens.
    s = s.strip("{}()[]")
    if not s:
        return ""
    # Reject obvious non-vars (numbers, operators, common LaTeX cmds).
    if s.lower() in _LATEX_NON_VARS:
        return ""
    if s.isdigit():
        return ""
    # Accept single Latin / Greek / unicode-Greek letters, or short
    # ALL-CAPS multi-letters that aren't English words (e.g. "RKHS",
    # "HK").  Reject prose tokens like "where", "the", etc.
    if len(s) == 1 and (s.isalpha() or s in _UNICODE_GREEK_VARS):
        return s
    if s in _LATEX_GREEK_VARS:
        return s
    if 2 <= len(s) <= 4 and s.isalpha() and s.isupper():
        return s   # e.g. "RKHS", "HK"
    if 2 <= len(s) <= 3 and s.isalpha() and s.islower():
        # Common variable-like multi-letter (e.g. "fn", "id") —
        # accept conservatively.
        return s
    return ""


_VAR_TOKEN_RE = re.compile(
    r"\\[A-Za-z]+|"           # LaTeX command (Greek letters etc.)
    r"[A-Za-z][A-Za-z0-9]*|"  # plain identifiers
    r"[α-ωΑ-Ω]"               # unicode Greek
)


def parse_formula_vars(latex: str) -> tuple[set[str], set[str], set[str]]:
    """Extract (used, bound, defined) variable name sets from *latex*.

    *Defined* is the LHS variable when the formula has the shape
    ``X = …`` or ``X(args) = …``.  *Bound* are variables introduced
    by big-operators (``\\sum_{i=…}``, ``\\int … d x``).  *Used* is
    every other variable mention.
    """
    used: set[str] = set()
    bound: set[str] = set()
    defined: set[str] = set()
    if not latex:
        return used, bound, defined

    # 1. Bound variables from \sum_{i=…} / \int … d x patterns.
    for m in re.finditer(
        r"\\(?:sum|prod|int|oint|iint|iiint|lim|max|min|sup|inf|"
        r"forall|exists|argmax|argmin)_\{([^{}]*)\}",
        latex,
    ):
        body = m.group(1)
        # Body is usually ``i=1`` or ``i \in S`` — keep the LHS letter.
        head = re.split(r"[=<>≤≥∈]", body, maxsplit=1)[0]
        for tok in _VAR_TOKEN_RE.findall(head):
            name = canonical_var_name(tok)
            if name:
                bound.add(name)
    # ``d x`` differentials inside integrals.
    for m in re.finditer(r"\bd\s*([A-Za-z])\b", latex):
        name = canonical_var_name(m.group(1))
        if name:
            bound.add(name)

    # 2. Defined variable from LHS = … shape.
    if "=" in latex:
        lhs = latex.split("=", 1)[0]
        # Strip a function-application suffix: ``f(x)`` → ``f``.
        lhs_func = re.match(r"\s*(\\?[A-Za-z][A-Za-z0-9]*)\s*\(",
                            lhs)
        if lhs_func:
            name = canonical_var_name(lhs_func.group(1))
            if name:
                defined.add(name)
        else:
            for tok in _VAR_TOKEN_RE.findall(lhs):
                name = canonical_var_name(tok)
                if name:
                    defined.add(name)
                    break  # first var on the LHS is the defined one

    # 3. Used variables: every var token in the LaTeX, minus bound +
    # defined.  We keep ``defined`` in ``used`` too — a defined
    # symbol is *also* used (this matters for shared_vars overlap).
    for tok in _VAR_TOKEN_RE.findall(latex):
        name = canonical_var_name(tok)
        if name:
            used.add(name)

    return used, bound, defined


# ---------------------------------------------------------------------------
# Variable-mention scanner for narration text
# ---------------------------------------------------------------------------

# Match ``X of Y`` (function-call notation as the verbalizer emits it)
# and ``X(...)`` (when raw LaTeX leaks through).  Used by
# ingest_passage to wire ``about(Passage, Var)`` edges.
_OF_CALL_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]|\\[A-Za-z]+)\s+of\s+",
    re.IGNORECASE,
)
_PAREN_CALL_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]|\\[A-Za-z]+)\s*\("
)


def _name_mentions(text: str) -> list[str]:
    """Return the canonical Var names mentioned in *text*."""
    out: set[str] = set()
    for m in _OF_CALL_RE.finditer(text):
        n = canonical_var_name(m.group(1))
        if n:
            out.add(n)
    for m in _PAREN_CALL_RE.finditer(text):
        n = canonical_var_name(m.group(1))
        if n:
            out.add(n)
    # Greek letters spelled out: "lambda", "alpha", …
    for greek in _LATEX_GREEK_VARS:
        if re.search(rf"(?<![A-Za-z]){greek}(?![A-Za-z])", text,
                     re.IGNORECASE):
            out.add(greek.lower())
    return sorted(out)


# ---------------------------------------------------------------------------
# Per-book persistence path
# ---------------------------------------------------------------------------

def _normalize_latex_for_containment(s: str) -> str:
    """Aggressive normalisation for substring containment tests.

    Strips whitespace, ``\\,``-style spacing commands, surrounding
    delimiters, and lowercases the alphabetic content.  The aim is
    to make ``J(f)`` match ``...+\\lambda J(f)`` even when the
    presentation differs slightly.
    """
    if not s:
        return ""
    t = s
    t = re.sub(r"\\[\\,;:! ]", "", t)
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]|\\left|\\right", "", t)
    t = re.sub(r"\s+", "", t)
    return t.lower()


def find_subexpression_parent(g: "MathGraph", latex: str) -> Optional[str]:
    """Return the nid of an existing Formula whose LaTeX *strictly
    contains* ``latex`` (after normalisation), or None.

    Used by the orchestrator to fold a small fragment (``J(f)``)
    into a larger card already on the board (``L(yi, f(xi)) + λJ(f)``)
    instead of emitting a duplicate.  Strict containment means the
    candidate parent must be longer than the child — same-length
    formulas don't subsume each other.
    """
    if not latex:
        return None
    needle = _normalize_latex_for_containment(latex)
    if len(needle) < 3:
        return None
    best_nid: Optional[str] = None
    best_len = 0
    for nid, f in g.formulas.items():
        haystack = _normalize_latex_for_containment(f.latex)
        if len(haystack) <= len(needle):
            continue
        if needle in haystack:
            if len(haystack) > best_len:
                best_nid = nid
                best_len = len(haystack)
    return best_nid


def graph_path_for_book(book_path: str) -> str:
    """Return the on-disk graph path next to *book_path*."""
    base, _ = os.path.splitext(book_path)
    return base + ".math_graph.json"


# ---------------------------------------------------------------------------
# Coverage audit helpers
# ---------------------------------------------------------------------------

# Heuristic signals that a *spoken* passage contains math content.
# Used by ``coverage_report`` to filter prose paragraphs from the
# uncovered-passage list — we don't want to flag every plain English
# sentence as missing a formula.
_MATH_SIGNAL_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    r"(?:[A-Za-z]\s+of\s+)|"          # "f of x"
    r"(?:[A-Za-z]\([^)]+\))|"           # "f(x)" raw
    r"(?:lambda|sigma|alpha|beta|gamma|theta|phi|psi|omega|delta|"
    r"epsilon|zeta|eta|kappa|mu|nu|xi|rho|tau|chi)|"
    r"(?:integral|summation|gradient|equation\s+\d|formula)|"
    r"(?:[=≤≥<>≠→∈∑∫∞±])"
    r")",
    re.IGNORECASE,
)


def _passage_has_math_signal(text: str) -> bool:
    return bool(text) and bool(_MATH_SIGNAL_RE.search(text))
