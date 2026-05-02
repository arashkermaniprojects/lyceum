"""Concept index extraction — Book → dict[cid, ConceptEntry].

Walks every BookNode in the book tree, scans its body_text for:

  1.  **Math literal templates** (LaTeX matrix envs, inline equations,
      bracketed cell arrays).  Each becomes a `ConceptTemplate` rooted at
      the home node; the concept id is "matrix" / "equation_block" / etc.

  2.  **Concept-name mentions** via the existing `sevim.math_lex.classify_math_label`
      vocabulary.  Each mention contributes a (potentially empty) template
      whose primitive type is the math_lex classification.

  3.  **Definitions**: when the home node's kind is "definition" and the
      first sentence introduces a noun phrase ("A *matrix* is a rectangular
      array of numbers."), capture the term and the prose.

The output is keyed by canonical concept id.  Multiple book contexts producing
the same concept yield multiple `ConceptTemplate` entries so the visual
resolver can pick by current chapter.

Determinism
-----------
- Walk order is pre-order over the BookNode tree (deterministic).
- All regexes are deterministic; no embeddings here.
"""
from __future__ import annotations

import re
from typing import Optional

from .ir import (
    Book, BookNode, ConceptEntry, ConceptTemplate,
    is_environment,
)

# Reuse SeVim's math vocabulary.
from sevim.math_lex import (
    classify_math_label,
    find_latex_spans,
    parse_matrix_literal,
)


# ---------------------------------------------------------------------------
# Concept-name canonicalisation
# ---------------------------------------------------------------------------

_NORM_PUNCT = re.compile(r"[^a-z0-9]+")

# Surface-form aliases that collapse to a canonical id.
_ALIASES: dict[str, str] = {
    "matrices": "matrix", "matrix": "matrix",
    "vectors": "vector", "vector": "vector",
    "sets": "set", "set": "set", "subset": "set", "superset": "set",
    "functions": "function", "function": "function", "map": "function",
    "mapping": "function", "morphism": "function",
    "groups": "group", "group": "group",
    "rings": "ring", "ring": "ring",
    "fields": "field", "field": "field",
    "spaces": "space", "space": "space", "vector space": "vector_space",
    "topological space": "topological_space",
    "metric space": "metric_space",
    "manifold": "manifold", "manifolds": "manifold",
    "tensor": "tensor", "tensors": "tensor",
    "tensor product": "tensor_product",
    "eigenvalue": "eigenvalue", "eigenvalues": "eigenvalue",
    "eigenvector": "eigenvector", "eigenvectors": "eigenvector",
    "determinant": "determinant", "determinants": "determinant",
    "polynomial": "polynomial", "polynomials": "polynomial",
    "derivative": "derivative", "derivatives": "derivative",
    "integral": "integral", "integrals": "integral",
    "limit": "limit", "limits": "limit",
    "sequence": "sequence", "sequences": "sequence",
    "series": "series",
    "graph": "graph", "graphs": "graph",
    "tree": "tree", "trees": "tree",
    "point": "point", "points": "point",
    "line": "line", "lines": "line",
    "curve": "curve", "curves": "curve",
    "circle": "circle", "circles": "circle",
    "triangle": "triangle", "triangles": "triangle",
    "polygon": "polygon", "polygons": "polygon",
    "axes": "axes", "axis": "axes",
    "coordinate system": "axes",
    "category": "category", "categories": "category",
    "functor": "functor", "functors": "functor",
    "natural transformation": "natural_transformation",
    "pullback": "pullback", "pushout": "pullback",
    "homomorphism": "homomorphism",
    "isomorphism": "isomorphism",
}


def _canonical_id(surface: str) -> Optional[str]:
    """Return the canonical concept id for a surface form, or None."""
    s = surface.strip().lower()
    if s in _ALIASES:
        return _ALIASES[s]
    # Try a soft normalisation: strip punctuation, collapse whitespace.
    norm = _NORM_PUNCT.sub("_", s).strip("_")
    return _ALIASES.get(norm)


# ---------------------------------------------------------------------------
# Definition heuristic
# ---------------------------------------------------------------------------

# "A matrix is a rectangular array of numbers."
# "A vector space V is a set …"
_DEF_INTRO_RE = re.compile(
    r"^\s*(?:[Aa]n?|[Tt]he)\s+"
    r"(?P<term>[A-Za-z][A-Za-z\- ]{2,40}?)"
    r"\s+(?:is|are|denotes|means)\s+",
)


def _extract_definition_term(body: str) -> Optional[str]:
    if not body:
        return None
    # Look at the first ~200 chars only — definitions usually start with the
    # introduced term.
    head = body.lstrip()[:240]
    m = _DEF_INTRO_RE.search(head)
    if not m:
        return None
    term = m.group("term").strip()
    return term


# ---------------------------------------------------------------------------
# Surface mention scanner
# ---------------------------------------------------------------------------

# Build a single regex covering all known surface forms, longest-first so
# multi-word forms ("vector space") match before single words ("vector").
_SURFACE_FORMS = sorted(_ALIASES.keys(), key=len, reverse=True)
_SURFACE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _SURFACE_FORMS) + r")\b",
    re.IGNORECASE,
)


def _scan_surface_mentions(text: str) -> list[tuple[int, str]]:
    """Return (offset, surface_form) pairs for every concept mention."""
    return [(m.start(), m.group(0)) for m in _SURFACE_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Template inference from a node's body
# ---------------------------------------------------------------------------

def _infer_node_templates(node: BookNode) -> list[tuple[str, ConceptTemplate]]:
    """Examine *node*'s body_text and emit (concept_id, template) pairs.

    Templates come from three sources:

    1. **LaTeX environments** found via `find_latex_spans`:
       - ``\\begin{pmatrix}…\\end{pmatrix}`` → matrix_bracket template
       - any LaTeX with structured constructs → equation_block template
    2. **Bare matrix literals** like `[[1,2],[3,4]]` in prose.
    3. **Math-vocabulary classification** of mention surface forms via
       `math_lex.classify_math_label`, when no literal is found nearby.

    The templates inherit `home_nid = node.nid` and an `evidence` payload
    citing page + snippet.
    """
    out: list[tuple[str, ConceptTemplate]] = []
    body = node.body_text or ""
    if not body:
        return out

    # 1. LaTeX spans → equation_block / matrix_bracket templates.
    for start, end, latex in find_latex_spans(body):
        snippet = latex[:120]
        mat = parse_matrix_literal(latex)
        if mat is not None:
            out.append(("matrix", ConceptTemplate(
                home_nid=node.nid,
                primitive="matrix_bracket",
                meta={
                    "kind": "matrix_bracket",
                    "nrows": mat["nrows"], "ncols": mat["ncols"],
                    "cells": mat["cells"], "delim": mat["delim"],
                },
                evidence={"page": node.page_start, "snippet": snippet,
                          "latex": latex},
            )))
        else:
            out.append(("equation", ConceptTemplate(
                home_nid=node.nid,
                primitive="equation_block",
                meta={"kind": "equation_block", "latex": latex},
                evidence={"page": node.page_start, "snippet": snippet},
            )))

    # 2. Bare matrix literals in prose.
    bare = parse_matrix_literal(body)
    if bare is not None:
        out.append(("matrix", ConceptTemplate(
            home_nid=node.nid,
            primitive="matrix_bracket",
            meta={
                "kind": "matrix_bracket",
                "nrows": bare["nrows"], "ncols": bare["ncols"],
                "cells": bare["cells"], "delim": bare["delim"],
            },
            evidence={"page": node.page_start, "snippet": body[:120],
                      "literal": True},
        )))

    # 3. Surface-form mentions → math_lex classification.
    for offset, surface in _scan_surface_mentions(body):
        cid = _canonical_id(surface)
        if cid is None:
            continue
        prim = classify_math_label(surface) or "rect"
        out.append((cid, ConceptTemplate(
            home_nid=node.nid,
            primitive=prim,
            meta={"kind": prim},
            evidence={"page": node.page_start, "offset": offset,
                      "surface": surface},
        )))

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_concepts(book: Book) -> dict[str, ConceptEntry]:
    """Walk *book.root*, extract concept mentions and templates, return the
    concept index keyed by canonical concept id.

    Side effect-free: the input ``book`` is not mutated.  Use
    ``book.concepts = extract_concepts(book)`` to attach the result.
    """
    index: dict[str, ConceptEntry] = {}

    def ensure(cid: str, surface: str) -> ConceptEntry:
        entry = index.get(cid)
        if entry is None:
            entry = ConceptEntry(
                cid=cid,
                canonical=cid.replace("_", " "),
                aliases=[],
                definitions=[],
                templates=[],
                figure_refs=[],
            )
            index[cid] = entry
        if surface and surface.lower() not in (a.lower() for a in entry.aliases):
            entry.aliases.append(surface)
        return entry

    for node in book.root.walk():
        # Definition extraction.
        if is_environment(node.kind) and node.kind == "definition":
            term = _extract_definition_term(node.body_text)
            if term:
                cid = _canonical_id(term)
                if cid:
                    entry = ensure(cid, term)
                    entry.definitions.append((node.nid, node.body_text.strip()))

        # Templates + mentions.
        for cid, tmpl in _infer_node_templates(node):
            entry = ensure(cid, "")
            entry.templates.append(tmpl)

    # Attach figures by home_nid: a figure becomes a candidate figure_ref of
    # every concept that occurs in the figure's home BookNode.
    nid_to_concepts: dict[str, set[str]] = {}
    for cid, entry in index.items():
        for tmpl in entry.templates:
            nid_to_concepts.setdefault(tmpl.home_nid, set()).add(cid)
    for fig in book.figures:
        for cid in nid_to_concepts.get(fig.home_nid, ()):
            if fig.fid not in index[cid].figure_refs:
                index[cid].figure_refs.append(fig.fid)

    return index
