"""Offline ingest: build the math semantic graph for a whole book.

The runtime narrator should NOT extract math on the fly — every
formula, sub-expression, citation, derivation, containment, and
concept must already be in the graph by the time narration starts.
This tool does that work in one batch pass per book.

What it does, per BookNode:
  1. Sentence-segment ``body_text`` (the same splitter the planner
     uses, so sentence indices line up at runtime).
  2. For each sentence, detect every math fragment + bare
     function-call notation (``L(yi, f(xi))``, ``J(f)``).
  3. Convert each to LaTeX, build a Formula node, run the Phase-0
     extractor (uses / binds / defines), and check sub-expression
     containment against ALL formulas already in the graph.
  4. Index the sentence as a Passage and wire ``about(P, F)`` for
     every formula that landed here.
  5. Run the Phase-1 enrichers (define / derive / equivalence /
     specialize) on the spoken-form sentence text.

Cross-passage pass at the end:
  * Citation graph — for every ``(N.M)`` reference inside a passage,
    link the passage to the formula(s) carrying ``N.M`` as a
    cite_label.
  * Containment cleanup — formulas that turned out to be strict
    sub-expressions of an existing card are kept in the graph but
    flagged via ``contains`` edges so the runtime can fold them.

Usage
-----

    uv run -- python -m tools.build_math_graph books/ESLII.json

Re-runnable: loads the existing ``books/<book>.math_graph.json`` and
merges new findings in (the persisted graph is monotonic —
enrichment, never rebuilding).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Iterator

from book import load_corpus
from book.ir import Book, BookNode
from narrator.planner import _split_sentences, _clean_body_for_outline
from narrator.qa import _sanitize_for_narration
from sevim.math_graph import (
    MathGraph, Formula, find_subexpression_parent, graph_path_for_book,
)
from sevim.math_graph_phase1 import enrich_clause
from serve.orchestrator import (
    _detect_math_fragments, _equation_citations_in,
    _verbalized_formula_keys, _normalize_formula_key,
    _repair_pymupdf_vstack,
)
from serve.refcontent import to_latex


# ---------------------------------------------------------------------------
# Walking the corpus
# ---------------------------------------------------------------------------

def _walk_nodes(book: Book) -> Iterator[BookNode]:
    """Pre-order walk that yields every BookNode with non-empty body."""
    def rec(n: BookNode) -> Iterator[BookNode]:
        body = (n.body_text or "").strip()
        if body:
            yield n
        for c in n.children:
            yield from rec(c)
    yield from rec(book.root)


# ---------------------------------------------------------------------------
# Per-passage ingest
# ---------------------------------------------------------------------------

def _ingest_passage(
    g: MathGraph, *,
    seq: int, raw_text: str, home_nid: str,
    fragments_seen_normkeys: set[str],
) -> dict:
    """Detect math, build Formula nodes + edges for a single sentence.

    Returns a per-passage stats dict.
    """
    spoken_text = _sanitize_for_narration(raw_text) or raw_text
    cite_labels = _equation_citations_in(raw_text)
    fragments = _detect_math_fragments(raw_text)
    formula_ids: list[str] = []
    contained: list[tuple[str, str]] = []   # (parent_nid, sub_key)

    # We synthesise a stable Formula nid that's deterministic across
    # rebuild runs — hash of (home_nid, normalised LaTeX).  This
    # matters because the persisted graph is keyed by nid; a fresh
    # build that produces a different nid would create duplicates.
    def _stable_nid(home_nid: str, key: str) -> str:
        from hashlib import blake2b
        h = blake2b(f"{home_nid}|{key}".encode("utf-8"),
                    digest_size=6).hexdigest()
        return f"n_offline_{h}"

    for frag, _frag_offset in fragments:
        latex = to_latex(frag).strip()
        if not latex:
            continue
        key = _normalize_formula_key(latex)
        if not key:
            continue

        # Containment fold — if this fragment is a strict
        # sub-expression of an already-ingested formula, mark
        # ``contains(parent, _subexpr:key)`` and skip the new node.
        parent_nid = find_subexpression_parent(g, latex)
        if parent_nid:
            g.add_edge(parent_nid, "contains", f"_subexpr:{key}",
                       meta={"latex": latex,
                             "passage_home": home_nid})
            contained.append((parent_nid, key))
            continue

        if key in fragments_seen_normkeys:
            # Already minted a Formula for this exact LaTeX in an
            # earlier passage; just re-anchor by emitting an
            # ``about`` edge from this passage to the existing
            # Formula (handled below in ``_link_passage_about``).
            continue
        fragments_seen_normkeys.add(key)

        nid = _stable_nid(home_nid, key)
        f = g.ingest_formula(
            nid=nid, latex=latex,
            cite_labels=list(cite_labels),
            home_nid=home_nid,
        )
        # Register the LaTeX surface for runtime layout / mention
        # scanning.
        f.surface = (frag or latex)[:60]
        formula_ids.append(nid)

    # Index this sentence as a Passage + wire about/paired_in_clause
    # edges at the end.
    g.ingest_passage(
        seq=seq, text=spoken_text, home_nid=home_nid,
        formula_ids_in_clause=tuple(formula_ids),
    )

    # Phase-1 enrichers — derived_from / specializes / related_to /
    # instance_of from prose patterns.
    enrich_clause(
        g, seq=seq, text=spoken_text, home_nid=home_nid,
        formula_ids_in_clause=tuple(formula_ids),
    )

    return {
        "fragments_kept": len(formula_ids),
        "fragments_contained": len(contained),
    }


def _link_passage_about(
    g: MathGraph, *, seq: int, home_nid: str,
) -> int:
    """Add ``about`` edges from this passage to every existing formula
    whose normalised LaTeX appears (verbatim, by surface form) in the
    passage's text.

    Phase-0 only ingested the formulas first-detected in *this*
    passage; this pass widens the link to cover re-mentions of formulas
    introduced in earlier passages.
    """
    pid = f"p:{seq}@{home_nid}" if home_nid else f"p:{seq}"
    p = g.passages.get(pid)
    if p is None:
        return 0
    # Track existing about(P, F) edges to avoid double-adding.
    existing = {(e.src, e.dst) for e in g.edges if e.type == "about"}
    n = 0
    text_lower = p.text.lower()
    for fid, f in g.formulas.items():
        # Cheap surface match: every cite label that appears in text
        # → wire about.
        for lab in f.cite_labels:
            if (f"({lab})" in p.text or f"equation {lab}" in text_lower
                    or f"eq. {lab}" in text_lower):
                if (pid, fid) not in existing:
                    g.add_edge(pid, "about", fid)
                    existing.add((pid, fid))
                    n += 1
                    break
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build(book_path: str, *, verbose: bool = True) -> str:
    """Build / enrich the math graph for *book_path*.  Returns the
    graph file path.
    """
    t0 = time.time()
    book = load_corpus(book_path)

    graph_path = graph_path_for_book(book_path)
    g = MathGraph.load(graph_path, book_id=book.title or book_path)
    if verbose:
        print(f"[build_math_graph] book={book.title!r} "
              f"existing graph: formulas={len(g.formulas)} "
              f"vars={len(g.vars)} edges={len(g.edges)}")

    fragments_seen_normkeys: set[str] = {
        _normalize_formula_key(f.latex) for f in g.formulas.values()
    } - {""}

    seq = 0
    n_nodes = 0
    n_passages = 0
    total_kept = 0
    total_contained = 0
    n_about_links = 0

    for node in _walk_nodes(book):
        body = _clean_body_for_outline(node.body_text or "")
        # Fold PyMuPDF's vertical-stack \sum/\prod/\int glyphs into
        # inline LaTeX BEFORE sentence-splitting, so equations spanning
        # multiple physical lines survive as one sentence and reach
        # ``_detect_math_fragments`` whole.  Also collapse single
        # newlines so an equation broken across lines isn't split into
        # one-token "sentences" by the sentence splitter.
        body = re.sub(r"\n(?!\s*\n)", " ", body)
        body = _repair_pymupdf_vstack(body)
        if len(body) < 12:
            continue
        n_nodes += 1
        for sentence in _split_sentences(body):
            if not sentence or not sentence.strip():
                continue
            stats = _ingest_passage(
                g, seq=seq,
                raw_text=sentence,
                home_nid=node.nid,
                fragments_seen_normkeys=fragments_seen_normkeys,
            )
            total_kept += stats["fragments_kept"]
            total_contained += stats["fragments_contained"]
            seq += 1
            n_passages += 1

    # Cross-passage pass: re-mention links via citation labels.
    seq = 0
    for node in _walk_nodes(book):
        body = _clean_body_for_outline(node.body_text or "")
        # Fold PyMuPDF's vertical-stack \sum/\prod/\int glyphs into
        # inline LaTeX BEFORE sentence-splitting, so equations spanning
        # multiple physical lines survive as one sentence and reach
        # ``_detect_math_fragments`` whole.  Also collapse single
        # newlines so an equation broken across lines isn't split into
        # one-token "sentences" by the sentence splitter.
        body = re.sub(r"\n(?!\s*\n)", " ", body)
        body = _repair_pymupdf_vstack(body)
        if len(body) < 12:
            continue
        for _ in _split_sentences(body):
            n_about_links += _link_passage_about(
                g, seq=seq, home_nid=node.nid,
            )
            seq += 1

    g.save(graph_path)
    elapsed = time.time() - t0
    if verbose:
        rep = g.coverage_report()
        edges_by_type = {}
        for e in g.edges:
            edges_by_type[e.type] = edges_by_type.get(e.type, 0) + 1
        print(f"[build_math_graph] DONE in {elapsed:.1f}s")
        print(f"  nodes walked:       {n_nodes}")
        print(f"  passages:           {n_passages}")
        print(f"  formulas kept:      {total_kept}")
        print(f"  fragments contained: {total_contained}")
        print(f"  cross-passage about links: {n_about_links}")
        print(f"  graph: formulas={len(g.formulas)} "
              f"vars={len(g.vars)} concepts={len(g.concepts)} "
              f"passages={len(g.passages)}")
        print(f"  edges by type:")
        for t, n in sorted(edges_by_type.items()):
            print(f"    {t}: {n}")
        print(f"  coverage: {rep['covered_passages']} covered, "
              f"{len(rep['uncovered_passages'])} uncovered")
        print(f"  saved → {graph_path}")
    return graph_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_math_graph",
                                description=__doc__.splitlines()[0])
    p.add_argument("book", nargs="+",
                   help="path(s) to ingested corpus JSON")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    for path in args.book:
        build(path, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
