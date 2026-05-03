"""Background ingest pipeline for an uploaded PDF.

Runs as a daemon thread spawned by ``POST /api/upload_book``; the
orchestrator and other LLM calls keep working while this thread does
its long, LLM-heavy job.  The pipeline updates ``books/_index.json``
phase-by-phase so the UI's status badge can show progress.

Phases (in order, in ``books_index.update_book(phase=…)``):

  1. extracting        — PDF → corpus JSON via ``book.parse.parse_pdf``
                          + concept / cross-ref extraction.
  2. building_math_graph
  3. discovering_chapters — read the chapters list from the corpus.
  4. building_chapter_maps — for each chapter, run the chapter-map
                          builder, the formula-explanation builder,
                          the latex repair, and the SeVim diagram
                          builder.
  5. registering       — hand the freshly-built corpus to the running
                          server's ``books_by_name`` so the dropdown
                          and TOC see it without a restart.
  6. ready             — done; ``ingested_chapters`` lists the nids
                          that successfully built sidecars.

A failure in any phase flips ``phase = "failed"`` and stashes the
exception in ``error``; the partial artifacts on disk are left for
inspection.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from typing import Optional, Callable

from . import books_index


def _set_phase(books_dir: str, book_id: str, phase: str,
               progress: str = "") -> None:
    print(f"[ingest:{book_id}] {phase} :: {progress}", flush=True)
    books_index.update_book(books_dir, book_id,
                            phase=phase, progress=progress)


def _run_phase(books_dir: str, book_id: str, phase: str,
               progress: str, fn: Callable[[], None]) -> None:
    _set_phase(books_dir, book_id, phase, progress)
    fn()


import re as _re

# Titles the PDF parser frequently mis-classifies as ``kind=chapter``
# even though they're not content the user wants narrated.  We skip
# them during ingest.
_FRONT_MATTER_TITLES = {
    "cover", "title page", "title", "copyright", "copyrights",
    "dedication", "contents", "table of contents",
    "index", "indices", "name index", "author index",
    "subject index", "topic index", "selected bibliography",
    "bibliography", "references", "preface", "front matter",
    "statement", "frontispiece",
    "acknowledgments", "acknowledgements",
    "about the author", "about the authors",
    "about the cover", "errata", "colophon",
    "list of figures", "list of tables", "list of symbols",
    "list of theorems", "list of definitions",
    "notation", "abbreviations", "glossary",
    "back matter",
}


_FRONT_MATTER_PREFIXES = (
    "author index", "name index", "subject index",
    "list of",  # "List of figures", "List of tables", …
)

# Section titles that *look* like chapters because the publisher
# wrapped them in a "Part" group (Sipser, Russell-Norvig, …).
# Matches "Ch 1", "Ch 1:", "Chapter 7 - Decision Trees", etc.
_CH_TITLE_RE = _re.compile(
    r"^\s*(?:Ch|Chapter)\.?\s*\d+\b", _re.IGNORECASE,
)


def _is_real_chapter(node: dict) -> bool:
    """Decide whether a corpus node should get its own chapter-zoom
    pipeline run.  Real content chapters get True; front matter,
    indices, bibliographies, and "Part X: …" containers get False.
    """
    title = (node.get("title") or "").strip()
    title_lower = title.lower()
    # Strip leading number/colon if present so "Ch 5: Basis Expansions"
    # matches the front-matter set when applicable.
    bare = _re.sub(r"^\s*\d+(\.\d+)*[:.\s-]+", "", title_lower).strip()
    if bare in _FRONT_MATTER_TITLES or title_lower in _FRONT_MATTER_TITLES:
        return False
    for pref in _FRONT_MATTER_PREFIXES:
        if title_lower.startswith(pref) or bare.startswith(pref):
            return False
    # Catch the parser's normalised forms ("ch_author_index" →
    # "author index" doesn't match because `_` survives).  Strip any
    # underscore-noise remnants in the lowered title.
    underscore_norm = title_lower.replace("_", " ").strip()
    if underscore_norm in _FRONT_MATTER_TITLES:
        return False
    for pref in _FRONT_MATTER_PREFIXES:
        if underscore_norm.startswith(pref):
            return False
    kind = (node.get("kind") or "").lower()
    if kind == "chapter":
        return True
    # Sections whose title looks like "Ch N" — Sipser-style books
    # where Parts wrap chapters and the chapter ends up as
    # ``kind=section`` in the parsed outline.
    if kind == "section" and _CH_TITLE_RE.match(title):
        return True
    return False


def _list_chapter_nids(corpus_json: str) -> list[str]:
    """Read the corpus JSON's tree and return the nids of every node
    that should get a chapter-zoom narration build."""
    import json as _json
    with open(corpus_json) as f:
        d = _json.load(f)
    chapters: list[str] = []

    def _walk(n: dict):
        if _is_real_chapter(n):
            chapters.append(n.get("nid") or "")
            # A node identified as a chapter; don't descend further
            # — its inner sections aren't themselves chapters.
            return
        for c in n.get("children") or []:
            _walk(c)

    root = d.get("root") or {}
    _walk(root)
    return [c for c in chapters if c]


def _build_one_chapter(corpus_json: str, chapter_nid: str) -> bool:
    """Run the full per-chapter sub-pipeline.  Returns True on
    successful chapter_map build (subsequent post-processing failures
    are logged but don't mark the chapter as failed — the chapter map
    alone is enough to play the chapter)."""
    print(f"  [chapter] {chapter_nid} → starting", flush=True)
    # 1. chapter map (uses local LLM for story_paragraph + role_in_parent)
    from tools.build_chapter_map import build as build_chapter_map
    safe_root = chapter_nid.replace("/", "_")
    stem = corpus_json[: -len(".json")] if corpus_json.endswith(".json") \
        else corpus_json
    cm_path = f"{stem}.chapter_map.{safe_root}.json"
    rc = build_chapter_map(corpus_json, chapter_nid, cm_path)
    if rc != 0:
        print(f"  [chapter] {chapter_nid} :: chapter_map failed (rc={rc})",
              flush=True)
        return False
    # 2. inline-formula extraction — runs BEFORE the explanation +
    # repair passes so they have a populated ``canonical_formula_latex``
    # to work against.  Targets sections whose math-graph attribution
    # missed an inline formula (Sipser-style tuple definitions, prose
    # theorem statements, …).
    try:
        from tools.extract_inline_formulas import repair as extract_inline
        extract_inline(cm_path)
    except Exception as e:
        print(f"  [chapter] {chapter_nid} :: inline formula error: {e}",
              flush=True)
    # 3. formula explanations
    try:
        from tools.build_formula_explanations import enrich
        enrich(cm_path)
    except Exception as e:
        print(f"  [chapter] {chapter_nid} :: explanations error: {e}",
              flush=True)
    # 4. latex repair
    try:
        from tools.repair_formula_latex import repair
        repair(cm_path)
    except Exception as e:
        print(f"  [chapter] {chapter_nid} :: latex repair error: {e}",
              flush=True)
    # 4. SeVim per-section diagrams
    try:
        from tools.build_sevim_diagrams import build as build_sd
        build_sd(cm_path, force=False)
    except Exception as e:
        print(f"  [chapter] {chapter_nid} :: sevim diagrams error: {e}",
              flush=True)
    print(f"  [chapter] {chapter_nid} → done", flush=True)
    return True


def _ingest(books_dir: str, book_id: str, pdf_path: str,
            corpus_json: str, *, register_with_server=None) -> None:
    """Run the full ingest pipeline for one PDF.  Mutates the index
    as it progresses.  Designed to be called inside a daemon thread."""
    try:
        # Phase 1: PDF → corpus JSON.
        _set_phase(books_dir, book_id, "extracting", "parsing PDF")
        from book.parse import parse_pdf
        from book.concepts import extract_concepts
        from book.crossref import extract_cross_refs
        from book.corpus import write_corpus
        figures_dir = corpus_json.replace(".json", "_figures")
        book = parse_pdf(pdf_path, figures_dir=figures_dir)
        _set_phase(books_dir, book_id, "extracting", "extracting concepts")
        try:
            extract_concepts(book)
        except Exception as e:
            print(f"[ingest:{book_id}] concept extraction warning: {e}",
                  flush=True)
        _set_phase(books_dir, book_id, "extracting",
                   "extracting cross-references")
        try:
            extract_cross_refs(book)
        except Exception as e:
            print(f"[ingest:{book_id}] crossref extraction warning: {e}",
                  flush=True)
        # ``write_corpus(path, book)`` — argument order matters.
        write_corpus(corpus_json, book)

        # Phase 2a: figures sidecar.  Must run BEFORE chapter_map so
        # the chapter-wide LLM call has the figures list to enforce
        # figure mentions in story_paragraphs.
        _set_phase(books_dir, book_id, "extracting", "extracting figures")
        try:
            from tools.extract_book_figures import main as extract_figs_main
            extract_figs_main([corpus_json])
        except SystemExit:
            pass            # the tool's main() can SystemExit cleanly
        except Exception as e:
            print(f"[ingest:{book_id}] figures sidecar warning: {e}",
                  flush=True)

        # Phase 2b: concept layer.  Used by chapter_map to fill cell
        # gists in plain English.  LLM-driven; SLOW on long books.
        _set_phase(books_dir, book_id, "building_concepts",
                   "L0..L4 concept narratives")
        try:
            from tools.build_concept_layer import main as build_cc_main
            build_cc_main([corpus_json])
        except SystemExit:
            pass
        except Exception as e:
            print(f"[ingest:{book_id}] concept layer warning: {e}",
                  flush=True)

        # Phase 2c: math graph.
        _set_phase(books_dir, book_id, "building_math_graph",
                   "scanning equations")
        try:
            from tools.build_math_graph import build as build_mg
            build_mg(corpus_json)
        except Exception as e:
            print(f"[ingest:{book_id}] math graph warning: {e}",
                  flush=True)

        # Phase 2d: formula layer (F0/F1/F2 explanations).  LLM-driven
        # per formula; SLOW.  Must run AFTER math graph.
        _set_phase(books_dir, book_id, "building_formulas",
                   "F0/F1/F2 explanations")
        try:
            from tools.build_formula_layer import main as build_fl_main
            build_fl_main([corpus_json])
        except SystemExit:
            pass
        except Exception as e:
            print(f"[ingest:{book_id}] formula layer warning: {e}",
                  flush=True)

        # Phase 3: per-chapter sub-pipeline.
        chapters = _list_chapter_nids(corpus_json)
        books_index.update_book(
            books_dir, book_id,
            total_chapters=len(chapters),
        )
        _set_phase(
            books_dir, book_id, "building_chapter_maps",
            f"0/{len(chapters)} chapters",
        )
        done: list[str] = []
        for i, nid in enumerate(chapters, 1):
            _set_phase(
                books_dir, book_id, "building_chapter_maps",
                f"{i-1}/{len(chapters)} → {nid}",
            )
            ok = _build_one_chapter(corpus_json, nid)
            if ok:
                done.append(nid)
                books_index.update_book(books_dir, book_id,
                                        ingested_chapters=list(done))
        # Phase 4: data-quality health check (Tier 1 inspector).
        # Walks every sidecar and writes <stem>.health.json with
        # per-check pass/fail/skip.  Cheap (<5 s) — always run.
        _set_phase(books_dir, book_id, "checking",
                   "running data-quality health check")
        try:
            from tools.data_quality_agent import run as dq_run
            dq_run(corpus_json)
        except Exception as e:
            print(f"[ingest:{book_id}] health check warning: {e}",
                  flush=True)

        # Phase 5: math-fidelity agent (Tier 2).  ONLY run when the
        # local VLM endpoint is reachable; otherwise every formula
        # would classify as ``unverifiable`` and waste an hour.
        try:
            from tools.vlm_engine import vlm_reachable
            if vlm_reachable(force=True):
                _set_phase(books_dir, book_id, "checking",
                           "math fidelity (VLM-backed)")
                from tools.fidelity_agent import run as fid_run
                fid_run(corpus_json, max_iters=2,
                        apply_repairs=True, verbose=False)
            else:
                print(f"[ingest:{book_id}] VLM not reachable; "
                      f"skipping fidelity_agent", flush=True)
        except Exception as e:
            print(f"[ingest:{book_id}] fidelity agent warning: {e}",
                  flush=True)

        # Phase 6: hand the new corpus to the running server.
        if register_with_server is not None:
            _set_phase(books_dir, book_id, "registering",
                       "registering with server")
            try:
                register_with_server(corpus_json)
            except Exception as e:
                print(f"[ingest:{book_id}] register warning: {e}",
                      flush=True)
        # Done.
        _set_phase(books_dir, book_id, "ready",
                   f"{len(done)}/{len(chapters)} chapters built")
    except Exception as e:
        traceback.print_exc()
        books_index.update_book(
            books_dir, book_id,
            phase="failed",
            error=f"{type(e).__name__}: {e}",
            progress="aborted on error",
        )


def spawn(books_dir: str, book_id: str, pdf_path: str,
          corpus_json: str, *,
          register_with_server: Optional[Callable[[str], None]] = None,
          ) -> threading.Thread:
    """Kick off ``_ingest`` in a daemon thread.  Returns the thread
    handle (caller usually doesn't keep it; the index is the source
    of truth for state)."""
    t = threading.Thread(
        target=_ingest,
        args=(books_dir, book_id, pdf_path, corpus_json),
        kwargs={"register_with_server": register_with_server},
        daemon=True,
        name=f"ingest:{book_id}",
    )
    t.start()
    return t
