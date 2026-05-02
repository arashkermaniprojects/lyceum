"""Run the per-chapter build pipeline against an existing corpus.

Used to backfill chapter-zoom sidecars for books that were already
loaded into the server (or that finished the PDF→corpus phase with an
older chapter detector).  For each chapter the new
``serve.ingest_pipeline._is_real_chapter`` accepts, we run

    chapter_map → formula_explanations → repair_formula_latex
                → sevim_diagrams

skipping chapters that already have a chapter_map sidecar on disk
unless ``--force`` is passed.

Usage:

    # Build every chapter's sidecars for ESLII
    python -m tools.ingest_existing_chapters books/ESLII.json

    # Force rebuild of chapter 5 specifically
    python -m tools.ingest_existing_chapters books/ESLII.json \\
        --only b/ch5 --force
"""
from __future__ import annotations

import argparse
import os
import sys


def _stem_for(corpus_json: str) -> str:
    return corpus_json[: -len(".json")] if corpus_json.endswith(".json") \
        else corpus_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", help="path to <stem>.json")
    p.add_argument("--only", action="append", default=[],
                   help="only build this chapter nid (repeatable); "
                        "default is every chapter the detector accepts")
    p.add_argument("--force", action="store_true",
                   help="rebuild even when sidecars already exist")
    p.add_argument("--list", action="store_true",
                   help="list the detected chapters and exit")
    args = p.parse_args(argv)

    if not os.path.isfile(args.corpus):
        print(f"error: {args.corpus} not found", file=sys.stderr)
        return 1
    from serve.ingest_pipeline import _list_chapter_nids
    chapters = _list_chapter_nids(args.corpus)
    if args.only:
        chapters = [c for c in chapters if c in set(args.only)]
    print(f"[ingest-chapters] {len(chapters)} chapter(s) to consider:")
    for c in chapters:
        print("   ", c)
    if args.list:
        return 0
    if not chapters:
        print("[ingest-chapters] nothing to do.")
        return 0

    stem = _stem_for(args.corpus)
    from tools.build_chapter_map import build as build_cm
    from tools.extract_inline_formulas import repair as extract_inline
    from tools.build_formula_explanations import enrich
    from tools.repair_formula_latex import repair
    from tools.build_sevim_diagrams import build as build_sd

    built = 0
    skipped = 0
    failed: list[str] = []
    for i, nid in enumerate(chapters, 1):
        flat = nid.replace("/", "_")
        cm_path = f"{stem}.chapter_map.{flat}.json"
        print(f"\n[{i}/{len(chapters)}] {nid}")
        if os.path.isfile(cm_path) and not args.force:
            print(f"  chapter_map exists ({cm_path}) — skip "
                  f"(pass --force to rebuild)")
            skipped += 1
            continue
        rc = build_cm(args.corpus, nid, cm_path)
        if rc != 0:
            print(f"  build_chapter_map failed (rc={rc})")
            failed.append(nid)
            continue
        try:
            extract_inline(cm_path)
        except Exception as e:
            print(f"  extract_inline_formulas error: {e}")
        try:
            enrich(cm_path)
        except Exception as e:
            print(f"  formula_explanations error: {e}")
        try:
            repair(cm_path)
        except Exception as e:
            print(f"  repair_formula_latex error: {e}")
        try:
            build_sd(cm_path, force=False)
        except Exception as e:
            print(f"  sevim_diagrams error: {e}")
        built += 1

    print(f"\n[ingest-chapters] built {built}, skipped {skipped}, "
          f"failed {len(failed)}")
    if failed:
        print(f"  failed nids: {failed}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
