"""sevim-ingest CLI — PDF → corpus JSON.

Usage:
    sevim-ingest BOOK.pdf [-o BOOK.json] [--figures DIR] [--no-concepts]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import (
    parse_pdf, extract_concepts, extract_cross_refs, write_corpus,
    embed_book, embeddings_available,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sevim-ingest",
        description="Parse a math textbook PDF into a SeVim corpus JSON.",
    )
    p.add_argument("pdf", help="path to source PDF")
    p.add_argument(
        "-o", "--out", default=None,
        help="output JSON path (default: replaces .pdf with .json)",
    )
    p.add_argument(
        "--figures", default=None,
        help="directory to extract embedded images into "
             "(default: <out>_figures/)",
    )
    p.add_argument(
        "--no-concepts", action="store_true",
        help="skip concept extraction (only structural tree + figures)",
    )
    p.add_argument(
        "--no-crossrefs", action="store_true",
        help="skip cross-reference extraction",
    )
    p.add_argument(
        "--title", default=None,
        help="override book title (otherwise read from PDF metadata)",
    )
    p.add_argument(
        "--embed", action="store_true",
        help="populate per-node + per-concept embeddings via local vLLM "
             "(requires the embedding server at LYCEUM_EMBED_URL or "
             "http://127.0.0.1:8003/v1)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.pdf):
        print(f"sevim-ingest: error: {args.pdf} not found", file=sys.stderr)
        return 2

    out_path = args.out or os.path.splitext(args.pdf)[0] + ".json"
    figures_dir = args.figures or os.path.splitext(out_path)[0] + "_figures"

    t0 = time.perf_counter()
    book = parse_pdf(args.pdf, figures_dir=figures_dir,
                     title_override=args.title)
    t_parse = time.perf_counter() - t0

    if not args.no_concepts:
        t1 = time.perf_counter()
        book.concepts = extract_concepts(book)
        t_concepts = time.perf_counter() - t1
    else:
        t_concepts = 0.0

    if not args.no_crossrefs:
        t2 = time.perf_counter()
        book.cross_refs = extract_cross_refs(book)
        t_xref = time.perf_counter() - t2
    else:
        t_xref = 0.0

    t_embed = 0.0
    embed_stats = None
    if args.embed:
        if not embeddings_available():
            print("[sevim-ingest] --embed requested but vLLM embed server "
                  "is not reachable; skipping embeddings", file=sys.stderr)
        else:
            t3 = time.perf_counter()
            n_nodes_pre = sum(1 for _ in book.root.walk())
            def _prog(done: int, total: int) -> None:
                pct = int(100 * done / max(total, 1))
                sys.stderr.write(f"\r[embed] {done}/{total} ({pct}%)")
                sys.stderr.flush()
            embed_stats = embed_book(book, progress=_prog)
            sys.stderr.write("\n")
            t_embed = time.perf_counter() - t3

    write_corpus(out_path, book)
    t_total = time.perf_counter() - t0

    n_nodes = sum(1 for _ in book.root.walk())
    print(
        f"ingested {args.pdf}\n"
        f"  → {out_path}\n"
        f"  pages={len(book.pages)}  "
        f"nodes={n_nodes}  depth={book.root.depth()}\n"
        f"  figures={len(book.figures)}  "
        f"concepts={len(book.concepts)}  "
        f"cross_refs={len(book.cross_refs)}\n"
        f"  timing: parse={t_parse*1000:.0f} ms  "
        f"concepts={t_concepts*1000:.0f} ms  "
        f"crossrefs={t_xref*1000:.0f} ms  "
        f"embed={t_embed*1000:.0f} ms  "
        f"total={t_total*1000:.0f} ms"
    )
    if embed_stats:
        print(f"  embeddings: {embed_stats['n_nodes_embedded']}/"
              f"{embed_stats['n_nodes_total']} nodes, "
              f"{embed_stats['n_concepts_embedded']}/"
              f"{embed_stats['n_concepts_total']} concepts, "
              f"dim={embed_stats['vector_dim']}, "
              f"model={embed_stats['model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
