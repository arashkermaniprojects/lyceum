"""Book corpus serialisation — Book ↔ JSON.

Determinism is the goal:
- ``write_corpus(path, book)`` writes byte-identical JSON for byte-identical
  inputs (sorted keys, fixed indent, no insertion timestamps in the IR).
- ``load_corpus(path)`` round-trips back to a fully-typed :class:`Book`.

The JSON schema is **versioned**: ``meta["schema_version"] = 1``.  Future
changes that break compatibility bump the integer; readers fall back
gracefully when older versions appear.
"""
from __future__ import annotations

import json
from typing import Any

from .ir import (
    Book, BookNode, ConceptEntry, ConceptTemplate,
    CrossRef, FigureRef,
)


SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _node_to_dict(n: BookNode) -> dict:
    return {
        "nid": n.nid,
        "kind": n.kind,
        "number": n.number,
        "title": n.title,
        "page_start": n.page_start,
        "page_end": n.page_end,
        "body_text": n.body_text,
        "children": [_node_to_dict(c) for c in n.children],
        "meta": dict(n.meta),
    }


def _figure_to_dict(f: FigureRef) -> dict:
    return {
        "fid": f.fid, "home_nid": f.home_nid, "page": f.page,
        "caption": f.caption, "bbox": list(f.bbox),
        "image_path": f.image_path, "meta": dict(f.meta),
    }


def _crossref_to_dict(c: CrossRef) -> dict:
    return {
        "from_nid": c.from_nid, "to_nid": c.to_nid,
        "label": c.label, "char_offset": c.char_offset,
    }


def _template_to_dict(t: ConceptTemplate) -> dict:
    return {
        "home_nid": t.home_nid, "primitive": t.primitive,
        "meta": dict(t.meta), "evidence": dict(t.evidence),
    }


def _entry_to_dict(e: ConceptEntry) -> dict:
    return {
        "cid": e.cid, "canonical": e.canonical,
        "aliases": list(e.aliases),
        "definitions": [list(d) for d in e.definitions],
        "templates": [_template_to_dict(t) for t in e.templates],
        "figure_refs": list(e.figure_refs),
        "embedding": list(e.embedding) if e.embedding else [],
    }


def book_to_dict(book: Book) -> dict[str, Any]:
    """Convert a :class:`Book` to a plain dict.  Round-trip-safe."""
    return {
        "schema_version": SCHEMA_VERSION,
        "title": book.title,
        "author": book.author,
        "source": book.source,
        "root": _node_to_dict(book.root),
        "figures": [_figure_to_dict(f) for f in book.figures],
        "cross_refs": [_crossref_to_dict(c) for c in book.cross_refs],
        "concepts": {cid: _entry_to_dict(e)
                     for cid, e in sorted(book.concepts.items())},
        "pages": list(book.pages),
        "meta": {**book.meta, "schema_version": SCHEMA_VERSION},
    }


def write_corpus(
    path: str, book: Book, *, indent: int = 2,
) -> None:
    """Write ``book`` to *path* as deterministic JSON.

    Keys are sorted; concept entries are alphabetised.  Identical inputs
    produce byte-identical files.
    """
    data = book_to_dict(book)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=indent, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Deserialisation
# ---------------------------------------------------------------------------

def _dict_to_node(d: dict) -> BookNode:
    return BookNode(
        nid=d["nid"], kind=d["kind"],
        number=d.get("number"),
        title=d.get("title", ""),
        page_start=int(d.get("page_start", -1)),
        page_end=int(d.get("page_end", -1)),
        body_text=d.get("body_text", ""),
        children=[_dict_to_node(c) for c in d.get("children", [])],
        meta=dict(d.get("meta", {})),
    )


def _dict_to_figure(d: dict) -> FigureRef:
    return FigureRef(
        fid=d["fid"], home_nid=d["home_nid"], page=int(d["page"]),
        caption=d.get("caption", ""),
        bbox=tuple(d.get("bbox", (0.0, 0.0, 0.0, 0.0))),
        image_path=d.get("image_path", ""),
        meta=dict(d.get("meta", {})),
    )


def _dict_to_crossref(d: dict) -> CrossRef:
    return CrossRef(
        from_nid=d["from_nid"], to_nid=d["to_nid"],
        label=d.get("label", ""), char_offset=int(d.get("char_offset", 0)),
    )


def _dict_to_template(d: dict) -> ConceptTemplate:
    return ConceptTemplate(
        home_nid=d["home_nid"],
        primitive=d.get("primitive", "rect"),
        meta=dict(d.get("meta", {})),
        evidence=dict(d.get("evidence", {})),
    )


def _dict_to_entry(d: dict) -> ConceptEntry:
    return ConceptEntry(
        cid=d["cid"], canonical=d.get("canonical", d["cid"]),
        aliases=list(d.get("aliases", [])),
        definitions=[(p[0], p[1]) for p in d.get("definitions", [])],
        templates=[_dict_to_template(t) for t in d.get("templates", [])],
        figure_refs=list(d.get("figure_refs", [])),
        embedding=tuple(d.get("embedding", [])),
    )


def dict_to_book(data: dict) -> Book:
    return Book(
        title=data.get("title", ""),
        author=data.get("author"),
        source=data.get("source", ""),
        root=_dict_to_node(data["root"]),
        figures=[_dict_to_figure(f) for f in data.get("figures", [])],
        cross_refs=[_dict_to_crossref(c) for c in data.get("cross_refs", [])],
        concepts={cid: _dict_to_entry(e)
                  for cid, e in data.get("concepts", {}).items()},
        pages=list(data.get("pages", [])),
        meta=dict(data.get("meta", {})),
    )


def load_corpus(path: str) -> Book:
    """Load a previously-written corpus JSON back into a :class:`Book`.

    Sidecars produced by ``tools/reingest_figures.py`` and
    ``tools/reingest_equations.py`` are merged in if present:

      * ``{stem}_figures_v2.json`` → augments / replaces ``Book.figures``
      * ``{stem}_equations.json``  → stored under ``Book.meta['equations']``

    This lets background re-ingestion deliver better metadata to the
    live system on the next server restart, with no schema migration.
    """
    import os
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    schema = data.get("schema_version") or data.get("meta", {}).get("schema_version")
    if schema is not None and schema > SCHEMA_VERSION:
        raise ValueError(
            f"corpus schema version {schema} is newer than reader's "
            f"{SCHEMA_VERSION}; upgrade book/corpus.py"
        )
    book = dict_to_book(data)

    # ---- Sidecar: re-ingested figures (Phase 2) -------------------------
    stem, _ = os.path.splitext(path)
    figs_v2 = stem + "_figures_v2.json"
    if os.path.isfile(figs_v2):
        try:
            with open(figs_v2, "r", encoding="utf-8") as f:
                payload = json.load(f)
            entries = payload.get("figures") or []
            existing_fids = {f.fid for f in book.figures}
            from .ir import FigureRef
            added = 0
            for e in entries:
                fid = e.get("fid")
                if not fid or fid in existing_fids:
                    continue
                book.figures.append(FigureRef(
                    fid=fid,
                    home_nid=e.get("home_nid", ""),
                    page=int(e.get("page") or 0),
                    caption=e.get("caption", ""),
                    bbox=tuple(e.get("bbox") or (0, 0, 0, 0)),
                    image_path=e.get("image_path", ""),
                    meta={"source": "reingest_v2",
                          "label": e.get("label", "")},
                ))
                added += 1
            if added:
                book.meta["figures_v2_added"] = added
        except Exception as exc:
            book.meta["figures_v2_load_error"] = str(exc)

    # ---- Sidecar: re-ingested equations (Phase 3) -----------------------
    eqs_path = stem + "_equations.json"
    if os.path.isfile(eqs_path):
        try:
            with open(eqs_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            book.meta["equations"] = payload.get("equations") or []
        except Exception as exc:
            book.meta["equations_load_error"] = str(exc)

    return book
