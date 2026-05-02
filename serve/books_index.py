"""Tiny JSON-backed registry for runtime-uploaded books.

Pre-loaded books (the ones passed on the ``serve.server`` CLI) do
*not* live here — they're tracked in the in-memory ``books_by_name``
map.  This file's job is to record what's been *uploaded by the user
through the UI* so a server restart doesn't lose them, and so the
ingest progress page has a place to look.

File location: ``books/_index.json`` (next to the per-book corpus
JSONs).  Schema:

    {
      "version": 1,
      "books": [
        {
          "id": "elements-of-statistical-learning-1xb2",
          "title": "The Elements of Statistical Learning",
          "source_pdf": "books/elements-of-statistical-learning-1xb2.pdf",
          "corpus_json": "books/elements-of-statistical-learning-1xb2.json",
          "phase": "ready",
          "progress": "all chapters built",
          "added_at": "2026-05-02T12:48:30Z",
          "ready_at": "2026-05-02T13:11:02Z",
          "error": "",
          "ingested_chapters": ["b/ch5", "b/ch6"],
          "total_chapters": 18
        }
      ]
    }

``phase`` is one of ``uploading``, ``extracting``, ``building_concepts``,
``building_math_graph``, ``building_chapter_maps``, ``ready``, ``failed``.

Concurrency: write is atomic via temp+rename; in-memory cache is
guarded by a module-level lock so two upload requests can't trample
each other's edits.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from typing import Any


_INDEX_FILENAME = "_index.json"
_LOCK = threading.RLock()
_VERSION = 1


def _index_path(books_dir: str) -> str:
    return os.path.join(books_dir, _INDEX_FILENAME)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(books_dir: str) -> dict:
    """Load the index, returning an empty registry if the file is
    missing or corrupt.  Always returns a dict shaped like the schema
    above, with at least an empty ``books`` list."""
    path = _index_path(books_dir)
    if not os.path.isfile(path):
        return {"version": _VERSION, "books": []}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {"version": _VERSION, "books": []}
    if not isinstance(data, dict):
        return {"version": _VERSION, "books": []}
    data.setdefault("version", _VERSION)
    books = data.get("books") or []
    if not isinstance(books, list):
        books = []
    data["books"] = [b for b in books if isinstance(b, dict)]
    return data


def save(books_dir: str, data: dict) -> None:
    """Atomic write to the index.  Caller holds whatever locking
    they need; this function only does fs-level atomicity (temp +
    rename)."""
    os.makedirs(books_dir, exist_ok=True)
    path = _index_path(books_dir)
    fd, tmp = tempfile.mkstemp(prefix="_index.", suffix=".json",
                                dir=books_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_SAFE_ID_RE = re.compile(r"[^a-z0-9]+")


def slugify_id(stem: str) -> str:
    """Turn a filename stem into a URL-safe id; collisions resolved
    by the caller via :func:`unique_id`."""
    s = _SAFE_ID_RE.sub("-", (stem or "book").lower()).strip("-")
    return s or "book"


def unique_id(books_dir: str, stem: str) -> str:
    """Pick a unique book id given a candidate stem.  Adds a short
    random suffix if the slug already exists in the index OR a file
    with the same stem already lives in ``books_dir``."""
    base = slugify_id(stem)
    with _LOCK:
        idx = load(books_dir)
        existing_ids = {b.get("id") for b in idx["books"]}

        def _collides(candidate: str) -> bool:
            if candidate in existing_ids:
                return True
            for ext in (".pdf", ".json"):
                if os.path.exists(os.path.join(
                        books_dir, candidate + ext)):
                    return True
            return False

        if not _collides(base):
            return base
        # 4-char random suffix; one retry path.
        import secrets
        for _ in range(8):
            cand = f"{base}-{secrets.token_hex(2)}"
            if not _collides(cand):
                return cand
    return f"{base}-{int(time.time())}"


def add_book(books_dir: str, *, book_id: str, title: str,
             source_pdf: str, corpus_json: str) -> dict:
    """Append a fresh book entry with phase=uploading.  Returns the
    new entry."""
    entry = {
        "id": book_id,
        "title": title,
        "source_pdf": source_pdf,
        "corpus_json": corpus_json,
        "phase": "uploading",
        "progress": "saved PDF, waiting to start ingest",
        "added_at": _now_iso(),
        "ready_at": "",
        "error": "",
        "ingested_chapters": [],
        "total_chapters": 0,
    }
    with _LOCK:
        data = load(books_dir)
        data["books"].append(entry)
        save(books_dir, data)
    return entry


def get_book(books_dir: str, book_id: str) -> dict | None:
    with _LOCK:
        for b in load(books_dir)["books"]:
            if b.get("id") == book_id:
                return b
    return None


def update_book(books_dir: str, book_id: str, **fields: Any) -> dict | None:
    """Patch an existing entry's fields atomically; returns the
    updated entry or None if the id is unknown."""
    with _LOCK:
        data = load(books_dir)
        for b in data["books"]:
            if b.get("id") != book_id:
                continue
            b.update(fields)
            if fields.get("phase") == "ready" and not b.get("ready_at"):
                b["ready_at"] = _now_iso()
            save(books_dir, data)
            return b
    return None


def list_books(books_dir: str) -> list[dict]:
    with _LOCK:
        return list(load(books_dir)["books"])
