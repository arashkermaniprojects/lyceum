"""Pin multi-book server state: loading multiple corpora, switching
the active one, per-book alias / concept caches, list endpoint.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from book.ir import Book


def _write_minimal_corpus(path: str, *, title: str, chapters: int = 3) -> None:
    """Write a tiny ESLII-shaped corpus JSON to *path*."""
    root = {
        "nid": "b", "kind": "book", "number": None, "title": title,
        "page_start": 1, "page_end": 999, "body_text": "",
        "children": [
            {"nid": f"b/ch{i}", "kind": "chapter", "number": str(i),
             "title": f"Chapter {i}", "page_start": i * 10,
             "page_end": (i + 1) * 10, "body_text": "",
             "children": []}
            for i in range(1, chapters + 1)
        ],
    }
    payload = {
        "title": title, "author": None, "source": "",
        "root": root, "concepts": {}, "pages": [], "figures": [],
        "cross_refs": [], "meta": {},
    }
    with open(path, "w") as f:
        json.dump(payload, f)


@pytest.fixture
def two_books(tmp_path):
    a = tmp_path / "alpha.json"
    b = tmp_path / "beta.json"
    _write_minimal_corpus(str(a), title="Alpha Book", chapters=3)
    _write_minimal_corpus(str(b), title="Beta Book", chapters=5)
    return [str(a), str(b)]


# ---------------------------------------------------------------------------
# Server initialisation accepts multiple paths
# ---------------------------------------------------------------------------

def test_server_loads_multiple_books(two_books, monkeypatch):
    """Constructing the Server with two paths populates books_by_name."""
    monkeypatch.setenv("SEVIM_DISABLE_ASR", "1")
    monkeypatch.setenv("SEVIM_SKIP_LLM_EQ_LATEX", "1")
    from serve.server import Server
    srv = Server(book_paths=two_books, prefer_kokoro=False, port=18001)
    try:
        assert "alpha" in srv.books_by_name
        assert "beta" in srv.books_by_name
        # First-given is active by default.
        assert srv.active_book_name == "alpha"
        # Per-book alias + concept_graph caches exist.
        assert "alpha" in srv.book_aliases
        assert "alpha" in srv.book_concept_graphs
    finally:
        # No serve() called → nothing to tear down.
        pass


def test_activate_book_swaps_qa_caches(two_books, monkeypatch):
    monkeypatch.setenv("SEVIM_DISABLE_ASR", "1")
    monkeypatch.setenv("SEVIM_SKIP_LLM_EQ_LATEX", "1")
    from serve.server import Server, SERVER_STATE
    from narrator import qa as qa_mod
    srv = Server(book_paths=two_books, prefer_kokoro=False, port=18002)
    # Spike known maps so we can detect the swap.
    srv.book_aliases["alpha"] = {"a-only": "alpha-canonical"}
    srv.book_aliases["beta"] = {"b-only": "beta-canonical"}
    srv._activate_book("alpha")
    assert qa_mod.get_alias_map().get("a-only") == "alpha-canonical"
    srv._activate_book("beta")
    assert qa_mod.get_alias_map().get("b-only") == "beta-canonical"
    assert "a-only" not in qa_mod.get_alias_map()
    assert SERVER_STATE.get("active_book_name") == "beta"


def test_activate_book_unknown_raises(two_books, monkeypatch):
    monkeypatch.setenv("SEVIM_DISABLE_ASR", "1")
    monkeypatch.setenv("SEVIM_SKIP_LLM_EQ_LATEX", "1")
    from serve.server import Server
    srv = Server(book_paths=two_books, prefer_kokoro=False, port=18003)
    with pytest.raises(KeyError):
        srv._activate_book("does-not-exist")


def test_existing_session_keeps_its_book(two_books, monkeypatch):
    """Switching the active book must not corrupt sessions already
    bound to the previous book — they hold a direct reference to the
    Book object, not a pointer."""
    monkeypatch.setenv("SEVIM_DISABLE_ASR", "1")
    monkeypatch.setenv("SEVIM_SKIP_LLM_EQ_LATEX", "1")
    from serve.server import Server
    from serve.session import build_session
    from narrator import NarrationClause, NarrationPlan
    from narrator.tts import NullTTS
    srv = Server(book_paths=two_books, prefer_kokoro=False, port=18004)
    alpha = srv.books_by_name["alpha"]
    plan = NarrationPlan(
        topic="<test>", book_title=alpha.title,
        clauses=[NarrationClause(text="t", home_nid="b",
                                  concepts=[], suggested_dur=1.0)],
        visited_nids=["b"], meta={"mode": "full"},
    )
    sess = build_session(
        plan_id="test-plan", book=alpha, plan=plan,
        tts_factory=lambda: NullTTS(),
    )
    srv._activate_book("beta")
    # Session still anchored on alpha.
    assert sess.book is alpha
    assert sess.book.title == "Alpha Book"
