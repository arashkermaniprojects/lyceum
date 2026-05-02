"""Tests for corpus serialisation and round-trip determinism."""
import json
import os
import pytest

from book.ir import (
    Book, BookNode, ConceptEntry, ConceptTemplate, CrossRef, FigureRef,
)
from book.corpus import (
    write_corpus, load_corpus, book_to_dict, dict_to_book, SCHEMA_VERSION,
)


def _toy_book() -> Book:
    """A small but representative book — exercises every IR field."""
    leaf = BookNode(
        nid="b/ch1/thm1.1", kind="theorem", number="1.1",
        title="Pythagoras", page_start=2, page_end=2,
        body_text="In a right triangle, $a^2 + b^2 = c^2$.",
        children=[
            BookNode(nid="b/ch1/thm1.1/proof", kind="proof", number=None,
                     title="", page_start=2, page_end=3,
                     body_text="By construction…"),
        ],
        meta={"author": "Euclid"},
    )
    chapter = BookNode(
        nid="b/ch1", kind="chapter", number="1", title="Geometry",
        page_start=1, page_end=10,
        body_text="Chapter on classical geometry.",
        children=[leaf],
    )
    root = BookNode(nid="b", kind="book", number=None, title="Toybook",
                    page_start=1, page_end=10, children=[chapter])

    return Book(
        title="Toybook", author="A. Mathematician", source="/tmp/toy.pdf",
        root=root,
        figures=[FigureRef(
            fid="fig_2_00_x", home_nid="b/ch1/thm1.1", page=2,
            caption="Right triangle", bbox=(10.0, 20.0, 30.0, 40.0),
            image_path="fig_2_00.png",
        )],
        cross_refs=[CrossRef(
            from_nid="b/ch1/thm1.1/proof", to_nid="b/ch1/thm1.1",
            label="Theorem 1.1", char_offset=0,
        )],
        concepts={
            "matrix": ConceptEntry(
                cid="matrix", canonical="matrix",
                aliases=["matrix", "matrices"],
                definitions=[("b/ch1", "A matrix is …")],
                templates=[ConceptTemplate(
                    home_nid="b/ch1", primitive="matrix_bracket",
                    meta={"nrows": 2, "ncols": 2},
                    evidence={"page": 2, "snippet": "[[1,2],[3,4]]"},
                )],
                figure_refs=["fig_2_00_x"],
            ),
        },
        pages=["page1 text", "page2 text"],
        meta={"isbn": "000-0", "ingested_at": "2026-04-25"},
    )


def test_book_to_dict_round_trips_via_dict_to_book():
    b = _toy_book()
    out = dict_to_book(book_to_dict(b))
    assert out.title == b.title
    assert out.root.find("b/ch1/thm1.1").body_text == \
        b.root.find("b/ch1/thm1.1").body_text
    assert len(out.figures) == 1
    assert len(out.cross_refs) == 1
    assert "matrix" in out.concepts
    assert out.concepts["matrix"].templates[0].meta["nrows"] == 2


def test_write_corpus_is_deterministic(tmp_path):
    b = _toy_book()
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    write_corpus(str(p1), b)
    write_corpus(str(p2), b)
    assert p1.read_bytes() == p2.read_bytes()


def test_write_corpus_sorted_concept_keys(tmp_path):
    """Concept keys are alphabetised in the JSON output."""
    b = _toy_book()
    # Add a second concept after the first.
    b.concepts["aaa"] = ConceptEntry(
        cid="aaa", canonical="aaa", aliases=[], definitions=[],
        templates=[], figure_refs=[],
    )
    out = tmp_path / "c.json"
    write_corpus(str(out), b)
    data = json.loads(out.read_text())
    assert list(data["concepts"].keys()) == sorted(data["concepts"].keys())


def test_load_corpus_round_trip(tmp_path):
    b = _toy_book()
    p = tmp_path / "round.json"
    write_corpus(str(p), b)
    b2 = load_corpus(str(p))
    assert b2.title == b.title
    assert b2.root.depth() == b.root.depth()
    # Body text and meta survive.
    assert b2.root.find("b/ch1/thm1.1").meta == {"author": "Euclid"}


def test_schema_version_in_meta(tmp_path):
    b = _toy_book()
    p = tmp_path / "v.json"
    write_corpus(str(p), b)
    data = json.loads(p.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["meta"]["schema_version"] == SCHEMA_VERSION


def test_load_rejects_newer_schema(tmp_path):
    p = tmp_path / "future.json"
    fake = {"schema_version": SCHEMA_VERSION + 1,
            "title": "Future Book", "root": {
                "nid": "b", "kind": "book", "number": None, "title": "",
                "page_start": 1, "page_end": 1, "body_text": "",
                "children": [], "meta": {}}}
    p.write_text(json.dumps(fake))
    with pytest.raises(ValueError, match="schema version"):
        load_corpus(str(p))
