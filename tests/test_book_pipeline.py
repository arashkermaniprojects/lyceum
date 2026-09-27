"""End-to-end tests against a local sample paper PDF (skipped when absent).

These are smoke tests over the full book pipeline: parse → concepts →
cross-refs → corpus serialise → corpus load → equivalence.

If the PDF is missing the tests skip cleanly so the suite still runs in
environments without it.
"""
import json
import os
import pytest

PDF_PATH = "tests/data/sample_paper.pdf"


def _pdf_available() -> bool:
    return os.path.exists(PDF_PATH)


pytestmark = pytest.mark.skipif(
    not _pdf_available(), reason=f"{PDF_PATH} not present"
)


def test_parse_pdf_returns_a_book(tmp_path):
    from book import parse_pdf
    b = parse_pdf(PDF_PATH, figures_dir=str(tmp_path / "figs"))
    assert b.title  # non-empty
    assert len(b.pages) > 0
    assert b.root.kind == "book"
    # Tree has at least chapter-level children.
    assert any(c.kind in ("chapter", "section", "introduction")
               for c in b.root.children)


def test_extract_concepts_finds_at_least_one_concept(tmp_path):
    from book import parse_pdf, extract_concepts
    b = parse_pdf(PDF_PATH, figures_dir=str(tmp_path / "figs"))
    concepts = extract_concepts(b)
    assert len(concepts) >= 1
    # Every entry has at least one template.
    for entry in concepts.values():
        assert len(entry.templates) >= 1


def test_extract_cross_refs_runs(tmp_path):
    from book import parse_pdf, extract_concepts, extract_cross_refs
    b = parse_pdf(PDF_PATH, figures_dir=str(tmp_path / "figs"))
    b.concepts = extract_concepts(b)
    refs = extract_cross_refs(b)
    # The sample paper may or may not cite numbered theorems; we just
    # require the function to return a list.
    assert isinstance(refs, list)


def test_corpus_round_trip_byte_identical(tmp_path):
    from book import parse_pdf, extract_concepts, write_corpus
    b = parse_pdf(PDF_PATH, figures_dir=str(tmp_path / "figs"))
    b.concepts = extract_concepts(b)
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    write_corpus(str(p1), b)
    write_corpus(str(p2), b)
    assert p1.read_bytes() == p2.read_bytes()


def test_corpus_load_round_trip_preserves_tree(tmp_path):
    from book import parse_pdf, extract_concepts, write_corpus, load_corpus
    b = parse_pdf(PDF_PATH, figures_dir=str(tmp_path / "figs"))
    b.concepts = extract_concepts(b)
    p = tmp_path / "rt.json"
    write_corpus(str(p), b)
    b2 = load_corpus(str(p))
    assert b2.title == b.title
    assert b2.root.depth() == b.root.depth()
    assert len(list(b2.root.walk())) == len(list(b.root.walk()))
    assert len(b2.concepts) == len(b.concepts)


def test_corpus_size_is_reasonable(tmp_path):
    """Sanity check: the sample paper should produce a corpus under ~5 MB."""
    from book import parse_pdf, extract_concepts, write_corpus
    b = parse_pdf(PDF_PATH, figures_dir=str(tmp_path / "figs"))
    b.concepts = extract_concepts(b)
    p = tmp_path / "size.json"
    write_corpus(str(p), b)
    size = p.stat().st_size
    assert size < 5 * 1024 * 1024  # 5 MB upper bound
