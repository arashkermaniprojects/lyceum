"""Tests for the recursive BookNode IR — depth, traversal, lookups."""
from book.ir import (
    BookNode, RECOMMENDED_KINDS, ENVIRONMENT_KINDS, STRUCTURAL_KINDS,
    is_environment, is_known_kind, is_structural,
    nid_segment, join_nid,
)


def _leaf(nid: str, kind: str = "remark", title: str = "leaf") -> BookNode:
    return BookNode(nid=nid, kind=kind, number=None, title=title,
                    page_start=1, page_end=1)


def test_depth_zero_for_leaf():
    n = _leaf("x")
    assert n.depth() == 0


def test_depth_counts_max_branch():
    root = BookNode(nid="r", kind="book", number=None, title="r",
                    page_start=1, page_end=10, children=[
                        _leaf("a"),
                        BookNode(nid="b", kind="chapter", number="1",
                                title="b", page_start=1, page_end=5,
                                children=[_leaf("c"), _leaf("d")]),
                    ])
    assert root.depth() == 2


def test_walk_preorder():
    root = BookNode(nid="r", kind="book", number=None, title="r",
                    page_start=1, page_end=10, children=[
                        BookNode(nid="r/a", kind="chapter", number="1",
                                title="a", page_start=1, page_end=3,
                                children=[_leaf("r/a/x")]),
                        _leaf("r/b"),
                    ])
    nids = [n.nid for n in root.walk()]
    assert nids == ["r", "r/a", "r/a/x", "r/b"]


def test_find_by_number():
    root = BookNode(nid="r", kind="book", number=None, title="r",
                    page_start=1, page_end=5, children=[
                        BookNode(nid="r/t1", kind="theorem", number="3.2",
                                title="t1", page_start=2, page_end=2),
                        BookNode(nid="r/t2", kind="theorem", number="3.3",
                                title="t2", page_start=3, page_end=3),
                    ])
    assert root.find_by_number("3.2").nid == "r/t1"
    assert root.find_by_number("9.9") is None


def test_find_by_nid():
    root = BookNode(nid="r", kind="book", number=None, title="r",
                    page_start=1, page_end=1, children=[_leaf("r/a"), _leaf("r/b")])
    assert root.find("r/b").title == "leaf"
    assert root.find("r/none") is None


def test_kind_classification():
    assert is_structural("chapter")
    assert is_environment("theorem")
    assert is_known_kind("definition")
    assert not is_structural("theorem")
    assert not is_environment("chapter")
    # Open-string: unknown kinds are accepted but report None for the helpers.
    assert not is_known_kind("custom_book_environment")


def test_kind_sets_are_disjoint():
    assert STRUCTURAL_KINDS.isdisjoint(ENVIRONMENT_KINDS)
    assert (STRUCTURAL_KINDS | ENVIRONMENT_KINDS) <= RECOMMENDED_KINDS


def test_nid_segment_handles_kind_prefix():
    assert nid_segment("chapter", "3", "Linear Maps") == "ch3"
    assert nid_segment("theorem", "3.2.1", "FTA") == "thm3_2_1"


def test_nid_segment_falls_back_to_title():
    seg = nid_segment("chapter", None, "Preliminaries")
    assert seg.startswith("ch_")
    assert "preliminaries" in seg


def test_join_nid_root_uses_segment_only():
    assert join_nid("", "ch1") == "ch1"
    assert join_nid("b", "ch1") == "b/ch1"
    assert join_nid("b/ch1", "s1.2") == "b/ch1/s1.2"


def test_all_text_concatenates_descendants():
    root = BookNode(nid="r", kind="book", number=None, title="r",
                    page_start=1, page_end=5,
                    body_text="root text",
                    children=[
                        BookNode(nid="r/a", kind="chapter", number="1",
                                title="a", page_start=1, page_end=3,
                                body_text="chapter text",
                                children=[
                                    BookNode(nid="r/a/x", kind="theorem",
                                             number="1.1", title="thm",
                                             page_start=2, page_end=2,
                                             body_text="theorem text"),
                                ]),
                    ])
    out = root.all_text()
    assert "root text" in out
    assert "chapter text" in out
    assert "theorem text" in out
