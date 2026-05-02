"""Book Intermediate Representation — recursive hierarchy + overlays.

The book is the source of truth for narrator-driven teaching.  Everything
the system visualises is *retrieved* from this corpus; nothing is invented.

The hierarchy is recursive and arbitrary-depth:

    Book.root  →  BookNode (kind="book")
      ├── BookNode (kind="part")
      │     └── BookNode (kind="chapter")
      │           └── BookNode (kind="section")
      │                 └── BookNode (kind="subsection")
      │                       └── BookNode (kind="theorem")
      │                             └── BookNode (kind="proof")
      │                                   └── BookNode (kind="lemma")  …

Three orthogonal overlays sit on top of the tree:

  * `figures`     — extracted images keyed to their owning node(s).
  * `cross_refs`  — flat citation graph between any two nodes.
  * `concepts`    — per-concept index with visual templates by context.

The IR is *deterministic*: parsing the same PDF produces the same JSON bytes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Kind taxonomy — open enum (string), with a recommended set.
# ---------------------------------------------------------------------------
# Open-string, not Literal: math books invent their own environments
# ("Construction", "Notation", "Convention", "Heuristic"…).  Any
# `\begin{custom}` becomes `kind="custom"` without forcing canonicalisation.
# Authors of downstream code should defensively fall back when a kind is
# unknown rather than assert against a closed set.

# Recommended kinds.  Use `book.ir.is_known_kind(s)` to test membership.
RECOMMENDED_KINDS: frozenset[str] = frozenset({
    # Structural
    "book", "part", "chapter", "section", "subsection",
    "subsubsection", "paragraph", "subparagraph",
    # Mathematical environments
    "definition", "theorem", "lemma", "corollary", "proposition",
    "claim", "fact", "remark", "example", "exercise", "solution",
    "proof", "problem", "hint", "construction", "convention", "notation",
    # Auxiliary
    "preface", "introduction", "appendix", "bibliography",
    "index", "glossary", "abstract", "acknowledgements",
})

# Structural kinds form the spine of the tree (chapters, sections, …).
# Environment kinds (theorem, definition, …) are leaves of the structural
# spine but may themselves contain children (a proof can hold lemmas).
STRUCTURAL_KINDS: frozenset[str] = frozenset({
    "book", "part", "chapter", "section", "subsection",
    "subsubsection", "paragraph", "subparagraph",
    "preface", "introduction", "appendix",
    "bibliography", "index", "glossary",
})

ENVIRONMENT_KINDS: frozenset[str] = frozenset({
    "definition", "theorem", "lemma", "corollary", "proposition",
    "claim", "fact", "remark", "example", "exercise", "solution",
    "proof", "problem", "hint", "construction", "convention", "notation",
    "abstract",
})


def is_known_kind(kind: str) -> bool:
    return kind in RECOMMENDED_KINDS


def is_structural(kind: str) -> bool:
    return kind in STRUCTURAL_KINDS


def is_environment(kind: str) -> bool:
    return kind in ENVIRONMENT_KINDS


# ---------------------------------------------------------------------------
# BookNode — the recursive container
# ---------------------------------------------------------------------------

# Sentinel for unknown page bounds.
PAGE_UNKNOWN: int = -1


@dataclass
class BookNode:
    """One node in the book's hierarchical tree.

    Attributes
    ----------
    nid:
        Stable, hierarchical identifier built from path components.
        Example: ``"b/p1/ch3/s3.2/ss3.2.1/thm3.2.1.5/proof"``
    kind:
        One of the recommended kinds, OR an arbitrary book-defined string.
    number:
        The book's own numbering (``"3.2.1"`` or ``"Theorem 4.5.a"``).  None
        when the node is unnumbered (preface, abstract, anonymous remark).
    title:
        Display title.  May be empty for unnamed environments.
    page_start, page_end:
        Inclusive page bounds.  ``PAGE_UNKNOWN`` (-1) when unknown.
    body_text:
        Text directly belonging to this node, *excluding* descendants.
        For a section that contains five subsections, this holds the prose
        between the section heading and the first subsection.  Empty when
        the node is a pure container with no own prose.
    children:
        Ordered list of child BookNodes.  May be empty for leaves.
    meta:
        Kind-specific extras: theorem dependencies, proof references,
        author of the chapter (in edited volumes), language, etc.
    """
    nid: str
    kind: str
    number: Optional[str]
    title: str
    page_start: int
    page_end: int
    body_text: str = ""
    children: list["BookNode"] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ----------------- traversal helpers --------------------------

    def walk(self) -> Iterator["BookNode"]:
        """Pre-order traversal (this node first, then each child's subtree)."""
        yield self
        for c in self.children:
            yield from c.walk()

    def depth(self) -> int:
        """Tree depth rooted at this node (a leaf has depth 0)."""
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def find(self, nid: str) -> Optional["BookNode"]:
        """Return the descendant with the given nid, or None."""
        for n in self.walk():
            if n.nid == nid:
                return n
        return None

    def find_by_number(self, number: str) -> Optional["BookNode"]:
        """Return the first descendant whose `number` matches exactly.
        Used by cross-reference resolution (``see Theorem 3.2.1``)."""
        for n in self.walk():
            if n.number == number:
                return n
        return None

    def all_text(self) -> str:
        """Concatenate body_text from this node and all descendants in
        traversal order.  Useful for full-text search."""
        return "\n".join(n.body_text for n in self.walk() if n.body_text)


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------

@dataclass
class FigureRef:
    """An image extracted from the book.

    A figure may be referenced from multiple BookNodes.  ``home_nid`` is the
    node where the figure physically lives in the typeset document; other
    nodes that mention the figure participate via cross_refs.
    """
    fid: str               # stable id, e.g. "fig_3_2"
    home_nid: str          # node that owns the figure (where it's rendered)
    page: int
    caption: str
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1) on page
    image_path: str        # relative path to the extracted image file
    meta: dict = field(default_factory=dict)


@dataclass
class CrossRef:
    """A citation overlay edge between two BookNodes.

    Captured by scanning body_text for patterns like
    ``see Theorem 3.2.1`` and resolving the number to a target node.
    """
    from_nid: str          # source BookNode where the citation appears
    to_nid: str            # target BookNode being cited
    label: str             # rendered text, e.g. "Theorem 3.2.1"
    char_offset: int       # offset within from_nid.body_text, for highlighting


# ---------------------------------------------------------------------------
# Concept index
# ---------------------------------------------------------------------------

@dataclass
class ConceptTemplate:
    """One context-specific visual template for a concept.

    The same word ("matrix") may carry different visual conventions across
    the book — bracketed grids in chapter 1, tensor diagrams in chapter 8.
    Each occurrence-context produces one template.
    """
    home_nid: str          # the BookNode where this template was inferred
    primitive: str         # SeVim primitive name (set_blob, matrix_bracket, …)
    meta: dict             # primitive-specific kwargs (cells, nrows, label, …)
    evidence: dict         # provenance: page, snippet, latex_source if any


@dataclass
class ConceptEntry:
    """All knowledge the corpus has about one concept name.

    Keyed in ``Book.concepts`` by a normalised concept id ("matrix",
    "linear_map", "eigenvalue").  Multiple natural-language surface forms
    can map to the same id (handled by ``aliases``).
    """
    cid: str                          # normalised id, e.g. "matrix"
    canonical: str                    # display label, e.g. "matrix"
    aliases: list[str]                # alternative surface forms
    definitions: list[tuple[str, str]] # [(home_nid, definition_text), …]
    templates: list[ConceptTemplate]
    figure_refs: list[str]            # FigureRef.fid values
    embedding: tuple[float, ...] = ()  # optional, for retrieval


# ---------------------------------------------------------------------------
# Book — the top-level corpus
# ---------------------------------------------------------------------------

@dataclass
class Book:
    """The full corpus extracted from one source document.

    Determinism: parsing the same PDF (with the same ingestion options)
    must produce a Book whose JSON serialisation is byte-identical across
    runs.
    """
    title: str
    author: Optional[str]
    source: str                       # original file path / URL
    root: BookNode                    # root of the recursive tree
    figures: list[FigureRef] = field(default_factory=list)
    cross_refs: list[CrossRef] = field(default_factory=list)
    concepts: dict[str, ConceptEntry] = field(default_factory=dict)
    pages: list[str] = field(default_factory=list)  # raw text per page index
    meta: dict = field(default_factory=dict)        # ISBN, year, ingestion ts

    # ----------------- convenience -----------------------------------------

    def find(self, nid: str) -> Optional[BookNode]:
        return self.root.find(nid)

    def find_by_number(self, number: str) -> Optional[BookNode]:
        return self.root.find_by_number(number)


# ---------------------------------------------------------------------------
# nid construction helpers
# ---------------------------------------------------------------------------
# nids are slash-joined path components.  Each segment is normalised to
# [a-z0-9_] so the full nid is filename-safe and URL-safe.

_SEG_RE = re.compile(r"[^a-z0-9]+")


def nid_segment(kind: str, number: Optional[str], title: str) -> str:
    """Build one path segment from a node's kind/number/title."""
    parts: list[str] = []
    # Short prefix from kind.
    prefix = {
        "book": "b", "part": "p", "chapter": "ch",
        "section": "s", "subsection": "ss", "subsubsection": "sss",
        "paragraph": "para", "subparagraph": "subpara",
        "definition": "def", "theorem": "thm", "lemma": "lem",
        "corollary": "cor", "proposition": "prop", "claim": "cl",
        "fact": "fact", "remark": "rem", "example": "ex",
        "exercise": "exer", "solution": "sol", "proof": "proof",
        "problem": "prob", "hint": "hint",
        "preface": "pref", "introduction": "intro",
        "appendix": "app", "bibliography": "biblio",
        "abstract": "abs",
    }.get(kind, kind)
    if number:
        parts.append(f"{prefix}{_SEG_RE.sub('_', number.lower()).strip('_')}")
    elif title:
        compact = _SEG_RE.sub("_", title.lower()).strip("_")[:24]
        parts.append(f"{prefix}_{compact}" if compact else prefix)
    else:
        parts.append(prefix)
    return parts[0]


def join_nid(parent_nid: str, segment: str) -> str:
    """Compose a child nid from its parent's nid + segment."""
    if not parent_nid:
        return segment
    return f"{parent_nid}/{segment}"
