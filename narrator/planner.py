"""Narration planner — topic query → ordered NarrationPlan.

Pipeline
--------
1.  **Retrieve** relevant BookNodes for the topic via BM25 over body_text.
2.  **Order** the retrieved nodes by their position in the book (preserving
    pedagogical sequence).
3.  **Segment** each node's body_text into sentence-level clauses.
4.  **Tag** every concept mention in each clause with its canonical id +
    char offset (the same surface scanner used during ingestion).
5.  **Estimate duration** for each clause from a fixed reading rate
    (configurable; default 15 chars / second ≈ 180 wpm).

The plan is **deterministic**: same (book, topic) → same plan.  Every
ranking tie is broken on (page_start, nid).  No timestamps are written
into the plan — those come later from TTS.

A plan can be **replayed** identically by the orchestrator; the orchestrator
attaches actual time stamps from the audio stream once Kokoro emits them.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from book.ir import Book, BookNode

# Reuse the same surface-form scanner used during ingestion so concept
# tags match the corpus index.
from .resolver import resolve as resolve_concept   # noqa: F401  (used by callers)


# ---------------------------------------------------------------------------
# Tokenisation / BM25
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(r"[A-Za-z][A-Za-z\-]{1,}")

# PDF text extractors emit Unicode ligatures (ﬁ, ﬂ, ﬃ, …) verbatim.  Our
# tokenizer's [A-Za-z] class drops them, so "overﬁt" becomes "over" and
# BM25 mismatches against "overfit" queries entirely.  Normalise to
# multi-letter ASCII before tokenizing.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "ft", "ﬆ": "st",
    "Æ": "AE", "æ": "ae", "Œ": "OE", "œ": "oe", "ß": "ss",
}
_LIG_TRANS = str.maketrans(_LIGATURES)


def _normalize_ligatures(text: str) -> str:
    return (text or "").translate(_LIG_TRANS)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(_normalize_ligatures(text))]


# Tiny English stoplist — keeps math terms.
_STOP = frozenset({
    "the", "and", "of", "to", "a", "an", "in", "on", "by", "for",
    "is", "are", "was", "were", "be", "as", "this", "that", "these",
    "those", "with", "from", "or", "if", "so", "but", "we", "it",
    "such", "any", "each", "all",
})


def _bm25_scores(
    query: list[str],
    docs: list[list[str]],
    k1: float = 1.5, b: float = 0.75,
) -> list[float]:
    """Standard BM25 over a fixed corpus.  Returns one score per doc."""
    n_docs = len(docs)
    if n_docs == 0:
        return []
    avg_dl = sum(len(d) for d in docs) / max(1, n_docs)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    scores = [0.0] * n_docs
    for term in query:
        if term in _STOP:
            continue
        n_qi = df.get(term, 0)
        if n_qi == 0:
            continue
        idf = math.log((n_docs - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
        for i, d in enumerate(docs):
            tf = sum(1 for w in d if w == term)
            if tf == 0:
                continue
            dl = len(d)
            denom = tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0))
            scores[i] += idf * (tf * (k1 + 1)) / max(denom, 1e-9)
    return scores


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

# Conservative sentence splitter: splits on .?! followed by whitespace + capital,
# avoiding common abbreviations.
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|cf|etc|Mr|Dr|St|Prof|Fig|Eq|Sec|Ch|Thm|Def)\.\s*$",
                     re.IGNORECASE)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(])")


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # Pre-process: collapse multiple newlines + whitespace.
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    raw = _SENT_BOUNDARY.split(cleaned)
    # Heuristic merge: if a sentence ends with an abbrev marker, glue the
    # next one back.
    merged: list[str] = []
    for s in raw:
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s.strip())
    return [s for s in merged if s]


# ---------------------------------------------------------------------------
# Concept tagging
# ---------------------------------------------------------------------------

# Re-use the same surface forms as concepts.py.  Build the regex lazily from
# the loaded book's concept index so any custom aliases the user added are
# honoured.

def _build_surface_regex(book: Book) -> tuple[re.Pattern, dict[str, str]]:
    """Compile a regex matching every surface form in the book's concept
    index, plus an alias→cid lookup table.

    Tolerant of two shapes for ``book.concepts``:
      * the canonical :class:`ConceptEntry` dataclass (production path
        via ``book.corpus.load_corpus``)
      * a plain dict ``{canonical, aliases, …}`` (test fixtures, raw
        JSON loaders).  Reading attributes off a dict raised
        ``AttributeError`` until this guard was added.
    """
    alias_to_cid: dict[str, str] = {}
    for cid, entry in book.concepts.items():
        if hasattr(entry, "canonical"):
            canonical = entry.canonical
            aliases = entry.aliases
        elif isinstance(entry, dict):
            canonical = entry.get("canonical", "")
            aliases = entry.get("aliases", []) or []
        else:
            continue
        for surface in {canonical, *aliases}:
            if surface:
                alias_to_cid[surface.lower()] = cid
    if not alias_to_cid:
        return re.compile(r"$^"), {}   # never matches
    sorted_forms = sorted(alias_to_cid.keys(), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(s) for s in sorted_forms) + r")\b"
    return re.compile(pattern, re.IGNORECASE), alias_to_cid


def _tag_clause(
    clause: str, surface_re: re.Pattern, alias_to_cid: dict[str, str],
) -> list[tuple[str, int]]:
    """Return [(cid, char_offset), …] for every concept mention in *clause*."""
    out: list[tuple[str, int]] = []
    for m in surface_re.finditer(clause):
        cid = alias_to_cid.get(m.group(0).lower())
        if cid:
            out.append((cid, m.start()))
    return out


# ---------------------------------------------------------------------------
# Plan dataclasses
# ---------------------------------------------------------------------------

# Reading rate: chars per second.  ~180 wpm × ~5 chars/word / 60 ≈ 15.
_DEFAULT_CPS = 15.0
_MIN_CLAUSE_DUR = 1.5
_MAX_CLAUSE_DUR = 30.0


@dataclass
class NarrationClause:
    """One spoken unit of the narration plan.

    Attributes
    ----------
    text:
        The exact prose to be spoken (whitespace-normalised).
    home_nid:
        The BookNode this clause came from — used by the resolver for
        context-sensitive template selection.
    concepts:
        Ordered list of (concept_id, char_offset_in_text) for every concept
        mention in this clause.  Preserves source order so downstream code
        can sync visuals to TTS word timestamps.
    suggested_dur:
        Estimated speaking duration in seconds (TTS will refine this).
    """
    text: str
    home_nid: str
    concepts: list[tuple[str, int]] = field(default_factory=list)
    suggested_dur: float = 0.0
    # Free-form per-clause hints: the chapter-zoom planner stores
    # ``{"clause_speed": 0.85}`` on the punch-line so the orchestrator
    # can tell Kokoro to slow that one clause down without touching the
    # session-wide rate.  Other planners leave this empty.
    meta: dict = field(default_factory=dict)


@dataclass
class NarrationPlan:
    """Ordered narration plan for a topic query.

    A plan is the deterministic output of (book, topic, options).  It is
    NOT timestamped — that happens later when TTS produces audio.

    When ``streaming`` is True, ``clauses`` may be an iterator instead
    of a list — used to yield clauses as the LLM generates them so the
    user starts hearing audio while later sentences are still being
    produced.  Code that needs the total length (``total_chars``,
    ``total_dur``, indexing) must guard on ``streaming``.
    """
    topic: str
    book_title: str
    clauses: list = field(default_factory=list)
    visited_nids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    streaming: bool = False

    def total_chars(self) -> int:
        if self.streaming:
            return 0
        return sum(len(c.text) for c in self.clauses)

    def total_dur(self) -> float:
        if self.streaming:
            return 0.0
        return sum(c.suggested_dur for c in self.clauses)


# ---------------------------------------------------------------------------
# Public planner
# ---------------------------------------------------------------------------

def _estimate_dur(text: str, cps: float = _DEFAULT_CPS) -> float:
    raw = len(text) / max(cps, 1.0)
    return max(_MIN_CLAUSE_DUR, min(raw, _MAX_CLAUSE_DUR))


def _candidate_nodes(book: Book) -> list[BookNode]:
    """Return every BookNode that has body_text worth retrieving."""
    return [n for n in book.root.walk() if n.body_text and n.body_text.strip()]


def plan(
    book: Book, topic: str,
    *,
    top_k: int = 6,
    cps: float = _DEFAULT_CPS,
    include_kinds: Optional[set[str]] = None,
    exclude_kinds: Optional[set[str]] = None,
) -> NarrationPlan:
    """Build a deterministic NarrationPlan for *topic* against *book*.

    Parameters
    ----------
    top_k:
        Maximum number of BookNodes to include in the plan.
    cps:
        Reading rate (characters per second) used for duration estimates.
    include_kinds / exclude_kinds:
        Optional kind filters.  If `include_kinds` is given, only nodes
        whose kind is in the set are considered.  `exclude_kinds` removes
        nodes from the candidate pool (useful to skip ``"bibliography"``).
    """
    candidates = _candidate_nodes(book)
    if include_kinds is not None:
        candidates = [n for n in candidates if n.kind in include_kinds]
    if exclude_kinds is not None:
        candidates = [n for n in candidates if n.kind not in exclude_kinds]

    q_tokens = _tokenize(topic)
    if not q_tokens or not candidates:
        return NarrationPlan(
            topic=topic, book_title=book.title, clauses=[],
            visited_nids=[],
            meta={"reason": "empty_topic_or_corpus"},
        )

    docs = [_tokenize(n.body_text) for n in candidates]
    scores = _bm25_scores(q_tokens, docs)

    # Hybrid: when embeddings are available, blend BM25 ranking with cosine
    # ranking via reciprocal rank fusion (same as narrator.qa._retrieve).
    from book import embeddings as _emb
    bm25_paired = sorted(
        ((sc, i) for i, sc in enumerate(scores)),
        key=lambda sc_i: (-sc_i[0], candidates[sc_i[1]].page_start,
                           candidates[sc_i[1]].nid),
    )
    bm25_ranks = {idx: r for r, (sc, idx) in enumerate(bm25_paired) if sc > 0}

    cand_vecs = [tuple(n.meta.get("embedding", ())) for n in candidates]
    use_dense = any(cand_vecs) and _emb.is_available()
    dense_ranks: dict[int, int] = {}
    if use_dense:
        qv = _emb.embed_text(topic)
        if qv:
            dpaired = _emb.ranked_by_cosine(qv, cand_vecs)
            dense_ranks = {idx: r for r, (sc, idx) in enumerate(dpaired)
                           if sc > 0}

    if dense_ranks:
        # RRF fusion.
        K = 60.0
        fused: list[tuple[float, int]] = []
        all_idx = set(bm25_ranks) | set(dense_ranks)
        for idx in all_idx:
            s = 0.0
            if idx in bm25_ranks:
                s += 1.0 / (K + bm25_ranks[idx])
            if idx in dense_ranks:
                s += 1.0 / (K + dense_ranks[idx])
            fused.append((s, idx))
        fused.sort(key=lambda s_i: (-s_i[0], candidates[s_i[1]].page_start,
                                     candidates[s_i[1]].nid))
        chosen = [candidates[i] for _s, i in fused[:top_k]]
    else:
        indexed = list(zip(scores, candidates))
        ranked = sorted(
            indexed,
            key=lambda sc_n: (-sc_n[0], sc_n[1].page_start, sc_n[1].nid),
        )
        chosen = [n for sc, n in ranked[:top_k] if sc > 0.0]

    if not chosen:
        top_score = (
            (fused[0][0] if dense_ranks else 0.0)
            if dense_ranks
            else (ranked[0][0] if 'ranked' in locals() and ranked else 0.0)
        )
        return NarrationPlan(
            topic=topic, book_title=book.title, clauses=[],
            visited_nids=[],
            meta={"reason": "no_match", "top_score": top_score},
        )

    # Single-rooted lecture.  The retriever can return matches scattered
    # across chapters (regularization shows up in §3.4, §5.4, §16.2,
    # §18.3…) and the previous design walked all of them, producing a
    # lecture that whiplashed between chapters.  A coherent topic
    # introduction stays inside one subtree: pick the highest-scoring
    # node as the lecture root and walk its descendants depth-first via
    # ``plan_outline`` — the same depth-decay walk the TOC click uses,
    # so the topic mode and the "drill chapter" mode produce visually
    # consistent lectures.  When the chosen root is too thin to fill a
    # session (a leaf with very little body), we widen one level up to
    # the parent so the listener still gets a proper introduction.
    # ``chosen`` is already in rank order (ties broken by page, then nid),
    # so ``chosen[0]`` is the best match.
    primary = _pick_primary_root(chosen, book)
    visited_alts = [n.nid for n in chosen if n.nid != primary.nid]
    sub_plan = plan_outline(
        book,
        cps=cps,
        root_nid=primary.nid,
        max_depth=3,
    )
    sub_plan.topic = topic
    sub_plan.meta = {
        **(sub_plan.meta or {}),
        "top_score": (fused[0][0] if dense_ranks
                      else (ranked[0][0] if 'ranked' in locals()
                            and ranked else 0.0)),
        "n_candidates": len(candidates),
        "n_chosen": len(chosen),
        "retrieval": "hybrid_rrf" if dense_ranks else "bm25_only",
        "primary_root": primary.nid,
        "alternate_roots": visited_alts,
        "mode": "topic_focused",
    }
    return sub_plan


def _pick_primary_root(chosen: list[BookNode], book: Book) -> BookNode:
    """Pick the single subtree to walk for a topic lecture.

    Use the highest-scoring retrieved node as-is.  The previous
    revision tried to widen up to a parent when the top match's
    subtree was thin, but that widening crossed kind boundaries
    (chapter → book) on small corpora and broke the kind contract
    callers rely on.  Keeping the walk rooted at the actual top-1
    gives a focused lecture even when the section is short, and
    callers who want breadth can iterate via follow-up tangents.
    """
    if not chosen:
        return book.root
    return chosen[0]


def _find_parent(root: BookNode, nid: str) -> Optional[BookNode]:
    """Locate the parent node of *nid* in the tree rooted at *root*."""
    if not nid:
        return None
    for c in root.children:
        if c.nid == nid:
            return root
        found = _find_parent(c, nid)
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------
# Topic path: prerequisite-first order + bridge clauses
# ---------------------------------------------------------------------------

def _path_through(nodes: list[BookNode], book: Book) -> list[BookNode]:
    """Order *nodes* so that prerequisites come before the sections that
    cite them, using the math semantic graph's ``references`` and
    ``derived_from`` edges as the dependency signal.

    Algorithm — Kahn's topological sort with a stable tiebreak on
    ``(page_start, nid)``.  When the graph has no actionable edges
    between any two chosen nodes (or no graph at all is reachable),
    degenerate to plain book order — same behaviour as before this
    function existed.

    The rationale for using *references* / *derived_from* as the
    dependency signal: those edges literally encode "section A's
    formula cites / is derived from section B's formula", which is the
    exact direction of conceptual prerequisite.  ``contains``,
    ``related_to`` and ``paired_in_clause`` are deliberately ignored —
    the first two are bidirectional and the third is co-occurrence
    rather than a prerequisite.
    """
    if len(nodes) <= 1:
        return list(nodes)

    # Try to load the book's offline math graph; degrade silently.
    g = _maybe_load_math_graph(book)

    nid_set = {n.nid for n in nodes}
    by_nid = {n.nid: n for n in nodes}

    # Build the dependency DAG: edge A → B means "A depends on B" so B
    # must surface first.  Detected by walking out_edges of every
    # formula whose home_nid is in our chosen set.
    deps: dict[str, set[str]] = {n.nid: set() for n in nodes}
    if g is not None:
        for f in list(g.formulas.values()):
            if not f.home_nid or f.home_nid not in nid_set:
                continue
            for e in g.out_edges(f.id):
                if e.type not in ("references", "derived_from"):
                    continue
                tgt = g.formulas.get(e.dst)
                if tgt is None or not tgt.home_nid:
                    continue
                if tgt.home_nid == f.home_nid:
                    continue   # intra-section, not a path edge
                if tgt.home_nid in nid_set:
                    deps[f.home_nid].add(tgt.home_nid)

    # Kahn: repeatedly emit any node whose unmet-dependency set is
    # empty, preferring lowest (page_start, nid) on ties.  When the
    # graph contains a cycle (rare; ESLII has none we've observed),
    # break it by emitting the lowest-page candidate from the
    # cycle so the loop always makes progress.
    ordered: list[BookNode] = []
    emitted: set[str] = set()
    remaining = list(nodes)
    while remaining:
        ready = [n for n in remaining if not (deps[n.nid] - emitted)]
        if not ready:
            ready = remaining   # fallback for cycles
        ready.sort(key=lambda n: (n.page_start, n.nid))
        first = ready[0]
        ordered.append(first)
        emitted.add(first.nid)
        remaining.remove(first)
    return ordered


def _maybe_load_math_graph(book: Book):
    """Return the per-book MathGraph if it can be loaded from disk;
    otherwise None.  The graph file path is derived from the book's
    source path via ``sevim.math_graph.graph_path_for_book``; when the
    book was constructed in-memory (unit tests) ``book.source`` is
    empty and we return None.
    """
    src = getattr(book, "source", "") or ""
    if not src:
        return None
    try:
        from sevim.math_graph import MathGraph, graph_path_for_book
        gpath = graph_path_for_book(src)
        import os as _os
        if not _os.path.isfile(gpath):
            return None
        return MathGraph.load(gpath, book_id=book.title or src)
    except Exception:
        return None


def _bridge_text(prev: BookNode, nxt: BookNode, topic: str) -> str:
    """Build the synthetic spoken sentence that bridges *prev* into *nxt*.

    Names the destination explicitly (kind + number + title) and gives
    the learner a one-clause reason for the transition: both stops on
    the same topic ``topic``.  Keeps the wording short — the orchestrator
    is going to follow this with the destination's passage-card banner
    plus the section's own first body sentences within a few seconds.
    """
    def _label(n: BookNode) -> str:
        kind = (n.kind or "section").replace("_", " ")
        if n.number and n.title:
            return f"{kind} {n.number}, {n.title}"
        if n.title:
            return n.title
        if n.number:
            return f"{kind} {n.number}"
        return kind
    src = _label(prev)
    dst = _label(nxt)
    topic = (topic or "this topic").strip() or "this topic"
    return (
        f"That covers how {src} approaches {topic}. "
        f"We now move to {dst}, which is the next stop on this path "
        f"through {topic}."
    )


# ---------------------------------------------------------------------------
# Mode B — read-the-book linear plan
# ---------------------------------------------------------------------------

def plan_full(
    book: Book,
    *,
    cps: float = _DEFAULT_CPS,
    include_kinds: Optional[set[str]] = None,
    exclude_kinds: Optional[set[str]] = None,
    skip_short_chars: int = 12,
    max_clause_chars: int = 800,
) -> NarrationPlan:
    """Pre-order linear plan covering every BookNode in book reading order.

    The narrator walks the tree depth-first (book → part → chapter → …),
    sentence-segments each node's body_text, and tags concept mentions
    just like ``plan()`` does for topic-driven sessions.

    Use this for "read the book from chapter 1" mode.

    Parameters
    ----------
    cps:
        Reading rate (chars per second).  Default 15 cps ≈ 180 wpm.
    include_kinds, exclude_kinds:
        Kind filters.  Defaults exclude bibliography, index, and
        glossary (their text is rarely worth narrating verbatim).
    skip_short_chars:
        Skip nodes whose body_text has fewer than this many non-space chars.
    max_clause_chars:
        Hard cap on a single clause length (long sentences get split at
        the nearest period after this offset).
    """
    if exclude_kinds is None:
        exclude_kinds = {"bibliography", "index", "glossary"}

    surface_re, alias_to_cid = _build_surface_regex(book)

    clauses: list[NarrationClause] = []
    visited: list[str] = []

    for n in book.root.walk():
        if include_kinds is not None and n.kind not in include_kinds:
            continue
        if n.kind in exclude_kinds:
            continue
        body = (n.body_text or "").strip()
        if len(body) < skip_short_chars:
            continue

        # Optional title preamble for structural nodes — turns the heading
        # into spoken context ("Chapter 3, eigenvalues.")
        if n.title and n.kind in {"chapter", "section", "subsection",
                                  "subsubsection", "part", "appendix"}:
            preamble = (
                f"{n.kind.capitalize()} {n.number}. {n.title}."
                if n.number else f"{n.kind.capitalize()}. {n.title}."
            )
            clauses.append(NarrationClause(
                text=preamble,
                home_nid=n.nid,
                concepts=_tag_clause(preamble, surface_re, alias_to_cid),
                suggested_dur=_estimate_dur(preamble, cps),
            ))

        for sent in _split_sentences(body):
            # Split overlong sentences at the nearest period.
            chunks = [sent]
            if len(sent) > max_clause_chars:
                chunks = _split_long_sentence(sent, max_clause_chars)
            for chunk in chunks:
                clauses.append(NarrationClause(
                    text=chunk,
                    home_nid=n.nid,
                    concepts=_tag_clause(chunk, surface_re, alias_to_cid),
                    suggested_dur=_estimate_dur(chunk, cps),
                ))
        visited.append(n.nid)

    return NarrationPlan(
        topic="<full-book>",
        book_title=book.title,
        clauses=clauses,
        visited_nids=visited,
        meta={"mode": "full", "cps": cps,
              "include_kinds": list(include_kinds) if include_kinds else None,
              "exclude_kinds": list(exclude_kinds) if exclude_kinds else None},
    )


_PDF_NOISE_RE = re.compile(
    r"^\s*(?:"
    r"this is page\b"            # PDF extractor running header
    r"|printer\s*:\s*opaq\w*"    # publisher-specific watermark
    r"|springer series\b"
    r"|isbn\b"
    r"|copyright\b"
    r"|to (?:my|our) (?:parents|families|wife|husband|teachers)\b"
    r"|library of congress\b"
    r"|all rights reserved\b"
    r")",
    re.I,
)

# Front-matter / page-furniture kinds we skip in outline mode.
_FRONT_MATTER_KINDS = frozenset({
    "introduction",        # ESLII labels prefaces as 'introduction'
    "preface", "foreword", "dedication", "acknowledgements",
    "frontmatter", "copyright_page", "title_page",
    "bibliography", "index", "glossary",
    "references", "author_index",
})

_NARRATABLE_KINDS = frozenset({
    "book", "part", "chapter", "section", "subsection",
    "subsubsection", "appendix",
})


def _is_pdf_noise(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return True
    if _PDF_NOISE_RE.match(s):
        return True
    # Standalone page numbers ("12", "3.4.").
    if re.fullmatch(r"\d+\.?", s):
        return True
    # Author-list style sentences: 3+ proper-name pairs, no verb.
    if re.fullmatch(
        r"(?:[A-Z][a-z]+\s+[A-Z][a-z]+,?\s*){3,}", s,
    ):
        return True
    return False


def _clean_body_for_outline(body: str) -> str:
    """Drop leading PDF page-header lines and dedication blocks before
    sentence-segmenting."""
    if not body:
        return ""
    lines = body.splitlines()
    cleaned: list[str] = []
    for ln in lines:
        if _PDF_NOISE_RE.match(ln.strip()):
            continue
        if ln.strip().lower() in (
            "to our parents", "dedication", "preface",
        ):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


def plan_outline(
    book: Book,
    *,
    cps: float = _DEFAULT_CPS,
    root_nid: str = "",
    max_depth: int = 3,
    # ``-1`` at any depth means "all sentences in this node's body".
    # Default: emit the root node's full body (the user clicked it,
    # they want the whole sub-chapter), then taper at deeper depths.
    sentences_at_depth: tuple = (-1, 4, 2, 1, 0),
    exclude_kinds: Optional[set[str]] = None,
    skip_short_chars: int = 8,
    skip_front_matter: bool = True,
) -> NarrationPlan:
    """Hierarchical, top-down narration plan.

    Walks the BookNode tree depth-first from *root_nid* (or the whole
    book when empty), emitting at each node:

      * the title preamble (always, when the kind is structural),
      * the first ``sentences_at_depth[depth]`` sentences of body_text,
        which decay with depth so chapters get a short intro and
        subsubsections get only their title.

    Front-matter (preface, dedication, ISBN page, etc.) and obvious
    PDF-extraction noise ("This is page v Printer: Opaq", standalone
    page numbers, author-list lines) are filtered out by default — set
    ``skip_front_matter=False`` to include them.
    """
    if exclude_kinds is None:
        exclude_kinds = set(_FRONT_MATTER_KINDS) if skip_front_matter else \
                        {"bibliography", "index", "glossary"}
    surface_re, alias_to_cid = _build_surface_regex(book)

    root = book.find(root_nid) if root_nid else book.root
    if root is None:
        root = book.root

    clauses: list[NarrationClause] = []
    visited: list[str] = []

    # Deferred import — qa imports planner, so we can only import qa
    # lazily here to avoid a circular import.
    from .qa import _sanitize_for_narration as _verbalize

    def _emit(node, depth: int) -> None:
        if node.kind in exclude_kinds:
            return
        # The book root carries title-page boilerplate as body_text;
        # only announce its title, never recite the body.
        is_root = (node is root and node.kind == "book")

        # Title preamble.
        if node.title and node.kind in _NARRATABLE_KINDS:
            preamble = (
                f"{node.kind.capitalize()} {node.number}. {node.title}."
                if node.number else
                f"{node.kind.capitalize()}. {node.title}."
            )
            spoken = _verbalize(preamble) or preamble
            clauses.append(NarrationClause(
                text=spoken, home_nid=node.nid,
                concepts=_tag_clause(spoken, surface_re, alias_to_cid),
                suggested_dur=_estimate_dur(spoken, cps),
            ))
            visited.append(node.nid)

        if is_root:
            return

        body = _clean_body_for_outline(node.body_text or "")
        if len(body) < skip_short_chars:
            return
        n_sent = sentences_at_depth[
            min(depth, len(sentences_at_depth) - 1)
        ]
        if n_sent == 0:
            return
        # ``-1`` (or any negative) → unlimited.
        cap = float("inf") if n_sent < 0 else n_sent
        emitted = 0
        for sent in _split_sentences(body):
            if emitted >= cap:
                break
            if _is_pdf_noise(sent):
                continue
            # Verbalize OCR'd math (Unicode symbols, function-call
            # notation, Greek letters) so Kokoro doesn't read raw
            # characters letter-by-letter.
            spoken = _verbalize(sent) or sent
            clauses.append(NarrationClause(
                text=spoken, home_nid=node.nid,
                concepts=_tag_clause(spoken, surface_re, alias_to_cid),
                suggested_dur=_estimate_dur(spoken, cps),
            ))
            emitted += 1

    def _walk(node, depth: int) -> None:
        _emit(node, depth)
        if depth >= max_depth:
            return
        for c in node.children:
            _walk(c, depth + 1)

    _walk(root, 0)

    return NarrationPlan(
        topic=f"<outline:{root.nid}>",
        book_title=book.title,
        clauses=clauses,
        visited_nids=visited,
        meta={"mode": "outline", "cps": cps,
              "root_nid": root.nid, "max_depth": max_depth,
              "sentences_at_depth": list(sentences_at_depth)},
    )


def _split_long_sentence(sent: str, max_chars: int) -> list[str]:
    """Split a long sentence into ≤ max_chars chunks at periods/semicolons."""
    chunks: list[str] = []
    cur = sent
    while len(cur) > max_chars:
        # Find nearest sentence-ish break before max_chars.
        cut = max(cur.rfind(". ", 0, max_chars),
                  cur.rfind("; ", 0, max_chars),
                  cur.rfind(", ", 0, max_chars))
        if cut < max_chars // 2:
            cut = max_chars   # hard split if no good break
        chunks.append(cur[:cut + 1].strip())
        cur = cur[cut + 1:].strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Chapter-zoom plan: punch-line first, then tour, then drill
# ---------------------------------------------------------------------------

def _spell_label(label: str) -> str:
    """Spell ``Equation 5.42`` as ``equation five point forty-two`` so
    Kokoro reads it as one phrase instead of pausing on the period."""
    if not label:
        return ""
    s = label.strip()
    # Pull the kind ("Equation" / "Figure" / etc.) and the dotted
    # number suffix.  Anything else passes through untouched.
    import re as _re
    m = _re.match(r"(?i)^(equation|figure|table|theorem|algorithm|"
                   r"section|subsection)\s+([\d.]+)\s*$", s)
    if not m:
        return s
    kind = m.group(1).lower()
    parts = m.group(2).split(".")
    spoken = []
    _DIGITS = {"0":"zero","1":"one","2":"two","3":"three","4":"four",
               "5":"five","6":"six","7":"seven","8":"eight","9":"nine"}
    def _num(p: str) -> str:
        try:
            n = int(p)
        except ValueError:
            return p
        if 0 <= n < 10:
            return _DIGITS[p]
        # 10-99: just write digits-with-spaces; Kokoro reads them fine
        # ("forty-two" is more natural but the digits version is also OK).
        return " ".join(_DIGITS.get(c, c) for c in p)
    for i, p in enumerate(parts):
        if i:
            spoken.append("point")
        spoken.append(_num(p))
    return f"{kind} {' '.join(spoken)}"


def _formula_pin_sentence(node: dict, cf_label: str, para: str) -> str:
    """One closing sentence appended to a node's narration that
    explicitly names its canonical equation, so the audio bridges
    to the rendered formula in the treemap cell.  Suppressed when
    the paragraph itself already names the label (rare — the LLM
    only got 47/58 in the chapter-essay pass) so we don't repeat.
    """
    if not cf_label:
        return ""
    spoken = _spell_label(cf_label)
    if not spoken:
        return ""
    # Skip when the paragraph already mentions the spelled-out form
    # or the raw "Equation N.M" — keeps audio clean for the few
    # nodes the LLM did cite in-paragraph.
    p_lower = (para or "").lower()
    if spoken.lower() in p_lower:
        return ""
    if cf_label.lower() in p_lower:
        return ""
    return f"This is the idea written down as {spoken}."


def _split_paragraph(text: str) -> list[str]:
    """Split a story paragraph into sentence-clauses for the planner.

    Reuses ``_split_sentences`` (the planner's own sentence splitter)
    so the behaviour matches every other plan path; collapses repeated
    whitespace and drops empties so trailing periods don't produce
    blank clauses.
    """
    text = (text or "").strip()
    if not text:
        return []
    out: list[str] = []
    for s in _split_sentences(text):
        s = s.strip()
        if s:
            out.append(s)
    return out

def plan_chapter_zoom(
    book: Book, *,
    chapter_map: dict,
    cps: float = _DEFAULT_CPS,
    drill_depth: int = 2,
) -> NarrationPlan:
    """Build the top-down chapter narration backed by a chapter-map sidecar.

    Three stages of clauses:

    1. **Punch-line.** The chapter root's gist is read at a deliberately
       slower rate (``meta["clause_speed"]`` per clause) so the
       opening sentence registers as the headline rather than the
       first of many lookalike body sentences.
    2. **Tour.** One clause per L1 child, of the form
       ``"<role_in_parent>. This is <kind> <number>: <title>."`` so each
       child is framed by the chapter's whole story before the listener
       hears anything specific about it.  The clause's ``home_nid``
       points at that child, which the rAF sync engine uses to highlight
       the matching treemap cell.
    3. **Drill.** For each L1 child whose subtree has its own children
       (and only down to ``drill_depth`` from the chapter root), repeat
       the tour pattern at that level using the L1's gist as the
       framing parent.  Stops cleanly when we run out of children.

    The plan also carries a single ``chapter_map`` visual op in
    ``meta["chapter_map_payload"]`` so the orchestrator emits the
    treemap shape exactly once on the first clause.

    Returns a NarrationPlan whose ``meta["mode"] == "chapter_zoom"``;
    the orchestrator switches off its usual passage-card / book-figure /
    formula emission for that mode (the treemap already shows the same
    structure, so duplicating banner cards would only crowd the board).
    """
    root = (chapter_map or {}).get("root") or {}
    root_nid = root.get("nid") or chapter_map.get("root_nid", "")
    if not root_nid or not root.get("children"):
        # Empty / malformed map → fall through to the depth-decay walk
        # so the user still gets *something* sensible.
        return plan_outline(book, root_nid=root_nid)

    surface_re, alias_to_cid = _build_surface_regex(book)
    clauses: list[NarrationClause] = []
    visited: list[str] = [root_nid]

    def _emit(text: str, home_nid: str, *, speed: float = 1.0) -> None:
        if not text:
            return
        clauses.append(NarrationClause(
            text=text,
            home_nid=home_nid,
            concepts=_tag_clause(text, surface_re, alias_to_cid),
            suggested_dur=_estimate_dur(text, cps) / max(0.4, speed),
            meta={"clause_speed": speed} if speed != 1.0 else {},
        ))

    # 1. Punch-line at the chapter root.  Prefer the offline-built
    # ``story_paragraph`` (a 3-5 sentence narrative paragraph that
    # opens the chapter as one flowing essay) and fall back to the
    # 1-sentence ``gist`` when the paragraph is missing.  The slow
    # speed of 0.85 marks this as the headline; everything below
    # plays at the session's default rate so the rest of the
    # narrative reads as one continuous voice.
    chapter_label = (
        f"Chapter {root.get('number')}: {root.get('title')}"
        if root.get("number") else (root.get("title") or "this chapter")
    )
    root_paragraph = (root.get("story_paragraph") or "").strip()
    headline = root_paragraph or (root.get("gist") or "").strip()
    if not headline:
        headline = f"This is the big picture of {chapter_label}."
    intro_lead = f"Here is the whole picture of {chapter_label}."
    # The opening clause slows down so the listener registers the
    # chapter title; the body of the punch-line plays at default
    # speed so the user is already inside the narrative voice by
    # the time the tour begins.
    _emit(intro_lead, root_nid, speed=0.85)
    for sent in _split_paragraph(headline):
        _emit(sent, root_nid)

    # Helper — emit one paragraph as one clause per sentence, all
    # anchored at the same home_nid so the rAF sync engine keeps the
    # treemap cell highlighted while the paragraph reads.  Falls back
    # to a single-clause "[role_in_parent]. This is [title]." line
    # when the offline paragraph is missing for that nid.  Every node
    # that has a canonical formula gets a closing "captured in
    # equation N.M" sentence appended so the audio explicitly names
    # the formula the treemap cell is showing — without that sentence
    # the visual and the spoken story stayed siloed.  Consecutive
    # sections that share a canonical label (parent-inheritance
    # fallback when the math graph missed an equation) skip the pin
    # sentence after the first one to keep the audio from chanting
    # "equation five point one" four times in a row.
    last_pin_label = {"v": ""}
    def _emit_node(node: dict) -> None:
        nid = (node.get("nid") or "").strip()
        if not nid:
            return
        para = (node.get("story_paragraph") or "").strip()
        cf_label = (node.get("canonical_formula_label") or "").strip()
        formula_pin = _formula_pin_sentence(node, cf_label, para)
        if formula_pin and cf_label == last_pin_label["v"]:
            formula_pin = ""
        if formula_pin:
            last_pin_label["v"] = cf_label
        if para:
            for sent in _split_paragraph(para):
                _emit(sent, nid)
            if formula_pin:
                _emit(formula_pin, nid)
                # Parameter-by-parameter walkthrough, when the
                # chapter-map builder generated one for this node.
                # ``formula_explanation`` is a paragraph of plain-
                # English sentences naming each symbol in the
                # canonical formula and tying it back to the section's
                # story; emitted right after the pin so the listener
                # hears "this is equation 5.1: [explanation]".
                expl = (node.get("formula_explanation") or "").strip()
                if expl:
                    for s in _split_paragraph(expl):
                        _emit(s, nid)
            visited.append(nid)
            return
        # Fallback for nodes the offline build didn't cover (typically
        # appendices / exercises / bibliographic notes).  We still
        # deserve one short sentence anchored at that nid so the user
        # at least hears the section's name and role.
        role = (node.get("role_in_parent") or "").strip()
        title = (node.get("title") or "").strip()
        kind = (node.get("kind") or "section").replace("_", " ")
        number = (node.get("number") or "").strip()
        label = (f"{kind} {number}: {title}" if number and title
                 else (title or kind))
        if role:
            role = role[0].upper() + role[1:]
            sent = f"{role} This is {label}."
        else:
            sent = f"This is {label}."
        _emit(sent, nid)
        if formula_pin:
            _emit(formula_pin, nid)
        visited.append(nid)

    # 2 + 3. Tour + drill in *depth-first* order — each L1 child is
    # narrated, and immediately after, its own children (subsections)
    # are narrated, before moving to the next L1 sibling.  This
    # matches the visual order of the chapter-map stack (DFS top-to-
    # bottom), so the active-cell highlight walks straight down the
    # column instead of jumping back to the top after the section
    # tour to start the subsection drill.  Skips appendices and
    # exercises at every level.
    SKIP_KINDS = {"exercise", "exercises", "bibliographic_notes"}
    l1_children = [c for c in root.get("children", [])
                   if (c.get("kind") or "") not in SKIP_KINDS]
    for child in l1_children:
        _emit_node(child)
        if drill_depth >= 2:
            grandkids = [g for g in child.get("children", [])
                         if (g.get("kind") or "") not in SKIP_KINDS]
            for gk in grandkids:
                _emit_node(gk)

    return NarrationPlan(
        topic=f"<chapter-zoom:{root_nid}>",
        book_title=book.title,
        clauses=clauses,
        visited_nids=visited,
        meta={
            "mode": "chapter_zoom",
            "cps": cps,
            "root_nid": root_nid,
            "drill_depth": drill_depth,
            # The orchestrator reads this on the first clause and emits
            # the treemap once; it is the *only* visual op the
            # chapter_zoom mode lets through.
            "chapter_map_payload": chapter_map,
        },
    )
