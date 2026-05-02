"""Extract concept-level prerequisites from a Book's body text.

Where ``book/aliases.py`` finds *synonyms* ("RBF" ↔ "radial basis
function"), this module finds *dependencies* — phrases that say
"X is built on Y", "to understand X, recall Y", "the X uses the Y",
etc.  The result is a graph the ``dependencies`` planner uses to
narrate prerequisites that come from the concept layer rather than
the citation layer.

Pure function over the in-memory Book; safe to cache per-book at
server startup.  No I/O, no LLM.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional

from .ir import Book


# A concept-shaped surface form (1-4 words) — kept conservative so
# the patterns don't grab whole clauses.
_CONCEPT_RE = (
    r"(?:[A-Za-z][A-Za-z0-9-]+(?:\s+[A-Za-z][A-Za-z0-9-]+){0,3})"
)

# Patterns that mark "X depends on Y" relationships.  Each captures
# ``head`` (the dependent concept) and ``prereq`` (the prerequisite).
_DEP_PATTERNS: list[re.Pattern] = [
    # "X is built on Y" / "X builds on Y" / "X is built upon Y"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s+"
        r"(?:is|are)\s+built\s+(?:on|upon)\s+"
        rf"(?:the\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
    # "X requires Y" / "X requires the Y"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s+requires?\s+"
        rf"(?:the\s+|a\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
    # "X uses Y" / "X uses the Y"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s+uses?\s+"
        rf"(?:the\s+|a\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
    # "X relies on Y"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s+relies\s+on\s+"
        rf"(?:the\s+|a\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
    # "X is based on Y"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s+is\s+based\s+on\s+"
        rf"(?:the\s+|a\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
    # "X generalizes Y" / "X is a generalization of Y"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s+generalizes?\s+"
        rf"(?:the\s+|a\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s+is\s+a\s+"
        r"generali[sz]ation\s+of\s+"
        rf"(?:the\s+|a\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
    # "X extends Y"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s+extends?\s+"
        rf"(?:the\s+|a\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
    # "to understand X, [we|you] need Y" / "to understand X, recall Y"
    re.compile(
        rf"to\s+understand\s+(?:the\s+)?(?P<head>{_CONCEPT_RE})"
        r"\s*,\s+(?:we|you)?\s*(?:need|recall|review)\s+"
        rf"(?:the\s+|a\s+)?(?P<prereq>{_CONCEPT_RE})\b",
    ),
]


# Heads that are obvious sentence-grammar fragments — pronouns,
# conjunctions, demonstratives.  Concept heads must NOT start with
# any of these.  Tighter than the alias noise list because the
# dependency patterns are more lenient on the verb side.
_HEAD_FORBIDDEN_STARTS = frozenset({
    # Pronouns + auxiliaries
    "we", "you", "he", "she", "it", "they", "i",
    "this", "that", "these", "those", "what", "which", "who",
    # Conjunctions + connectives
    "and", "or", "but", "however", "thus", "hence", "therefore",
    "so", "yet", "while", "since", "because", "if", "when", "where",
    "as", "than", "though", "although",
    # Discourse fillers
    "here", "there", "now", "then", "next", "afterwards",
    "first", "second", "third", "finally", "lastly",
    # Articles / quantifiers
    "the", "a", "an", "any", "all", "some", "many", "few", "every",
    "each", "no", "such",
    # Structural / typographic
    "section", "chapter", "figure", "table", "equation",
    "example", "exercise", "page", "appendix", "definition",
    "theorem", "lemma", "proof", "note", "remark", "see", "refer",
    "case", "way", "form", "kind", "type", "sort",
    # Prepositions
    "in", "on", "at", "to", "from", "by", "with", "without", "for",
    "of", "into", "onto", "out", "up", "down", "off", "over", "under",
    # Misc verbs that tend to start clauses, not concepts
    "let", "consider", "suppose", "assume", "given", "set",
})


def _is_concept_phrase(s: str) -> bool:
    if not s or not s.strip():
        return False
    s = s.strip()
    if len(s) < 3 or len(s) > 60:
        return False
    toks = [t.lower() for t in s.split()]
    if toks[0] in _HEAD_FORBIDDEN_STARTS:
        return False
    # At least one word must look like a content word (≥ 3 chars) and
    # not be a forbidden start — otherwise the phrase is structural.
    has_content = any(
        len(t) >= 3 and t not in _HEAD_FORBIDDEN_STARTS
        for t in toks
    )
    return has_content


def _normalise(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _known_terms(book: Book) -> set[str]:
    """Return a normalised set of every "real" concept-bearing term
    we recognise in the book — concept index keys + their aliases,
    chapter / section titles, and the title's individual content
    words.  Used as a filter for both head and prereq slots so only
    known entities form dependency edges.
    """
    terms: set[str] = set()
    # Concept index entries.
    for cid, ent in (book.concepts or {}).items():
        terms.add(_normalise(cid))
        if hasattr(ent, "canonical"):
            cano = (ent.canonical or "").strip()
            if cano:
                terms.add(_normalise(cano))
            for a in (getattr(ent, "aliases", None) or []):
                if a:
                    terms.add(_normalise(a))
        elif isinstance(ent, dict):
            cano = (ent.get("canonical") or "").strip()
            if cano:
                terms.add(_normalise(cano))
            for a in ent.get("aliases", []) or []:
                if a:
                    terms.add(_normalise(a))
    # Section / chapter / subsection titles — strip leading numbering.
    for n in book.root.walk():
        if not n.title:
            continue
        title = re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", n.title).strip()
        if title:
            terms.add(_normalise(title))
    terms.discard("")
    return terms


def _term_appears_in(text_lc: str, term: str) -> bool:
    """Word-boundary check that *term* (lower-case) appears in
    *text_lc* (also lower-case)."""
    if not term or not text_lc:
        return False
    return bool(re.search(rf"\b{re.escape(term)}\b", text_lc))


def extract_concept_graph(
    book: Book, *, min_count: int = 1, max_edges: int = 5_000,
) -> dict[str, list[tuple[str, int]]]:
    """Return ``{concept → [(prereq, count), …]}``.

    Pure heuristic extraction over body_text — high-precision pattern
    matching, *restricted to known concept terms* (book.concepts keys
    + section titles).  This filter is what keeps the graph
    trustworthy: textbook prose contains thousands of "X uses Y"
    sentence fragments, but only a small fraction relate two real
    concepts.  Phrases like "we use" / "one can" / "best-subset
    selection chose to" are dropped because their head isn't a
    recognised concept name.
    """
    known = _known_terms(book)
    if not known:
        return {}
    edge_counts: Counter = Counter()
    for node in book.root.walk():
        body = (node.body_text or "")
        if not body:
            continue
        for pat in _DEP_PATTERNS:
            for m in pat.finditer(body):
                head = (m.group("head") or "").strip()
                prereq = (m.group("prereq") or "").strip()
                if (not _is_concept_phrase(head)
                        or not _is_concept_phrase(prereq)):
                    continue
                head_lc = _normalise(head)
                prereq_lc = _normalise(prereq)
                if not head_lc or not prereq_lc or head_lc == prereq_lc:
                    continue
                # Match against known terms.  Either side must contain
                # a recognised concept term as a word-boundary match.
                # We canonicalise to the longest matching known term
                # so "linear regression model" → "linear regression"
                # if that's the indexed concept.
                head_match = _best_known_match(head_lc, known)
                prereq_match = _best_known_match(prereq_lc, known)
                if not head_match or not prereq_match:
                    continue
                if head_match == prereq_match:
                    continue
                # Drop trivial subset relations.
                if (head_match in prereq_match.split()
                        or prereq_match in head_match.split()):
                    continue
                edge_counts[(head_match, prereq_match)] += 1

    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (head, prereq), count in edge_counts.most_common(max_edges):
        if count < min_count:
            break
        out[head].append((prereq, count))
    return dict(out)


def _best_known_match(phrase: str, known: set[str]) -> Optional[str]:
    """Return the longest known term that's a word-boundary substring
    of *phrase* (both lower-case).  ``None`` if no known term matches.
    """
    if not phrase or not known:
        return None
    if phrase in known:
        return phrase
    candidates = [t for t in known
                  if t and _term_appears_in(phrase, t)]
    if not candidates:
        return None
    return max(candidates, key=len)


def prerequisites(
    graph: dict[str, list[tuple[str, int]]],
    concept: str, *, max_items: int = 6,
) -> list[tuple[str, int]]:
    """Return the top-N prerequisites for *concept*.

    Lookup is case- and article-insensitive.  Returns ``[]`` when the
    concept isn't in the graph.
    """
    if not graph or not concept:
        return []
    key = _normalise(concept)
    edges = graph.get(key, [])
    if not edges:
        # Try matching as a substring of any head — handles
        # "ridge regression" lookup against "linear regression
        # ridge regression" if the entry is there literally.
        for k, v in graph.items():
            if key in k:
                edges = v
                break
    return edges[:max_items]
