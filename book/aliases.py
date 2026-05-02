"""Extract concept synonyms from a Book's body text.

Textbooks introduce alternative names for concepts inline:

  * "the Gram matrix (also called the *kernel matrix*)"
  * "the bias-variance tradeoff, also known as the bias-variance dilemma"
  * "the support vector machine, or SVM"
  * "the radial basis function, RBF for short"
  * "fitted values, i.e., model predictions"

This module walks every BookNode's ``body_text`` once and applies a
small set of patterns to harvest these alternate-naming pairs.  The
result is a ``{alias_lc → canonical}`` map the QA retriever uses to
expand a query like "what is the kernel matrix" so the dense / BM25
search also pulls passages mentioning "Gram matrix".

Pure function over the in-memory Book; safe to cache per-book at
server startup.  No I/O, no LLM, no external dependencies.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

from .ir import Book


# A "concept-shaped" surface form: 1-5 words, lower- or Title-cased
# (no sentence punctuation).  We constrain length so the regex doesn't
# grab whole clauses; the noise filter then drops obvious prose tails.
_CONCEPT_RE = (
    r"(?:[A-Za-z][A-Za-z0-9-]+(?:\s+[A-Za-z][A-Za-z0-9-]+){0,3})"
)

# Acronym shape — used only on the alias side of parenthetical pairs.
# Two-to-six capital letters with no lower-case interior.
_ACRONYM_RE = r"(?:[A-Z][A-Z0-9]{1,5})"

# Patterns that mark "X is also called Y" relationships.  Each captures
# ``head`` (the canonical concept) and ``alias`` (the alternate name).
# Order matters — more specific patterns first.
_PATTERNS: list[re.Pattern] = [
    # "the Gram matrix (also called the kernel matrix)"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s*"
        r"\(\s*(?:also\s+(?:called|known\s+as)|or)\s+"
        rf"(?:the\s+)?(?P<alias>{_CONCEPT_RE})\s*\)",
    ),
    # "support vector machine, also known as SVM"
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE}),\s+"
        r"(?:also\s+(?:called|known\s+as)|sometimes\s+called)\s+"
        rf"(?:the\s+)?(?P<alias>{_CONCEPT_RE})\b",
    ),
    # "support vector machine, or SVM,"  (acronym alias only — bare
    # ", or X" is too noisy as a free-form pattern in math prose).
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE}),\s+or\s+"
        rf"(?P<alias>{_ACRONYM_RE})\b\s*[,)]",
    ),
    # "radial basis function (RBF)" — parenthetical acronym only.
    # Earlier we accepted lower-case parentheticals here, but that
    # picked up tons of clarifying phrases like "(up to constants)".
    re.compile(
        rf"\b(?:the\s+)?(?P<head>{_CONCEPT_RE})\s*\(\s*"
        rf"(?P<alias>{_ACRONYM_RE})\s*\)",
    ),
]


def _acronym_matches_head(head: str, alias: str) -> bool:
    """``RBF`` must be the initials of ``radial basis function``."""
    head_words = [w for w in re.split(r"[\s-]+", head) if w]
    alias = alias.strip()
    if not head_words or not alias:
        return False
    if not alias.isupper() or not (2 <= len(alias) <= 6):
        return True  # not an acronym → skip the check
    initials = "".join(w[0] for w in head_words).upper()
    # Allow alias to be a strict prefix of initials (e.g., "ML" for
    # "machine learning method") or to match exactly.
    return alias == initials[:len(alias)] or alias == initials


# Tokens that are nearly always noise as either head or alias.  Drop
# matches where one side is just a generic structural word.
_NOISE_TOKENS = frozenset({
    "case", "section", "chapter", "figure", "table", "equation",
    "example", "exercise", "page", "appendix", "definition",
    "theorem", "lemma", "proof", "note", "remark", "see", "refer",
    "this", "that", "these", "those", "such", "each", "every",
    "the", "a", "an", "any", "all", "some", "many", "few",
    "fig", "eq", "ch", "thm",
    # Sub-set of common English nouns / adjectives that look like
    # concept names but rarely act as synonyms in practice:
    "way", "form", "kind", "type", "sort", "case",
    "left", "right", "above", "below",
})


def _is_concept_phrase(s: str) -> bool:
    """Reject obvious non-concept matches."""
    if not s or not s.strip():
        return False
    s = s.strip()
    if len(s) < 2 or len(s) > 60:
        return False
    # First token must be either lower-case noun-shaped or a proper noun.
    first = s.split()[0].lower()
    if first in _NOISE_TOKENS:
        return False
    # If every token is in the noise set, drop.
    toks = [t.lower() for t in s.split()]
    if all(t in _NOISE_TOKENS for t in toks):
        return False
    return True


def _normalise(s: str) -> str:
    """Strip articles, lowercase, collapse whitespace."""
    s = (s or "").strip().lower()
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_aliases(
    book: Book, *, min_count: int = 1, max_aliases: int = 5_000,
) -> dict[str, str]:
    """Walk the book once and return ``{alias_lc → canonical_lc}``.

    Pairs are kept only when both sides are concept-shaped and not in
    the noise list.  ``min_count`` lets callers raise the bar if they
    want fewer false positives (default 1 because most ML synonyms in
    a textbook only appear once).
    """
    pair_counts: Counter = Counter()
    pair_canonical: dict[tuple[str, str], tuple[str, str]] = {}

    for node in book.root.walk():
        body = (node.body_text or "")
        if not body:
            continue
        for pat in _PATTERNS:
            for m in pat.finditer(body):
                head = (m.group("head") or "").strip()
                alias = (m.group("alias") or "").strip()
                if not _is_concept_phrase(head) or not _is_concept_phrase(alias):
                    continue
                # When the alias is an acronym shape, it must actually
                # be the initials of the head — otherwise we accept
                # noise like "(see Figure 3.2)" attached to whichever
                # noun phrase precedes it.
                if alias.isupper() and 2 <= len(alias) <= 6:
                    if not _acronym_matches_head(head, alias):
                        continue
                head_lc = _normalise(head)
                alias_lc = _normalise(alias)
                if not head_lc or not alias_lc or head_lc == alias_lc:
                    continue
                # Reject pairs where one side is wholly contained in
                # the other ("matrix" ↔ "kernel matrix") — those are
                # specialisations, not synonyms, and they hurt
                # retrieval expansion.
                if head_lc in alias_lc.split() or alias_lc in head_lc.split():
                    continue
                if head_lc == alias_lc.split()[-1] or alias_lc == head_lc.split()[-1]:
                    continue
                key = tuple(sorted([head_lc, alias_lc]))
                pair_counts[key] += 1
                # First-seen surface forms win for the canonical.
                pair_canonical.setdefault(key, (head_lc, alias_lc))

    # Build the map in both directions so retrieval can look up either side.
    out: dict[str, str] = {}
    for key, count in pair_counts.most_common(max_aliases):
        if count < min_count:
            break
        a, b = pair_canonical[key]
        # When a single alias maps to multiple canonicals (rare), keep
        # the most-frequent one — Counter ordering above ensures this.
        out.setdefault(a, b)
        out.setdefault(b, a)
    return out


def expand_query(query: str, alias_map: dict[str, str]) -> str:
    """Append every alias variant for terms appearing in *query* so a
    BM25 / dense search hits passages that use the alternate phrasing.

    Returns *query* unchanged when no alias matches.
    """
    if not query or not alias_map:
        return query
    q_lower = query.lower()
    extras: list[str] = []
    seen: set[str] = set()
    for surface, canonical in alias_map.items():
        # Cheap substring check first; a real word-boundary search is
        # enforced by ``re.search`` only for matches we'd otherwise emit.
        if surface in q_lower:
            if not re.search(rf"\b{re.escape(surface)}\b", q_lower):
                continue
            if canonical in q_lower:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            extras.append(canonical)
    if not extras:
        return query
    return query + " " + " ".join(extras)
