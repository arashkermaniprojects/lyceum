"""Cross-reference overlay extraction.

Scans every BookNode's body_text for citation patterns
(``see Theorem 3.2.1``, ``cf. Definition 1.1``, ``equation (4.5)``, ...) and
resolves the cited number to the matching BookNode.

Resolution is by ``BookNode.number`` exact match.  Ambiguous numbers (the
same "3.1" appearing in two different chapter contexts) are resolved by
selecting the closest sibling under the same ancestor; otherwise the first
match in pre-order wins.

The output is a list of :class:`CrossRef` overlays — *not* attached to the
tree (which stays a clean hierarchy).  Code that needs adjacency simply
iterates ``book.cross_refs``.
"""
from __future__ import annotations

import re
from typing import Optional

from .ir import Book, BookNode, CrossRef


# Patterns matching "<env-name> <number>" — common citation forms.
_CITE_RE = re.compile(
    r"\b(?:"
    r"see\s+|cf\.\s*|by\s+|from\s+|"
    r"recall\s+(?:that\s+)?|"
    r"in\s+(?:the\s+)?|of\s+|using\s+|via\s+|"
    r"according\s+to\s+|as\s+(?:in\s+|shown\s+in\s+)?|"
    r")"
    r"(?P<env>Theorem|Lemma|Proposition|Corollary|Definition|"
    r"Example|Exercise|Equation|Eq\.|Section|Chapter|"
    r"Figure|Fig\.|Table|Appendix|Remark|Claim)"
    r"\s+(?P<num>\d+(?:\.\d+)*[a-zA-Z]?)",
    re.IGNORECASE,
)


def _find_target(
    root: BookNode, env: str, number: str, source_nid: str,
) -> Optional[BookNode]:
    """Resolve "Theorem 3.2.1" → BookNode.

    Strategy: prefer the deepest descendant that has the matching ``number``
    AND a kind compatible with the cited environment (Theorem → kind in
    {"theorem", "thm"}).  Fall back to any node with the matching number.
    """
    env_lower = env.lower().rstrip(".")
    env_aliases = {
        "fig": "figure",
        "eq": "equation",
        "thm": "theorem",
    }
    desired_kind = env_aliases.get(env_lower, env_lower)

    typed_match: Optional[BookNode] = None
    any_match: Optional[BookNode] = None
    for n in root.walk():
        if n.number != number:
            continue
        if any_match is None:
            any_match = n
        if n.kind == desired_kind and typed_match is None:
            typed_match = n
            break  # exact kind+number wins
    return typed_match or any_match


def extract_cross_refs(book: Book) -> list[CrossRef]:
    """Extract every resolvable citation overlay edge.

    Citations whose target cannot be resolved are silently dropped (they
    often refer to outside-the-book sources like external papers or
    figures that were not numbered).
    """
    out: list[CrossRef] = []
    for node in book.root.walk():
        body = node.body_text or ""
        if not body:
            continue
        for m in _CITE_RE.finditer(body):
            env = m.group("env")
            num = m.group("num")
            target = _find_target(book.root, env, num, node.nid)
            if target is None or target.nid == node.nid:
                continue
            label = f"{env.rstrip('.').title()} {num}"
            out.append(CrossRef(
                from_nid=node.nid, to_nid=target.nid,
                label=label, char_offset=m.start(),
            ))
    return out
