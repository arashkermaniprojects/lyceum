"""Citation-graph navigation over a Book's ``cross_refs``.

Each :class:`book.ir.CrossRef` is one citation edge: a ``from_nid``
that mentions a ``to_nid`` with a human-readable ``label``
("Theorem 3.2", "Figure 5.7", "Chapter 12", …).  Stacked across
the whole book they form a small directed graph the tutor can
walk to answer "what does this section reference?", "where else
is this idea used?", "which chapter is most heavily cited?".

All helpers are pure functions over an in-memory Book — no I/O,
no state, no LLM.  Caching is the caller's concern.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Optional

from .ir import Book, CrossRef


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _is_self_ref(cr: CrossRef, *, root_nid: str) -> bool:
    """A cross-ref to oneself or to one's own subtree is uninformative —
    e.g. running headers picking up "Chapter 5" inside ``b/ch5``."""
    if not cr.to_nid or not cr.from_nid:
        return True
    if cr.to_nid == cr.from_nid:
        return True
    if cr.from_nid.startswith(cr.to_nid + "/"):
        return True
    if cr.to_nid.startswith(cr.from_nid + "/"):
        return True
    return False


def _within(nid: str, root_nid: str) -> bool:
    """True if ``nid`` is at-or-under ``root_nid``."""
    if not root_nid:
        return True
    return nid == root_nid or nid.startswith(root_nid + "/")


# ---------------------------------------------------------------------------
# Outgoing / incoming
# ---------------------------------------------------------------------------

def outgoing(
    book: Book, nid: str, *,
    include_descendants: bool = True,
    drop_self_refs: bool = True,
) -> list[CrossRef]:
    """Return the citations *originating from* ``nid`` (and, by default,
    from its descendants)."""
    if not nid:
        return []
    out: list[CrossRef] = []
    for cr in book.cross_refs:
        src = cr.from_nid
        if not src:
            continue
        if include_descendants:
            if not _within(src, nid):
                continue
        else:
            if src != nid:
                continue
        if drop_self_refs and _is_self_ref(cr, root_nid=nid):
            continue
        out.append(cr)
    return out


def incoming(
    book: Book, nid: str, *,
    include_descendants: bool = True,
    drop_self_refs: bool = True,
    drop_intra_subtree: bool = True,
) -> list[CrossRef]:
    """Return the citations that *point at* ``nid`` (or, by default,
    at its descendants).

    ``drop_intra_subtree`` (default True) excludes refs whose source
    is itself within the focus subtree — otherwise §5.8.1 citing
    §5.8 would surface as "§5.8 is cited by §5.8 (16×)", which is
    misleading.  We want incoming citations from *outside* the focus.
    """
    if not nid:
        return []
    out: list[CrossRef] = []
    for cr in book.cross_refs:
        tgt = cr.to_nid
        src = cr.from_nid
        if not tgt or not src:
            continue
        if include_descendants:
            if not _within(tgt, nid):
                continue
        else:
            if tgt != nid:
                continue
        if drop_self_refs and _is_self_ref(cr, root_nid=nid):
            continue
        if drop_intra_subtree and _within(src, nid):
            continue
        out.append(cr)
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def most_cited(
    book: Book, *, k: int = 10, kinds: Optional[Iterable[str]] = None,
) -> list[tuple[str, int]]:
    """Return ``[(to_nid, citation_count), …]`` for the top *k* nodes.

    ``kinds`` (optional) restricts to BookNodes of those kinds — e.g.,
    ``("section", "subsection")`` to skip chapter-level captures.
    """
    counts: Counter = Counter()
    nid_to_kind = {}
    if kinds is not None:
        kinds = set(kinds)
        for n in book.root.walk():
            nid_to_kind[n.nid] = n.kind
    for cr in book.cross_refs:
        if not cr.to_nid:
            continue
        if _is_self_ref(cr, root_nid=cr.to_nid):
            continue
        if kinds is not None:
            if nid_to_kind.get(cr.to_nid) not in kinds:
                continue
        counts[cr.to_nid] += 1
    return counts.most_common(k)


def by_label(book: Book, label: str) -> list[CrossRef]:
    """All cross_refs whose human label equals *label* (e.g.,
    ``"Theorem 3.2"``).  Useful for "where is X used?"."""
    if not label:
        return []
    return [cr for cr in book.cross_refs if cr.label == label]


def group_by_label(refs: list[CrossRef]) -> list[tuple[str, int]]:
    """Compress a ref list into ``[(label, count), …]`` sorted by
    descending count, ties broken alphabetically."""
    counts: Counter = Counter(cr.label for cr in refs if cr.label)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def group_by_target(
    refs: list[CrossRef],
) -> list[tuple[str, int, str]]:
    """Compress to ``[(to_nid, count, sample_label), …]`` sorted by
    descending count.  ``sample_label`` is the first label seen
    pointing at that nid, useful for narration."""
    by_nid: dict[str, list[str]] = defaultdict(list)
    for cr in refs:
        if not cr.to_nid:
            continue
        by_nid[cr.to_nid].append(cr.label or cr.to_nid)
    out: list[tuple[str, int, str]] = []
    for nid, labels in by_nid.items():
        out.append((nid, len(labels), labels[0]))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def group_by_source(
    refs: list[CrossRef],
) -> list[tuple[str, int, str]]:
    """Compress to ``[(from_nid, count, sample_label), …]`` sorted by
    descending count — useful for "where else is this cited?" so the
    user sees the *originating* sections, not the targets within
    their own focus subtree."""
    by_nid: dict[str, list[str]] = defaultdict(list)
    for cr in refs:
        if not cr.from_nid:
            continue
        by_nid[cr.from_nid].append(cr.label or cr.from_nid)
    out: list[tuple[str, int, str]] = []
    for nid, labels in by_nid.items():
        out.append((nid, len(labels), labels[0]))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


# ---------------------------------------------------------------------------
# Path between two nodes
# ---------------------------------------------------------------------------

def _chapter_of(nid: str) -> str:
    """Return the ``b/chN`` ancestor of *nid*, or ``""`` for non-chapter
    paths.  Stable rule: walk path components left-to-right, return the
    prefix that ends with ``chN``."""
    if not nid:
        return ""
    parts = nid.split("/")
    out: list[str] = []
    for p in parts:
        out.append(p)
        if p.startswith("ch") and p[2:].isdigit():
            return "/".join(out)
    return ""


def chapter_graph(book: Book) -> dict:
    """Aggregate cross_refs to chapter level.

    Returns ``{nodes, edges}`` where:
      * ``nodes`` is a list of ``{nid, number, title, kind, in_count,
        out_count}`` for every chapter that has at least one
        incoming or outgoing edge (or always for chapter-kind
        BookNodes — the visual map always wants every chapter).
      * ``edges`` aggregates inter-chapter citations into
        ``{src, dst, count, sample_label}`` entries.  Self-edges
        (chapter citing itself) are dropped.

    Used by ``/api/citation_graph`` to render the visual navigation
    map.  Pure function over the in-memory Book.
    """
    nodes: list[dict] = []
    nid_to_node: dict[str, dict] = {}
    for n in book.root.walk():
        if n.kind != "chapter":
            continue
        node = {
            "nid": n.nid,
            "number": (n.number or "").strip(),
            "title": (n.title or "").strip(),
            "kind": n.kind,
            "in_count": 0,
            "out_count": 0,
        }
        nodes.append(node)
        nid_to_node[n.nid] = node

    edge_counts: Counter = Counter()
    edge_samples: dict[tuple[str, str], str] = {}
    for cr in book.cross_refs:
        src = _chapter_of(cr.from_nid)
        dst = _chapter_of(cr.to_nid)
        if not src or not dst or src == dst:
            continue
        if src not in nid_to_node or dst not in nid_to_node:
            continue
        key = (src, dst)
        edge_counts[key] += 1
        edge_samples.setdefault(key, cr.label or "")
        nid_to_node[src]["out_count"] += 1
        nid_to_node[dst]["in_count"] += 1

    edges = [
        {
            "src": src, "dst": dst, "count": count,
            "sample_label": edge_samples.get((src, dst), ""),
        }
        for (src, dst), count in edge_counts.most_common()
    ]
    # Sort nodes by chapter number for stable layout.
    def _num_key(n: dict) -> tuple:
        try:
            return (int(n["number"]),)
        except (TypeError, ValueError):
            return (10_000, n["nid"])
    nodes.sort(key=_num_key)
    return {"nodes": nodes, "edges": edges}


def citation_path(
    book: Book, src_nid: str, dst_nid: str, *, max_hops: int = 3,
) -> list[CrossRef]:
    """Return a short citation path from ``src_nid`` to ``dst_nid``.

    Walks the directed graph BFS-style up to ``max_hops``.  At each
    node the outgoing edges considered are those from the node *and
    every descendant* — otherwise an intermediate hop like ``Chapter 5``
    (which itself doesn't cite anything but whose subsections do)
    would dead-end the search.

    Returns the list of edges in order, or ``[]`` when no path is
    found within budget.
    """
    if not src_nid or not dst_nid or src_nid == dst_nid:
        return []
    # Build subtree-aware adjacency.  Each from_nid contributes its
    # edges to *every ancestor* in the chain so a BFS step from
    # ``b/ch5`` can find the edges actually originating in
    # ``b/ch5/s5_8`` etc.
    adj: dict[str, list[CrossRef]] = defaultdict(list)
    for cr in book.cross_refs:
        if not cr.from_nid or not cr.to_nid:
            continue
        # Walk every ancestor prefix of from_nid and register the edge.
        parts = cr.from_nid.split("/")
        for i in range(1, len(parts) + 1):
            ancestor = "/".join(parts[:i])
            adj[ancestor].append(cr)

    # BFS with parent pointers.
    visited = {src_nid: None}
    frontier = [(src_nid, 0)]
    edge_to: dict[str, CrossRef] = {}
    while frontier:
        nxt: list[tuple[str, int]] = []
        for node, depth in frontier:
            if depth >= max_hops:
                continue
            for cr in adj.get(node, []):
                # Skip intra-subtree edges entirely — citation paths
                # that return to descendants of where they came from
                # are uninformative for navigation.
                if _is_self_ref(cr, root_nid=node):
                    continue
                if cr.to_nid in visited:
                    continue
                visited[cr.to_nid] = node
                edge_to[cr.to_nid] = cr
                if cr.to_nid == dst_nid or _within(dst_nid, cr.to_nid):
                    # Found dst (or a node that contains dst).
                    final_to = cr.to_nid
                    path: list[CrossRef] = []
                    cur = final_to
                    while cur != src_nid:
                        e = edge_to[cur]
                        path.append(e)
                        cur = visited[cur]
                        if cur is None:
                            break
                    path.reverse()
                    return path
                nxt.append((cr.to_nid, depth + 1))
        frontier = nxt
    return []
