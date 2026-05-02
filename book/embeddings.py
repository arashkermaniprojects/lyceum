"""Embedding layer — local vLLM-served Qwen3 → BookNode / concept vectors.

The vLLM server at ``http://127.0.0.1:8003/v1`` (configurable via
``LYCEUM_EMBED_URL``) speaks the OpenAI Embeddings API.  We call it,
batch-style, to populate:

  * ``BookNode.meta["embedding"]`` for every node with body_text
  * ``ConceptEntry.embedding`` for every indexed concept
  * Per-passage embeddings cached separately so the corpus JSON
    stays compact

Determinism
-----------
- vLLM serves a single model deterministically (greedy pooling).
- Vectors are unit-normalised to float32 before storage so cosine
  similarity is just a dot product downstream.
- All tensors are stored as plain Python floats in JSON — no binary
  blobs in the corpus.

Cost
----
- Per call: ~30 ms for a batch of 32 short passages on a single 5090.
- Per book ingestion: 539 nodes batched at 32/call → ~17 calls →
  half a second of GPU time + JSON serialisation.

Public API
----------
``embed_text(text)``                  → tuple[float, ...]
``embed_batch(texts)``                → list[tuple[float, ...]]
``embed_book(book, ...)``             → mutates book in place
``cosine(a, b)``                      → float
``ranked_by_cosine(query_vec, ...)``  → list of (score, idx)
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Optional

from .ir import Book, BookNode, ConceptEntry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.environ.get("LYCEUM_EMBED_URL",
                                  "http://127.0.0.1:8003/v1")
DEFAULT_MODEL = os.environ.get("LYCEUM_EMBED_MODEL",
                               "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_BATCH = int(os.environ.get("LYCEUM_EMBED_BATCH", "32"))
DEFAULT_TIMEOUT = float(os.environ.get("LYCEUM_EMBED_TIMEOUT", "60"))

# Cap each passage at this many characters before embedding.  Most retrieval
# benefits from coherent ~1-2k-char chunks; longer needs splitting upstream.
DEFAULT_MAX_CHARS = 1800


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

def _post_embed(
    texts: list[str], *, base_url: str, model: str, timeout: float,
) -> list[list[float]]:
    """One POST → batch of vectors.  Raises on error."""
    url = base_url.rstrip("/") + "/embeddings"
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local-vllm"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [item["embedding"] for item in data.get("data", [])]


def _normalise(v: list[float]) -> tuple[float, ...]:
    """L2-normalise to unit length, return as a tuple of float32-precision."""
    s = sum(x * x for x in v) ** 0.5
    if s == 0.0:
        return tuple(v)
    inv = 1.0 / s
    return tuple(x * inv for x in v)


# ---------------------------------------------------------------------------
# Public single / batch
# ---------------------------------------------------------------------------

def is_available(*, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 2.0) -> bool:
    """True iff the embedding server responds with a model list."""
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url, headers={"Authorization": "Bearer local-vllm"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return bool(data.get("data"))
    except Exception:
        return False


def embed_text(
    text: str, *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[float, ...]:
    """Embed a single string.  Returns an empty tuple on failure."""
    s = (text or "")[:max_chars].strip()
    if not s:
        return ()
    try:
        vecs = _post_embed([s], base_url=base_url, model=model, timeout=timeout)
    except (urllib.error.URLError, ValueError, KeyError) as e:
        print(f"[book.embeddings] embed_text failed: {e}")
        return ()
    if not vecs:
        return ()
    return _normalise(vecs[0])


def embed_batch(
    texts: list[str], *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    batch_size: int = DEFAULT_BATCH,
    max_chars: int = DEFAULT_MAX_CHARS,
    progress: Optional[callable] = None,
) -> list[tuple[float, ...]]:
    """Embed N strings in batches.  Empty/blank inputs get an empty tuple."""
    out: list[tuple[float, ...]] = []
    n = len(texts)
    i = 0
    while i < n:
        chunk = texts[i: i + batch_size]
        # Filter blanks; vLLM rejects empty strings.
        cleaned = [(j, (t or "")[:max_chars].strip())
                   for j, t in enumerate(chunk)]
        non_empty = [(j, s) for j, s in cleaned if s]
        if non_empty:
            payload = [s for _j, s in non_empty]
            try:
                vecs = _post_embed(payload, base_url=base_url,
                                    model=model, timeout=timeout)
            except Exception as e:
                print(f"[book.embeddings] batch {i}/{n} failed: {e}")
                vecs = []
            # Splice non-empty results back into chunk order.
            chunk_out: list[tuple[float, ...]] = [()] * len(chunk)
            for (j, _s), v in zip(non_empty, vecs):
                chunk_out[j] = _normalise(v)
            out.extend(chunk_out)
        else:
            out.extend([()] * len(chunk))
        i += batch_size
        if progress:
            progress(min(i, n), n)
    return out


# ---------------------------------------------------------------------------
# Cosine + ranking
# ---------------------------------------------------------------------------

def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity for vectors that may or may not be unit-normalised."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # If both inputs are unit-normalised (the case after _normalise), this
    # is already the cosine.  Otherwise normalise on the fly.
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if abs(na - 1.0) < 1e-3 and abs(nb - 1.0) < 1e-3:
        return dot
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def ranked_by_cosine(
    query_vec: tuple[float, ...],
    doc_vecs: list[tuple[float, ...]],
) -> list[tuple[float, int]]:
    """Return ``[(score, index), …]`` sorted by descending cosine score.

    Empty doc vectors get score 0; they sink to the bottom.
    """
    scored: list[tuple[float, int]] = []
    for i, v in enumerate(doc_vecs):
        scored.append((cosine(query_vec, v), i))
    scored.sort(key=lambda s_i: (-s_i[0], s_i[1]))
    return scored


# ---------------------------------------------------------------------------
# Whole-book population
# ---------------------------------------------------------------------------

def embed_book(
    book: Book, *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    embed_concepts: bool = True,
    progress: Optional[callable] = None,
) -> dict:
    """Populate embeddings on every BookNode that has body_text, plus on every
    ConceptEntry.  Mutates *book* in place; returns a small stats dict.

    The vector for each BookNode is stored under ``node.meta["embedding"]``
    so it round-trips through corpus JSON without schema changes.
    """
    nodes = [n for n in book.root.walk() if n.body_text and n.body_text.strip()]
    texts = [n.body_text for n in nodes]
    vectors = embed_batch(texts, base_url=base_url, model=model,
                           progress=progress)
    n_node_ok = 0
    for n, v in zip(nodes, vectors):
        if v:
            n.meta["embedding"] = list(v)
            n_node_ok += 1

    n_concept_ok = 0
    if embed_concepts:
        cids = list(book.concepts.keys())
        # Build concept text by concatenating definitions + canonical name
        # + a few mention contexts.  Keeps the embedding semantically
        # focused on the concept, not on whichever node it first appeared in.
        c_texts: list[str] = []
        for cid in cids:
            entry = book.concepts[cid]
            parts: list[str] = [entry.canonical]
            for nid, defn in entry.definitions[:2]:
                parts.append(defn)
            if not entry.definitions:
                # Fall back to the home_nid body_text of the first template.
                if entry.templates:
                    home = book.find(entry.templates[0].home_nid)
                    if home and home.body_text:
                        parts.append(home.body_text[:600])
            c_texts.append("\n".join(parts))
        cvecs = embed_batch(c_texts, base_url=base_url, model=model,
                             progress=progress)
        for cid, v in zip(cids, cvecs):
            if v:
                book.concepts[cid].embedding = v
                n_concept_ok += 1

    return {
        "n_nodes_embedded": n_node_ok,
        "n_nodes_total": len(nodes),
        "n_concepts_embedded": n_concept_ok,
        "n_concepts_total": len(book.concepts),
        "vector_dim": len(next((v for v in vectors if v), ())),
        "model": model,
    }


def passage_vectors(book: Book) -> tuple[list[BookNode], list[tuple[float, ...]]]:
    """Return ``(nodes, vectors)`` for every BookNode that has a stored embedding.

    Used by retrieval code: pre-builds the parallel arrays so retrieval is
    just a single pass over the doc-vector list.
    """
    nodes: list[BookNode] = []
    vecs: list[tuple[float, ...]] = []
    for n in book.root.walk():
        v = n.meta.get("embedding")
        if v:
            nodes.append(n)
            vecs.append(tuple(v))
    return nodes, vecs
