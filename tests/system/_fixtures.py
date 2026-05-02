"""Shared fixtures for the system-level test suite.

These tests drive the orchestration / session / persistence layers
end-to-end against the real ESLII corpus, with NullTTS so the
fast-path runs deterministically.  Live-LLM tests use a separate
fixture set and are skipped when vLLM isn't reachable.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import pytest

from book.ir import Book, BookNode, CrossRef
from chalkboard import Chalkboard
from narrator import NarrationClause, NarrationPlan
from narrator.tts import NullTTS

from serve.session import Session, build_session


_ESLII_PATH = os.environ.get(
    "SEVIM_TEST_BOOK",
    os.path.join(
        os.path.dirname(__file__), "..", "..", "books", "ESLII.json",
    ),
)


def _build_node(d: dict) -> BookNode:
    n = BookNode(
        nid=d["nid"], kind=d.get("kind", "passage"),
        number=d.get("number"), title=d.get("title", ""),
        page_start=d.get("page_start", 0),
        page_end=d.get("page_end", 0),
        body_text=d.get("body_text", ""),
    )
    for c in d.get("children", []):
        n.children.append(_build_node(c))
    return n


@pytest.fixture(scope="session")
def eslii() -> Book:
    """The real ESLII corpus, loaded once per session via the
    production loader (``book.corpus.load_corpus``) so concept
    entries and figures arrive as proper dataclass instances —
    not raw dicts.  Hand-rolling the deserialisation here used to
    crash ``planner._build_surface_regex`` because dicts don't have
    an ``.canonical`` attribute.
    """
    if not os.path.isfile(_ESLII_PATH):
        pytest.skip(f"ESLII corpus not found at {_ESLII_PATH}")
    from book.corpus import load_corpus
    return load_corpus(_ESLII_PATH)


def _trivial_plan(book: Book) -> NarrationPlan:
    """Tiny seed plan so build_session has something to wrap."""
    return NarrationPlan(
        topic="<system-test>", book_title=book.title or "t",
        clauses=[NarrationClause(
            text="ready", home_nid=book.root.nid,
            concepts=[], suggested_dur=0.5,
        )],
        visited_nids=[book.root.nid], meta={"mode": "full"},
    )


@pytest.fixture
def session(eslii) -> Session:
    """A fresh session bound to the ESLII corpus.  NullTTS so we
    exercise the orchestration without spinning a real Kokoro
    subprocess.  Wired with an alias map so retrieval expansion is
    a real signal."""
    from book.aliases import extract_aliases
    from book.concept_graph import extract_concept_graph
    from narrator import qa as qa_mod
    qa_mod.set_alias_map(extract_aliases(eslii))
    qa_mod.set_concept_graph(extract_concept_graph(eslii))
    yield build_session(
        plan_id="system-test-1",
        book=eslii,
        plan=_trivial_plan(eslii),
        tts_factory=lambda: NullTTS(),
    )
    qa_mod.set_alias_map({})
    qa_mod.set_concept_graph({})
    qa_mod.set_streaming_backend(None)


# ---------------------------------------------------------------------------
# Live-LLM availability probe
# ---------------------------------------------------------------------------

_VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")


def vllm_reachable(timeout: float = 1.5) -> bool:
    """True when the local Qwen endpoint is up.  Cached per process."""
    if "_VLLM_OK" in globals():
        return globals()["_VLLM_OK"]
    try:
        req = urllib.request.Request(
            _VLLM_URL.rstrip("/") + "/models",
            headers={"Authorization": "Bearer local-vllm"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = resp.status == 200
    except Exception:
        ok = False
    globals()["_VLLM_OK"] = ok
    return ok


live_llm = pytest.mark.skipif(
    not vllm_reachable(),
    reason="vLLM endpoint not reachable",
)


# ---------------------------------------------------------------------------
# Helpers for asserting visualisation invariants
# ---------------------------------------------------------------------------

def aabb_overlap(a, b) -> bool:
    return (a.x < b.x + b.w and b.x < a.x + a.w
            and a.y < b.y + b.h and b.y < a.y + a.h)


def assert_no_overlap(chalkboard: Chalkboard) -> None:
    shapes = list(chalkboard.shapes)
    for i, a in enumerate(shapes):
        for b in shapes[i + 1:]:
            assert not aabb_overlap(a, b), (
                f"shapes overlap: {a.nid}@({a.x},{a.y},{a.w},{a.h}) "
                f"vs {b.nid}@({b.x},{b.y},{b.w},{b.h})"
            )


def collect_visual_ops(orch) -> list[dict]:
    """Drain a session's main orch for one full pass and collect
    every visual op emitted (across every clause)."""
    ops: list[dict] = []
    for ev in orch.stream():
        from serve.orchestrator import (
            AudioChunkEvent, AudioCompleteEvent, StreamEvent,
        )
        if isinstance(ev, StreamEvent):
            ops.extend(ev.visual_ops or [])
    return ops
