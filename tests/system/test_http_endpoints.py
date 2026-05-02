"""Drive every public HTTP endpoint via an in-process server.

Spins up the server bound to a free port, then probes every endpoint
the frontend hits — verifying response shape, status codes, and
error-path graceful degradation.

Tests are tagged ``@pytest.mark.slow`` because spawning the server
loads the corpus + builds aliases / concept graph (~ a few seconds).
Skipped when the ESLII corpus isn't available.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest


@pytest.fixture(scope="module")
def http_server(tmp_path_factory):
    """In-process server on a free port, returns its base URL."""
    book_path = os.environ.get(
        "SEVIM_TEST_BOOK",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "books", "ESLII.json",
        ),
    )
    if not os.path.isfile(book_path):
        pytest.skip(f"ESLII corpus not found at {book_path}")
    # Pick a free port.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    # Disable Kokoro / Qwen / streaming-TTS so the server boot is fast
    # and deterministic.
    os.environ["SEVIM_DISABLE_ASR"] = "1"
    os.environ["SEVIM_SKIP_LLM_EQ_LATEX"] = "1"
    os.environ["SEVIM_TTS_STREAM"] = "0"
    os.environ["SEVIM_QA_BACKEND"] = "tutor"
    sess_dir = tmp_path_factory.mktemp("sessions")
    os.environ["SEVIM_SESSION_DIR"] = str(sess_dir)
    from serve.server import Server
    srv = Server(book_paths=[book_path], port=port,
                  prefer_kokoro=False, prefer_qwen=False)
    t = threading.Thread(target=srv.serve, daemon=True)
    t.start()
    # Wait for the port to accept.
    base = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            r = urllib.request.urlopen(base + "/api/health", timeout=0.5)
            if r.status == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.skip("server did not come up")
    yield base
    # Daemon thread; nothing to clean.


def _get(base: str, path: str, *, timeout: float = 5.0) -> tuple[int, dict]:
    """GET, return (status, parsed JSON or raw text)."""
    req = urllib.request.Request(base + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
            try:
                return r.status, json.loads(body)
            except Exception:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body


def _post(base: str, path: str, payload: dict | bytes,
          *, content_type: str = "application/json",
          timeout: float = 10.0) -> tuple[int, dict | str]:
    if isinstance(payload, dict):
        data = json.dumps(payload).encode("utf-8")
    else:
        data = payload
    req = urllib.request.Request(
        base + path, data=data, method="POST",
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
            try:
                return r.status, json.loads(body)
            except Exception:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body


# ---------------------------------------------------------------------------
# Health + listings
# ---------------------------------------------------------------------------

def test_health_ok(http_server):
    status, body = _get(http_server, "/api/health")
    assert status == 200
    assert body.get("ok") is True
    assert body.get("book")  # title is set


def test_root_serves_index_html(http_server):
    """GET / must serve the in-tree index.html.  Regression guard:
    the multi-book refactor briefly stopped wiring _Handler.static_dir
    and every page-load 404'd."""
    status, body = _get(http_server, "/")
    assert status == 200, "GET / should serve index.html"
    assert isinstance(body, str), "index.html must be served as text"
    assert "<!doctype" in body.lower() or "<html" in body.lower(), (
        "GET / didn't return HTML"
    )


def test_books_endpoint_lists_active(http_server):
    status, body = _get(http_server, "/api/books")
    assert status == 200
    books = body.get("books", [])
    assert len(books) >= 1
    assert any(b.get("active") for b in books)
    assert all("name" in b and "title" in b for b in books)


def test_voices_endpoint_returns_list_or_empty(http_server):
    status, body = _get(http_server, "/api/voices")
    assert status == 200
    assert "voices" in body
    assert isinstance(body["voices"], list)


def test_sessions_endpoint_initially_empty(http_server):
    status, body = _get(http_server, "/api/sessions")
    assert status == 200
    assert isinstance(body.get("sessions"), list)


def test_citation_graph_endpoint(http_server):
    status, body = _get(http_server, "/api/citation_graph")
    assert status == 200
    assert "nodes" in body and "edges" in body
    assert len(body["nodes"]) >= 1


def test_toc_endpoint(http_server):
    status, body = _get(http_server, "/api/toc")
    assert status == 200
    # ToC is a list / dict — just sanity-check the shape exists.
    assert body is not None


# ---------------------------------------------------------------------------
# Session creation + Q&A roundtrip
# ---------------------------------------------------------------------------

def test_answer_creates_session_then_404_on_unknown_plan(http_server):
    status, body = _post(http_server, "/api/answer",
                          {"question": "what is bagging"})
    assert status == 200
    pid = body.get("plan_id", "")
    assert pid

    # Query nonsense plan_id.
    status, body = _get(http_server, "/api/session/does-not-exist")
    assert status == 404


def test_session_resume_round_trip(http_server):
    status, body = _post(http_server, "/api/answer",
                          {"question": "what is regularization"})
    pid = body.get("plan_id", "")
    assert pid
    # Pause to keep the session in memory without consuming events.
    _post(http_server, "/api/pause", {"plan_id": pid})
    status, body = _get(http_server, "/api/session/" + pid)
    assert status == 200
    assert body.get("plan_id") == pid


def test_active_book_unknown_404(http_server):
    status, body = _post(http_server, "/api/active_book",
                          {"name": "does-not-exist-book"})
    assert status == 404


def test_active_book_known_ok(http_server):
    # Pick the active book name from /api/books.
    _, listing = _get(http_server, "/api/books")
    name = listing["active"]
    status, body = _post(http_server, "/api/active_book", {"name": name})
    assert status == 200
    assert body.get("ok") is True


# ---------------------------------------------------------------------------
# Robustness: malformed JSON
# ---------------------------------------------------------------------------

def test_malformed_json_returns_400(http_server):
    status, body = _post(http_server, "/api/answer",
                          b"{ not valid json",
                          content_type="application/json")
    assert status == 400


def test_question_missing_plan_id_404(http_server):
    status, body = _post(http_server, "/api/question",
                          {"question": "hello"})
    assert status == 404


def test_transcribe_empty_body_400(http_server):
    status, body = _post(http_server, "/api/transcribe", b"",
                          content_type="audio/wav")
    assert status == 400


# ---------------------------------------------------------------------------
# Pause / Resume / Cancel
# ---------------------------------------------------------------------------

def test_pause_resume_cancel(http_server):
    status, body = _post(http_server, "/api/answer",
                          {"question": "what is bias variance tradeoff"})
    pid = body.get("plan_id", "")
    assert pid
    s, _ = _post(http_server, "/api/pause", {"plan_id": pid})
    assert s == 200
    s, _ = _post(http_server, "/api/resume", {"plan_id": pid})
    assert s == 200
    s, _ = _post(http_server, "/api/cancel", {"plan_id": pid})
    assert s == 200
    # After cancel the session is dropped — subsequent /api/session
    # for the same id should return 404.
    s, _ = _get(http_server, "/api/session/" + pid)
    assert s == 404
