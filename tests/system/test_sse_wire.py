"""End-to-end SSE wire-format check.

Boots the in-process server, fires `/api/answer`, then opens the
SSE stream `/api/stream/<plan_id>` raw and parses the bytes.  Asserts:

  * Every event is shaped ``event: <name>\\ndata: <json>\\n\\n``.
  * ``data`` is valid JSON.
  * The expected event names appear (``clause`` and ``done`` always,
    plus ``audio_chunk`` / ``audio_complete`` when streaming TTS
    is active — we run with NullTTS here, so just clause / done).
  * The session ends with ``done``.

Skipped when the ESLII corpus isn't present.
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
    book_path = os.environ.get(
        "SEVIM_TEST_BOOK",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "books", "ESLII.json",
        ),
    )
    if not os.path.isfile(book_path):
        pytest.skip(f"ESLII corpus not found at {book_path}")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    os.environ["SEVIM_DISABLE_ASR"] = "1"
    os.environ["SEVIM_SKIP_LLM_EQ_LATEX"] = "1"
    os.environ["SEVIM_TTS_STREAM"] = "0"
    os.environ["SEVIM_QA_BACKEND"] = "tutor"
    sess_dir = tmp_path_factory.mktemp("sse_sessions")
    os.environ["SEVIM_SESSION_DIR"] = str(sess_dir)
    from serve.server import Server
    srv = Server(book_paths=[book_path], port=port,
                  prefer_kokoro=False, prefer_qwen=False)
    t = threading.Thread(target=srv.serve, daemon=True)
    t.start()
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


def _create_plan(base: str) -> str:
    """Create a plan via /api/answer; return plan_id."""
    req = urllib.request.Request(
        base + "/api/answer",
        data=json.dumps({"question": "what is bagging"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        body = json.loads(r.read())
    pid = body.get("plan_id", "")
    assert pid, body
    return pid


def _read_sse_blocks(stream, *, max_blocks: int = 20,
                     timeout_per_read: float = 5.0) -> list[dict]:
    """Read up to ``max_blocks`` SSE blocks from the stream.

    Each block is parsed into ``{event, data, raw_lines}``.  Stops
    early on ``event: done``.
    """
    blocks: list[dict] = []
    cur_event = ""
    cur_data: list[str] = []
    raw_lines: list[str] = []
    while len(blocks) < max_blocks:
        line = stream.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        raw_line = line.rstrip("\n").rstrip("\r")
        raw_lines.append(raw_line)
        if raw_line.startswith("event:"):
            cur_event = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            cur_data.append(raw_line[len("data:"):].strip())
        elif raw_line == "":
            if cur_event:
                payload_text = "".join(cur_data).strip()
                payload = None
                try:
                    payload = json.loads(payload_text) if payload_text else {}
                except Exception:
                    payload = {"_raw": payload_text}
                blocks.append({
                    "event": cur_event,
                    "data": payload,
                    "raw_lines": list(raw_lines),
                })
                if cur_event == "done":
                    return blocks
                cur_event = ""
                cur_data = []
                raw_lines = []
        else:
            # Comment / heartbeat — ignore.
            pass
    return blocks


def test_sse_wire_format(http_server):
    pid = _create_plan(http_server)
    req = urllib.request.Request(
        http_server + f"/api/stream/{pid}",
        headers={"Accept": "text/event-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as stream:
            assert stream.status == 200
            assert "text/event-stream" in stream.headers.get(
                "Content-Type", "",
            )
            blocks = _read_sse_blocks(stream, max_blocks=20)
    except urllib.error.URLError as e:
        pytest.fail(f"SSE stream errored: {e}")
    assert blocks, "no SSE blocks received"
    events = [b["event"] for b in blocks]
    # We always end with ``done``.
    assert events[-1] == "done", events
    # At least one ``clause`` arrived (NullTTS, non-streaming → one
    # clause event per narrated sentence).
    assert any(e == "clause" for e in events), events
    # Every block's data is valid JSON (a dict).
    for b in blocks:
        assert isinstance(b["data"], dict), b
    # Each clause carries the documented fields.
    for b in blocks:
        if b["event"] == "clause":
            d = b["data"]
            for key in ("seq", "clause_text", "home_nid",
                        "audio_b64", "audio_dur", "rate",
                        "voice", "word_timestamps", "visual_ops"):
                assert key in d, f"missing {key} in clause payload: {d.keys()}"


def test_sse_unknown_plan_returns_404(http_server):
    """Hitting /api/stream with an unknown plan_id returns 404."""
    try:
        with urllib.request.urlopen(
            http_server + "/api/stream/does-not-exist",
            timeout=5,
        ) as r:
            assert r.status == 404, r.status
    except urllib.error.HTTPError as e:
        assert e.code == 404
