"""Stdlib HTTP+SSE server for the Lyceum / SeVim_math teaching session.

Endpoints
---------
  GET  /                          static index.html
  GET  /static/*                  static assets
  POST /api/narrate               body: {"topic": str, ...} → {"plan_id": str}
  POST /api/read_book             body: {"max_content": int?, ...} → {"plan_id": str}
                                  Mode B: read entire book in pre-order.
  GET  /api/stream/{plan_id}      SSE stream of narration events
  POST /api/question              body: {"plan_id", "question"}
                                  Mode C: inject a tangent.
  POST /api/pause                 body: {"plan_id"} → halts dispatch.
  POST /api/resume                body: {"plan_id"} → resumes dispatch.
  POST /api/cancel                body: {"plan_id"} → tears down session.
  GET  /api/snapshot/{plan_id}    SVG of the main chalkboard.
  GET  /api/snapshot/{plan_id}/tangent  SVG of the tangent panel.
  GET  /api/health                {"ok": true}

All payloads are JSON.  SSE events:
    event: clause          → {panel, ...event-fields, is_tangent_start, is_tangent_end, tangent_id}
    event: paused          → {}
    event: resumed         → {}
    event: tangent_start   → {tangent_id}
    event: tangent_end     → {tangent_id}
    event: done            → {}
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from book import load_corpus
from chalkboard import Chalkboard
from chalkboard.policy import ReadingOrderPolicy
from narrator import plan as plan_topic, plan_full
from narrator.overview import (
    render_toc, render_chapter, overview_stats,
)
from narrator.crossref_view import (
    render_chapter_flow, render_top_cited, render_node_neighbourhood,
)
from narrator.tts import _BackendBase, NullTTS, auto_tts
from .session import Session, build_session


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, Session] = {}
_SESSIONS_LOCK = threading.Lock()


def _autosave(session: Session) -> None:
    """Persist the session to disk after each turn.  Failure is non-
    fatal — a crashed save logs and the user keeps going."""
    try:
        from . import persistence
        persistence.save_session_dict(
            session.plan_id, session.snapshot_dict(),
        )
    except Exception as e:
        print(f"[serve] autosave failed for {session.plan_id}: {e}")


def _store(session: Session) -> None:
    """Register the session and wire its auto-save callback so every
    turn snapshots to disk (resumable across server restarts)."""
    if session._save_callback is None:
        session._save_callback = _autosave
    with _SESSIONS_LOCK:
        _SESSIONS[session.plan_id] = session
    # Snapshot immediately so even a session that takes no turns is
    # discoverable on the next reload.
    _autosave(session)


def _get(plan_id: str) -> Optional[Session]:
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(plan_id)
    if sess is not None:
        return sess
    # Lazy-rehydrate from disk if not in memory.
    return _resume_from_disk(plan_id)


def _resume_from_disk(plan_id: str) -> Optional[Session]:
    """Try to rebuild a session from a saved snapshot.  Returns the
    session if successfully restored; ``None`` otherwise.  Adds it to
    the in-memory registry so subsequent calls hit the cache.

    When the snapshot carries a ``book_name`` and that book is loaded
    on the server, the session is bound to that specific book — so a
    user resuming a Bishop conversation while ESLII is currently
    active still continues with the right corpus.
    """
    try:
        from . import persistence
        snap = persistence.load_session_dict(plan_id)
    except Exception:
        return None
    if not snap:
        return None
    # Prefer the snapshot's recorded book; fall back to the server's
    # active book.
    srv = SERVER_STATE.get("server")
    book = None
    snap_book_name = (snap.get("book_name") or "").strip()
    if srv is not None and snap_book_name:
        book = srv.books_by_name.get(snap_book_name)
    if book is None:
        book = SERVER_STATE.get("book")
    if book is None:
        return None
    sess = Session(
        plan_id=plan_id, book=book,
        main_orch=None,                   # main narration is gone
    )
    sess.restore_from_dict(snap)
    sess._save_callback = _autosave
    with _SESSIONS_LOCK:
        _SESSIONS[plan_id] = sess
    return sess


def _drop(plan_id: str) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.pop(plan_id, None)
    # Also remove the saved snapshot so the user can't reload back
    # into a session they explicitly cancelled.
    try:
        from . import persistence
        persistence.delete_session(plan_id)
    except Exception:
        pass


# Tiny shared state for callbacks that need a back-reference to the
# loaded book / config.  Populated from the run() entry point.
SERVER_STATE: dict = {}


# ---------------------------------------------------------------------------
# Visual pre-compute helper — runs the SVG synth during plan-build for
# the narrate-node path so the diagram is ready before clause 0's audio
# starts.  The previously-existing canonical-formulas side-channel was
# removed because it produced spoken-formula cards that the narration
# never actually spoke.
# ---------------------------------------------------------------------------

def _prepopulate_node_visuals(plan, node) -> None:
    """Best-effort: kick off LLM-SVG diagram synth and stash the result
    on ``plan.meta["intro_svg"]``.  No-op when the LLM endpoint is
    unreachable.  Adds at most ~5 s to plan-build."""
    title = (node.title or "").strip()
    if not title:
        return
    if not plan.meta:
        plan.meta = {}
    try:
        from viz.llm_synth import synthesise as _synth
        svg_res = _synth(title, budget_s=5.0)
    except Exception:
        return
    if svg_res is not None:
        plan.meta["intro_svg"] = {
            "svg_body": svg_res.svg_body,
            "width": svg_res.width,
            "height": svg_res.height,
            "elapsed_ms": svg_res.elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "Lyceum/0.2"

    book = None  # type: ignore[assignment]
    book_path = ""
    static_dir = ""
    image_dir = ""
    tts_factory = lambda self: NullTTS()  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:
        return

    # ---- routing -----------------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if path == "/favicon.ico":
            return self._favicon()
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):], None)
        if path.startswith("/api/figure/"):
            fid = path[len("/api/figure/"):]
            return self._figure(fid)
        if path.startswith("/api/stream/"):
            plan_id = path[len("/api/stream/"):]
            return self._sse_stream(plan_id)
        if path.startswith("/api/snapshot/"):
            rest = path[len("/api/snapshot/"):]
            if rest.endswith("/tangent"):
                return self._snapshot(rest[:-len("/tangent")], tangent=True)
            return self._snapshot(rest, tangent=False)
        if path == "/api/health":
            return self._json({"ok": True, "book": self.book.title if self.book else None})
        if path == "/api/overview":
            return self._overview()
        if path == "/api/chapter_maps":
            return self._chapter_maps_list()
        if path.startswith("/api/ingest_status/"):
            from urllib.parse import unquote
            book_id = unquote(path[len("/api/ingest_status/"):])
            return self._ingest_status(book_id)
        if path.startswith("/api/chapter_viz/"):
            from urllib.parse import unquote
            nid = unquote(path[len("/api/chapter_viz/"):])
            return self._chapter_viz(nid)
        if path.startswith("/api/sevim_diagram/"):
            from urllib.parse import unquote
            nid = unquote(path[len("/api/sevim_diagram/"):])
            return self._sevim_diagram(nid)
        if path.startswith("/api/section_formula/"):
            from urllib.parse import unquote
            nid = unquote(path[len("/api/section_formula/"):])
            return self._section_formula(nid)
        if path.startswith("/api/figures/"):
            from urllib.parse import unquote
            nid = unquote(path[len("/api/figures/"):])
            return self._figures_for_nid(nid)
        if path.startswith("/api/figure_image/"):
            from urllib.parse import unquote
            fid = unquote(path[len("/api/figure_image/"):])
            return self._figure_image(fid)
        if path == "/api/overview/stats":
            return self._overview_stats()
        if path == "/api/toc":
            return self._toc_json()
        if path.startswith("/api/chapter/"):
            from urllib.parse import unquote
            nid = unquote(path[len("/api/chapter/"):])
            return self._chapter(nid)
        if path == "/api/xref/flow":
            return self._xref_flow()
        if path == "/api/xref/top":
            return self._xref_top()
        if path == "/api/citation_graph":
            return self._citation_graph()
        if path == "/api/sessions":
            return self._sessions_list()
        if path == "/api/voices":
            return self._voices_list()
        if path == "/api/books":
            return self._books_list()
        if path.startswith("/api/session/"):
            from urllib.parse import unquote
            plan_id = unquote(path[len("/api/session/"):])
            return self._session_resume(plan_id)
        if path.startswith("/api/xref/focus/"):
            from urllib.parse import unquote
            nid = unquote(path[len("/api/xref/focus/"):])
            return self._xref_focus(nid)
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        path = self.path
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        # Binary audio endpoint — handled before the JSON parser so we
        # never try to JSON-decode a webm/opus blob.
        if path == "/api/transcribe":
            return self._transcribe(body)
        if path == "/api/upload_book":
            return self._upload_book(body)
        if path == "/api/active_book":
            try:
                data = json.loads(body) if body else {}
            except Exception:
                self.send_error(400, "bad json")
                return
            return self._set_active_book(data)
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self.send_error(400, "bad json")
            return
        if path == "/api/narrate":
            return self._narrate(data)
        if path == "/api/narrate_node":
            return self._narrate_node(data)
        if path == "/api/read_book":
            return self._read_book(data)
        if path == "/api/question":
            return self._question(data)
        if path == "/api/answer":
            return self._answer(data)
        if path == "/api/pause":
            return self._control(data, action="pause")
        if path == "/api/resume":
            return self._control(data, action="resume")
        if path == "/api/cancel":
            return self._control(data, action="cancel")
        self.send_error(404, "not found")

    # ---- static ------------------------------------------------------------

    def _serve_static(self, rel: str, content_type: Optional[str]) -> None:
        if ".." in rel or rel.startswith("/"):
            self.send_error(400, "bad path")
            return
        full = os.path.join(self.static_dir, rel)
        if not os.path.isfile(full):
            self.send_error(404, "not found")
            return
        if content_type is None:
            ext = os.path.splitext(rel)[1].lower()
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml",
                ".json": "application/json",
                ".png": "image/png",
                ".wav": "audio/wav",
            }.get(ext, "application/octet-stream")
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # ``no-store`` instead of ``no-cache``: ``no-cache`` lets the
        # browser revalidate against a stale ETag; ``no-store`` forces
        # a full re-fetch every time, so visual changes (like the
        # edge-rendering rework) propagate without the user having to
        # remember Ctrl+Shift+R.
        self.send_header("Cache-Control",
                         "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    def _favicon(self) -> None:
        """Silence the browser's default /favicon.ico request."""
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()

    def _figure(self, fid: str) -> None:
        """Serve an extracted figure image by FigureRef.fid.

        Three lookup paths:
          * Original ingestion dir (book/parse_pdf.py output);
          * v2 sidecar dir (tools/reingest_figures.py);
          * On-demand crop cache (serve/figure_ondemand.py) — for
            references that the original ingestion missed but that
            we crop on the fly from the source PDF.
        """
        if self.book is None:
            self.send_error(503, "no figure store")
            return
        # On-demand fids have a known prefix and a label suffix; for them
        # we re-derive the crop (cached) directly from the PDF.
        if fid.startswith("od_"):
            return self._figure_ondemand(fid)
        if not self.image_dir:
            self.send_error(503, "no figure store")
            return
        ref = next((f for f in self.book.figures if f.fid == fid), None)
        if ref is None or not ref.image_path:
            self.send_error(404, "figure not found")
            return
        if "/" in ref.image_path or ".." in ref.image_path:
            self.send_error(400, "bad path")
            return
        v2_dir = self.image_dir.rstrip("/")
        if not v2_dir.endswith("_v2"):
            v2_dir = v2_dir + "_v2"
        candidates = [
            os.path.join(self.image_dir, ref.image_path),
            os.path.join(v2_dir, ref.image_path),
        ]
        full = next((p for p in candidates if os.path.isfile(p)), None)
        if full is None:
            self.send_error(404, "image missing on disk")
            return
        with open(full, "rb") as f:
            data = f.read()
        ext = os.path.splitext(ref.image_path)[1].lower()
        ctype = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml", ".gif": "image/gif",
        }.get(ext, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _figure_ondemand(self, fid: str) -> None:
        """Serve an on-demand PDF-cropped figure.  ``fid`` has form
        ``od_{label}`` where label is the figure number (e.g.
        ``od_6.14`` for "Figure 6.14").
        """
        from .figure_ondemand import crop_figure_by_label
        label = fid[len("od_"):]
        if not re.fullmatch(r"\d+(?:\.\d+){0,2}", label or ""):
            self.send_error(400, "bad on-demand fid")
            return
        pdf_path = (self.book.source or "").strip()
        if not pdf_path or not os.path.isfile(pdf_path):
            # Try sibling .pdf of the corpus json (book_path).
            stem = os.path.splitext(self.book_path or "")[0]
            cand = stem + ".pdf"
            if os.path.isfile(cand):
                pdf_path = cand
        if not pdf_path:
            self.send_error(503, "no source PDF")
            return
        od = crop_figure_by_label(pdf_path, label)
        if od is None or not os.path.isfile(od.full_path):
            self.send_error(404, "figure not found in PDF")
            return
        with open(od.full_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    # ---- session lifecycle -------------------------------------------------

    def _build(self, plan, *, max_content: int, canvas_w: float,
               canvas_h: float, voice: Optional[str], speed: float,
               detail_level: str = "L1_story") -> Session:
        plan_id = uuid.uuid4().hex[:12]
        session = build_session(
            plan_id=plan_id, book=self.book, plan=plan,
            tts_factory=type(self).tts_factory,
            canvas_w=canvas_w, canvas_h=canvas_h,
            max_content=max_content,
            voice=voice, speed=speed,
            book_path=self.book_path or "",
            detail_level=detail_level,
        )
        _store(session)
        return session

    def _overview(self) -> None:
        """Render the chapter-level book overview (depth=1) as SVG."""
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        svg = render_toc(
            self.book, max_depth=1, include_environments=False,
            canvas_w=2400, canvas_h=1500,
        )
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.end_headers()
        self.wfile.write(svg.encode("utf-8"))

    def _chapter_viz(self, root_nid: str) -> None:
        """Serve the cached chapter-visualization SVG for *root_nid*.

        The file lives next to the book at
        ``<stem>.chapter_viz.<flat_nid>.svg`` and is generated offline
        by ``tools/build_chapter_visualization.py``.  When missing
        we kick off a background build so the frontend can re-poll
        and pick it up once it lands — narration never waits on this
        round-trip; the chapter-zoom stack starts immediately and the
        right-side panel fills in asynchronously.

        Returns the SVG bytes with ``Content-Type: image/svg+xml`` on
        hit, ``404`` with a small JSON status while the build is
        still running so the frontend's poll loop can distinguish
        "not generated yet" from a hard failure.
        """
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        # Resolve active book path (same logic as _chapter_maps_list).
        book_path = ""
        for name, bk in (getattr(self, "books_by_name", {}) or {}).items():
            if bk is self.book:
                book_path = (getattr(self, "book_paths_by_name", {}) or {}
                             ).get(name, "")
                break
        if not book_path:
            book_path = self.book.source or ""
        if not book_path:
            return self._json({"error": "no book path"}, status=503)
        stem, _ = os.path.splitext(book_path)
        flat = root_nid.replace("/", "_")
        svg_path = f"{stem}.chapter_viz.{flat}.svg"
        if os.path.isfile(svg_path):
            try:
                with open(svg_path, "rb") as f:
                    body = f.read()
            except OSError as e:
                return self._json({"error": f"read failed: {e}"}, status=500)
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # Not cached.  Spawn the LLM build in a daemon thread (idempotent
        # — the tool's own build() refuses to overwrite without --force,
        # and our presence-check below guards against duplicate spawns).
        sidecar = f"{stem}.chapter_map.{flat}.json"
        if os.path.isfile(sidecar):
            self._maybe_spawn_chapter_viz_build(sidecar, svg_path)
        return self._json(
            {"status": "pending", "nid": root_nid},
            status=404,
        )

    # Track in-flight chapter-viz builds so we don't fork the same
    # Qwen call ten times while the frontend polls.  Keyed by output
    # path so two different chapters can build in parallel.
    _viz_builds_in_flight: dict[str, float] = {}

    def _maybe_spawn_chapter_viz_build(self, sidecar: str,
                                        out_path: str) -> None:
        import threading
        import time as _t
        now = _t.monotonic()
        # GIL serialises single dict ops, so a plain get/set is enough
        # to dedupe overlapping polls without a separate lock.  Treat a
        # build older than 5 minutes as gone (failed silently or the
        # process restarted between polls).
        started = type(self)._viz_builds_in_flight.get(out_path, 0.0)
        if started > 0.0 and now - started < 300.0:
            return
        type(self)._viz_builds_in_flight[out_path] = now

        def _run() -> None:
            try:
                from tools.build_chapter_visualization import build
                build(sidecar)
            except Exception as e:
                print(f"[chapter-viz] background build failed: {e}",
                      file=sys.stderr)
            finally:
                type(self)._viz_builds_in_flight.pop(out_path, None)

        t = threading.Thread(target=_run, daemon=True,
                              name="chapter-viz-build")
        t.start()

    # Cache: ``<book_path> -> {nid -> svg_string}`` so we don't reload
    # the JSON sidecar on every chapter-map navigation event.  Cleared
    # implicitly by server restart (which is when sidecars get rebuilt
    # via ``tools/build_sevim_diagrams``).
    _sevim_diagram_cache: dict[str, dict[str, str]] = {}
    # Cache: ``<book_path> -> {nid -> {"label": str, "latex": str}}``
    # built lazily from the chapter_map sidecars on first request.
    _section_formula_cache: dict[str, dict[str, dict]] = {}
    # Cache: ``<book_path> -> (figures_by_nid_dict, image_dir_abs_path,
    # fid_to_path_dict)``.  Built lazily on first figures request.
    _figures_cache: dict[str, tuple] = {}

    # Cache: ``<book_path> -> {nid -> body_text}`` so the figures
    # endpoint can match figures by *label mention* in the active
    # section's narration text, not just by home_nid.  Built lazily
    # from the same chapter_map sidecars the formula cache reads.
    _section_text_cache: dict[str, dict[str, str]] = {}

    def _section_texts_for_active_book(self) -> dict[str, str]:
        """Return ``{nid: text}`` covering every chapter-map node of
        the active book, where ``text`` is the concatenation of
        gist + story_paragraph + formula_explanation.  Used to scan
        for figure-label mentions."""
        srv = SERVER_STATE.get("server")
        if srv is None:
            return {}
        active_name = getattr(srv, "active_book_name", "")
        book_path = (getattr(srv, "book_paths_by_name", {}) or {}
                     ).get(active_name, "")
        if not book_path:
            return {}
        cached = type(self)._section_text_cache.get(book_path)
        if cached is not None:
            return cached
        out: dict[str, str] = {}
        import glob as _glob
        stem, _ = os.path.splitext(book_path)

        def _walk(n: dict):
            nid = (n.get("nid") or "").strip()
            if nid:
                pieces = [
                    (n.get("gist") or "").strip(),
                    (n.get("story_paragraph") or "").strip(),
                    (n.get("formula_explanation") or "").strip(),
                ]
                txt = " \n".join(p for p in pieces if p)
                if txt:
                    out[nid] = txt
            for c in n.get("children", []) or []:
                _walk(c)

        for sidecar in _glob.glob(f"{stem}.chapter_map.*.json"):
            try:
                with open(sidecar) as f:
                    payload = json.load(f)
            except Exception:
                continue
            root = (payload or {}).get("root") or {}
            if root:
                _walk(root)
        type(self)._section_text_cache[book_path] = out
        return out

    def _figures_index_for_active_book(self) -> Optional[tuple]:
        """Return ``(by_nid, image_dir, fid_to_path)`` for the active
        book.  Loads the sidecar lazily on first call.  ``None`` when
        no sidecar exists for the active book."""
        srv = SERVER_STATE.get("server")
        if srv is None:
            return None
        # Resolve active book path via the (name → path) map.
        active_name = getattr(srv, "active_book_name", "")
        book_path = (getattr(srv, "book_paths_by_name", {}) or {}
                     ).get(active_name, "")
        if not book_path:
            active_book = (getattr(srv, "books_by_name", {}) or {}
                           ).get(active_name)
            if active_book is not None:
                book_path = getattr(active_book, "source", "") or ""
        if not book_path:
            return None
        cached = type(self)._figures_cache.get(book_path)
        if cached is not None:
            return cached
        stem, _ = os.path.splitext(book_path)
        sidecar_path = stem + ".figures.json"
        if not os.path.isfile(sidecar_path):
            type(self)._figures_cache[book_path] = ({}, "", {})
            return ({}, "", {})
        try:
            with open(sidecar_path) as f:
                payload = json.load(f)
        except Exception:
            type(self)._figures_cache[book_path] = ({}, "", {})
            return ({}, "", {})
        by_nid = payload.get("by_nid") or {}
        image_dir_rel = payload.get("image_dir") or ""
        # ``image_dir`` in the sidecar is just the basename
        # (``..._figures_v2``); resolve against the book's directory.
        image_dir = os.path.join(os.path.dirname(book_path), image_dir_rel) \
                    if image_dir_rel else ""
        # Build fid → absolute path map for the image endpoint.
        fid_to_path: dict[str, str] = {}
        for nid, entries in by_nid.items():
            if not isinstance(entries, list):
                continue
            for e in entries:
                fid = (e.get("fid") or "").strip()
                rel = (e.get("image_path") or "").strip()
                if fid and rel and image_dir:
                    fid_to_path[fid] = os.path.join(image_dir, rel)
        result = (by_nid, image_dir, fid_to_path)
        type(self)._figures_cache[book_path] = result
        return result

    def _figures_for_nid(self, nid: str) -> None:
        """``GET /api/figures/<nid>`` — figures whose home is *nid*
        OR any of nid's ancestors.  Walking up the tree means a
        narration anchored at a deeply-nested subsection still gets
        the figures attached at the section / chapter level (their
        usual home in scientific layouts).
        """
        idx = self._figures_index_for_active_book()
        if idx is None:
            return self._json({"figures": []})
        by_nid, _image_dir, _fid_to_path = idx
        if not by_nid:
            return self._json({"figures": []})
        # Return *every* figure that lives under the same chapter as
        # the queried nid.  The chapter prefix is the first 1-3 path
        # segments of the nid (covers ``b/ch5`` for ESLII and
        # ``b/p1/s_ch_1_regular_languages`` for Sipser).  Walking up
        # the nid like this avoids needing a chapter-aware index on
        # the server side; the frontend then highlights the
        # mentioned figure as the narration moves through cells.
        target = (nid or "").strip()
        if not target:
            return self._json({"nid": nid, "figures": []})
        segs = [s for s in target.split("/") if s]
        # Use the first 3 segments (or fewer if the nid is shorter)
        # — covers both the ``b/ch5`` and ``b/p1/s_ch_1_*`` shapes.
        chapter_prefix = "/".join(segs[: min(3, len(segs))])

        def _is_related(home: str) -> bool:
            if not home:
                return False
            return home == chapter_prefix \
                or home.startswith(chapter_prefix + "/") \
                or chapter_prefix.startswith(home + "/") \
                or chapter_prefix == home

        seen_fids: set[str] = set()
        out: list[dict] = []

        def _push(entry: dict) -> None:
            fid = entry.get("fid") or ""
            if not fid or fid in seen_fids:
                return
            seen_fids.add(fid)
            out.append({
                "fid": fid,
                "label": entry.get("label") or "",
                "caption": entry.get("caption") or "",
                "page": entry.get("page") or 0,
                "home_nid": entry.get("home_nid") or "",
                "aliases": list(entry.get("aliases") or []),
                "image_url": "/api/figure_image/" + fid,
            })

        for home, entries in by_nid.items():
            if not _is_related(home):
                continue
            for entry in (entries or []):
                _push(entry)

        # Also pull in any figure whose LABEL ("Figure 1.14") appears
        # in the active section's narration text.  Story paragraphs
        # often cite figures attached at completely different home
        # nids (a section text saying "Figure 1.14 illustrates …"
        # references a figure rooted at a different theorem-level
        # nid in the math-graph attribution).  Matching by mention
        # closes that gap so the panel shows what the narrator just
        # named.
        section_texts = self._section_texts_for_active_book()
        active_text = section_texts.get(target, "")
        if active_text:
            import re as _re_lbl
            # Build a label → entry map for fast lookup.
            label_to_entry: dict[str, dict] = {}
            for entries in by_nid.values():
                for e in entries or []:
                    lab = (e.get("label") or "").strip()
                    if lab and lab not in label_to_entry:
                        label_to_entry[lab] = e
            mentioned = set()
            # Match "Figure 1.14", "FIGURE 1.14", "Table 2.3", etc.
            for m in _re_lbl.finditer(
                r"\b(?:Figure|FIGURE|Table|TABLE|Diagram|DIAGRAM)"
                r"\s+(\d+(?:\.\d+)?)\b",
                active_text,
            ):
                lbl_num = m.group(1)
                # Try several casing variants of the label.
                for cand in (
                    f"Figure {lbl_num}", f"FIGURE {lbl_num}",
                    f"Table {lbl_num}", f"TABLE {lbl_num}",
                    f"Diagram {lbl_num}", f"DIAGRAM {lbl_num}",
                ):
                    e = label_to_entry.get(cand)
                    if e is not None and cand not in mentioned:
                        mentioned.add(cand)
                        _push(e)
                        break
        # Sort by page so the figures appear in book order.
        out.sort(key=lambda f: (f.get("page") or 0, f.get("label") or ""))
        return self._json({"nid": nid, "figures": out})

    def _figure_image(self, fid: str) -> None:
        """``GET /api/figure_image/<fid>`` — serve the PNG bytes for
        a figure id.  404 if the fid isn't in the active book's
        index."""
        idx = self._figures_index_for_active_book()
        if idx is None:
            self.send_error(404, "no figures index")
            return
        _by_nid, _image_dir, fid_to_path = idx
        path = fid_to_path.get(fid)
        if not path or not os.path.isfile(path):
            self.send_error(404, "figure not found")
            return
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError as e:
            return self._json({"error": f"read failed: {e}"}, status=500)
        ext = (os.path.splitext(path)[1] or "").lower()
        ctype = "image/png"
        if ext in (".jpg", ".jpeg"):
            ctype = "image/jpeg"
        elif ext == ".webp":
            ctype = "image/webp"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "max-age=3600")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----- Upload + ingest --------------------------------------------------

    def _parse_multipart_pdf(
        self, body: bytes,
    ) -> tuple[str, bytes] | None:
        """Tiny multipart/form-data parser scoped to ONE file field
        called ``pdf``.  Returns (filename, bytes) on success, None
        otherwise.  Avoids the deprecated ``cgi`` module and the
        full ``email.parser`` ceremony — we only need this one
        shape for one endpoint."""
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype.lower():
            return None
        # Pull out boundary=...
        m_bound = None
        for part in ctype.split(";"):
            part = part.strip()
            if part.lower().startswith("boundary="):
                m_bound = part[len("boundary="):].strip().strip('"')
                break
        if not m_bound:
            return None
        boundary = b"--" + m_bound.encode("latin-1")
        # Each section starts with the boundary, ends with the next.
        parts = body.split(boundary)
        for raw in parts:
            raw = raw.lstrip(b"\r\n")
            if not raw or raw.startswith(b"--"):
                continue
            # Split headers from body on the first \r\n\r\n.
            head_end = raw.find(b"\r\n\r\n")
            if head_end < 0:
                continue
            headers_blob = raw[:head_end].decode("latin-1", "replace")
            body_blob = raw[head_end + 4:]
            # Strip the trailing \r\n that precedes the next boundary.
            if body_blob.endswith(b"\r\n"):
                body_blob = body_blob[:-2]
            # Look for a Content-Disposition with name="pdf" and a filename.
            if "name=\"pdf\"" not in headers_blob:
                continue
            filename = ""
            for line in headers_blob.split("\r\n"):
                if line.lower().startswith("content-disposition:"):
                    for piece in line.split(";"):
                        piece = piece.strip()
                        if piece.lower().startswith("filename="):
                            filename = piece[len("filename="):].strip().strip('"')
                            break
            return (filename or "uploaded.pdf", body_blob)
        return None

    def _upload_book(self, body: bytes) -> None:
        """``POST /api/upload_book`` — multipart/form-data with one PDF
        file field named ``pdf``.  Saves the PDF, registers an entry
        in ``books/_index.json``, kicks off the ingest pipeline in a
        daemon thread, and returns immediately with the book id so
        the client can poll ``/api/ingest_status/<id>``.
        """
        srv = SERVER_STATE.get("server")
        if srv is None:
            return self._json({"error": "server not initialised"},
                              status=503)
        if not body:
            return self._json({"error": "empty body"}, status=400)
        parsed = self._parse_multipart_pdf(body)
        if parsed is None:
            return self._json(
                {"error": "expected multipart/form-data with a 'pdf' field"},
                status=400,
            )
        filename, pdf_bytes = parsed
        if not filename.lower().endswith(".pdf"):
            return self._json({"error": "filename must end in .pdf"},
                              status=400)
        if len(pdf_bytes) < 1024:
            return self._json({"error": "PDF is suspiciously small"},
                              status=400)

        # Persist next to the existing books.  Books-dir defaults to
        # the dirname of the first loaded book; falls back to ``books/``
        # under the cwd.
        from . import books_index, ingest_pipeline
        books_dir = ""
        if srv.book_paths_by_name:
            first = next(iter(srv.book_paths_by_name.values()))
            books_dir = os.path.dirname(first) or ""
        if not books_dir:
            books_dir = os.path.join(os.getcwd(), "books")
        os.makedirs(books_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(filename))[0]
        book_id = books_index.unique_id(books_dir, stem)
        pdf_path = os.path.join(books_dir, f"{book_id}.pdf")
        corpus_json = os.path.join(books_dir, f"{book_id}.json")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        # Pre-register the entry with a placeholder title; the ingest
        # pipeline overwrites it once the PDF metadata is parsed.
        books_index.add_book(
            books_dir,
            book_id=book_id,
            title=stem.replace("-", " ").replace("_", " ").title(),
            source_pdf=pdf_path,
            corpus_json=corpus_json,
        )

        def _register(corpus_path: str) -> None:
            """Hand the freshly-built corpus to the running server's
            in-memory book registry so the next ``/api/books`` call
            sees it without a server restart."""
            try:
                from book.corpus import load_corpus
                book = load_corpus(corpus_path)
                stem_now = os.path.splitext(os.path.basename(corpus_path))[0]
                srv.books_by_name[stem_now] = book
                srv.book_paths_by_name[stem_now] = corpus_path
                # Also patch the index entry's title from the parsed PDF.
                books_index.update_book(
                    books_dir, book_id, title=(book.title or stem_now),
                )
            except Exception as e:
                print(f"[upload] register failed: {e}")

        ingest_pipeline.spawn(
            books_dir, book_id, pdf_path, corpus_json,
            register_with_server=_register,
        )
        return self._json({
            "ok": True,
            "id": book_id,
            "title": stem.replace("-", " ").replace("_", " ").title(),
            "pdf": pdf_path,
        })

    def _ingest_status(self, book_id: str) -> None:
        """``GET /api/ingest_status/<id>`` — current phase of the
        background ingest for *book_id*.  Returns 404 when the id is
        unknown."""
        srv = SERVER_STATE.get("server")
        if srv is None:
            return self._json({"error": "server not initialised"},
                              status=503)
        from . import books_index
        books_dir = ""
        if srv.book_paths_by_name:
            first = next(iter(srv.book_paths_by_name.values()))
            books_dir = os.path.dirname(first) or ""
        if not books_dir:
            books_dir = os.path.join(os.getcwd(), "books")
        entry = books_index.get_book(books_dir, book_id)
        if entry is None:
            return self._json(
                {"error": "unknown book id", "id": book_id},
                status=404,
            )
        return self._json(entry)

    def _section_formula(self, nid: str) -> None:
        """Return the canonical formula label + LaTeX for *nid*.

        Read from the chapter-map sidecars belonging to the active
        book.  Used by the frontend's right panel to display the
        section's equation underneath the SeVim concept diagram.
        """
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        book_path = ""
        for name, bk in (getattr(self, "books_by_name", {}) or {}).items():
            if bk is self.book:
                book_path = (getattr(self, "book_paths_by_name", {}) or {}
                             ).get(name, "")
                break
        if not book_path:
            book_path = self.book.source or ""
        if not book_path:
            return self._json({"error": "no book path"}, status=503)
        cache = type(self)._section_formula_cache.get(book_path)
        if cache is None:
            cache = {}
            import glob as _glob
            stem, _ = os.path.splitext(book_path)

            # The chapter-map's canonical_formula_latex is whatever
            # PyMuPDF spat out, which often needs the same cleaning
            # the chalkboard cell renderer does (``M X m=1 \beta mhm``
            # → ``\sum_{m=1}^{M} \beta_m h_m`` etc.).  Apply it here
            # so the right-panel formula card receives renderable
            # LaTeX, not raw OCR.
            try:
                from viz.treemap import _clean_ocr_latex
            except Exception:
                _clean_ocr_latex = lambda s: s   # type: ignore

            def _walk(n: dict):
                cur_nid = (n.get("nid") or "").strip()
                cur_label = (n.get("canonical_formula_label") or "").strip()
                cur_latex = (n.get("canonical_formula_latex") or "").strip()
                if cur_nid and cur_label and cur_latex:
                    cleaned = _clean_ocr_latex(cur_latex) or cur_latex
                    cache[cur_nid] = {
                        "label": cur_label,
                        "latex": cleaned,
                    }
                for c in n.get("children", []) or []:
                    _walk(c)

            for sidecar in _glob.glob(f"{stem}.chapter_map.*.json"):
                try:
                    with open(sidecar) as f:
                        d = json.load(f)
                    root = d.get("root") or {}
                    if root:
                        _walk(root)
                except Exception:
                    continue
            type(self)._section_formula_cache[book_path] = cache
        meta = cache.get(nid)
        if not meta:
            return self._json({"status": "missing", "nid": nid}, status=404)
        return self._json({"nid": nid, **meta})

    def _sevim_diagram(self, nid: str) -> None:
        """Return the per-section SeVim concept diagram SVG for *nid*.

        The concept diagrams are pre-built by
        ``tools/build_sevim_diagrams`` and stored in
        ``<book_stem>.sevim_diagrams.<root>.json`` keyed by nid.  We
        glob over every such sidecar belonging to the active book and
        merge them into one in-memory map; the lookup is O(1) after
        the first request.
        """
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        # Resolve active book path.
        book_path = ""
        for name, bk in (getattr(self, "books_by_name", {}) or {}).items():
            if bk is self.book:
                book_path = (getattr(self, "book_paths_by_name", {}) or {}
                             ).get(name, "")
                break
        if not book_path:
            book_path = self.book.source or ""
        if not book_path:
            return self._json({"error": "no book path"}, status=503)
        cache = type(self)._sevim_diagram_cache.get(book_path)
        if cache is None:
            cache = {}
            import glob as _glob
            stem, _ = os.path.splitext(book_path)
            for sidecar in _glob.glob(f"{stem}.sevim_diagrams.*.json"):
                try:
                    with open(sidecar) as f:
                        d = json.load(f)
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if isinstance(k, str) and isinstance(v, str):
                                cache[k] = v
                except Exception:
                    continue
            type(self)._sevim_diagram_cache[book_path] = cache
        svg = cache.get(nid)
        if not svg:
            return self._json(
                {"status": "missing", "nid": nid},
                status=404,
            )
        body = svg.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chapter_maps_list(self) -> None:
        """Return the nids that have a chapter-map sidecar on disk so
        the frontend's TOC click handler can switch chapter-zoom on a
        single click instead of forcing the user to shift-click.

        Cheap glob — re-evaluated every call so a sidecar that lands
        on disk between page loads is discovered without a server
        restart.
        """
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        # Resolve the active book's source path so we can scan the
        # right ``books/<stem>.chapter_map.<nid>.json`` glob.
        book_path = ""
        for name, bk in (getattr(self, "books_by_name", {}) or {}).items():
            if bk is self.book:
                book_path = (getattr(self, "book_paths_by_name", {}) or {}
                             ).get(name, "")
                break
        if not book_path:
            book_path = self.book.source or ""
        nids: list[str] = []
        if book_path:
            import glob as _glob
            stem, _ = os.path.splitext(book_path)
            prefix = f"{stem}.chapter_map."
            # Read the ``root_nid`` field inside each sidecar — that's
            # the canonical nid the file refers to, no reverse-mangling
            # heuristics needed.
            for sidecar in _glob.glob(f"{prefix}*.json"):
                try:
                    with open(sidecar) as f:
                        d = json.load(f)
                    nid = (d.get("root_nid") or "").strip()
                    if nid:
                        nids.append(nid)
                except Exception:
                    continue
        return self._json({"nids": sorted(set(nids))})

    def _overview_stats(self) -> None:
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        return self._json(overview_stats(
            self.book, max_depth=2, include_environments=False,
        ))

    def _toc_json(self) -> None:
        """Plain-JSON table of contents — frontend uses this for keyboard
        navigation and search."""
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        from book.ir import is_structural

        def entry(node):
            return {
                "nid": node.nid, "kind": node.kind,
                "number": node.number, "title": node.title,
                "page_start": node.page_start, "page_end": node.page_end,
                "children": [entry(c) for c in node.children
                             if is_structural(c.kind)],
            }
        return self._json({
            "title": self.book.title, "author": self.book.author,
            "root": entry(self.book.root),
        })

    def _chapter(self, nid: str) -> None:
        """Render a single chapter sub-tree at deeper detail."""
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        if not self.book.find(nid):
            return self._json({"error": f"no node {nid}"}, status=404)
        svg = render_chapter(
            self.book, nid, max_depth=4,
            canvas_w=2000, canvas_h=1200,
        )
        if not svg:
            return self._json({"error": "could not render chapter"},
                              status=500)
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.end_headers()
        self.wfile.write(svg.encode("utf-8"))

    def _xref_flow(self) -> None:
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        svg = render_chapter_flow(self.book)
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.end_headers()
        self.wfile.write(svg.encode("utf-8"))

    def _xref_top(self) -> None:
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        svg = render_top_cited(self.book)
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.end_headers()
        self.wfile.write(svg.encode("utf-8"))

    def _citation_graph(self) -> None:
        """GET /api/citation_graph — chapter-level citation graph as
        JSON.  Used by the visual map panel to render an interactive
        map the user can click to jump into any chapter.
        """
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        from book import xref as xref_mod
        return self._json(xref_mod.chapter_graph(self.book))

    def _sessions_list(self) -> None:
        """GET /api/sessions — list recent persisted sessions."""
        from . import persistence
        return self._json({"sessions": persistence.list_session_summaries()})

    def _books_list(self) -> None:
        """GET /api/books — list every loaded book + every uploaded
        book whose ingest hasn't finished yet.

        Returns ``{books: [{name, title, n_chapters, active, phase,
        progress}], active}``.  ``phase`` is ``"ready"`` for books
        the orchestrator can play right now, anything else for
        in-flight ingests.
        """
        srv = SERVER_STATE.get("server")
        if srv is None:
            return self._json({"books": [], "active": ""})
        out = []
        seen_names: set[str] = set()
        for name, book in srv.books_by_name.items():
            chapters = [
                n for n in book.root.walk()
                if n.kind == "chapter"
            ]
            out.append({
                "name": name,
                "title": (book.title or name).strip(),
                "n_chapters": len(chapters),
                "active": name == srv.active_book_name,
                "phase": "ready",
                "progress": "",
            })
            seen_names.add(name)
        # Merge in-flight uploads from the JSON index.  The
        # background ingest thread will register the book in
        # ``books_by_name`` once its corpus is built; until then we
        # surface it here with its current phase so the dropdown
        # can show a "(ingesting…)" hint.
        try:
            from . import books_index
            books_dir = ""
            if srv.book_paths_by_name:
                first = next(iter(srv.book_paths_by_name.values()))
                books_dir = os.path.dirname(first) or ""
            if not books_dir:
                books_dir = os.path.join(os.getcwd(), "books")
            for entry in books_index.list_books(books_dir):
                book_id = entry.get("id") or ""
                if not book_id or book_id in seen_names:
                    continue
                phase = entry.get("phase") or "uploading"
                out.append({
                    "name": book_id,
                    "title": entry.get("title") or book_id,
                    "n_chapters": entry.get("total_chapters") or 0,
                    "active": False,
                    "phase": phase,
                    "progress": entry.get("progress") or "",
                })
        except Exception as e:
            print(f"[books] merging index failed: {e}")
        return self._json({
            "books": out,
            "active": srv.active_book_name,
        })

    def _set_active_book(self, data: dict) -> None:
        """POST /api/active_book {name: ...}.  Switches the active
        book; new sessions use the new book.  Existing sessions keep
        their original book."""
        srv = SERVER_STATE.get("server")
        if srv is None:
            return self._json({"error": "server not initialised"},
                              status=503)
        name = (data.get("name") or "").strip()
        if not name or name not in srv.books_by_name:
            return self._json(
                {"error": f"unknown book {name!r}"}, status=404,
            )
        srv._activate_book(name)
        return self._json({"ok": True, "active": name})

    def _voices_list(self) -> None:
        """GET /api/voices — list available Kokoro voices.

        Empty list when no Kokoro install is detected.  Frontend
        falls back to the default and disables the picker.
        """
        try:
            from narrator.tts import list_kokoro_voices, _DEFAULT_VOICE
            voices = list_kokoro_voices()
            return self._json({
                "voices": voices,
                "default": _DEFAULT_VOICE,
            })
        except Exception as e:
            return self._json(
                {"voices": [], "default": "", "error": str(e)},
                status=200,
            )

    def _session_resume(self, plan_id: str) -> None:
        """GET /api/session/<plan_id> — return a session's snapshot
        so the frontend can rebuild its UI after a page reload.

        Lazy-rehydrates the in-memory registry from disk if needed.
        Responds 404 when the plan_id is unknown.
        """
        sess = _get(plan_id)
        if sess is None:
            return self._json(
                {"error": f"no session {plan_id}"}, status=404,
            )
        return self._json(sess.snapshot_dict())

    def _xref_focus(self, nid: str) -> None:
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        if not self.book.find(nid):
            return self._json({"error": f"no node {nid}"}, status=404)
        svg = render_node_neighbourhood(self.book, nid)
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.end_headers()
        self.wfile.write(svg.encode("utf-8"))

    def _narrate_node(self, data: dict) -> None:
        """Start narration scoped to a specific BookNode (clicked from TOC).

        Default mode is *outline* (hierarchical tour of the subtree);
        callers can pass ``mode=full`` to recite the entire subtree
        sequentially.
        """
        from narrator import plan_outline, plan_chapter_zoom
        from narrator.planner import plan_full
        nid = data.get("nid", "")
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        node = self.book.find(nid)
        if node is None:
            return self._json({"error": f"no node {nid}"}, status=404)
        mode = (data.get("mode") or "outline").lower()
        # When the clicked node has a chapter-map sidecar on disk and
        # the caller didn't override the mode, switch to chapter-zoom:
        # render the whole-chapter treemap, narrate top-down via
        # punch-line + per-section role-in-parent tour.  Falls back to
        # the depth-decay outline when no sidecar exists.
        chapter_map_payload = None
        if mode == "outline":
            book_path = ""
            try:
                from .server import SERVER_STATE  # type: ignore
                srv = SERVER_STATE.get("server")
                if srv is not None:
                    for name, bk in srv.books_by_name.items():
                        if bk is self.book:
                            book_path = srv.book_paths_by_name.get(name, "")
                            break
            except Exception:
                pass
            if not book_path:
                book_path = self.book.source or ""
            if book_path:
                from tools.build_chapter_map import (
                    chapter_map_path_for_root,
                )
                cmap_path = chapter_map_path_for_root(book_path, nid)
                if os.path.isfile(cmap_path):
                    try:
                        chapter_map_payload = json.load(open(cmap_path))
                        mode = "chapter_zoom"
                    except Exception as e:
                        print(f"[serve] chapter-map load failed: {e}")
        if mode == "full":
            sub = type(self.book)(
                title=self.book.title, author=self.book.author,
                source=self.book.source, root=node,
                figures=self.book.figures, cross_refs=self.book.cross_refs,
                concepts=self.book.concepts, pages=self.book.pages,
                meta=dict(self.book.meta),
            )
            p = plan_full(sub)
        elif mode == "chapter_zoom" and chapter_map_payload is not None:
            p = plan_chapter_zoom(
                self.book,
                chapter_map=chapter_map_payload,
                drill_depth=int(data.get("drill_depth", 2)),
            )
        else:
            p = plan_outline(
                self.book,
                root_nid=nid,
                max_depth=int(data.get("max_depth", 3)),
            )
        if not p.clauses:
            return self._json({"error": f"node {nid} has no narratable content"},
                              status=404)
        # Pre-compute the canonical SVG diagram and formula list in
        # parallel with the rest of plan-build, so the user gets the
        # diagram + formulas with the very first clause's audio
        # rather than 5 s into the narration.  The orchestrator's
        # ``_canonical_visual_op`` reads from ``plan.meta["intro_svg"]``
        # / the formula card emitter reads from ``plan.meta["intro_formulas"]``.
        _prepopulate_node_visuals(p, node)
        session = self._build(
            p,
            max_content=int(data.get("max_content", 8)),
            canvas_w=float(data.get("canvas_w", 1280)),
            canvas_h=float(data.get("canvas_h", 720)),
            voice=data.get("voice"),
            speed=float(data.get("speed", 1.0)),
            detail_level=str(data.get("detail_level", "L1_story")),
        )
        return self._json({
            "plan_id": session.plan_id, "mode": mode,
            "nid": nid,
            "n_clauses": len(p.clauses), "n_visited": len(p.visited_nids),
            "intro_svg": bool((p.meta or {}).get("intro_svg")),
            "intro_formulas": len((p.meta or {}).get("intro_formulas") or []),
        })

    def _narrate(self, data: dict) -> None:
        topic = (data.get("topic") or "").strip()
        if not topic:
            return self._json({"error": "topic required"}, status=400)
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        p = plan_topic(self.book, topic, top_k=int(data.get("top_k", 4)))
        if not p.clauses:
            return self._json({"error": "no relevant content for topic",
                               "meta": p.meta}, status=404)
        session = self._build(
            p,
            max_content=int(data.get("max_content", 8)),
            canvas_w=float(data.get("canvas_w", 1280)),
            canvas_h=float(data.get("canvas_h", 720)),
            voice=data.get("voice"),
            speed=float(data.get("speed", 1.0)),
            detail_level=str(data.get("detail_level", "L1_story")),
        )
        return self._json({
            "plan_id": session.plan_id, "topic": topic,
            "n_clauses": len(p.clauses), "n_visited": len(p.visited_nids),
            "mode": "topic",
        })

    def _read_book(self, data: dict) -> None:
        """Read the book.  Default mode is *outline*: a top-down tour
        that visits each chapter title + a short intro, then descends
        into sections / subsections.  ``mode=full`` (legacy) reads the
        entire body sequentially.

        Body params:
            mode:        "outline" (default) | "full"
            root_nid:    when given, scope the outline to that subtree.
            max_depth:   outline depth (default 3 — chapter / section /
                         subsection).
        """
        from narrator import plan_outline
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        mode = (data.get("mode") or "outline").lower()
        if mode == "full":
            p = plan_full(self.book)
        else:
            p = plan_outline(
                self.book,
                root_nid=data.get("root_nid", "") or "",
                max_depth=int(data.get("max_depth", 3)),
            )
        if not p.clauses:
            return self._json({"error": "book has no narratable content"},
                              status=404)
        session = self._build(
            p,
            max_content=int(data.get("max_content", 8)),
            canvas_w=float(data.get("canvas_w", 1280)),
            canvas_h=float(data.get("canvas_h", 720)),
            voice=data.get("voice"),
            speed=float(data.get("speed", 1.0)),
            detail_level=str(data.get("detail_level", "L1_story")),
        )
        return self._json({
            "plan_id": session.plan_id, "mode": mode,
            "root_nid": p.meta.get("root_nid", ""),
            "n_clauses": len(p.clauses),
            "n_visited": len(p.visited_nids),
        })

    def _question(self, data: dict) -> None:
        plan_id = data.get("plan_id")
        question = (data.get("question") or "").strip()
        session = _get(plan_id)
        if session is None:
            return self._json({"error": "no such plan"}, status=404)
        if not question:
            return self._json({"error": "question required"}, status=400)
        try:
            tid = session.ask(
                question,
                top_k=int(data.get("top_k", 3)),
                voice=data.get("voice"),
                speed=float(data.get("speed", 1.0)),
                detail_level=str(data.get("detail_level", "L1_story")),
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self._json(
                {"error": f"question handler failed: {exc}"}, status=500,
            )
        return self._json({"tangent_id": tid})

    def _answer(self, data: dict) -> None:
        """Standalone Q&A: create a tangent-only session, run the
        answer in the right panel, leave the main board untouched.

        When the user clicks Ask without a running lecture, they want
        a focused answer — they do NOT want the system to commandeer
        the main flow and start narrating a chapter on its own.  Once
        the tangent finishes, the session is idle and ready for the
        next question.  The user can explicitly start a lecture via
        Topic / Read book if they want narration.
        """
        question = (data.get("question") or "").strip()
        if self.book is None:
            return self._json({"error": "no book loaded"}, status=503)
        if not question:
            return self._json({"error": "question required"}, status=400)
        # Tangent-only session: main_orch=None so no main events are
        # ever emitted, but the tangent iterator is set by ask().
        plan_id = uuid.uuid4().hex[:12]
        session = Session(
            plan_id=plan_id, book=self.book, main_orch=None,
        )
        session.tangent_board = Chalkboard(
            canvas_w=float(data.get("canvas_w", 1280)) * 0.6,
            canvas_h=float(data.get("canvas_h", 720)),
            max_content=int(data.get("max_content", 8)),
            policy=ReadingOrderPolicy(),
        )
        session.tts_factory = type(self).tts_factory
        session._save_callback = _autosave
        with _SESSIONS_LOCK:
            _SESSIONS[plan_id] = session
        try:
            tid = session.ask(
                question,
                top_k=int(data.get("top_k", 3)),
                voice=data.get("voice"),
                speed=float(data.get("speed", 1.0)),
                detail_level=str(data.get("detail_level", "L1_story")),
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with _SESSIONS_LOCK:
                _SESSIONS.pop(plan_id, None)
            return self._json(
                {"error": f"answer handler failed: {exc}"}, status=500,
            )
        return self._json({
            "plan_id": plan_id,
            "tangent_id": tid,
            "mode": "answer",
            "question": question,
        })

    def _control(self, data: dict, *, action: str) -> None:
        plan_id = data.get("plan_id")
        session = _get(plan_id)
        if session is None:
            return self._json({"error": "no such plan"}, status=404)
        if action == "pause":
            session.pause()
        elif action == "resume":
            session.resume()
        elif action == "cancel":
            session.cancel()
            _drop(plan_id)
        return self._json({"ok": True, "action": action})

    def _transcribe(self, body: bytes) -> None:
        """POST /api/transcribe — body is raw audio (webm/opus, mp3,
        wav, …); return ``{"text", "duration", "language", "elapsed"}``.

        First request loads the Whisper model (~75 MB tiny, ~5s on
        cold cache).  Subsequent requests reuse the singleton.
        """
        from . import asr
        if not body:
            return self._json({"error": "empty audio body"}, status=400)
        try:
            res = asr.transcribe(body)
        except asr.AudioDecodeError as e:
            return self._json({"error": f"decode failed: {e}"}, status=400)
        except ValueError as e:
            return self._json({"error": str(e)}, status=400)
        except RuntimeError as e:
            return self._json({"error": str(e)}, status=503)
        return self._json({
            "text": res.text,
            "duration": round(res.duration, 3),
            "language": res.language,
            "elapsed": round(res.elapsed, 3),
        })

    # ---- streaming ---------------------------------------------------------

    def _sse_stream(self, plan_id: str) -> None:
        session = _get(plan_id)
        if session is None:
            self.send_error(404, "no such plan")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while session.is_active():
                pe = session.next_event()
                if pe is None:
                    break
                if pe.is_tangent_start:
                    self._sse_emit("tangent_start",
                                   {"tangent_id": pe.tangent_id})
                if pe.is_tangent_end:
                    self._sse_emit("tangent_end",
                                   {"tangent_id": pe.tangent_id})
                    continue
                ev = pe.event
                # Streaming-TTS subevents — the orchestrator interleaves
                # AudioChunkEvent / AudioCompleteEvent between the
                # clause's StreamEvent so the browser can begin
                # playback while later chunks are still rendering.
                from .orchestrator import (
                    AudioChunkEvent as _ACE,
                    AudioCompleteEvent as _ACO,
                )
                if isinstance(ev, _ACE):
                    self._sse_emit("audio_chunk", {
                        "panel": pe.panel,
                        "tangent_id": pe.tangent_id,
                        "seq": ev.seq,
                        "chunk_idx": ev.chunk_idx,
                        "pcm_b64": ev.pcm_b64,
                        "rate": ev.rate,
                        "text": ev.text,
                    })
                    continue
                if isinstance(ev, _ACO):
                    self._sse_emit("audio_complete", {
                        "panel": pe.panel,
                        "tangent_id": pe.tangent_id,
                        "seq": ev.seq,
                        "duration": ev.duration,
                        "n_chunks": ev.n_chunks,
                    })
                    continue
                payload = {
                    "panel": pe.panel,
                    "tangent_id": pe.tangent_id,
                    "seq": ev.seq,
                    "clause_text": ev.clause_text,
                    "home_nid": ev.home_nid,
                    "audio_b64": ev.audio_b64,
                    "audio_dur": ev.audio_dur,
                    "rate": ev.rate,
                    "voice": ev.voice,
                    "word_timestamps": ev.word_timestamps,
                    "visual_ops": ev.visual_ops,
                    "streaming": getattr(ev, "streaming", False),
                    "graph_stats": getattr(ev, "graph_stats", {}) or {},
                }
                self._sse_emit("clause", payload)
            self._sse_emit("done", {})
            # Persist the math semantic graph so enrichment from this
            # session survives across runs (per-book persistence).
            # Tangent-only sessions (no_main_orch) skip this — there's
            # no main orchestrator to ask.
            if session.main_orch is not None:
                try:
                    session.main_orch.save_math_graph()
                except Exception as e:
                    print(f"[server] math_graph.save failed: {e}")
        except (ConnectionResetError, BrokenPipeError):
            session.cancel()

    def _sse_emit(self, event: str, payload: dict) -> None:
        try:
            self.wfile.write(b"event: ")
            self.wfile.write(event.encode("ascii"))
            self.wfile.write(b"\ndata: ")
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            self.wfile.write(b"\n\n")
            self.wfile.flush()
        except (ConnectionResetError, BrokenPipeError):
            raise

    def _snapshot(self, plan_id: str, *, tangent: bool) -> None:
        session = _get(plan_id)
        if session is None:
            self.send_error(404, "no such plan")
            return
        board = session.tangent_board if tangent else session.main_board
        svg = board.snapshot()
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.end_headers()
        self.wfile.write(svg.encode("utf-8"))

    # ---- helpers -----------------------------------------------------------

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

class Server:
    def __init__(
        self, *,
        book_paths: list[str],
        static_dir: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8001,
        prefer_kokoro: bool = True,
        prefer_qwen: bool = False,
    ) -> None:
        if not book_paths:
            raise ValueError("at least one book path required")
        self.host = host
        self.port = port
        self.static_dir = static_dir or os.path.join(
            os.path.dirname(__file__), "static",
        )
        # Wire static_dir on the Handler so / and /static/* serve
        # the in-tree index.html.  The multi-book refactor lost this
        # line; without it every page request 404'd.
        _Handler.static_dir = self.static_dir
        # Load every book the operator passed; key by the file's stem.
        # The first one is the default active book.
        self.books_by_name: dict = {}
        self.book_paths_by_name: dict = {}
        self.book_aliases: dict = {}
        self.book_concept_graphs: dict = {}
        self.book_image_dirs: dict = {}
        from book.aliases import extract_aliases
        from book.concept_graph import extract_concept_graph
        from narrator import qa as qa_mod
        for path in book_paths:
            book = load_corpus(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            self.books_by_name[stem] = book
            self.book_paths_by_name[stem] = path
            try:
                amap = extract_aliases(book)
            except Exception as e:
                print(f"[serve] alias map for {stem!r} failed: {e}")
                amap = {}
            self.book_aliases[stem] = amap
            try:
                cg = extract_concept_graph(book)
            except Exception as e:
                print(f"[serve] concept graph for {stem!r} failed: {e}")
                cg = {}
            self.book_concept_graphs[stem] = cg
            n_edges = sum(len(v) for v in cg.values()) if cg else 0
            cand = os.path.join(os.path.dirname(path), f"{stem}_figures")
            self.book_image_dirs[stem] = cand if os.path.isdir(cand) else ""
            print(f"[serve] loaded book {stem!r} — "
                  f"{len(amap)} aliases, {n_edges} concept edges")
            # Math semantic graph — pre-build once per book if missing
            # OR if the corpus JSON is newer than the graph file (the
            # book was re-ingested).  This is the offline-first
            # extraction so the runtime narrator can look up formulas
            # / containment / references / derivations rather than
            # racing the audio to detect them.
            try:
                from sevim.math_graph import graph_path_for_book
                gpath = graph_path_for_book(path)
                needs_build = (
                    not os.path.isfile(gpath) or
                    os.path.getmtime(path) > os.path.getmtime(gpath)
                )
                if needs_build:
                    print(f"[serve] building math graph for {stem!r} "
                          f"— this may take a minute…")
                    from tools.build_math_graph import build as _mg_build
                    _mg_build(path, verbose=False)
                else:
                    from sevim.math_graph import MathGraph
                    g = MathGraph.load(gpath)
                    print(f"[serve] math graph for {stem!r}: "
                          f"{len(g.formulas)} formulas, "
                          f"{len(g.passages)} passages, "
                          f"{len(g.edges)} edges")
            except Exception as e:
                print(f"[serve] math graph for {stem!r} failed: {e}")
        # Active book = first one given; can be switched at runtime via
        # POST /api/active_book.
        self.active_book_name = next(iter(self.books_by_name))
        # Stash registries on SERVER_STATE so the handler + persistence
        # callbacks can find them (the Handler class is a singleton-ish
        # configuration target, so we'd hit problems sharing complex
        # state directly).
        SERVER_STATE["server"] = self
        self._activate_book(self.active_book_name)

        # ---- TTS + QA backend setup (one-time, not per active-book swap) ----
        if prefer_kokoro:
            # Build ONE long-lived Kokoro worker shared across every
            # session and tangent — its synthesize() is already thread-
            # safe via an internal lock, so serialising N concurrent
            # callers is fine.  Saves the per-session subprocess spawn
            # (~1-2 s) on every tangent.  Warm it with a tiny phrase
            # so the first real synth doesn't pay the cold-start tax.
            #
            # Default backend: KokoroStreamTTS (per-phrase chunks
            # streamed to the browser; first audio in <1 s on long
            # answers).  Set SEVIM_TTS_STREAM=0 to fall back to the
            # legacy single-WAV-per-clause path.
            shared_tts = None
            if os.environ.get("SEVIM_TTS_STREAM", "1") != "0":
                try:
                    from narrator.tts import KokoroStreamTTS
                    shared_tts = KokoroStreamTTS()
                    print("[serve] kokoro streaming TTS loaded "
                          "(per-phrase audio chunks)")
                except Exception as e:
                    print(f"[serve] streaming TTS unavailable ({e}); "
                          "falling back to single-WAV synth")
            if shared_tts is None:
                shared_tts = auto_tts(prefer="kokoro")
            try:
                import time as _t
                _w0 = _t.perf_counter()
                shared_tts.synthesize("ready.")
                print(f"[serve] kokoro warmed in "
                      f"{_t.perf_counter() - _w0:.2f}s "
                      f"(backend: {shared_tts.name})")
            except Exception as e:
                print(f"[serve] kokoro warmup failed: {e}")
            _Handler.tts_factory = staticmethod(lambda: shared_tts)
        else:
            _Handler.tts_factory = staticmethod(lambda: NullTTS())
        # Q&A backends: closed-book synth (uses passages) and the
        # low-similarity intro path (open-book — used only when the
        # question's match into the book is weak).  Both target the
        # same local vLLM endpoint.  We try both whenever vLLM is
        # reachable, even if --qwen wasn't passed, because the intro
        # backend is a strict improvement over the "I cannot find
        # anything in this book" template.
        from narrator import qa as qa_mod
        from narrator import (
            make_vllm_backend, make_vllm_tutor_backend,
            make_vllm_intro_backend,
        )
        from narrator.qa import (
            make_vllm_tutor_streaming_backend, set_streaming_backend,
        )
        vllm_url = os.environ.get("VLLM_BASE_URL",
                                  "http://127.0.0.1:8000/v1")
        vllm_model = os.environ.get("VLLM_MODEL",
                                    "Qwen/Qwen2.5-14B-Instruct-AWQ")
        # Default behaviour: install the chat-style tutor backend whenever
        # vLLM is reachable.  Lets Qwen choose answer length naturally
        # from the user's phrasing and pick up dialogue history for
        # follow-ups.  Set SEVIM_QA_BACKEND=strict to fall back to the
        # legacy 2-4-sentence plain-prose backend.
        backend_choice = os.environ.get("SEVIM_QA_BACKEND", "tutor").lower()
        if prefer_qwen or backend_choice in ("tutor", "strict"):
            if backend_choice == "strict":
                backend = make_vllm_backend(
                    base_url=vllm_url, model=vllm_model,
                )
                tag = "strict closed-book"
            else:
                backend = make_vllm_tutor_backend(
                    base_url=vllm_url, model=vllm_model,
                )
                tag = "tutor (adaptive length)"
            if backend:
                qa_mod.set_synth_backend(backend)
                print(f"[serve] using vLLM at {vllm_url} for Q&A — {tag}")
            else:
                print("[serve] vLLM unreachable; using retrieval-only Q&A "
                      "(start it with: bash /home/ara/vllm-restart/"
                      "start_vllm_8000_qwen_text.sh)")
        # Always try to install the intro backend (silent if vLLM is down).
        intro = make_vllm_intro_backend(base_url=vllm_url, model=vllm_model)
        if intro:
            qa_mod.set_intro_backend(intro)
            print(f"[serve] using vLLM at {vllm_url} for low-similarity intro")

        # Streaming tutor backend — opt out via SEVIM_QA_STREAM=0 if you
        # want the full response buffered before TTS starts.
        if os.environ.get("SEVIM_QA_STREAM", "1") != "0":
            stream_bk = make_vllm_tutor_streaming_backend(
                base_url=vllm_url, model=vllm_model,
            )
            if stream_bk:
                set_streaming_backend(stream_bk)
                print(f"[serve] streaming tutor enabled — first audio "
                      f"in <1 s after the LLM produces its first sentence")

    def _activate_book(self, name: str) -> None:
        """Switch the default book that NEW sessions will use.

        Existing sessions keep their original book — switching only
        affects which corpus the next ``/api/answer`` / ``/api/narrate``
        etc. bind to.  Swaps the qa module's alias / concept caches so
        retrieval and dependencies use the right book's signals.
        Idempotent and safe to call mid-runtime.
        """
        if name not in self.books_by_name:
            raise KeyError(f"unknown book {name!r}")
        self.active_book_name = name
        active = self.books_by_name[name]
        from narrator import qa as qa_mod
        qa_mod.set_alias_map(self.book_aliases.get(name, {}))
        qa_mod.set_concept_graph(self.book_concept_graphs.get(name, {}))
        _Handler.book = active
        _Handler.book_path = self.book_paths_by_name[name]
        _Handler.image_dir = self.book_image_dirs.get(name, "")
        SERVER_STATE["book"] = active
        SERVER_STATE["active_book_name"] = name
        print(f"[serve] active book set to {name!r}")

    def serve(self) -> None:
        httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        active = self.books_by_name[self.active_book_name]
        n_books = len(self.books_by_name)
        print(f"Lyceum serving {active.title!r}"
              + (f" (+{n_books-1} more)" if n_books > 1 else "")
              + f" on http://{self.host}:{self.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="sevim-serve",
                                description="Run the SeVim_math teaching server.")
    p.add_argument("book", nargs="+",
                   help="path(s) to ingested corpus JSON; multiple "
                        "books load all of them and let the user "
                        "switch between them at runtime")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--no-tts", action="store_true",
                   help="skip Kokoro; use silent NullTTS")
    p.add_argument("--qwen", action="store_true",
                   help="enable local Qwen synthesis for Q&A answers")
    args = p.parse_args(argv)
    Server(
        book_paths=args.book, host=args.host, port=args.port,
        prefer_kokoro=not args.no_tts, prefer_qwen=args.qwen,
    ).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
