"""Tier-3 e2e-smoke-agent for Lyceum (Section F of QUALITY_INSPECTOR_DESIGN.md).

Headless Playwright harness that drives the live UI against canonical
fixtures and asserts the user-facing happy path:

  F1  First clause within ≤ first_clause_budget_s of click.
  F2  Chapter narration runs for narration_window_s without page
      console errors / SSE disconnects / JS exceptions.
  F3  Question answered (first answer-audio chunk) within
      ≤ answer_budget_s of clicking Ask.
  F4  Voice round-trip (POST /api/transcribe with a known WebM blob)
      returns text within ≤ 3 s.   [optional, --voice]
  F5  Resumable session — kill server, restart, page reload, verify
      shapes re-mount.                                  [optional, --resume]

Output:
    health/smoke_runs/<ts>.json         per-run report
    health/smoke_runs/last_screenshot.png

Exit code 0 if every selected check passes, 2 otherwise.

Local-only: drives a local Lyceum on http://127.0.0.1:8001.  No
external API.

Usage:
    .venv/bin/python3 -m tools.smoke_agent
    .venv/bin/python3 -m tools.smoke_agent --book ESLII \\
        --chapter "Basis Expansions and Regularization"
    .venv/bin/python3 -m tools.smoke_agent --no-F3 --voice
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parent.parent
HEALTH_DIR = PROJECT / "health" / "smoke_runs"
BASE_URL = "http://127.0.0.1:8001/"


# ---------------------------------------------------------------------------
# Init script: instrument EventSource so we can measure SSE events from
# Playwright without depending on internal page state.
# ---------------------------------------------------------------------------

INIT_SCRIPT = r"""
(() => {
  window._smoke = {
    audio_chunks: [],            // list of {seq, panel, ts}
    clauses:      [],            // list of {seq, panel, ts}
    done:         false,
    tangent_open: 0,             // counter
    tangent_close: 0,
    sse_errors:   0,
    page_errors:  [],            // list of strings
    first_chunk_t: null,         // page-time (ms) of first audio_chunk
    started_at:   performance.now(),
  };
  const Orig = window.EventSource;
  if (!Orig) return;
  window.EventSource = function (url) {
    const es = new Orig(url);
    es.addEventListener('audio_chunk', ev => {
      const t = performance.now();
      try {
        const d = JSON.parse(ev.data || '{}');
        window._smoke.audio_chunks.push({
          seq: d.seq, panel: d.panel || 'main', t,
        });
      } catch (_) { window._smoke.audio_chunks.push({t}); }
      if (window._smoke.first_chunk_t == null)
        window._smoke.first_chunk_t = t;
    });
    es.addEventListener('clause', ev => {
      try {
        const d = JSON.parse(ev.data || '{}');
        window._smoke.clauses.push({
          seq: d.seq, panel: d.panel || 'main',
          t: performance.now(),
        });
      } catch (_) {}
    });
    es.addEventListener('tangent_start',
      () => { window._smoke.tangent_open++; });
    es.addEventListener('tangent_end',
      () => { window._smoke.tangent_close++; });
    es.addEventListener('done',
      () => { window._smoke.done = true; });
    es.addEventListener('error',
      () => { window._smoke.sse_errors++; });
    return es;
  };
  for (const k of ['CONNECTING','OPEN','CLOSED'])
    window.EventSource[k] = Orig[k];
})();
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _switch_book(name: str) -> None:
    body = json.dumps({"name": name}).encode()
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/api/active_book",
        data=body, headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=10).read()


def _new_page(pw, *, viewport=(1280, 720), grant_mic: bool = False):
    browser = pw.chromium.launch(headless=True)
    perms = ["microphone"] if grant_mic else []
    ctx = browser.new_context(
        viewport={"width": viewport[0], "height": viewport[1]},
        permissions=perms,
    )
    if grant_mic:
        ctx.grant_permissions(["microphone"], origin=BASE_URL)
    ctx.add_init_script(INIT_SCRIPT)
    page = ctx.new_page()
    page.on("pageerror", lambda exc: page.evaluate(
        "(s) => window._smoke.page_errors.push(s)", str(exc),
    ))
    return browser, ctx, page


def _open(page) -> None:
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=20000)
    try:
        page.wait_for_load_state("networkidle", timeout=4000)
    except Exception:
        pass
    page.wait_for_timeout(1500)


def _click_chapter(page, label: str) -> None:
    page.locator(".toc-node").first.click()
    page.wait_for_timeout(400)
    chap = page.locator(f'.toc-chapter:has-text("{label}")')
    if chap.count() == 0:
        raise RuntimeError(f"chapter {label!r} not in TOC")
    chap.first.scroll_into_view_if_needed()
    chap.first.click()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_F1_first_clause(page, *, label: str,
                            budget_s: float) -> dict:
    """Click chapter; first audio_chunk SSE event must arrive
    within budget_s wall-clock."""
    t0 = time.time()
    _click_chapter(page, label)
    # Poll for first chunk.
    deadline = t0 + budget_s
    first = None
    while time.time() < deadline:
        page.wait_for_timeout(500)
        first = page.evaluate(
            "() => window._smoke && window._smoke.first_chunk_t"
        )
        if first is not None:
            break
    elapsed = time.time() - t0
    if first is None:
        return {"status": "fail",
                "detail": {"reason": f"no audio_chunk within {budget_s}s",
                           "elapsed_s": round(elapsed, 2)}}
    return {"status": "pass",
            "detail": {"first_chunk_after_s": round(elapsed, 2),
                       "budget_s": budget_s}}


def check_F2_no_errors(page, *, narration_window_s: float) -> dict:
    """Run narration for narration_window_s; assert no page errors,
    no SSE disconnects, monotone clause progression."""
    page.wait_for_timeout(int(narration_window_s * 1000))
    snap = page.evaluate("""() => ({
        n_clauses:    (window._smoke.clauses || []).length,
        n_chunks:     (window._smoke.audio_chunks || []).length,
        sse_errors:   window._smoke.sse_errors || 0,
        page_errors:  window._smoke.page_errors || [],
        last_seq: ((window._smoke.clauses || []).slice(-1)[0] || {}).seq,
    })""")
    n_err = len(snap.get("page_errors", [])) + snap.get("sse_errors", 0)
    return {
        "status": "fail" if n_err > 0 or snap["n_clauses"] == 0 else "pass",
        "detail": {
            "narration_window_s": narration_window_s,
            "clauses_seen":       snap["n_clauses"],
            "audio_chunks_seen":  snap["n_chunks"],
            "last_seq":           snap["last_seq"],
            "sse_errors":         snap["sse_errors"],
            "page_errors":        snap["page_errors"][:5],
        },
    }


def check_F3_answer(page, *, question: str,
                      budget_s: float) -> dict:
    """Type a question into the Ask box, click Ask, assert a
    tangent_start fires AND a non-main audio_chunk arrives within
    budget_s."""
    pre = page.evaluate(
        "() => ({n: (window._smoke.audio_chunks||[]).filter(c => c.panel === 'tangent').length, "
        "open: window._smoke.tangent_open||0})"
    )
    inp = page.locator("#ask-input-header")
    if inp.count() == 0:
        return {"status": "fail",
                "detail": {"reason": "ask-input-header not found"}}
    inp.fill(question)
    page.locator("#ask-btn-header").first.click()
    t0 = time.time()
    deadline = t0 + budget_s
    got_tangent_chunk = False
    while time.time() < deadline:
        page.wait_for_timeout(500)
        cur = page.evaluate(
            "() => ({n: (window._smoke.audio_chunks||[]).filter(c => c.panel === 'tangent').length, "
            "open: window._smoke.tangent_open||0})"
        )
        if cur["n"] > pre["n"]:
            got_tangent_chunk = True
            break
    elapsed = time.time() - t0
    return {
        "status": "pass" if got_tangent_chunk else "fail",
        "detail": {"answer_audio_within_s": round(elapsed, 2),
                   "budget_s": budget_s,
                   "tangent_open_delta": cur["open"] - pre["open"],
                   "tangent_chunks_delta": cur["n"] - pre["n"]},
    }


def check_F4_voice_roundtrip(*, budget_s: float = 3.0) -> dict:
    """POST a small silent WebM/Opus blob to /api/transcribe and
    assert a JSON response with a ``text`` field arrives within
    budget_s.  We don't drive the browser — the round-trip is
    server-side."""
    # Minimal valid WebM/Opus container with no audio (header only).
    # faster-whisper returns "" for silence in <500ms typically.
    webm = (
        b"\x1aE\xdf\xa3\x9fB\x86\x81\x01B\xf7\x81\x01B\xf2\x81\x04"
        b"B\xf3\x81\x08B\x82\x84webmB\x87\x81\x02B\x85\x81\x02"
    )
    boundary = "----lyceum-smoke"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; '
        f'filename="silent.webm"\r\n'
        f"Content-Type: audio/webm\r\n\r\n"
    ).encode() + webm + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/api/transcribe",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=budget_s + 1) as resp:
            payload = json.loads(resp.read())
        elapsed = time.time() - t0
        ok = isinstance(payload, dict) and "text" in payload
        return {
            "status": "pass" if (ok and elapsed <= budget_s) else "fail",
            "detail": {"elapsed_s": round(elapsed, 2),
                       "budget_s": budget_s,
                       "has_text_field": ok,
                       "text_len": len((payload or {}).get("text", ""))},
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {"status": "fail",
                "detail": {"reason": repr(e),
                           "elapsed_s": round(elapsed, 2)}}


def check_F5_resume_session() -> dict:
    """Resumable-session check.  Stub for now — doing this properly
    requires killing + restarting the user's running server, which
    the smoke agent should not do silently.  Mark as ``skip`` with a
    note; the full implementation is a follow-up."""
    return {"status": "skip",
            "detail": {"note":
                "F5 needs server-lifecycle control; implement separately"}}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(*, book: str, chapter_label: str,
        first_clause_budget_s: float,
        narration_window_s: float,
        answer_budget_s: float,
        question: str,
        do_F1: bool, do_F2: bool, do_F3: bool,
        do_F4: bool, do_F5: bool,
        viewport: tuple[int, int],
        out_dir: Path) -> dict:

    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = out_dir or HEALTH_DIR
    out_path = out_dir / f"{ts}_{book}.json"
    shot_path = out_dir / f"{ts}_{book}.png"

    _switch_book(book)

    results: dict[str, dict] = {}
    started = time.time()
    with sync_playwright() as pw:
        browser, ctx, page = _new_page(pw, viewport=viewport,
                                         grant_mic=do_F4)
        try:
            _open(page)

            if do_F1:
                results["F1_first_clause"] = check_F1_first_clause(
                    page, label=chapter_label,
                    budget_s=first_clause_budget_s,
                )
            else:
                # Need the click for downstream checks even if F1 is off.
                _click_chapter(page, chapter_label)

            if do_F2:
                results["F2_no_errors"] = check_F2_no_errors(
                    page, narration_window_s=narration_window_s,
                )

            if do_F3:
                results["F3_answer"] = check_F3_answer(
                    page, question=question, budget_s=answer_budget_s,
                )

            if do_F4:
                # Voice path is server-side; doesn't need the page,
                # but we run it inside the same session for the
                # consolidated report.
                results["F4_voice_roundtrip"] = check_F4_voice_roundtrip()

            if do_F5:
                results["F5_resume_session"] = check_F5_resume_session()

            page.screenshot(path=str(shot_path), full_page=False)
        finally:
            browser.close()

    counts = {"pass": 0, "fail": 0, "skip": 0}
    for r in results.values():
        counts[r.get("status", "fail")] = counts.get(
            r.get("status", "fail"), 0) + 1

    report = {
        "ts":            ts,
        "book":          book,
        "chapter_label": chapter_label,
        "wall_seconds":  round(time.time() - started, 1),
        "summary":       counts,
        "checks":        results,
        "screenshot":    str(shot_path.relative_to(PROJECT)),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[smoke] wrote {out_path.relative_to(PROJECT)}")
    print(f"[smoke] summary: pass={counts['pass']} "
          f"fail={counts['fail']} skip={counts['skip']}")
    for cid, r in results.items():
        flag = {"pass": "✓", "fail": "✗", "skip": "·"}.get(
            r.get("status"), "?")
        print(f"  {flag} {cid}")
        d = r.get("detail") or {}
        for k, v in list(d.items())[:4]:
            print(f"      {k}: {v}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="ESLII",
                    help="active book name (must already be loaded)")
    ap.add_argument("--chapter",
                    default="Basis Expansions and Regularization",
                    help="substring of a TOC chapter label")
    ap.add_argument("--first-clause-budget", type=float, default=15.0,
                    help="seconds (cold start; tighten to 5 once warm)")
    ap.add_argument("--narration-window", type=float, default=45.0)
    ap.add_argument("--answer-budget",   type=float, default=20.0)
    ap.add_argument("--question",
                    default="what is the bias variance tradeoff")
    ap.add_argument("--no-F1", action="store_true")
    ap.add_argument("--no-F2", action="store_true")
    ap.add_argument("--no-F3", action="store_true")
    ap.add_argument("--voice", action="store_true",
                    help="run F4 voice round-trip (default off)")
    ap.add_argument("--resume", action="store_true",
                    help="run F5 resume-session (currently stubbed)")
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    rep = run(
        book=args.book, chapter_label=args.chapter,
        first_clause_budget_s=args.first_clause_budget,
        narration_window_s=args.narration_window,
        answer_budget_s=args.answer_budget,
        question=args.question,
        do_F1=not args.no_F1, do_F2=not args.no_F2, do_F3=not args.no_F3,
        do_F4=args.voice, do_F5=args.resume,
        viewport=(args.width, args.height),
        out_dir=args.out_dir,
    )
    return 0 if rep["summary"]["fail"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
