"""Record an MP4 video of the Lyceum app narrating one chapter, with the
Kokoro TTS voice as the audio track.

The page itself is the only valid SSE subscriber (the producer/consumer
queue inside ``serve.server`` hands each event to whoever consumes it
first, so two parallel subscribers would silently split the audio).  We
therefore monkey-patch ``EventSource`` from a Playwright init script so
the page's own subscription mirrors every ``audio_chunk`` PCM payload
into a JS-side buffer that we read out at the end of the run; the
browser keeps playing the audio normally so the visual side-effects
stay locked to the audio clock.

PCM chunks: base64-encoded 16-bit signed little-endian mono at 24 000 Hz
(Kokoro default, hard-coded in serve/orchestrator.py).

Usage:
    .venv/bin/python3 tools/record_narration.py \\
        --book sipser-introduction-to-the-theory-of-computation-3e-3a09 \\
        --chapter "Ch 1: Regular Languages" \\
        --max-seconds 180 \\
        --width 1920 --height 1080 \\
        --out paper/figures/lyceum_sipser_ch1.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT = Path('/home/ara/Documents/Programming/sevim_math')
BASE_URL = 'http://127.0.0.1:8001/'
SAMPLE_RATE = 24_000  # Kokoro default; see serve/orchestrator.py


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def http_post(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL.rstrip('/') + path,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def write_wav(path: Path, pcm16: bytes, rate: int) -> None:
    """Write a minimal mono 16-bit PCM WAV.  Pure stdlib, no scipy."""
    n_samples = len(pcm16) // 2
    n_channels = 1
    bits_per_sample = 16
    byte_rate = rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = len(pcm16)
    chunk_size = 36 + data_size
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', chunk_size))
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))                # PCM format
        f.write(struct.pack('<H', n_channels))
        f.write(struct.pack('<I', rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm16)


# ---------------------------------------------------------------------------
# init script: rewrite EventSource so audio_chunk PCM mirrors into JS state
# ---------------------------------------------------------------------------

INIT_SCRIPT = r'''
(() => {
  window._lyceumPCM = [];        // base64 chunks in arrival order
  window._lyceumDone = false;    // flips true on the SSE 'done' event
  window._lyceumStartedAt = 0;   // performance.now() when first chunk lands
  const Orig = window.EventSource;
  if (!Orig) return;
  window.EventSource = function (url) {
    const es = new Orig(url);
    es.addEventListener('audio_chunk', (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if (d && d.pcm_b64) {
          if (window._lyceumPCM.length === 0)
            window._lyceumStartedAt = performance.now();
          window._lyceumPCM.push(d.pcm_b64);
        }
      } catch (_) {}
    });
    es.addEventListener('done', () => {
      window._lyceumDone = true;
    });
    return es;
  };
  // Preserve constants on the wrapper so feature checks still pass.
  for (const k of ['CONNECTING', 'OPEN', 'CLOSED']) {
    window.EventSource[k] = Orig[k];
  }
})();
'''


# ---------------------------------------------------------------------------
# main capture
# ---------------------------------------------------------------------------

def record(book: str, chapter_substr: str, *,
           max_seconds: float, width: int, height: int,
           out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / '.video_tmp'
    # Wipe stale .webm / .wav from previous runs.  Playwright names
    # videos by random hex; if an older run's file outlives the run,
    # ``webms[-1]`` could pick the stale file and we'd mux yesterday's
    # video with today's audio.
    if work_dir.exists():
        for old in work_dir.iterdir():
            try:
                old.unlink()
            except Exception:
                pass
    work_dir.mkdir(parents=True, exist_ok=True)

    # Make sure the requested book is the active one.
    print(f'[setup] switching active book → {book}')
    http_post('/api/active_book', {'name': book})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=['--mute-audio'],   # don't try to render audio in headless;
                                     # we capture PCM via SSE.
        )
        ctx = browser.new_context(
            viewport={'width': width, 'height': height},
            record_video_dir=str(work_dir),
            record_video_size={'width': width, 'height': height},
        )
        ctx.add_init_script(INIT_SCRIPT)
        page = ctx.new_page()
        try:
            print(f'[load] {BASE_URL}')
            page.goto(BASE_URL, wait_until='domcontentloaded', timeout=20000)
            try:
                page.wait_for_load_state('networkidle', timeout=4000)
            except Exception:
                pass
            page.wait_for_timeout(1500)

            # Expand the book in the TOC.
            page.locator('.toc-node').first.click()
            page.wait_for_timeout(500)

            # Find the chapter row whose label contains chapter_substr.
            target = page.locator(
                f'.toc-chapter:has-text("{chapter_substr}")'
            )
            if target.count() == 0:
                raise RuntimeError(
                    f'no TOC chapter matched substring {chapter_substr!r}'
                )
            target.first.scroll_into_view_if_needed()
            print(f'[click] {chapter_substr}')
            target.first.click()

            # Wait until either the SSE 'done' event lands or max_seconds.
            t0 = time.time()
            tick = 0
            while time.time() - t0 < max_seconds:
                page.wait_for_timeout(2000)
                tick += 2
                got = page.evaluate(
                    '() => ({n: window._lyceumPCM.length, '
                    'done: window._lyceumDone, '
                    'started: window._lyceumStartedAt})'
                )
                if tick % 10 == 0:
                    print(f'  t={tick}s chunks={got["n"]} done={got["done"]}')
                if got['done']:
                    print(f'[done] SSE done event after {tick}s, '
                          f'{got["n"]} audio chunks')
                    break
            else:
                print(f'[cap] hit max_seconds={max_seconds}s before done')

            # Drain the PCM buffer + read warmup timestamp.
            print('[pull] reading audio buffer from page')
            state = page.evaluate(
                '() => ({pcm: window._lyceumPCM, '
                'startedAt: window._lyceumStartedAt})'
            )
            chunks_b64 = state['pcm']
            warmup_ms = float(state['startedAt'] or 0.0)
        finally:
            print('[close] finishing video file')
            page.close()
            ctx.close()        # this is what flushes the .webm
            browser.close()

    # Concatenate audio.
    pcm_bytes = b''.join(base64.b64decode(c) for c in chunks_b64)
    audio_seconds = len(pcm_bytes) / 2 / SAMPLE_RATE
    print(f'[audio] {len(chunks_b64)} chunks, '
          f'{len(pcm_bytes)} bytes, ≈{audio_seconds:.1f}s @ {SAMPLE_RATE}Hz')

    wav_path = work_dir / 'audio.wav'
    write_wav(wav_path, pcm_bytes, SAMPLE_RATE)

    # Find the .webm Playwright dropped.
    webms = sorted(work_dir.glob('*.webm'))
    if not webms:
        raise RuntimeError(f'no .webm video produced in {work_dir}')
    webm = webms[-1]
    print(f'[video] {webm.name} ({webm.stat().st_size/1e6:.1f} MB)')

    # Front-trim the video: Kokoro can synthesize audio faster than
    # realtime, so PCM chunks pile up ahead of browser playback.  The
    # browser starts playing audio when the first chunk arrives, at
    # wall-clock ``warmup_s``; visual ops on the recording are timed
    # to that playback, so the captured PCM[0..N] aligns with video
    # frames at wall-clock [warmup_s..warmup_s+N/sr].  Strip the silent
    # warmup off the video front so audio time 0 = video time 0.
    warmup_s = max(0.0, warmup_ms / 1000.0)
    print(f'[trim] warmup detected = {warmup_s:.2f}s; '
          f'cutting front of video to align audio time 0')

    # Mux.  ``-shortest`` then bounds the output to the shorter of
    # (trimmed video, captured audio); the audio buffer typically
    # outruns playback by a few seconds because Kokoro synthesises
    # ahead, so the audio gets the trim.
    print(f'[mux] → {out_path}')
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{warmup_s:.3f}', '-i', str(webm),
        '-i', str(wav_path),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print('--- ffmpeg stderr ---')
        print(proc.stderr[-2000:])
        raise RuntimeError(f'ffmpeg failed: rc={proc.returncode}')

    size_mb = out_path.stat().st_size / 1e6
    print(f'[ok] {out_path}  ({size_mb:.1f} MB, '
          f'audio≈{audio_seconds:.1f}s, warmup-trimmed)')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', required=True)
    ap.add_argument('--chapter', required=True,
                    help='substring matched against TOC row text')
    ap.add_argument('--max-seconds', type=float, default=180.0)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    record(
        book=args.book,
        chapter_substr=args.chapter,
        max_seconds=args.max_seconds,
        width=args.width,
        height=args.height,
        out_path=args.out,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
