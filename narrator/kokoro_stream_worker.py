#!/usr/bin/env python3
"""Long-lived Kokoro streaming worker.

Reads one JSON request per line on stdin, writes MULTIPLE responses
per request: one ``chunk`` line for each audio chunk produced by
``Kokoro.create_stream`` (an async generator), then one ``final``
line marking the end.  This lets the orchestrator forward chunks to
the browser as they're produced, cutting first-audio latency from
~half-a-second to ~tens of milliseconds for long sentences.

Request:  {"id": str, "text": str, "voice": str?, "speed": float?, "lang": str?}
Chunk:    {"id": str, "kind": "chunk", "idx": int,
           "pcm_b64": str, "rate": int}
Final:    {"id": str, "kind": "final", "rate": int, "duration": float, "n_chunks": int}
Error:    {"id": str, "error": str}

PCM is 16-bit signed little-endian mono at the model's native rate
(usually 24 kHz).  The browser stitches chunks back together with
Web Audio API's AudioContext.decodeAudioData / direct buffer copy.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import traceback


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _samples_to_pcm16(samples) -> bytes:
    """Float32 [-1, 1] → 16-bit signed little-endian PCM bytes."""
    import numpy as np
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    return pcm.tobytes()


def _split_phrases(text: str, *, min_words: int = 5) -> list[str]:
    """Split *text* at strong punctuation while keeping tokens of at
    least ``min_words``.  Used to coax sub-sentence chunking out of
    a TTS engine that otherwise buffers a whole sentence internally.

    The split is non-destructive — concatenating the result with
    spaces reproduces the original input.
    """
    import re
    out: list[str] = []
    parts = re.split(r"([.!?]\s+|[,;]\s+)", text)
    cur = ""
    for piece in parts:
        if not piece:
            continue
        cur += piece
        if cur.endswith((". ", "! ", "? ", ", ", "; ")):
            words = cur.split()
            if len(words) >= min_words:
                out.append(cur.strip())
                cur = ""
    tail = cur.strip()
    if tail:
        if out and len(tail.split()) < 2:
            out[-1] = (out[-1] + " " + tail).strip()
        else:
            out.append(tail)
    return out or [text]


async def _serve_one(k, req: dict) -> None:
    req_id = req.get("id")
    text = (req.get("text") or "").strip()
    if not text:
        _emit({"id": req_id, "error": "empty text"})
        return
    voice = req.get("voice", os.environ.get("KOKORO_VOICE", "af_heart"))
    speed = float(req.get("speed", 1.0))
    lang = req.get("lang", os.environ.get("KOKORO_LANG", "en-us"))
    rate_seen = 24000
    n_chunks = 0
    total_samples = 0
    try:
        # kokoro_onnx's ``create_stream`` buffers a whole sentence
        # before yielding, so to get *real* per-phrase streaming we
        # split the input into shorter phrases and synthesise each
        # as its own chunk.  The browser then plays the first phrase
        # while later ones are still being computed.
        pieces = _split_phrases(text)
        for piece in pieces:
            samples, rate = k.create(
                piece, voice=voice, speed=speed, lang=lang,
            )
            rate_seen = int(rate)
            pcm = _samples_to_pcm16(samples)
            total_samples += len(samples)
            _emit({
                "id": req_id, "kind": "chunk", "idx": n_chunks,
                "pcm_b64": base64.b64encode(pcm).decode("ascii"),
                "rate": rate_seen, "text": piece,
            })
            n_chunks += 1
        duration = total_samples / float(max(rate_seen, 1))
        _emit({
            "id": req_id, "kind": "final",
            "rate": rate_seen, "duration": duration,
            "n_chunks": n_chunks,
        })
    except Exception as e:
        _log("[kokoro_stream] error: " + traceback.format_exc())
        _emit({"id": req_id, "error": str(e)})


async def _amain() -> None:
    model_path = os.environ.get(
        "KOKORO_MODEL", ".kokoro-models/kokoro-v1.0.onnx",
    )
    voices_path = os.environ.get(
        "KOKORO_VOICES", ".kokoro-models/voices-v1.0.bin",
    )
    _log(f"[kokoro_stream] loading model={model_path} voices={voices_path}")
    from kokoro_onnx import Kokoro
    k = Kokoro(model_path, voices_path)
    _log("[kokoro_stream] ready")
    _emit({"event": "ready"})

    loop = asyncio.get_event_loop()
    # Read stdin in a thread because asyncio.run_in_executor is the
    # canonical way to bridge blocking IO into the event loop.
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _emit({"id": None, "error": f"bad json: {e}"})
            continue
        await _serve_one(k, req)


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
