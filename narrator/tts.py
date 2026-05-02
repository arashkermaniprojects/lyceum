"""TTS adapter — Kokoro in production, silent fallback for tests.

Two backends, selected at construction time:

1.  **KokoroTTS** — wraps a long-lived ``kokoro_onnx`` subprocess.  Each
    request is one JSON line over stdin; each response is one JSON line on
    stdout.  Mirror's the existing TextVis-Game / TextVis-3D-Interactive
    pattern (``agents/kokoro-tts-server.py``).  Audio is returned as
    base64-encoded WAV bytes.

2.  **NullTTS** — emits silent WAV of the correct estimated duration with
    linearly-interpolated word timestamps.  Lets the rest of the pipeline
    run in CI / test environments without Kokoro models present.

Both backends produce :class:`AudioClip` with the same fields, so callers
don't need to branch.

The adapter is intentionally narrow: synthesise(text) → AudioClip.  No
batching, no streaming chunks (TTS produces a complete clause at a time;
streaming happens at the orchestration layer).
"""
from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import threading
import wave
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .timing import linear_word_timestamps


# ---------------------------------------------------------------------------
# AudioClip
# ---------------------------------------------------------------------------

@dataclass
class AudioClip:
    """One TTS-synthesised clause.

    Attributes
    ----------
    text:
        The spoken text (echoed back for traceability).
    wav_bytes:
        Complete WAV-encoded audio.  Self-contained — frontends can
        decode directly without out-of-band metadata.
    duration:
        Total audio duration in seconds (may differ slightly from the
        WAV header due to padding).
    rate:
        Sample rate (typically 24000 for Kokoro).
    word_timestamps:
        ``[(word, t_start_s, t_end_s), …]`` — one entry per whitespace-
        separated word.  Backends without phoneme alignment populate this
        via linear interpolation.
    voice:
        Voice identifier the backend used.
    backend:
        ``"kokoro"`` or ``"null"`` — useful for trace logs.
    """
    text: str
    wav_bytes: bytes
    duration: float
    rate: int = 24000
    word_timestamps: list[tuple[str, float, float]] = field(default_factory=list)
    voice: str = ""
    backend: str = ""


# ---------------------------------------------------------------------------
# Backend interface (duck-typed, no abstract base required)
# ---------------------------------------------------------------------------

class _BackendBase:
    name: str = ""

    def synthesize(self, text: str, *, voice: Optional[str] = None,
                   speed: float = 1.0) -> AudioClip:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# NullTTS — silent fallback
# ---------------------------------------------------------------------------

# Approximate Kokoro speaking rate at speed=1.0: ~12 chars/sec for clear
# English narration (slightly slower than the planner's 15 cps).
_NULL_CPS = 12.0
_NULL_RATE = 24000


def _silent_wav(duration: float, rate: int = _NULL_RATE) -> bytes:
    """Build a silent 16-bit mono WAV of the given duration."""
    n_samples = max(1, int(round(duration * rate)))
    buf = bytearray()
    # WAV uses little-endian; ``wave`` module writes a proper header.
    import io
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)   # 16-bit
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_samples)
    return bio.getvalue()


class NullTTS(_BackendBase):
    """Silent placeholder audio + linear word-timing.  Use in tests / CI.

    The wav_bytes carry only ~100 ms of silence regardless of the announced
    `duration`.  The orchestrator's clients are expected to use
    ``audio_dur`` for actual clause-pacing — playing 30 seconds of zeroes
    over SSE is wasteful (≈23 MB per clause base64'd) and chokes the
    browser's EventSource buffer.
    """
    name = "null"

    # Real duration of the placeholder WAV.  Long enough that a play() call
    # actually fires `play` (some browsers refuse near-zero-length audio),
    # short enough that the bytes fit in a single SSE event.
    _PLACEHOLDER_DUR = 0.1

    def __init__(self, cps: float = _NULL_CPS, rate: int = _NULL_RATE) -> None:
        self.cps = cps
        self.rate = rate

    def synthesize(self, text: str, *, voice: Optional[str] = None,
                   speed: float = 1.0) -> AudioClip:
        if not text or not text.strip():
            wav = _silent_wav(self._PLACEHOLDER_DUR, self.rate)
            return AudioClip(text=text, wav_bytes=wav,
                             duration=self._PLACEHOLDER_DUR,
                             rate=self.rate, word_timestamps=[],
                             voice=voice or "null", backend=self.name)
        chars = max(1, len(text))
        announced_dur = chars / max(self.cps * speed, 1.0)
        # Always emit only a tiny silent WAV — keeps SSE payloads tractable.
        wav = _silent_wav(self._PLACEHOLDER_DUR, self.rate)
        # Word timestamps still describe the FULL announced duration so
        # downstream timing of visual ops + word highlighting stays correct.
        words = linear_word_timestamps(text, announced_dur)
        return AudioClip(
            text=text, wav_bytes=wav, duration=announced_dur, rate=self.rate,
            word_timestamps=words, voice=voice or "null",
            backend=self.name,
        )


# ---------------------------------------------------------------------------
# KokoroTTS — long-lived subprocess
# ---------------------------------------------------------------------------

# Default paths: pull the same models that TextVis-Game uses.  Override via
# env vars if your install lives elsewhere.
_DEFAULT_MODEL = os.environ.get(
    "KOKORO_MODEL",
    "/home/ara/Documents/Programming/agentic_systems/TextVis 3D/.kokoro-models/kokoro-v1.0.onnx",
)
_DEFAULT_VOICES = os.environ.get(
    "KOKORO_VOICES",
    "/home/ara/Documents/Programming/agentic_systems/TextVis 3D/.kokoro-models/voices-v1.0.bin",
)
_DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")


def list_kokoro_voices(voices_path: str = _DEFAULT_VOICES) -> list[str]:
    """Return every voice id available in the voices.bin file.

    Returns ``[]`` when the file is missing or unreadable so the
    caller can degrade gracefully.  The file is an ``np.savez``
    archive whose key list IS the voice catalog.
    """
    if not voices_path or not os.path.isfile(voices_path):
        return []
    try:
        import numpy as _np
        with _np.load(voices_path, allow_pickle=True) as data:
            return sorted(data.files)
    except Exception as e:
        print(f"[narrator.tts] failed to read voices from {voices_path}: {e}")
        return []


class KokoroTTS(_BackendBase):
    """Long-lived stdin/stdout Kokoro worker.

    Spawns a Python subprocess running ``kokoro-tts-server.py`` (the same
    script used by TextVis-Game).  One request per line; one response per
    line.
    """
    name = "kokoro"

    def __init__(
        self,
        *,
        worker_script: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        voices: str = _DEFAULT_VOICES,
        default_voice: str = _DEFAULT_VOICE,
    ) -> None:
        self.default_voice = default_voice
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._worker_script = worker_script or self._find_worker_script()
        self._spawn(model, voices)

    @staticmethod
    def _find_worker_script() -> str:
        candidates = [
            os.path.join(os.path.dirname(__file__), "kokoro_worker.py"),
            "/home/ara/Documents/Programming/agentic_systems/TextVis-Game/agents/kokoro-tts-server.py",
            "/home/ara/Documents/Programming/agentic_systems/TextVis 3D/agents/kokoro-tts-server.py",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise RuntimeError(
            "Kokoro worker script not found — set worker_script= or "
            "drop kokoro-tts-server.py next to narrator/tts.py."
        )

    def _spawn(self, model: str, voices: str) -> None:
        env = dict(os.environ)
        env["KOKORO_MODEL"] = model
        env["KOKORO_VOICES"] = voices
        # Use the project venv's Python so kokoro_onnx is found.
        py = os.environ.get(
            "SEVIM_PYTHON",
            "/home/ara/Documents/Programming/sevim_math/.venv/bin/python3",
        )
        self._proc = subprocess.Popen(
            [py, self._worker_script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env, bufsize=1,
        )
        # Wait for the "ready" event line.
        ready = self._proc.stdout.readline()
        try:
            payload = json.loads(ready)
        except Exception as e:
            self._proc.kill()
            raise RuntimeError(
                f"kokoro worker did not emit JSON ready event; "
                f"first line was: {ready!r}"
            ) from e
        if payload.get("event") != "ready":
            raise RuntimeError(f"kokoro worker first message was: {payload}")

    def synthesize(self, text: str, *, voice: Optional[str] = None,
                   speed: float = 1.0) -> AudioClip:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Kokoro worker is not running")
        req = {
            "id": "1",
            "text": text,
            "voice": voice or self.default_voice,
            "speed": speed,
        }
        with self._lock:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Kokoro worker closed unexpectedly")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"Kokoro error: {resp['error']}")
        wav_bytes = base64.b64decode(resp["wav_b64"])
        rate = int(resp.get("rate", 24000))
        dur = float(resp.get("duration", len(wav_bytes) / max(rate * 2, 1)))
        words = linear_word_timestamps(text, dur)
        return AudioClip(
            text=text, wav_bytes=wav_bytes, duration=dur, rate=rate,
            word_timestamps=words, voice=req["voice"],
            backend=self.name,
        )

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                self._proc.kill()


# ---------------------------------------------------------------------------
# KokoroStreamTTS — long-lived chunked-output worker
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    """One audio chunk yielded by streaming TTS.

    ``pcm16`` is 16-bit signed little-endian mono PCM at ``rate`` Hz.
    The frontend's Web Audio API converts these to AudioBuffers and
    schedules them in sequence so playback starts as soon as the
    first chunk arrives — not after the full sentence is rendered.
    """
    text: str
    pcm16: bytes
    rate: int
    chunk_idx: int
    voice: str = ""
    is_final: bool = False        # True for the closing marker
    duration: float = 0.0         # accumulated duration up to here


class KokoroStreamTTS(_BackendBase):
    """Long-lived chunked-output Kokoro worker.

    Same stdin/stdout protocol as :class:`KokoroTTS`, but each
    request elicits N+1 response lines: N chunks plus one final
    marker.  ``synthesize_stream(text)`` is a generator yielding
    :class:`AudioChunk` instances as they arrive — the orchestrator
    forwards each as an SSE ``audio_chunk`` event so the browser
    plays the first chunk while later ones are still being computed.
    """
    name = "kokoro_stream"

    def __init__(
        self,
        *,
        worker_script: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        voices: str = _DEFAULT_VOICES,
        default_voice: str = _DEFAULT_VOICE,
    ) -> None:
        self.default_voice = default_voice
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._worker_script = worker_script or self._find_worker_script()
        self._spawn(model, voices)

    @staticmethod
    def _find_worker_script() -> str:
        # Prefer the in-repo streaming worker alongside this file.
        candidate = os.path.join(
            os.path.dirname(__file__), "kokoro_stream_worker.py",
        )
        if os.path.exists(candidate):
            return candidate
        raise RuntimeError(
            "kokoro_stream_worker.py not found next to narrator/tts.py"
        )

    def _spawn(self, model: str, voices: str) -> None:
        env = dict(os.environ)
        env["KOKORO_MODEL"] = model
        env["KOKORO_VOICES"] = voices
        py = os.environ.get(
            "SEVIM_PYTHON",
            "/home/ara/Documents/Programming/sevim_math/.venv/bin/python3",
        )
        self._proc = subprocess.Popen(
            [py, self._worker_script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env, bufsize=1,
        )
        # First line should be {"event": "ready"}.
        ready = self._proc.stdout.readline()
        try:
            payload = json.loads(ready)
        except Exception as e:
            self._proc.kill()
            raise RuntimeError(
                f"kokoro_stream_worker: no ready event; "
                f"first line was: {ready!r}"
            ) from e
        if payload.get("event") != "ready":
            raise RuntimeError(
                f"kokoro_stream_worker first message was: {payload}"
            )

    def synthesize_stream(
        self, text: str, *,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> "Iterator[AudioChunk]":
        """Yield one :class:`AudioChunk` per chunk, plus a closing
        ``is_final=True`` marker.  Holds the worker's lock for the
        duration of the request so concurrent callers serialise."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Kokoro stream worker is not running")
        req = {
            "id": "1",
            "text": text,
            "voice": voice or self.default_voice,
            "speed": speed,
        }
        with self._lock:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("kokoro_stream_worker closed")
                resp = json.loads(line)
                if "error" in resp:
                    raise RuntimeError(f"kokoro stream error: {resp['error']}")
                kind = resp.get("kind")
                if kind == "chunk":
                    pcm = base64.b64decode(resp["pcm_b64"])
                    # CRITICAL: use the worker's per-chunk ``text`` (the
                    # phrase that this chunk's PCM actually contains),
                    # NOT the function argument ``text`` which is the
                    # whole clause.  Word-level browser sync paces the
                    # yellow read-marker over chunk.text spread across
                    # chunk.dur — using the full clause for every chunk
                    # makes the marker race through every clause word
                    # during the first phrase's audio (5/25 = 20% of
                    # actual speaking time → "yellow goes too fast").
                    chunk_text = resp.get("text", text)
                    yield AudioChunk(
                        text=chunk_text, pcm16=pcm,
                        rate=int(resp.get("rate", 24000)),
                        chunk_idx=int(resp.get("idx", 0)),
                        voice=req["voice"],
                        is_final=False,
                    )
                elif kind == "final":
                    yield AudioChunk(
                        text=text, pcm16=b"",
                        rate=int(resp.get("rate", 24000)),
                        chunk_idx=int(resp.get("n_chunks", 0)),
                        voice=req["voice"],
                        is_final=True,
                        duration=float(resp.get("duration", 0.0)),
                    )
                    return
                else:
                    # Unknown line — log and keep reading.
                    print(f"[kokoro_stream] unknown response: {resp}",
                          file=__import__("sys").stderr)

    def synthesize(
        self, text: str, *, voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> AudioClip:
        """Compatibility shim — assemble all chunks into a single
        :class:`AudioClip` so callers that don't care about streaming
        keep working unchanged."""
        if not text or not text.strip():
            wav = _silent_wav(0.1, 24000)
            return AudioClip(text=text, wav_bytes=wav, duration=0.1,
                             rate=24000, word_timestamps=[],
                             voice=voice or self.default_voice,
                             backend=self.name)
        all_pcm = bytearray()
        rate = 24000
        duration = 0.0
        for chunk in self.synthesize_stream(text, voice=voice, speed=speed):
            if chunk.is_final:
                rate = chunk.rate
                duration = chunk.duration
                break
            rate = chunk.rate
            all_pcm.extend(chunk.pcm16)
        # Wrap PCM in a WAV header so the existing audio.src path works.
        wav_bytes = _pcm16_to_wav(bytes(all_pcm), rate)
        if duration <= 0.0:
            duration = len(all_pcm) / float(rate * 2)
        words = linear_word_timestamps(text, duration)
        return AudioClip(
            text=text, wav_bytes=wav_bytes, duration=duration,
            rate=rate, word_timestamps=words,
            voice=voice or self.default_voice,
            backend=self.name,
        )

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                self._proc.kill()


def _pcm16_to_wav(pcm: bytes, rate: int) -> bytes:
    """Wrap raw PCM16 mono bytes in a minimal RIFF/WAVE header."""
    import struct
    n = len(pcm)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + n)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", n)
        + pcm
    )


# ---------------------------------------------------------------------------
# Auto-detect convenience
# ---------------------------------------------------------------------------

def auto_tts(*, prefer: str = "kokoro") -> _BackendBase:
    """Return a usable TTS backend.

    Tries the preferred backend first; falls back to ``NullTTS`` if it
    can't be initialised (model files missing, kokoro_onnx not installed,
    worker script not found, etc.).
    """
    if prefer == "kokoro":
        try:
            return KokoroTTS()
        except Exception as e:
            import sys
            print(f"[narrator.tts] kokoro unavailable ({e}); using NullTTS",
                  file=sys.stderr)
    return NullTTS()
