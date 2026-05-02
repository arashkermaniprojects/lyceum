"""Local automatic speech recognition for the voice-tutor loop.

Wraps ``faster_whisper`` with a process-level singleton so the model
loads once per server lifetime, not once per request.  Default model
is ``tiny.en`` (CPU, ~75 MB, ~1s for a 5s utterance) — small enough
to keep the loop snappy on a developer laptop while still robust on
ML-jargon vocabulary.

Local-only: never calls Anthropic, OpenAI, or any cloud STT.

Honors the ``SEVIM_DISABLE_ASR=1`` env var so unit tests stay
deterministic without loading the model.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional


ASR_MODEL_NAME = os.environ.get("SEVIM_ASR_MODEL", "tiny.en")
ASR_DEVICE = os.environ.get("SEVIM_ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.environ.get("SEVIM_ASR_COMPUTE", "int8")
ASR_MAX_DURATION = float(os.environ.get("SEVIM_ASR_MAX_DURATION", "30"))


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    text: str
    duration: float       # input audio length, seconds
    language: str
    elapsed: float        # wall-clock time spent transcribing


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model = None  # WhisperModel | None
_model_load_error: Optional[str] = None


def is_disabled() -> bool:
    return os.environ.get("SEVIM_DISABLE_ASR") == "1"


def _get_model():
    """Lazy-load the Whisper model.  Cached for the process lifetime."""
    global _model, _model_load_error
    if _model is not None or _model_load_error is not None:
        return _model
    with _model_lock:
        if _model is not None or _model_load_error is not None:
            return _model
        if is_disabled():
            _model_load_error = "ASR disabled via SEVIM_DISABLE_ASR=1"
            return None
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            _model_load_error = (
                f"faster_whisper not installed: {e}.  "
                "Run: uv pip install faster-whisper"
            )
            return None
        try:
            _model = WhisperModel(
                ASR_MODEL_NAME,
                device=ASR_DEVICE,
                compute_type=ASR_COMPUTE_TYPE,
            )
        except Exception as e:
            _model_load_error = f"Whisper load failed: {e}"
            return None
        return _model


def model_status() -> dict:
    """Return ``{available: bool, name: str, error: str}`` for /status."""
    if is_disabled():
        return {"available": False, "name": "", "error": "disabled"}
    if _model is not None:
        return {"available": True, "name": ASR_MODEL_NAME, "error": ""}
    if _model_load_error:
        return {"available": False, "name": "", "error": _model_load_error}
    return {"available": True, "name": ASR_MODEL_NAME,
            "error": "not yet loaded"}


# ---------------------------------------------------------------------------
# Audio handling — accept anything ffmpeg can decode
# ---------------------------------------------------------------------------

class AudioDecodeError(RuntimeError):
    pass


def _decode_to_wav_pcm16(blob: bytes) -> tuple[bytes, float]:
    """Decode an arbitrary audio blob (webm/opus, mp4, mp3, wav…) to
    16 kHz mono PCM16 WAV bytes via ffmpeg.

    Returns ``(wav_bytes, duration_seconds)``.  Raises ``AudioDecodeError``
    when ffmpeg is missing or the input is undecodable.
    """
    if not blob:
        raise AudioDecodeError("empty audio body")
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("ffmpeg not installed")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0",
             "-ac", "1", "-ar", "16000",
             "-f", "wav", "pipe:1"],
            input=blob, capture_output=True, timeout=15.0,
        )
    except subprocess.TimeoutExpired:
        raise AudioDecodeError("ffmpeg timed out")
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace")[:200]
        raise AudioDecodeError(f"ffmpeg failed: {msg}")
    wav = proc.stdout
    if not wav:
        raise AudioDecodeError("ffmpeg produced empty output")
    # Compute duration from the actual data size, not the WAV header —
    # ffmpeg writes ``nframes = INT32_MAX`` when its output is a pipe
    # because the size field can't be back-patched.  We derive it from
    # ``rate * channels * sampwidth`` and the byte count of the data
    # chunk.  Falls back to 0.0 if we can't parse the header.
    duration = 0.0
    try:
        with wave.open(io.BytesIO(wav), "rb") as wf:
            rate = wf.getframerate() or 1
            channels = wf.getnchannels() or 1
            sampwidth = wf.getsampwidth() or 1
        # WAV header is 44 bytes for the standard PCM layout we asked
        # ffmpeg to produce (RIFF/WAVE/fmt/data chunks).
        data_bytes = max(0, len(wav) - 44)
        duration = data_bytes / float(rate * channels * sampwidth)
    except Exception:
        duration = 0.0
    return wav, duration


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def transcribe(blob: bytes) -> TranscriptionResult:
    """Decode and transcribe *blob* (any common browser audio format).

    Raises ``RuntimeError`` when ASR is unavailable, ``AudioDecodeError``
    when the audio can't be decoded, ``ValueError`` for input that
    exceeds ``ASR_MAX_DURATION``.
    """
    model = _get_model()
    if model is None:
        raise RuntimeError(_model_load_error or "ASR unavailable")
    wav_bytes, duration = _decode_to_wav_pcm16(blob)
    if duration > ASR_MAX_DURATION:
        raise ValueError(
            f"audio too long: {duration:.1f}s > {ASR_MAX_DURATION}s cap"
        )
    started = time.monotonic()
    # faster-whisper accepts a file path or a NumPy float32 array.
    # Cheapest path: write to a temp file (in-memory tempfile would
    # require a NumPy decode of the WAV, which we'd rather avoid).
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        segments, info = model.transcribe(
            tmp.name,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    elapsed = time.monotonic() - started
    language = getattr(info, "language", "") or ""
    return TranscriptionResult(
        text=text, duration=duration,
        language=language, elapsed=elapsed,
    )
