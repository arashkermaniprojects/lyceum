"""Pin the streaming-TTS phrase splitter and the worker JSON
protocol shape.  We don't actually load the Kokoro model in tests
(it pulls ONNX runtime + the voices file) — instead we exercise
the helpers and stub the worker process.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _load_worker_module():
    """Import narrator/kokoro_stream_worker.py as a module without
    invoking its main() — we just want the helper functions."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "narrator", "kokoro_stream_worker.py",
    )
    spec = importlib.util.spec_from_file_location(
        "kokoro_stream_worker", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Phrase splitter
# ---------------------------------------------------------------------------

def test_split_phrases_keeps_short_input_intact():
    mod = _load_worker_module()
    out = mod._split_phrases("Bagging averages predictions.")
    assert out == ["Bagging averages predictions."]


def test_split_phrases_on_periods():
    mod = _load_worker_module()
    text = ("Bagging averages predictions across many models. "
            "It reduces the prediction variance significantly. "
            "The trade-off is added computational cost.")
    out = mod._split_phrases(text)
    # Each sentence is ≥ 5 words → three chunks.
    assert len(out) == 3
    assert out[0].startswith("Bagging")
    assert out[1].startswith("It reduces")
    assert out[2].startswith("The trade-off")


def test_split_phrases_on_commas_after_min_words():
    mod = _load_worker_module()
    text = "We minimize the regularized loss function carefully, then we proceed to the proof."
    out = mod._split_phrases(text)
    # Should split at the comma since the lead-in has > 5 words.
    assert len(out) == 2


def test_split_phrases_skips_short_lead_in():
    """``Bagging,`` is too short to be its own chunk."""
    mod = _load_worker_module()
    text = "Bagging, or bootstrap aggregating, is a method that reduces variance."
    out = mod._split_phrases(text)
    # First two clauses are too short to break on; expect one chunk
    # OR two with the lead-ins glued.
    assert all(len(p.split()) >= 2 for p in out)


def test_split_phrases_returns_at_least_one():
    mod = _load_worker_module()
    out = mod._split_phrases("")
    assert out == [""]
    out2 = mod._split_phrases("hi")
    assert out2 == ["hi"]


# ---------------------------------------------------------------------------
# PCM conversion helper
# ---------------------------------------------------------------------------

def test_samples_to_pcm16_round_trip():
    mod = _load_worker_module()
    import numpy as np
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5],
                       dtype=np.float32)
    pcm = mod._samples_to_pcm16(samples)
    # 7 samples × 2 bytes = 14 bytes.
    assert len(pcm) == 14
    # Round-trip: extreme values clip to ±32767.
    decoded = np.frombuffer(pcm, dtype="<i2")
    assert decoded[0] == 0
    assert decoded[3] == 32767
    assert decoded[4] == -32767
    # Out-of-range values clip too.
    assert decoded[5] == 32767
    assert decoded[6] == -32767


# ---------------------------------------------------------------------------
# AudioChunk dataclass shape
# ---------------------------------------------------------------------------

def test_audio_chunk_dataclass_has_required_fields():
    """The orchestrator + frontend agree on this shape — pin it."""
    from narrator.tts import AudioChunk
    c = AudioChunk(
        text="hi", pcm16=b"\x00\x00", rate=24000, chunk_idx=0,
        voice="af_heart", is_final=False, duration=0.0,
    )
    for attr in ("text", "pcm16", "rate", "chunk_idx",
                 "voice", "is_final", "duration"):
        assert hasattr(c, attr)


def test_pcm16_to_wav_header_shape():
    """Wrapping PCM in WAV produces a 44-byte header + the data."""
    from narrator.tts import _pcm16_to_wav
    pcm = b"\x00\x00" * 100  # 100 silent samples
    wav = _pcm16_to_wav(pcm, 24000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert wav[36:40] == b"data"
    assert len(wav) == 44 + len(pcm)


def test_synthesize_stream_yields_per_chunk_text_not_full_clause():
    """``synthesize_stream`` must pass through the worker's per-chunk
    ``text`` (the phrase the chunk's PCM actually contains), not the
    function-argument ``text`` (the whole clause).

    Regression: an earlier version set ``AudioChunk(text=text, ...)``
    using the function arg, so every chunk advertised the full clause
    as its text.  The browser's word-level read-marker scheduler then
    laid out *all* clause words inside the first chunk's audio
    duration — yellow raced through 100% of words during the first
    phrase's 20% of the audio (5 words shown across 25 words of clause
    in 5/25 = 20% of speaking time → "the yellow indicator goes too
    fast").  Subsequent chunks then advanced the wordOffset past the
    DOM, so no further yellow movement happened — looking like the
    panel was stuck.
    """
    from narrator.tts import KokoroStreamTTS

    full_clause = (
        "Bagging averages predictions across many models. "
        "It reduces the prediction variance significantly."
    )

    class _StubProc:
        """Pretends to be the kokoro_stream_worker subprocess.

        Reads requests off stdin, replies with two ``chunk`` lines
        carrying *different* per-chunk ``text`` fields, then a
        ``final``.
        """
        def __init__(self):
            self._inbox = []
            self._outbox = [
                ('{"id":"1","kind":"chunk","idx":0,"pcm_b64":"AAA=",'
                 '"rate":24000,"text":"Bagging averages predictions '
                 'across many models."}'),
                ('{"id":"1","kind":"chunk","idx":1,"pcm_b64":"AAA=",'
                 '"rate":24000,"text":"It reduces the prediction '
                 'variance significantly."}'),
                ('{"id":"1","kind":"final","rate":24000,'
                 '"duration":3.5,"n_chunks":2}'),
            ]
            self.stdin = self
            self.stdout = self

        def write(self, s):
            self._inbox.append(s)
            return len(s)

        def flush(self):
            pass

        def readline(self):
            if self._outbox:
                return self._outbox.pop(0) + "\n"
            return ""

        def poll(self):
            return None

    stub = _StubProc()

    # Build a KokoroStreamTTS without spawning a real worker.
    tts = KokoroStreamTTS.__new__(KokoroStreamTTS)
    tts._proc = stub
    import threading
    tts._lock = threading.Lock()
    tts.default_voice = "af_heart"
    tts.name = "kokoro_stream"

    chunks = list(tts.synthesize_stream(full_clause))

    # Drop the final marker.
    body = [c for c in chunks if not c.is_final]
    assert len(body) == 2, "expected two body chunks"

    # CRITICAL: each chunk's text is the *phrase* the worker
    # synthesised, not the whole clause.
    assert body[0].text == "Bagging averages predictions across many models."
    assert body[1].text == "It reduces the prediction variance significantly."

    # No chunk should carry the full concatenated clause text — that's
    # the regression we're guarding against.
    for c in body:
        assert c.text != full_clause, (
            "chunk text must be the per-phrase synthesised text, not "
            "the whole clause — the browser's word-level sync paces "
            "yellow over chunk.text * chunk.dur, so a full-clause text "
            "in every chunk makes the marker race"
        )
