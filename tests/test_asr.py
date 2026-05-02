"""Pin the ASR module: model status, decode failure paths, disabled
mode.  We don't load the real Whisper model in unit tests — that
takes ~5s on cold cache and pulls in network at first run.  Tests
exercise everything *around* the model load.
"""
from __future__ import annotations

import os

import pytest

from serve import asr


@pytest.fixture(autouse=True)
def _disable_asr(monkeypatch):
    """Force ASR-disabled so tests never load the real model."""
    monkeypatch.setenv("SEVIM_DISABLE_ASR", "1")
    # Reset the singleton between tests in case another test loaded it.
    asr._model = None
    asr._model_load_error = None
    yield
    asr._model = None
    asr._model_load_error = None


def test_is_disabled_honors_env():
    assert asr.is_disabled() is True


def test_get_model_returns_none_when_disabled():
    assert asr._get_model() is None
    # Subsequent calls keep returning None without re-trying.
    assert asr._get_model() is None


def test_model_status_when_disabled():
    status = asr.model_status()
    assert status["available"] is False
    assert status["error"] == "disabled"


def test_transcribe_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="ASR"):
        asr.transcribe(b"\x00\x01\x02\x03")


def test_decode_to_wav_pcm16_rejects_empty():
    with pytest.raises(asr.AudioDecodeError, match="empty"):
        asr._decode_to_wav_pcm16(b"")


def test_decode_to_wav_pcm16_rejects_garbage(monkeypatch):
    """ffmpeg returns non-zero for non-audio input."""
    # Skip the test if ffmpeg isn't installed on this dev machine —
    # the import-time check already covers the no-ffmpeg path.
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    with pytest.raises(asr.AudioDecodeError):
        asr._decode_to_wav_pcm16(b"this is not audio data, just text")
