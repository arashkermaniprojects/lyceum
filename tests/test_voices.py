"""Pin Kokoro voice listing — degrades gracefully when the file is
missing or unreadable.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from narrator.tts import list_kokoro_voices


def test_list_voices_returns_empty_for_missing_file():
    assert list_kokoro_voices("/this/path/does/not/exist.bin") == []


def test_list_voices_reads_npz_keys(tmp_path):
    """The voices.bin format is a numpy npz; key list IS the voice
    catalog."""
    target = tmp_path / "voices.bin"
    # np.savez auto-appends .npz; write explicitly so ``target`` is
    # the actual file the loader sees.
    with open(target, "wb") as f:
        np.savez(
            f,
            af_alloy=np.zeros(10, dtype=np.float32),
            af_heart=np.zeros(10, dtype=np.float32),
            bf_isabella=np.zeros(10, dtype=np.float32),
        )
    voices = list_kokoro_voices(str(target))
    assert voices == ["af_alloy", "af_heart", "bf_isabella"]


def test_list_voices_handles_corrupt_file(tmp_path):
    """Garbage in the file → empty list, no exception."""
    p = tmp_path / "broken.bin"
    p.write_bytes(b"not a valid npz file")
    assert list_kokoro_voices(str(p)) == []
