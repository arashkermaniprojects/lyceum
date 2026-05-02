"""Word-timestamp utilities for narration.

The orchestrator needs to know *when* each word is spoken so it can fire
visual operations at the right moment.  Real phoneme alignment is the
right answer, but most TTS engines (including Kokoro at the time of
writing) don't expose phoneme timestamps.  This module provides:

  * **Linear interpolation** — split the clause into whitespace-separated
    words; each word's duration is proportional to its character length.
    Approximate but stable; sufficient for "show the matrix shape when the
    word 'matrix' is spoken".

  * **Concept-mention scheduling** — given (clause, concept_offsets,
    audio_duration), return the time each concept mention is spoken.

Both functions are pure / deterministic.
"""
from __future__ import annotations

import re


# Word-tokeniser: keep punctuation attached to its word so timing aligns
# with what the TTS will actually pronounce.
_WORD_RE = re.compile(r"\S+")


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, word), …] over *text*."""
    return [(m.start(), m.end(), m.group(0)) for m in _WORD_RE.finditer(text)]


def linear_word_timestamps(
    text: str, duration: float,
) -> list[tuple[str, float, float]]:
    """Return ``[(word, t_start, t_end), …]`` for *text* spread across *duration*.

    Each word's duration is proportional to its character length (so longer
    words get more time).  The whole list always sums to ``duration``.
    """
    if duration <= 0 or not text or not text.strip():
        return []
    spans = _word_spans(text)
    if not spans:
        return []
    total_chars = sum(end - start for start, end, _ in spans)
    if total_chars == 0:
        return []
    out: list[tuple[str, float, float]] = []
    cursor = 0.0
    for start, end, word in spans:
        share = (end - start) / total_chars
        word_dur = share * duration
        t_start = cursor
        t_end = cursor + word_dur
        out.append((word, t_start, t_end))
        cursor = t_end
    # Fix possible float drift on the last word so the total matches dur.
    if out:
        last_word, last_start, _last_end = out[-1]
        out[-1] = (last_word, last_start, duration)
    return out


def concept_event_times(
    text: str,
    concept_offsets: list[tuple[str, int]],
    word_timestamps: list[tuple[str, float, float]],
) -> list[tuple[str, float]]:
    """Return ``[(concept_id, t_seconds), …]`` for each concept mention.

    Maps each char offset to the word containing that offset and emits the
    word's start time.  When multiple concept mentions hit the same word,
    each gets the same timestamp (downstream code can stagger by ε if
    needed).
    """
    if not concept_offsets or not word_timestamps:
        return []
    spans = _word_spans(text)
    if not spans:
        return []
    # Pair each span (by index in spans list) with its (start_t, end_t)
    # from word_timestamps.  Lengths should match by construction.
    n = min(len(spans), len(word_timestamps))
    out: list[tuple[str, float]] = []
    for cid, offset in concept_offsets:
        # Find the word span containing this offset.
        word_idx = -1
        for i in range(n):
            ws, we, _w = spans[i]
            if ws <= offset < we:
                word_idx = i
                break
            if offset < ws:  # offset fell on whitespace; assign to next word
                word_idx = i
                break
        if word_idx < 0:
            word_idx = n - 1   # past end → assign to final word
        t_start = word_timestamps[word_idx][1]
        out.append((cid, t_start))
    out.sort(key=lambda x: x[1])
    return out
