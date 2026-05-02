"""Pin the streaming-TTS SSE event flow.

The orchestrator must:
  * yield a StreamEvent (streaming=True, audio_b64="") for every clause
  * follow it with one AudioChunkEvent per phrase
  * close with an AudioCompleteEvent

…when the active TTS is :class:`KokoroStreamTTS`.  When the TTS is
plain :class:`NullTTS` / :class:`KokoroTTS`, the orchestrator yields
exactly one StreamEvent per clause as before.
"""
from __future__ import annotations

from book.ir import Book, BookNode
from chalkboard import Chalkboard
from narrator import NarrationClause, NarrationPlan
from narrator.tts import (
    AudioChunk, AudioClip, NullTTS, _pcm16_to_wav,
)

from serve.orchestrator import (
    AudioChunkEvent, AudioCompleteEvent, Orchestrator, StreamEvent,
)


def _book() -> Book:
    root = BookNode(nid="b", kind="book", number=None, title="t",
                    page_start=1, page_end=999)
    return Book(title="t", author=None, source="", root=root,
                concepts={}, pages=[], figures=[], cross_refs=[])


def _plan(*texts) -> NarrationPlan:
    return NarrationPlan(
        topic="<test>", book_title="t",
        clauses=[NarrationClause(text=t, home_nid="b",
                                  concepts=[], suggested_dur=1.0)
                 for t in texts],
        visited_nids=["b"], meta={"mode": "full"},
    )


# ---------------------------------------------------------------------------
# Non-streaming backend → one StreamEvent per clause
# ---------------------------------------------------------------------------

def test_orchestrator_yields_single_stream_event_for_null_tts():
    orch = Orchestrator(book=_book(), plan=_plan("first.", "second."),
                        chalkboard=Chalkboard(), tts=NullTTS())
    events = list(orch.stream())
    assert all(isinstance(e, StreamEvent) for e in events)
    assert [e.streaming for e in events] == [False, False]
    assert all(e.audio_b64 != "" for e in events)


# ---------------------------------------------------------------------------
# Fake KokoroStreamTTS → chunks interleaved
# ---------------------------------------------------------------------------

class _FakeStreamTTS:
    """Mimics KokoroStreamTTS protocol without a real model."""
    name = "kokoro_stream"

    def synthesize_stream(self, text, *, voice=None, speed=1.0):
        # Pretend "text" splits into 2 phrases of equal size.
        n = max(1, len(text) // 2)
        for i in range(2):
            pcm = b"\x00\x10" * 50  # silent-ish
            yield AudioChunk(
                text=text[i * n:(i + 1) * n], pcm16=pcm, rate=24000,
                chunk_idx=i, voice=voice or "af_heart", is_final=False,
            )
        yield AudioChunk(
            text="", pcm16=b"", rate=24000, chunk_idx=2,
            voice=voice or "af_heart", is_final=True, duration=0.20,
        )

    def synthesize(self, text, *, voice=None, speed=1.0):
        clip = AudioClip(
            text=text, wav_bytes=_pcm16_to_wav(b"\x00\x00" * 50, 24000),
            duration=0.05, rate=24000, word_timestamps=[],
            voice=voice or "af_heart", backend=self.name,
        )
        return clip


def test_orchestrator_emits_chunks_for_streaming_tts(monkeypatch):
    """The orchestrator should detect the streaming backend and weave
    AudioChunk / AudioComplete events between StreamEvents."""
    # Patch the isinstance check to accept our fake.
    from serve import orchestrator as orch_mod
    from narrator import tts as tts_mod
    monkeypatch.setattr(
        tts_mod, "KokoroStreamTTS", _FakeStreamTTS,
    )
    orch = Orchestrator(book=_book(), plan=_plan("first.", "second."),
                        chalkboard=Chalkboard(), tts=_FakeStreamTTS())
    events = list(orch.stream())
    # For each clause: 1 StreamEvent + 2 AudioChunk + 1 AudioComplete.
    types = [type(e).__name__ for e in events]
    assert types == [
        "StreamEvent", "AudioChunkEvent", "AudioChunkEvent",
        "AudioCompleteEvent",
        "StreamEvent", "AudioChunkEvent", "AudioChunkEvent",
        "AudioCompleteEvent",
    ]
    # First clause's StreamEvent flagged as streaming, audio_b64 empty.
    se = events[0]
    assert isinstance(se, StreamEvent)
    assert se.streaming is True
    assert se.audio_b64 == ""
    # AudioChunk seq matches its clause; idx is 0-based.
    chunks = [e for e in events if isinstance(e, AudioChunkEvent)]
    assert chunks[0].seq == 0 and chunks[0].chunk_idx == 0
    assert chunks[1].seq == 0 and chunks[1].chunk_idx == 1
    assert chunks[2].seq == 1
    # AudioComplete carries final duration and chunk count.
    completes = [e for e in events if isinstance(e, AudioCompleteEvent)]
    assert len(completes) == 2
    assert completes[0].seq == 0
    assert completes[0].n_chunks == 2
