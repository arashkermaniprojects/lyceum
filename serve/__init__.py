"""Serve package --- HTTP + Server-Sent-Events delivery of narration.

This package owns everything that runs *during* a live narration
session:

* :class:`Server` - the HTTP server with the narrate / stream / Q&A /
  voice / book-selection endpoints (see ``serve.server``).
* :class:`Orchestrator` - the per-session state machine that turns a
  ``NarrationPlan`` into an interleaved stream of audio chunks plus
  typed visual ops, locked to the audio clock.  See
  ``serve.orchestrator`` for the full per-clause pipeline.
* ``serve.session`` - the ``Session`` object that pairs a main
  orchestrator + chalkboard with an on-demand tangent orchestrator +
  chalkboard and shares session knowledge between the two.
* ``serve.persistence`` - resumable-session JSON snapshots.
* ``serve.asr`` - faster-whisper voice-input endpoint.
* ``serve.refcontent`` - reference-card content extraction (Figure /
  Equation / Algorithm / Theorem / Section).
* ``serve.figure_ondemand`` - on-demand PDF crop for figures the
  ingestion pipeline missed.
* ``serve.ingest_pipeline`` - the auto-pipeline phases that run when
  a fresh PDF is uploaded.

The runtime is fully local: every model call lands on the user's own
machine (vLLM endpoints + Kokoro TTS subprocess), and no external
URL is reachable from any code path under ``serve``.

Citation
--------
If you use this package in your research, please cite the Lyceum
paper.  See ``CITATION.cff`` and ``NOTICE`` at the repository root.
"""

from .orchestrator import Orchestrator, StreamEvent
from .server import Server, main

__all__ = ["Orchestrator", "StreamEvent", "Server", "main"]
