"""Narrator package — bridges book corpus to SeVim's visual pipeline.

Public API:

    from narrator import resolve, render_resolved, ResolvedShape

    book = book.load_corpus("textbook.json")
    rs = resolve(book, "matrix", current_nid="b/ch8/s8.4")
    svg = render_resolved(rs)

Citation
--------
If you use this package in your research, please cite the Lyceum
paper.  See ``CITATION.cff`` and ``NOTICE`` at the repository root.
"""
from .resolver import (
    ResolvedShape, resolve, render_resolved,
)
from .planner import (
    NarrationClause, NarrationPlan, plan, plan_full, plan_outline,
    plan_chapter_zoom,
)
from .tts import (
    AudioClip, NullTTS, KokoroTTS, auto_tts,
)
from .timing import (
    linear_word_timestamps, concept_event_times,
)
from .qa import (
    answer, set_synth_backend, get_synth_backend,
    set_intro_backend, get_intro_backend,
    make_qwen_backend, make_vllm_backend, make_vllm_tutor_backend,
    make_vllm_intro_backend,
    auto_qwen_backend, is_low_similarity,
    RetrievedPassage,
)

__all__ = [
    "ResolvedShape", "resolve", "render_resolved",
    "NarrationClause", "NarrationPlan", "plan", "plan_full", "plan_outline",
    "plan_chapter_zoom",
    "AudioClip", "NullTTS", "KokoroTTS", "auto_tts",
    "linear_word_timestamps", "concept_event_times",
    "answer", "set_synth_backend", "get_synth_backend",
    "set_intro_backend", "get_intro_backend",
    "make_qwen_backend", "make_vllm_backend", "make_vllm_tutor_backend",
    "make_vllm_intro_backend",
    "auto_qwen_backend", "is_low_similarity",
    "RetrievedPassage",
]
