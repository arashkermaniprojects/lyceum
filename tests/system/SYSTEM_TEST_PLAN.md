# SeVim_math system test plan

**Goal**: a real-time, locally-served, GPT-style private math tutor with
live visualization.  Every aspect a curious student would exercise must
be covered.

Tests live in `tests/system/`.  They are tiered:

  * **A — deterministic structural** (no LLM, no real TTS).  Run on
    every commit.  Cover routing, retrieval, planner shapes, persistence,
    visualization invariants, robustness.
  * **B — live-LLM** (Qwen + Kokoro reachable).  Cover content quality
    that's hard to assert structurally (length adaptation, grounding,
    smoothness).  Marked `pytest.mark.live_llm`; skipped when vLLM /
    Kokoro aren't up.
  * **H — human judgment** (cannot be automated).  Listed for
    completeness; verified via manual playback before each release.

Every checkbox below maps to a named test.

---

## 1. System awareness about the book

  * `[A]` Loads the corpus, knows title / author / chapters / sections.
  * `[A]` Citation map exposes the chapter graph with > 0 edges.
  * `[A]` Concept index keys are queryable.
  * `[A]` Alias map expands queries (`RBF` ↔ `radial basis function`).

## 2. Conversation dispatch / routing

  * `[A]` `"what is this book about"` → book_overview.
  * `[A]` `"explain chapter 5"` → chapter_overview, target_nid resolved.
  * `[A]` `"section 5.8"` / `"§5.8"` → section_overview.
  * `[A]` `"tell me more"` → follow_up anchored on last focus.
  * `[A]` `"pause"` / `"stop"` / `"go on"` → control, no LLM call.
  * `[A]` `"what have we covered"` → recap.
  * `[A]` `"show me X again"` / `"repeat that"` → reshow.
  * `[A]` `"what do I need to know first"` → dependencies.
  * `[A]` `"see also"` → xref_explore.
  * `[A]` Plain question → topic_qa default.
  * `[A]` Citation shortcut bypasses BM25 for explicit numbers.

## 3. Multi-turn conversation flow

  * `[A]` Dialogue history grows monotonically up to cap.
  * `[A]` Focus pointer advances on content turns; not on control / follow-up.
  * `[A]` Follow-up after pause still anchors on the last content focus.
  * `[A]` Recap surfaces topics the user asked about.
  * `[A]` Forget clears knowledge state; next tangent treats everything as new.
  * `[A]` Long conversation (≥ 12 turns) stays bounded in memory + on disk.
  * `[A]` Switching topics doesn't drag old focus into new tangent.

## 4. Knowledge state

  * `[A]` SessionKnowledge sets shared across orchestrators.
  * `[A]` Re-ask after coverage dedupes formulas / canonical figures.
  * `[A]` Reshow gives the tangent fresh seen_* (doesn't pollute long-term).
  * `[A]` Forget control flushes everything.

## 5. Visualization quality

  * `[A]` ReadingOrderPolicy never produces overlapping shapes.
  * `[A]` Formula cards size to content (no big-empty-frame regression).
  * `[A]` Equation citations land on the same formula card.
  * `[A]` Variable definitions ("where x is …") land on the same card.
  * `[A]` Reference cards: clean LaTeX → KaTeX path; garbled → text fallback.
  * `[A]` Caps drop overflow ops AND erase from chalkboard cleanly.
  * `[A]` Tier-3 semantic cards size tightly for single equations.
  * `[A]` Empty / vector-only semantic graphs get filtered out.
  * `[H]` KaTeX renders math correctly visually.
  * `[H]` Animations look smooth.

## 6. Citation-graph navigation

  * `[A]` Outgoing/incoming/most-cited helpers are correct.
  * `[A]` Intra-subtree refs filtered out by default.
  * `[A]` Citation path BFS finds multi-hop paths via descendants.
  * `[A]` xref_explore plan covers both directions.
  * `[A]` Dependencies plan includes citation-graph + concept-graph signals.

## 7. Robustness / error paths

  * `[A]` Empty / whitespace question doesn't crash.
  * `[A]` Nonsense input falls through to topic_qa gracefully.
  * `[A]` Malformed JSON to API → 400.
  * `[A]` Missing plan_id → 404.
  * `[A]` Unknown book switch → 404.
  * `[A]` Unknown section/equation in citation shortcut → falls back to BM25.
  * `[A]` ASR endpoint with empty body → 400.
  * `[A]` Persistence path-traversal rejected.
  * `[A]` Save-callback failure does not crash the turn.
  * `[A]` Streaming TTS unavailable → falls back to single WAV.

## 8. Persistence + resume

  * `[A]` Snapshot round-trip captures dialogue + knowledge + chalkboard.
  * `[A]` Resumed session has main_orch=None, still handles tangents.
  * `[A]` Auto-save fires on every turn.
  * `[A]` Sessions tagged with book_name; resume re-binds correctly.
  * `[A]` Sessions sidebar listing surfaces stable fields.
  * `[A]` Cancel deletes the on-disk snapshot.

## 9. Multi-book

  * `[A]` Server loads multiple books; first is active.
  * `[A]` `_activate_book` swaps qa caches without leaking.
  * `[A]` Existing session keeps original book on switch.
  * `[A]` `/api/books` lists with active flag.

## 10. Voices

  * `[A]` Voice list parsed from voices.bin (Kokoro npz).
  * `[A]` Missing / corrupt file → empty list, no crash.
  * `[A]` Voice flows from frontend → API → Session → Orchestrator → TTS.

## 11. Streaming

  * `[A]` LLM phrase peeling honours math regions + min-words.
  * `[A]` Orchestrator yields StreamEvent + audio_chunks + audio_complete.
  * `[A]` Non-streaming TTS path yields one StreamEvent per clause.
  * `[A]` PCM16 → WAV header round-trips.
  * `[B]` First-audio latency under 1.5 s on a deep response.

## 12. Pedagogy / content quality (live-LLM)

  * `[B]` `"briefly"` → ≤ 3 clauses; `"in detail step by step"` → ≥ 6.
  * `[B]` Same question across runs is structurally consistent.
  * `[B]` Responses cite equation/section numbers when relevant.
  * `[B]` LaTeX delimiters used for math; survive sanitisation.
  * `[B]` Follow-up to a topic doesn't re-introduce primitives.

## 13. Voice loop (interaction)

  * `[A]` Push-to-talk auto-pauses session.
  * `[A]` ASR endpoint accepts webm/opus / wav / mp3 (decode test).
  * `[H]` Browser MediaRecorder → /api/transcribe → Ask flow works end-to-end.

## 14. Latency budget

  * `[B]` First sentence in < 0.6 s on a brief topic question.
  * `[B]` First audio chunk in < 1.0 s on a brief topic question.

## 15. Mobile + UX polish

  * `[H]` Layout collapses to single column < 900 px.
  * `[H]` Map / Sessions / Tangent overlays work on mobile.
  * `[H]` Mic FAB, pause pulse, drawer toggle are reachable.

---

## Weaknesses found during this round

**W1 (fixed) — `planner._build_surface_regex` crashed on dict-shaped
concept entries.**  The function read `entry.canonical` /
`entry.aliases` as attributes, which works for the production
loader (`book.corpus.load_corpus` constructs `ConceptEntry`
dataclasses) but raises `AttributeError` for any caller that hands
the planner a `Book` whose `concepts` is a plain dict (e.g., custom
test fixtures, raw-JSON loaders, future re-ingest tools).  Fixed by
making the function tolerate both shapes:

  * `if hasattr(entry, "canonical"):` → dataclass path
  * `elif isinstance(entry, dict):` → dict path
  * else: silently skip the entry

This prevents the orchestrator from blowing up on any non-canonical
corpus shape; the test suite now exercises both paths.

**W2 (none found in remaining categories).**  The 28 deterministic
journey + visual + robustness tests, the 14 HTTP endpoint tests,
and the 4 live-LLM pedagogy tests all pass on the first run after
W1 was fixed.  Real ESLII conversation through 12 turns produces no
overlapping shapes, knowledge state grows monotonically, recap
surfaces prior topics, and live Qwen produces noticeably-longer
responses for "in detail" vs "in one sentence".

---

## Final tally (this round)

  * **542** unit tests in `tests/` (pre-existing) — green
  * **28**  deterministic system tests (journey, visual, robustness) — green
  * **14**  in-process HTTP endpoint tests — green
  * **4**   live-LLM pedagogy tests — green (skip when vLLM is down)
  * **3**   cross-clause dedup tests (repeated citations / formulas /
            passage cards) — green
  * **2**   SSE wire-format tests (event-line shape + unknown-plan 404) — green
  * **= 593+ tests pass**, no regressions

## Follow-ups that landed alongside the test sweep

  * **Word-level audio sync.** Frontend now logs every audio chunk's
    AudioContext schedule time + duration + text in
    ``state.audioChunkLog``.  When ``playEvent`` opens a streaming
    clause, ``flushWordHighlightsForSeq`` re-walks the chunks and
    schedules per-word DOM highlights against the *actual* audio
    timeline.  Late-arriving chunks schedule their own highlights
    immediately, so the highlight tracks the voice even when Kokoro
    runs faster or slower than the LLM-estimated ``audio_dur``.  The
    LLM-timestamps path stays for the non-streaming fallback.

  * **Cross-clause dedup pinned.** ``test_cross_clause_dedup.py``
    drives explicit "the user mentions Equation 5.42 in five
    clauses" / "five clauses share the same y = m x + b" / "twice
    asked to explain chapter 5" sequences.  All three pass without
    code changes — confirming the existing ``seen_refs`` /
    ``seen_formulas`` / ``seen_canonical_topics`` shared sets behave
    correctly across long answers.

  * **SSE wire-format pinned.** ``test_sse_wire.py`` boots the
    server in-process, hits ``/api/stream/<plan_id>``, parses the
    raw bytes, and asserts every event block is shaped
    ``event:<name>\\ndata:<json>\\n\\n`` with valid JSON, the
    expected event names, and a closing ``done``.  Unknown plan_ids
    return 404 cleanly.

