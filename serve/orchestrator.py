"""Session orchestrator — wraps a Book + Chalkboard + TTS into a stream.

Given a NarrationPlan, the orchestrator drives the session:

  for each clause in the plan:
      1.  TTS-synthesise the clause           → audio + word_timestamps
      2.  Compute concept_event_times         → when each concept is spoken
      3.  Resolve each concept to a shape     → SVG via narrator.resolver
      4.  Append shape to the chalkboard      → ChalkOp("add", …)
      5.  Yield a stream event with audio + ops + timing

The orchestrator is a generator: it yields :class:`StreamEvent` records as
each clause completes, so the HTTP server can SSE them to the client in
near-real-time.

Determinism
-----------
- The plan, resolver, and chalkboard placement policy are all deterministic.
- TTS audio is the only nondeterministic component; even there, with a
  fixed Kokoro model + voice + speed, output is byte-identical run-to-run.
- The orchestrator does *not* introduce any wall-clock dependencies in
  what it emits — only the chalkboard's ChalkOp.t carries time.
"""
from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from typing import ClassVar, Iterator, Optional

from book.ir import Book
from narrator import (
    NarrationPlan, ResolvedShape, resolve, render_resolved,
)
from narrator.timing import concept_event_times
from narrator.tts import _BackendBase, NullTTS
from chalkboard import Chalkboard
from sevim.math_graph import (
    MathGraph, graph_path_for_book, find_subexpression_parent,
)

from .refcontent import RefContent, resolve_reference, to_latex


# ---------------------------------------------------------------------------
# Stream event
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """One SSE-friendly event emitted as a clause completes.

    The frontend uses this to:
      * play ``audio_b64`` (a complete WAV for this clause)
      * insert each shape SVG at ``concept_times[i]`` ms after audio start
      * highlight active concept at the right time

    When ``streaming=True``, ``audio_b64`` may be empty; the audio
    arrives as a sequence of :class:`AudioChunkEvent` instances after
    this event, then closed by an :class:`AudioCompleteEvent`.
    """
    seq: int
    clause_text: str
    home_nid: str
    audio_b64: str
    audio_dur: float
    rate: int
    voice: str
    word_timestamps: list[tuple[str, float, float]]
    visual_ops: list[dict] = field(default_factory=list)
    streaming: bool = False
    # Phase-0 / Phase-1 math-graph stats snapshot at clause-end:
    # ``{n_formulas, n_vars, n_passages, covered, uncovered}``.  The
    # frontend's coverage badge reads this so the user can see at a
    # glance whether all narrated math made it onto the chalkboard.
    graph_stats: dict = field(default_factory=dict)
    # visual_ops example:
    #   {"t": 0.43, "kind": "add",   "nid": "n_matrix", "svg": "<g>…</g>",
    #    "label": "matrix", "primitive": "matrix_bracket"}
    #   {"t": 0.87, "kind": "highlight", "nid": "n_matrix", "on": True}


@dataclass
class AudioChunkEvent:
    """One PCM-16 audio chunk for an in-progress streaming clause.

    The frontend's Web Audio scheduler decodes ``pcm_b64`` into an
    AudioBuffer and queues it after every previously-queued chunk
    for the same ``seq`` so playback flows continuously.
    """
    seq: int            # clause sequence number this chunk belongs to
    chunk_idx: int      # 0-based index within the clause
    pcm_b64: str        # base64 of 16-bit signed LE mono PCM
    rate: int           # sample rate in Hz
    text: str = ""      # phrase synthesised in this chunk (debug)


@dataclass
class AudioCompleteEvent:
    """Closes a streaming clause's audio.  Carries the final
    duration so the frontend's pacing can lock in (visual ops +
    word highlighting were scheduled against an estimate)."""
    seq: int
    duration: float
    n_chunks: int


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class Orchestrator:
    book: Book
    plan: NarrationPlan
    chalkboard: Chalkboard
    tts: _BackendBase = field(default_factory=NullTTS)
    voice: Optional[str] = None
    speed: float = 1.0
    seen_nids: set[str] = field(default_factory=set)
    seen_home_nids: set[str] = field(default_factory=set)
    seen_refs: set[str] = field(default_factory=set)
    seen_figure_fids: set[str] = field(default_factory=set)
    seen_canonical_topics: set[str] = field(default_factory=set)
    seen_formulas: set[str] = field(default_factory=set)
    # Citation labels (normalized) whose card has already landed on
    # the chalkboard.  Once "5.43" appears, no second card for the
    # same citation should — even if its latex differs (OCR-broken
    # vs. clean).  The learner sees one card per equation, period.
    seen_cites: set[str] = field(default_factory=set)
    # Hard caps per session — protect the user from runaway emission
    # when every detector grabs a different slice of the same broken
    # OCR text (the §5.8.1 "a mess" report had 5+ formula cards plus
    # 2 duplicate canonical figures for one passage).
    _emitted_formula_cards: int = 0
    _emitted_canonical_figures: int = 0
    # Cached query embedding for the current Q&A — populated lazily on
    # first figure-relevance check, reused across passages.
    _query_vec: Optional[tuple] = None
    _query_vec_loaded: bool = False
    # Signatures of semantic graphs we've already rendered, so a clause
    # that re-mentions the same function/matrix/operation set doesn't
    # double-render its diagram.
    seen_semantic_keys: set[str] = field(default_factory=set)
    # Reverse index from "speakable artifact key" → list of chalkboard
    # nids that should highlight when the narrator says that key.  When
    # a later clause re-mentions a citation, formula, or canonical topic
    # that's already on the board, we look it up here and emit a
    # ``highlight`` visual op for *every* matching card so the user
    # sees them all glow at the moment the narrator says it.  Multi-nid
    # mapping is essential because common verbalized phrases like
    # "f of x" appear in several formula cards on the same board (e.g.
    # the kernel-form card and the loss-functional card both contain
    # ``f(x)``); a previous single-nid map made all but the last
    # registration silent.
    _artifact_index: dict = field(default_factory=dict)
    # Most-recent function-bearing card on the chalkboard (formula_card
    # OR equation reference_card with body).  Cross-clause annotation
    # attachment uses this: when a later clause mentions
    # ``where x is …`` or ``Equation 5.42`` but emits no fresh formula,
    # we re-render this card with the new annotation folded in via an
    # ``update`` visual-op so the user sees variable definitions and
    # equation IDs *on the same box as the function they describe*.
    _last_function_nid: str = ""
    # Per-nid annotation set so repeated cite/var-defs don't grow the
    # card unboundedly; structure: ``{nid: {"kind": "formula"|"equation",
    # "fragment": str, "latex": str, "cites": list[str],
    # "var_defs": list[(sym, defn)], "ref_label": str}}``.
    _function_card_state: dict = field(default_factory=dict)
    # Cache of LLM-recovered LaTeX for OCR-fragmented equations,
    # keyed by ref_label ("5.42" → r"f \in H \\ \sum_{i=1}^N L(...)…").
    # Populated in ``__post_init__`` via parallel calls to the local
    # text Qwen so the streaming hot path never blocks on the LLM.
    _eq_latex_cache: dict = field(default_factory=dict)
    # Cache of LLM-cleaned reference-body HTML (math wrapped in
    # ``\(..\)`` / ``\[..\]``) for non-Equation references.  Keyed by
    # ``(kind, ref_label)``.  Hits flip the reference card to a
    # foreignObject so KaTeX auto-renders the math inline with prose.
    _ref_body_cache: dict = field(default_factory=dict)
    # Set when the LLM cache has been warmed once for this session.
    _eq_latex_warmed: bool = False
    # Phase-0 math semantic graph (sevim/math_graph.py).  Per-book
    # persistent: the orchestrator loads it on init, ingests every
    # emitted Formula + Passage during streaming, and the server
    # writes it back to disk on session shutdown.  The graph drives
    # the layout-clustering + variable-level mention highlighting +
    # coverage audit features.
    math_graph: Optional[MathGraph] = None
    book_path: str = ""
    # Pre-computed multi-level concept narration (built offline by
    # ``tools.build_concept_layer``).  When present, the orchestrator
    # prepends the chosen level's prose as a leading clause for each
    # newly-entered section.  Default detail-level is ``L1_story`` —
    # opens with the *why* before any formula appears.  Per-session
    # users can dial via ``detail_level``.
    concept_layer: Optional["ConceptLayer"] = None
    formula_layer: Optional["FormulaLayer"] = None
    detail_level: str = "L1_story"
    # Track which Formula nids were emitted in the current clause so
    # we can wire ``paired_in_clause`` + ``about(Passage, Formula)``
    # edges at clause-end.
    _clause_formula_nids: list[str] = field(default_factory=list)
    # Sections whose concept-layer preamble has already been delivered
    # this session, keyed by ``home_nid``.  Prevents the "story" from
    # repeating when a section is re-entered (e.g. tangents).
    _concept_preamble_done: set = field(default_factory=set)
    # Formulas whose F1_meaning has already been spoken aloud.  Once
    # per session — the learner heard "Equation 5.42 is the
    # regularization functional that minimizes…" once; subsequent
    # mentions don't re-explain.
    _formulas_explained: set = field(default_factory=set)
    # Sections whose L-level prose threads formulas into the
    # narrative itself (L2/L3 with inline citations).  For these,
    # post-clause auto-explanation is silent — the prose already
    # introduces each formula's role at the right moment.
    _narrative_sections: set = field(default_factory=set)
    # Citation labels (normalized: ``"5.42"`` not ``"Equation 5.42"``)
    # whose per-formula intro clause has already been generated.
    # Cross-section dedup: when ss5_8_1 and ss5_8_2 both contain a
    # formula labelled 5.42, we only generate one intro clause for
    # it across the whole session — repeating the same explanation
    # twice was a real complaint.
    _explained_cites: set = field(default_factory=set)
    # Session-level ``cite_label → canonical Formula`` map.  Used by
    # the forward-reference pre-emitter: when clause text mentions
    # "Equation 5.51" before 5.51's own per-formula clause has run,
    # the orchestrator looks up the canonical formula for "5.51"
    # here and pre-emits it onto the board so the user has a card
    # to look at while the narrator names it.  When 5.51's own
    # per-formula clause runs later, the existing nid dedups the
    # add op so the card just highlights, not re-emits.
    _canonical_by_cite: dict = field(default_factory=dict)
    # Cites whose card has been pre-emitted (via forward-reference
    # detection).  Prevents re-emit on every subsequent mention.
    _pre_emitted_cites: set = field(default_factory=set)

    def __post_init__(self) -> None:
        """Pre-warm the LLM cache for every equation cited in the plan.

        Each clause is scanned for ``Equation N.M`` / ``(N.M)``
        patterns; the OCR body for each unique label is fetched via
        ``resolve_reference`` and sent to the local Qwen in parallel
        for clean-LaTeX recovery.  By the time ``_reference_visual_ops``
        emits the card, ``self._eq_latex_cache[label]`` is already
        populated, so the user sees real KaTeX-rendered math instead
        of monospace OCR fragments.
        """
        self._warm_eq_latex_cache()
        # Load (or create) the per-book math semantic graph.  The
        # graph persists across sessions per the user's locked
        # decision (per-book + enrichment).  When the orchestrator is
        # constructed without a book_path (most unit tests), an
        # in-memory graph is created and discarded at shutdown.
        if self.math_graph is None:
            if self.book_path:
                self.math_graph = MathGraph.load(
                    graph_path_for_book(self.book_path),
                    book_id=self.book.title or self.book_path,
                )
            else:
                self.math_graph = MathGraph(
                    book_id=self.book.title or "<unsaved>",
                )
        # Load the pre-computed concept layer if it exists for this
        # book.  Path mirrors the math-graph convention:
        # ``<book>.json`` → ``<book>.concepts.json``.  Missing file is
        # benign — orchestrator falls back to reading the source text
        # verbatim, same behaviour as before this layer existed.
        if self.concept_layer is None:
            from narrator.concept_layer import ConceptLayer
            cl_path = ""
            if self.book_path:
                cl_path = self.book_path.replace(
                    ".json", ".concepts.json")
            self.concept_layer = (ConceptLayer.load(cl_path)
                                  if cl_path else ConceptLayer.empty())
        # Per-formula explanations (F0_role / F1_meaning / F2_walk),
        # built offline by ``tools.build_formula_layer``.  When
        # present, the orchestrator stitches the explanation onto each
        # formula card as it emits — the learner reads what the
        # formula MEANS right next to the symbols.
        if self.formula_layer is None:
            from narrator.formula_layer import FormulaLayer
            fl_path = ""
            if self.book_path:
                fl_path = self.book_path.replace(
                    ".json", ".formulas.json")
            self.formula_layer = (FormulaLayer.load(fl_path)
                                  if fl_path else FormulaLayer.empty())
        # Build the session-wide ``cite_label → canonical Formula``
        # index.  Used by the forward-reference pre-emitter so a
        # mention of "Equation 5.51" in an earlier clause's text
        # surfaces 5.51's card on the board *before* 5.51's own
        # per-formula clause runs.
        self._build_canonical_by_cite()
        # Concept-layer SCRIPT REPLACEMENT.  At detail_level != L0 (and
        # whenever the concept layer has prose for a section), replace
        # the section's source clauses with synthetic clauses built
        # from the chosen level's prose.  The PDF text becomes
        # background data; the L-level prose IS the lesson.  Visuals
        # (formula-cards, references) still emit because the prose
        # mentions formula labels by citation number.
        self._apply_concept_layer_rewrite()

    def _build_canonical_by_cite(self) -> None:
        """Pick one canonical Formula per citation label, session-wide.

        Same scoring used by the per-section canonical filter:
        offline ids preferred (stable), longest latex preferred
        (richest content).  Stored on ``self._canonical_by_cite``.
        """
        if self.math_graph is None:
            return
        def _norm(s):
            s = (s or "").strip()
            for p in ("Equation ", "Eq. ", "Eq "):
                if s.lower().startswith(p.lower()):
                    return s[len(p):]
            return s
        groups: dict[str, list] = {}
        for f in self.math_graph.formulas.values():
            if not f.cite_labels or not f.latex:
                continue
            for c in f.cite_labels:
                cn = _norm(c)
                if cn:
                    groups.setdefault(cn, []).append(f)
        for cite, group in groups.items():
            group.sort(key=lambda f: (
                0 if f.id.startswith("n_offline_") else 1,
                -len(f.latex or ""),
            ))
            self._canonical_by_cite[cite] = group[0]

    # ------------------------------------------------------------------
    # Per-formula audible explanations
    # ------------------------------------------------------------------

    def _yield_formula_explanations(self, visual_ops: list[dict]):
        """Generator: for each *new* formula card in *visual_ops* that
        has a pre-computed F1_meaning, synthesize a brief "what this
        formula is" interjection and yield it as a synthetic
        StreamEvent (+ audio chunks + complete) so the frontend reads
        and plays it the same as any other clause.

        Once-per-session: each formula id is explained at most once;
        subsequent mentions re-use the visual-only path.

        Failure-tolerant: TTS errors fall back to a silent clip so the
        learner still SEES the F1_meaning on the card even if Kokoro
        hiccups.
        """
        if (self.formula_layer is None
                or len(self.formula_layer) == 0):
            return
        def _norm(s):
            s = (s or "").strip()
            for p in ("Equation ", "Eq. ", "Eq "):
                if s.lower().startswith(p.lower()):
                    return s[len(p):]
            return s
        new_formula_ids: list[str] = []
        for op in visual_ops:
            if (op.get("kind") != "add"
                    or op.get("primitive") != "formula_card"):
                continue
            nid = op.get("nid", "")
            if not nid or nid in self._formulas_explained:
                continue
            # Cite-label dedup: each section has a different fid for
            # the same equation (e.g., "5.42" → fid_A in section X,
            # fid_B in section Y).  Plain nid-based dedup misses these
            # cross-section duplicates, so the narrator ends up
            # explaining 5.42 twice — once before 5.48 and again
            # after.  Track the citation label too.
            cite_n = ""
            if self.math_graph is not None:
                f = self.math_graph.formulas.get(nid)
                if f and f.cite_labels:
                    cite_n = _norm(f.cite_labels[0])
            if cite_n and cite_n in self._explained_cites:
                # Already explained under this citation; mark this
                # nid so subsequent visits also short-circuit.
                self._formulas_explained.add(nid)
                continue
            new_formula_ids.append(nid)
            if cite_n:
                self._explained_cites.add(cite_n)
        if not new_formula_ids:
            return
        from narrator.timing import linear_word_timestamps
        for fid in new_formula_ids:
            self._formulas_explained.add(fid)
            # Look up by formula_id first; fall back to cite-label so
            # inline-emitted formulas (synthetic n_formula_X_Y nids)
            # find their offline-graph explanation.
            fe = self.formula_layer.get(fid)
            if fe is None:
                # Last-resort: search by cite_label embedded in the op.
                # The label was attached to the visual op when it was
                # built; not all paths populate this, hence best-effort.
                continue
            text = fe.F1_meaning or fe.F0_role
            if not text:
                continue
            # cite_label can already start with "Equation " (runtime
            # extractor stores it that way; offline build stores just
            # "5.42").  Normalise so we don't say "Equation Equation".
            cite_norm = (fe.cite_label or "").strip()
            for prefix in ("Equation ", "Eq. ", "Eq "):
                if cite_norm.lower().startswith(prefix.lower()):
                    cite_norm = cite_norm[len(prefix):]
                    break
            cite = (f"Equation {cite_norm}"
                    if cite_norm else "That equation")
            # Compose a teacher-voice interjection: name what we just
            # saw, then what it means.  Keep it short — this fires
            # right after the formula card lands and we want the
            # learner to look-while-listening, not lose the thread.
            #
            # Cap at ~180 chars: longer interjections strain Kokoro's
            # streaming chunker and stack up multi-second pauses
            # before the next regular clause can play.
            meaning_clip = text
            if len(meaning_clip) > 160:
                # Trim at the last sentence boundary that fits.
                cut = meaning_clip.rfind(".", 0, 160)
                if cut < 80:
                    cut = 160
                meaning_clip = meaning_clip[:cut + 1].rstrip()
            spoken = (
                f"{cite} — "
                f"{(fe.F0_role + ': ') if fe.F0_role else ''}"
                f"{meaning_clip}"
            )
            # Use the same TTS path as a normal clause.  If synth
            # fails OR returns an unplayable clip (no chunks AND no
            # WAV bytes), SKIP this explanation entirely — yielding
            # an empty-audio StreamEvent stalls the frontend's rAF
            # loop because ``audioStart`` is set from chunk arrival,
            # which never happens.
            try:
                exp_seq = self._next_explain_seq()
                from narrator.planner import NarrationClause
                exp_clause = NarrationClause(
                    text=spoken, home_nid="", concepts=[],
                )
                audio = self._synthesize_for_clause(
                    exp_clause, exp_seq, text=spoken)
            except Exception as e:
                print(f"[explain] TTS failed for {fid}: {e}")
                continue
            has_chunks = bool(getattr(audio, "chunks", None) or [])
            has_wav = bool(getattr(audio, "wav_bytes", b"") or b"")
            if not has_chunks and not has_wav:
                # Silent fallback would leave the frontend stuck on
                # this seq forever (no audio start signal) — skip.
                print(f"[explain] dropping unplayable explanation for "
                      f"{fid} (no audio)")
                continue
            # Highlight the formula card being explained so the
            # learner's eye lands on it while the narrator describes
            # it.  The focus-dim engine on the frontend reads
            # visual_ops; including the nid here promotes that card
            # to opacity 1.0 and dims the rest.
            visual_ops = [
                {"t": 0.0, "kind": "highlight",
                 "nid": fid, "on": True},
            ]
            yield StreamEvent(
                seq=exp_seq,
                clause_text=spoken,
                home_nid="",
                audio_b64=("" if getattr(audio, "streamed", False)
                           else base64.b64encode(audio.wav_bytes)
                                       .decode("ascii")),
                audio_dur=audio.duration,
                rate=audio.rate,
                voice=audio.voice,
                word_timestamps=audio.word_timestamps,
                visual_ops=visual_ops,
                streaming=getattr(audio, "streamed", False),
                graph_stats={},
            )
            for ch in getattr(audio, "chunks", []) or []:
                yield ch
            if getattr(audio, "streamed", False):
                yield AudioCompleteEvent(
                    seq=exp_seq, duration=audio.duration,
                    n_chunks=len(getattr(audio, "chunks", []) or []),
                )

    def _next_explain_seq(self) -> int:
        """Allocate a synthetic seq number for an explanation clause.
        Uses negative offsets from a high base so explanation seqs
        never collide with regular plan-clause seqs (which start at
        0 and grow upward)."""
        if not hasattr(self, "_explain_seq_counter"):
            self._explain_seq_counter = 1_000_000
        self._explain_seq_counter += 1
        return self._explain_seq_counter

    # ------------------------------------------------------------------
    # Concept-layer script-replacement helpers
    # ------------------------------------------------------------------

    def _canonical_formulas_for_section(self, home_nid: str) -> list:
        """Return the deduped canonical formulas for *home_nid*.

        Same filter + dedup pipeline used elsewhere; factored out so
        the narrative-builder and the legacy per-formula path agree
        on which formulas exist.
        """
        if self.math_graph is None:
            return []
        candidates: list = []
        for f in self.math_graph.formulas.values():
            if f.home_nid != home_nid or not f.latex:
                continue
            is_offline = f.id.startswith("n_offline_")
            has_cite = bool(f.cite_labels)
            is_intro = f.id.startswith("n_intro_eq_")
            if not (is_offline or has_cite or is_intro):
                continue
            candidates.append(f)
        def _norm(s):
            s = (s or "").strip()
            for p in ("Equation ", "Eq. ", "Eq "):
                if s.lower().startswith(p.lower()):
                    return s[len(p):]
            return s
        # Dedup by (cite, latex).
        by_sig: dict = {}
        for f in candidates:
            cite_n = _norm(f.cite_labels[0]) if f.cite_labels else ""
            latex_n = _normalize_formula_key(f.latex)
            by_sig.setdefault((cite_n, latex_n), []).append(f)
        deduped: list = []
        for grp in by_sig.values():
            grp.sort(key=lambda f: (
                0 if f.id.startswith("n_offline_") else 1,
                -len(f.latex or ""),
            ))
            deduped.append(grp[0])
        # One per cite-label.
        by_cite: dict = {}
        no_cite: list = []
        for f in deduped:
            c = _norm(f.cite_labels[0]) if f.cite_labels else ""
            if c:
                by_cite.setdefault(c, []).append(f)
            else:
                no_cite.append(f)
        out: list = []
        for grp in by_cite.values():
            grp.sort(key=lambda f: (
                0 if f.id.startswith("n_offline_") else 1,
                -len(f.latex or ""),
            ))
            out.append(grp[0])
        out.extend(no_cite[:3])
        return out

    def _build_narrative_clauses(
        self, *, prose: str, home_nid: str,
        cite_to_formula: dict,
    ):
        """Split ``[FORMULA:N.M]``-anchored prose into clauses,
        attaching each formula to the clause that *introduced* it.

        Returns a list of ``NarrationClause`` instances.  Empty list
        on parse failure (caller falls back to legacy structure).

        Splitting strategy: every ``[FORMULA:N.M]`` marker becomes a
        formula-attachment point; the prose preceding the marker is
        broken at sentence boundaries; the LAST sentence in that
        preceding chunk gets ``_inject_formula_ids = [fid]`` so the
        card lands while the narrator is wrapping that sentence's
        role description.  Pre-marks the formula in
        ``_formulas_explained`` so the post-clause auto-explainer
        skips it.
        """
        import re as _re
        from narrator.planner import NarrationClause
        if not prose:
            return []
        # Pattern captures the cite-label inside the marker.
        parts = _re.split(r"\[FORMULA:(\d+(?:\.\d+)*)\]", prose)
        # parts alternates: prose, cite, prose, cite, ..., prose.
        clauses: list = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                # prose segment
                for sent in _split_into_clauses(part):
                    s = sent.strip()
                    if not s:
                        continue
                    clauses.append(NarrationClause(
                        text=s, home_nid=home_nid, concepts=[],
                        suggested_dur=max(2.0, len(s) * 0.05),
                    ))
            else:
                cite = part.strip()
                f = cite_to_formula.get(cite)
                if f is None or not clauses:
                    continue
                # Cross-section dedup: skip the attach (no card emit,
                # no per-formula explanation) when this citation has
                # already been introduced in a prior clause.  The
                # current clause's prose still reads — the narrator
                # may legitimately come back to a formula for further
                # discussion — but we don't re-introduce it visually
                # or auditorily.
                if cite in self._explained_cites:
                    continue
                self._explained_cites.add(cite)
                # Attach the formula card to the preceding clause so
                # it lands while the narrator is finishing the
                # sentence that introduces its role.
                last = clauses[-1]
                existing = list(getattr(last, "_inject_formula_ids", []))
                if f.id not in existing:
                    existing.append(f.id)
                    last._inject_formula_ids = existing  # type: ignore[attr-defined]
                # The narrative clause IS the explanation; pre-mark
                # so the post-clause explainer doesn't double up.
                self._formulas_explained.add(f.id)
        return clauses

    # ------------------------------------------------------------------
    # Concept-layer script-replacement
    # ------------------------------------------------------------------

    def _apply_concept_layer_rewrite(self) -> None:
        """Replace each section's spoken script with the L-level prose,
        while distributing the section's formulas across the synthetic
        clauses so all visual cards still land.

        Behaviour by ``detail_level``:

          * ``L0_gist``  — additive: insert the one-line gist as a
                           preamble clause; keep all PDF clauses as
                           the spoken body.  Terse mode preserves the
                           original-text experience for users who want
                           the shortest-possible framing.
          * ``L1_story`` … ``L4_anchor`` — *replacement*: PDF clauses
                           for sections WITH a concept entry are
                           dropped entirely; the L-level prose becomes
                           the spoken script.  All formulas the
                           dropped PDF would have emitted are
                           re-attached to the synthetic clauses as
                           injected visual ops, so the learner still
                           sees every card.

        Sections without a concept entry are untouched at every level
        (graceful degradation while extraction is incomplete).
        """
        if (self.concept_layer is None
                or len(self.concept_layer) == 0):
            return
        # Chapter-zoom plans are already gist-rewritten clauses keyed
        # on the chapter map; running another concept-layer pass over
        # them would replace the punch-line / role-in-parent prose with
        # the per-section L1 prose and double-emit formulas.
        if (self.plan and self.plan.meta
                and (self.plan.meta or {}).get("mode") == "chapter_zoom"):
            return
        from narrator.planner import NarrationClause
        new_clauses: list = []
        clauses = list(self.plan.clauses)
        n = len(clauses)
        i = 0
        n_replaced_sections = 0
        n_appended_clauses = 0
        n_kept_clauses = 0
        replace_mode = (self.detail_level != "L0_gist")
        while i < n:
            home_nid = clauses[i].home_nid
            # Group consecutive clauses sharing this home_nid.
            j = i + 1
            while j < n and clauses[j].home_nid == home_nid:
                j += 1
            if (home_nid
                    and self.concept_layer.has(home_nid)):
                prose = self.concept_layer.text_for(
                    home_nid, level=self.detail_level)
                synth: list = []
                if prose:
                    for sent in _split_into_clauses(prose):
                        s = sent.strip()
                        if not s:
                            continue
                        synth.append(NarrationClause(
                            text=s, home_nid=home_nid, concepts=[],
                            suggested_dur=max(2.0, len(s) * 0.05),
                        ))
                if synth and replace_mode:
                    # Narrative-with-anchors: detect ``[FORMULA:N.M]``
                    # markers in the prose.  When present, split the
                    # prose at markers, treat each segment as its own
                    # clause, and attach each marker's formula to the
                    # preceding clause so the card lands as the
                    # narrator wraps up the role-sentence — no
                    # isolated per-formula clauses, one continuous
                    # story.
                    import re as _re
                    if "[FORMULA:" in prose:
                        # Look up canonical formulas (dedup) so
                        # marker citations resolve to a single card.
                        section_canonical = (
                            self._canonical_formulas_for_section(home_nid))
                        cite_to_formula: dict[str, object] = {}
                        for f in section_canonical:
                            if f.cite_labels:
                                cn = (f.cite_labels[0]
                                      .replace("Equation ", "")
                                      .strip())
                                cite_to_formula.setdefault(cn, f)
                        narrative_clauses = (
                            self._build_narrative_clauses(
                                prose=prose,
                                home_nid=home_nid,
                                cite_to_formula=cite_to_formula,
                            ))
                        if narrative_clauses:
                            self._narrative_sections.add(home_nid)
                            new_clauses.extend(narrative_clauses)
                            n_appended_clauses += len(narrative_clauses)
                            n_replaced_sections += 1
                            i = j
                            continue
                    # Soft fallback: prose has no anchor markers but
                    # mentions formulas as "Equation N.M".  Inline
                    # citation detection will land them; suppress the
                    # post-clause auto-explainer to avoid duplication.
                    if _re.search(r"\bEquation\s+\d+\.\d+", prose):
                        self._narrative_sections.add(home_nid)
                        new_clauses.extend(synth)
                        n_appended_clauses += len(synth)
                        n_replaced_sections += 1
                        i = j
                        continue
                    # ---- prose has no formulas: legacy story+per-formula ----
                    # Pick the section's canonical formulas (offline,
                    # cited, or intro), dedup near-duplicates, and
                    # collapse to ONE per citation label.  See
                    # earlier comments for the dedup rationale.
                    canonical: list = []
                    if self.math_graph is not None:
                        for f in self.math_graph.formulas.values():
                            if f.home_nid != home_nid or not f.latex:
                                continue
                            is_offline = f.id.startswith("n_offline_")
                            has_cite = bool(f.cite_labels)
                            is_intro = f.id.startswith("n_intro_eq_")
                            if not (is_offline or has_cite or is_intro):
                                continue
                            canonical.append(f)
                    def _norm_cite(s: str) -> str:
                        s = (s or "").strip()
                        for p in ("Equation ", "Eq. ", "Eq "):
                            if s.lower().startswith(p.lower()):
                                return s[len(p):]
                        return s
                    by_sig: dict[tuple, list] = {}
                    for f in canonical:
                        cite_n = (_norm_cite(f.cite_labels[0])
                                  if f.cite_labels else "")
                        latex_n = _normalize_formula_key(f.latex)
                        by_sig.setdefault((cite_n, latex_n), []).append(f)
                    deduped: list = []
                    for sig, group in by_sig.items():
                        group.sort(key=lambda f: (
                            0 if f.id.startswith("n_offline_") else 1,
                            -len(f.latex or ""),
                        ))
                        deduped.append(group[0])
                    by_cite: dict[str, list] = {}
                    no_cite: list = []
                    for f in deduped:
                        c = (_norm_cite(f.cite_labels[0])
                             if f.cite_labels else "")
                        if c:
                            by_cite.setdefault(c, []).append(f)
                        else:
                            no_cite.append(f)
                    final: list = []
                    for c, group in by_cite.items():
                        group.sort(key=lambda f: (
                            0 if f.id.startswith("n_offline_") else 1,
                            -len(f.latex or ""),
                        ))
                        final.append(group[0])
                    final.extend(no_cite[:3])
                    final.sort(key=lambda f: (
                        0 if f.cite_labels else 1,
                        _norm_cite(f.cite_labels[0]) if f.cite_labels else "z",
                    ))
                    final = final[:10]
                    # Build the per-section clause sequence:
                    #   (1) the L-level story clauses (no formulas) —
                    #       set the why before any symbol appears;
                    #   (2) ONE clause per canonical formula whose
                    #       spoken text IS the formula's explanation
                    #       AND whose visual op emits the formula
                    #       card.  The card lands at the moment the
                    #       narrator names it, and the focus-dim
                    #       engine puts the spotlight on it.
                    section_clauses: list = list(synth)
                    for f in final:
                        cite_n = (_norm_cite(f.cite_labels[0])
                                  if f.cite_labels else "")
                        # Cross-section dedup: if a per-formula intro
                        # for this citation has already been built in
                        # an earlier section's clauses, skip — the
                        # learner has heard the same "Equation 5.42 —
                        # regularization functional: …" once already.
                        if cite_n and cite_n in self._explained_cites:
                            continue
                        if cite_n:
                            self._explained_cites.add(cite_n)
                        fe = (self.formula_layer.lookup(
                                  formula_id=f.id, cite_label=cite_n)
                              if self.formula_layer is not None else None)
                        role = (fe.F0_role if fe else "")
                        meaning = (fe.F1_meaning if fe else "")
                        # Cap meaning length so Kokoro's chunker
                        # doesn't choke on very long sentences.
                        clip = meaning
                        if clip and len(clip) > 160:
                            cut = clip.rfind(".", 0, 160)
                            if cut < 80:
                                cut = 160
                            clip = clip[:cut + 1].rstrip()
                        cite_phrase = (f"Equation {cite_n}"
                                       if cite_n else "This formula")
                        if role and clip:
                            spoken = f"{cite_phrase}, {role}: {clip}"
                        elif role:
                            spoken = f"{cite_phrase}, {role}."
                        elif clip:
                            spoken = f"{cite_phrase}: {clip}"
                        else:
                            spoken = f"{cite_phrase}."
                        c = NarrationClause(
                            text=spoken, home_nid=home_nid,
                            concepts=[],
                            suggested_dur=max(2.0, len(spoken) * 0.05),
                        )
                        c._inject_formula_ids = [f.id]  # type: ignore[attr-defined]
                        # Per-formula clauses: card lands as the
                        # narrator names it (early in the audio), so
                        # the visual ↔ audio ↔ explanation are all
                        # synchronous instead of card-after-name.
                        c._inject_position = "start"  # type: ignore[attr-defined]
                        # Mark this formula as already explained so
                        # the post-clause explainer doesn't re-explain.
                        self._formulas_explained.add(f.id)
                        section_clauses.append(c)
                    new_clauses.extend(section_clauses)
                    n_appended_clauses += len(section_clauses)
                    n_replaced_sections += 1
                    i = j
                    continue
                if synth and not replace_mode:
                    # L0: additive — preamble + PDF clauses.
                    new_clauses.extend(synth)
                    n_appended_clauses += len(synth)
            for k in range(i, j):
                new_clauses.append(clauses[k])
                n_kept_clauses += 1
            i = j
        if n_appended_clauses > 0:
            mode = ("replace" if replace_mode else "additive")
            print(f"[concept-layer] detail={self.detail_level} "
                  f"mode={mode}: {n_replaced_sections} sections "
                  f"replaced, {n_appended_clauses} synthetic clauses "
                  f"emitted, {n_kept_clauses} PDF clauses preserved")
            self.plan.clauses = new_clauses

    # ------------------------------------------------------------------
    # Math-graph helpers — called from each formula-card emission site
    # ------------------------------------------------------------------

    def _graph_ingest_formula(self, *, nid: str, latex: str,
                              cite_labels: Optional[list[str]] = None,
                              home_nid: str = "") -> Optional[str]:
        """Add the Formula node + uses/binds/defines edges to the graph
        and return the layout anchor nid (the existing formula card
        that shares the most variables with this new one).

        Idempotent: re-ingesting the same nid merges new cite labels
        rather than duplicating.  Tracks the nid in the current
        clause's formula list so ``paired_in_clause`` + ``about``
        edges land at clause-end.

        The returned ``anchor_nid`` is included in the emitted ADD op
        so the frontend can place related formulas physically close.
        Returns ``None`` when the graph is unavailable or no related
        formula exists yet.
        """
        if self.math_graph is None:
            return None
        self.math_graph.ingest_formula(
            nid=nid, latex=latex or "",
            cite_labels=list(cite_labels or []),
            home_nid=home_nid,
        )
        if nid not in self._clause_formula_nids:
            self._clause_formula_nids.append(nid)
        return self.math_graph.best_anchor_for(nid)

    # ------------------------------------------------------------------
    # Visual connectors between formula cards
    # ------------------------------------------------------------------

    # Edge styles borrowed from sevim/s5_render.py (see
    # ARCHITECTURE.md of that project).  Each style is a dict the
    # frontend's renderEdge handler converts into an SVG path.
    # ClassVar — otherwise dataclass would reject the dict default.
    _EDGE_STYLES: ClassVar[dict] = {
        # Each edge type gets a distinct colour + line style so a
        # quick glance reveals the kind of relationship without
        # reading the label.
        # ``B is derived from A`` — orange, solid, filled arrowhead.
        "derived_from": {
            "stroke": "#e65100", "stroke_width": 3.0,
            "dash": "", "marker": "arrow", "label": "derives from",
        },
        # ``B specializes A`` — purple, dashed, hollow triangle.
        "specializes": {
            "stroke": "#4a148c", "stroke_width": 3.0,
            "dash": "10,6", "marker": "hollow", "label": "special case of",
        },
        # ``B references A`` — indigo blue, solid, filled arrowhead.
        "references": {
            "stroke": "#0d47a1", "stroke_width": 2.8,
            "dash": "", "marker": "arrow", "label": "cites",
        },
        # ``A ≈ B`` — teal-green, dotted, ``≈`` label.  Symmetric, so
        # it gets no directional arrow but still has a visible double-
        # arrow marker so the bidirectional relationship reads at a
        # glance.
        "related_to": {
            "stroke": "#00695c", "stroke_width": 2.8,
            "dash": "2,4", "marker": "biarrow", "label": "≈",
        },
    }

    def _collect_edge_visual_ops(self, *, clause_formula_nids: list[str],
                                 audio_dur: float) -> list[dict]:
        """Return ``edge`` visual ops for every semantic relationship
        between formulas already on the chalkboard.

        Both endpoints must already be on-board — the chalkboard never
        back-jumps to materialize a formula that lived in a previous
        section.  Cross-section references are dropped instead of
        rendered, leaving the learner free to revisit prior formulas
        on demand if they want to.

        Each edge fires near the END of the clause.
        """
        if self.math_graph is None or not clause_formula_nids:
            return []
        ops: list[dict] = []
        seen_pairs: set[tuple[str, str, str]] = set()
        edge_t = max(0.5, audio_dur - 0.4)
        candidate_types = ("derived_from", "specializes",
                           "references", "related_to")

        # Content-signature → canonical chalkboard nid.  The offline
        # math graph extracts the same equation several times (once per
        # passage that contains it), each as a distinct Formula node
        # with its own id.  We collapse those duplicates: an edge whose
        # dst signature already matches an on-board formula gets
        # redirected to the existing card instead of looking like a
        # different formula.
        def _sig(cite_labels, latex: str) -> str:
            cite = ",".join(sorted(cite_labels or []))
            return f"{cite}::{_normalize_formula_key(latex)}"
        sig_to_nid: dict[str, str] = {}
        for s in self.chalkboard.shapes:
            if s.primitive != "formula_card":
                continue
            sig = _sig(s.meta.get("cites") or [],
                       s.meta.get("latex") or "")
            if sig and sig not in sig_to_nid:
                sig_to_nid[sig] = s.nid

        def _norm_cite(s: str) -> str:
            s = (s or "").strip()
            for p in ("Equation ", "Eq. ", "Eq "):
                if s.lower().startswith(p.lower()):
                    return s[len(p):]
            return s
        # Pre-compute a cite → set-of-formula-ids map so we can
        # expand each clause source to "all math-graph siblings that
        # share its cite-label".  The chalkboard card we emit is
        # often the prose-rich ``n_ref_Equation_5_42_2`` (best for
        # display), but its sibling ``n_formula_2_X`` is what owns
        # the ``references`` edges.  Without expansion, the edge
        # collector queries the prose-rich card's edge list and
        # finds nothing.
        cite_to_fids: dict[str, list[str]] = {}
        for f in self.math_graph.formulas.values():
            for c in (f.cite_labels or []):
                cite_to_fids.setdefault(_norm_cite(c), []).append(f.id)
        for fid in clause_formula_nids:
            src_formula = self.math_graph.formulas.get(fid)
            src_cites = (set(_norm_cite(c)
                             for c in (src_formula.cite_labels or []))
                         if src_formula else set())
            edge_source_fids: set[str] = {fid}
            for c in src_cites:
                for sib_id in cite_to_fids.get(c, ()):
                    edge_source_fids.add(sib_id)
            # Iterate the union of edges from this formula and all
            # its same-cite siblings.  ``seen_pairs`` later dedups
            # multiple incoming edges to the same canonical dst.
            edges_iter = []
            for src_id in edge_source_fids:
                for ee in self.math_graph.out_edges(src_id):
                    edges_iter.append(ee)
            for e in edges_iter:
                if e.type not in candidate_types:
                    continue
                if e.dst not in self.math_graph.formulas:
                    continue
                # Skip self-cite references: the math graph wires every
                # pair of formulas that share a citation label as a
                # ``references`` edge by default ("Equation 5.42 → 5.42").
                # These are duplicates, not pedagogical relationships,
                # and after canonical-dedup they collapse into self-edges
                # that get filtered later — so just drop them upfront.
                dst_formula = self.math_graph.formulas[e.dst]
                dst_cites = set(_norm_cite(c)
                                for c in (dst_formula.cite_labels or []))
                if (e.type == "references"
                        and src_cites
                        and dst_cites
                        and src_cites & dst_cites):
                    continue
                # The edge's recorded ``src`` may be a sibling not on
                # the chalkboard (we expanded edge sources to all
                # same-cite formulas).  Re-anchor the edge's source
                # to ``fid`` — the formula actually rendered on the
                # board for this clause.
                canonical_src = (fid if self._chalkboard_has(fid)
                                 else e.src)
                if not self._chalkboard_has(canonical_src):
                    continue
                # Resolve the dst to its canonical (already-on-board)
                # nid by content signature, so multiple offline Formula
                # entries that differ only by id collapse onto a single
                # chalkboard card.  ``dst_formula`` was already
                # fetched above for the self-cite filter.
                dst_sig = _sig(dst_formula.cite_labels, dst_formula.latex)
                canonical_dst = sig_to_nid.get(dst_sig, e.dst)
                # Self-edge after dedup (src and dst collapse to the
                # same card) — drop, it would draw a tiny loop.
                if canonical_dst == canonical_src:
                    continue
                # Per learner preference, the chalkboard never
                # back-jumps to a previously-shown formula.  If the
                # cross-section target isn't already on the board, we
                # drop the edge — there is no card to point at and we
                # don't spawn a "ghost reference" for the user.  If the
                # user wants to revisit Equation 5.42 from a later
                # section, they ask for it explicitly.
                if not self._chalkboard_has(canonical_dst):
                    continue
                key = (canonical_src, e.type, canonical_dst)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                style = self._EDGE_STYLES.get(e.type, {})
                # Bridge label: instead of the generic edge-type word
                # ("cites", "derives from"), surface the *concrete
                # reference* in the label so the learner reads the
                # logical connection at a glance.  E.g. "from (5.42)"
                # instead of "derives from".  This is the smallest
                # version of "explain, don't just read"; richer prose
                # bridges (worked-example sentences) come in iter #5+.
                dst_cite = ""
                if dst_formula.cite_labels:
                    dst_cite = f"({dst_formula.cite_labels[0]})"
                if e.type == "derived_from":
                    label = f"from {dst_cite}".strip() or "from"
                elif e.type == "specializes":
                    label = (f"special case of {dst_cite}".strip()
                             if dst_cite else "special case of")
                elif e.type == "references":
                    label = f"uses {dst_cite}".strip() or "uses"
                elif e.type == "related_to":
                    label = "≈"
                else:
                    label = style.get("label", "")
                ops.append({
                    "t": edge_t, "kind": "edge",
                    "src_nid": canonical_src, "dst_nid": canonical_dst,
                    "edge_type": e.type,
                    "stroke": style.get("stroke", "#37474f"),
                    "stroke_width": style.get("stroke_width", 1.5),
                    "dash": style.get("dash", ""),
                    "marker": style.get("marker", "arrow"),
                    "label": label,
                })
        return ops

    def save_math_graph(self) -> None:
        """Write the math graph back to disk.  Called by the server
        on session shutdown / cancel so enrichment persists."""
        if self.math_graph is None:
            return
        # Coverage audit — list math-bearing passages with no Formula
        # node attached.  Phase 0 surfaces this as a console warning
        # so the developer / user can see what's missing.  Phase 1
        # rules + Phase 4 LLM will close the gap; for now, visibility
        # is the goal.
        try:
            report = self.math_graph.coverage_report()
            n_unc = len(report.get("uncovered_passages", []))
            print(f"[math_graph] formulas={report['n_formulas']} "
                  f"vars={report['n_vars']} "
                  f"covered={report['covered_passages']} "
                  f"uncovered={n_unc}")
            if n_unc:
                for u in report["uncovered_passages"][:8]:
                    print(f"  [uncovered] {u['home_nid']}  "
                          f"{u['text'][:120]!r}")
        except Exception as e:
            print(f"[orchestrator] coverage_report failed: {e}")
        if not self.book_path:
            return
        try:
            self.math_graph.save(graph_path_for_book(self.book_path))
        except Exception as e:
            print(f"[orchestrator] math_graph.save failed: {e}")

    def _warm_eq_latex_cache(self) -> None:
        """Fan out parallel LLM calls for every equation referenced in
        the plan.  Bounded by an overall 6-second deadline so a
        slow / down LLM doesn't delay session start.
        """
        if self._eq_latex_warmed:
            return
        self._eq_latex_warmed = True
        # Disable warm-up via env so unit tests can keep determinism.
        if os.environ.get("SEVIM_SKIP_LLM_EQ_LATEX") == "1":
            return
        # When the plan is a streaming plan, ``self.plan.clauses`` is a
        # generator — iterating it here would consume the LLM's
        # sentence stream before it's narrated.  Skip the warm-up
        # entirely; equation LaTeX recovery still happens on-demand
        # in the LLM's own response (the tutor prompt asks for clean
        # LaTeX delimiters, so KaTeX renders them directly).
        if getattr(self.plan, "streaming", False):
            return
        # Collect unique references mentioned across all clauses,
        # split into Equation labels (-> _eq_latex_cache) and
        # everything else with renderable bodies (-> _ref_body_cache).
        eq_labels: list[str] = []
        seen_eq: set[str] = set()
        body_refs: list[tuple[str, str]] = []  # (kind, label)
        seen_body: set[tuple[str, str]] = set()
        for cl in self.plan.clauses:
            for kind, lab, _off, _tnid in _scan_references(
                cl.text, self.book, cl.home_nid,
            ):
                if kind == "Equation":
                    if lab not in seen_eq:
                        seen_eq.add(lab)
                        eq_labels.append(lab)
                elif kind in _BODY_LLM_KINDS:
                    key = (kind, lab)
                    if key not in seen_body:
                        seen_body.add(key)
                        body_refs.append(key)
        if not eq_labels and not body_refs:
            return

        # Fetch OCR bodies up-front (cheap, deterministic).
        from .eq_latex import clean_via_llm, clean_body_via_llm
        eq_jobs: list[tuple[str, str]] = []
        for lab in eq_labels:
            content = resolve_reference(
                self.book, "Equation", lab, hint_nid="",
            )
            if content.latex and content.latex.strip():
                self._eq_latex_cache[lab] = content.latex.strip()
                continue
            if not content.text:
                continue
            if not _looks_garbled_equation(content.text):
                continue
            eq_jobs.append((lab, content.text))

        body_jobs: list[tuple[str, str, str]] = []  # (kind, label, text)
        for kind, lab in body_refs:
            content = resolve_reference(
                self.book, kind, lab, hint_nid="",
            )
            if not content.text or len(content.text.strip()) < 30:
                # Too little to bother rephrasing; the monospace card
                # is fine for short titles / single-line refs.
                continue
            body_jobs.append((kind, lab, content.text))

        if not eq_jobs and not body_jobs:
            return

        # Parallel LLM fan-out — shared 4-worker pool, 6-second deadline.
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        import time
        deadline = time.monotonic() + 6.0
        with ThreadPoolExecutor(max_workers=4) as ex:
            future_to_target: dict = {}
            for lab, ocr in eq_jobs:
                fut = ex.submit(clean_via_llm, ocr,
                                ref_label=lab, timeout=4.0)
                future_to_target[fut] = ("eq", lab)
            for kind, lab, text in body_jobs:
                fut = ex.submit(clean_body_via_llm, text,
                                kind=kind, ref_label=lab, timeout=5.0)
                future_to_target[fut] = ("body", (kind, lab))
            pending = set(future_to_target)
            while pending:
                remaining = max(0.05, deadline - time.monotonic())
                done, pending = wait(
                    pending, timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    break
                for fut in done:
                    target_kind, target_key = future_to_target[fut]
                    try:
                        result = fut.result(timeout=0.1)
                    except Exception:
                        result = ""
                    if not result:
                        continue
                    if target_kind == "eq":
                        self._eq_latex_cache[target_key] = result
                    else:
                        self._ref_body_cache[target_key] = result

    def _enforce_caps(self, ops: list[dict]) -> list[dict]:
        """Drop visual ops that would exceed the per-session caps.

        Keeps the first ``FORMULA_CARDS_MAX`` formula cards and
        ``CANONICAL_FIGURES_MAX`` canonical figures across the whole
        session.  All other primitives (passage banners, book figures,
        reference cards, …) pass through unchanged — those are anchor
        content the user explicitly asked for.

        Also removes the corresponding entries from
        ``self.chalkboard`` so over-the-cap ops never make it into
        snapshots.
        """
        kept: list[dict] = []
        for op in ops:
            prim = op.get("primitive")
            if prim == "formula_card":
                if self._emitted_formula_cards >= FORMULA_CARDS_MAX:
                    self.chalkboard.erase(op.get("nid", ""))
                    continue
                self._emitted_formula_cards += 1
            elif prim == "canonical_figure":
                if self._emitted_canonical_figures >= CANONICAL_FIGURES_MAX:
                    self.chalkboard.erase(op.get("nid", ""))
                    continue
                self._emitted_canonical_figures += 1
            kept.append(op)
        return kept

    def _synthesize_for_clause(self, clause, seq: int, *,
                               text: Optional[str] = None):
        """Run TTS for *clause*.

        Caller may pass an explicit *text* override — typically the
        verbalized (sanitized) form of ``clause.text``, computed by
        :meth:`stream` before iteration.  When omitted, falls back to
        ``clause.text``.  Feeding the sanitized form into TTS is what
        lets Kokoro speak math as English ("f of x") instead of
        reading raw symbols character-by-character.

        When the active backend is :class:`KokoroStreamTTS`, drain its
        per-phrase chunk generator into a list of
        :class:`AudioChunkEvent` objects and attach them to the
        returned audio clip via ``.chunks`` and ``.streamed = True``.
        The clause emitter then yields the clause's StreamEvent first,
        followed by every chunk, then an :class:`AudioCompleteEvent` —
        so the browser can begin playback while later chunks are still
        being computed by Kokoro.

        For non-streaming backends, behaves exactly like the old
        ``self.tts.synthesize`` call.
        """
        from narrator.tts import KokoroStreamTTS, AudioClip
        spoken_text = text if text is not None else clause.text
        # Per-clause speed override — chapter-zoom's punch-line clause
        # asks for a slower rate (0.85) so the headline registers; all
        # other clauses inherit the session speed.
        clause_speed = float(
            (getattr(clause, "meta", None) or {}).get("clause_speed",
                                                      self.speed)
        )
        if isinstance(self.tts, KokoroStreamTTS):
            try:
                chunks: list[AudioChunkEvent] = []
                rate = 24000
                duration = 0.0
                for ch in self.tts.synthesize_stream(
                    spoken_text, voice=self.voice, speed=clause_speed,
                ):
                    if ch.is_final:
                        rate = ch.rate
                        duration = ch.duration
                        break
                    rate = ch.rate
                    chunks.append(AudioChunkEvent(
                        seq=seq,
                        chunk_idx=ch.chunk_idx,
                        pcm_b64=base64.b64encode(ch.pcm16).decode("ascii"),
                        rate=ch.rate,
                        text=ch.text,
                    ))
                if duration <= 0.0 and chunks:
                    # Estimate duration from total PCM size.
                    total_bytes = sum(
                        len(base64.b64decode(c.pcm_b64)) for c in chunks
                    )
                    duration = total_bytes / float(rate * 2)
                from narrator.timing import linear_word_timestamps
                clip = AudioClip(
                    text=spoken_text, wav_bytes=b"",
                    duration=duration, rate=rate,
                    word_timestamps=linear_word_timestamps(
                        spoken_text, duration,
                    ),
                    voice=self.voice or "kokoro_stream",
                    backend="kokoro_stream",
                )
                clip.streamed = True
                clip.chunks = chunks
                return clip
            except Exception as e:
                # Streaming path fell over — degrade to a normal synth
                # so the user still gets audio for this clause.
                print(f"[orchestrator] streamed synth failed: {e}; "
                      f"falling back to single synth for this clause")
        try:
            return self.tts.synthesize(
                spoken_text, voice=self.voice, speed=clause_speed,
            )
        except Exception as e:
            # Final fallback: TTS broke entirely on this clause (Kokoro
            # ``index 510`` errors, model crashes, etc.).  Don't let one
            # bad clause kill the whole session — synthesise a silent
            # AudioClip whose duration is a reasonable read-time
            # estimate (≈ 0.06 s per character, matching natural speech
            # cadence) so the rAF loop still advances and the visual
            # ops fire on schedule.  The learner reads the clause text
            # in silence; the next clause attempts TTS normally.
            print(f"[orchestrator] TTS failed for seq={seq}: {e}; "
                  f"emitting silent clip so session continues")
            est_dur = max(1.5, 0.06 * len(spoken_text or ""))
            from narrator.timing import linear_word_timestamps
            clip = AudioClip(
                text=spoken_text, wav_bytes=b"",
                duration=est_dur, rate=24000,
                word_timestamps=linear_word_timestamps(
                    spoken_text, est_dur,
                ),
                voice=self.voice or "silent",
                backend="silent",
            )
            return clip

    def stream(self) -> Iterator[StreamEvent]:
        """Yield one StreamEvent per narration clause."""
        # Lazy import — narrator.qa imports from this package.
        try:
            from narrator.qa import _sanitize_for_narration
        except Exception:
            _sanitize_for_narration = None  # type: ignore[assignment]
        for seq, clause in enumerate(self.plan.clauses):
            # Compute the **spoken form** of the clause text up front.
            # ``clause.text`` from plan_full / plan() is raw OCR'd text
            # ("f(x) = ∫ k(x,y)φ(y)dy"); the user hears the verbalized
            # form ("f of x equals the integral of k of x y phi of y dy").
            # Splitting raw vs spoken is essential:
            #   * raw   → math fragment / citation / var-def detection
            #             (those scanners need the LaTeX / Unicode symbols).
            #   * spoken → TTS input, mention-scanning, frontend transcript
            #              text (so what's matched and shown matches what's
            #              heard).
            raw_text = clause.text
            # Skip page-number-only clauses ("168 5.", "162", etc.)
            # that the OCR'd book emitted as standalone sentences.  The
            # narrator reading "one hundred sixty-eight" mid-lesson is
            # pure noise; cognitive science calls this *extraneous load*.
            stripped = (raw_text or "").strip()
            if stripped and len(stripped) <= 12:
                tokens = [t for t in stripped.replace(".", " ").split()
                          if t]
                if tokens and all(
                    t.replace("-", "").replace(",", "").isdigit()
                    for t in tokens
                ):
                    continue
            if _sanitize_for_narration is not None:
                spoken_text = _sanitize_for_narration(raw_text) or raw_text
            else:
                spoken_text = raw_text
            # Strip mid-word truncations at the end of a clause (e.g.
            # "...where ˜f den") — when the OCR splits a sentence in
            # the middle of a word, the trailing partial word is read
            # as garbage by TTS.  Heuristic: drop a trailing single
            # token of <= 3 chars that follows a space.  Conservative —
            # only fires when we're confident it's a truncation.
            if spoken_text:
                parts = spoken_text.rstrip().rsplit(None, 1)
                if (len(parts) == 2 and 1 <= len(parts[1]) <= 3
                        and parts[1].isalpha()
                        and parts[1].lower() not in {
                            "is", "of", "in", "to", "as", "or", "be",
                            "at", "on", "if", "so", "we", "it", "no",
                            "an", "a",
                        }):
                    spoken_text = parts[0].rstrip()
            # Smooth-out citation labels for the TTS: "Equation 5.42"
            # is one unit of information that should read as a single
            # fluid phrase, but Kokoro's chunker treats the period
            # inside ``5.42`` as a sentence boundary and inserts a
            # pause.  Substitute "N.M" with "N point M" so the
            # narrator says "Equation five point forty-two" smoothly.
            # Both the displayed transcript and the audio see this
            # substitution — they stay in sync.
            spoken_text = _smooth_citation_speech(spoken_text)
            # The concept-layer preamble used to be prepended here at
            # runtime — that produced a "read twice" bug because the
            # offline-built rewriter (``_apply_concept_layer_rewrite``)
            # already inserts the preamble as its own synthetic
            # clauses.  The rewriter is now the single source of
            # truth for preamble injection.
            audio = self._synthesize_for_clause(clause, seq, text=spoken_text)
            event_times = concept_event_times(
                spoken_text, clause.concepts, audio.word_timestamps,
            )
            visual_ops: list[dict] = []
            # Chapter-zoom mode owns the canvas: a single nested
            # ``chapter_map`` SVG carries every section banner,
            # subsection cell and canonical-formula label as nested
            # ``data-nid`` groups.  We emit it exactly once on the first
            # clause and skip all the per-clause passage / book figure /
            # formula machinery for the rest of the lecture, since that
            # content is already on the board (just zoomed out).  The
            # rAF sync engine still highlights the right cell because
            # every clause's ``home_nid`` matches a nested cell's
            # ``data-nid``; ``highlightActiveClause`` was extended to
            # walk inside the outer shape for nested cells.
            _mode_zoom = (self.plan.meta or {}).get("mode") == "chapter_zoom"
            if _mode_zoom:
                if seq == 0:
                    payload = (self.plan.meta or {}).get(
                        "chapter_map_payload"
                    )
                    cmap_op = self._emit_chapter_map_op(payload)
                    if cmap_op is not None:
                        visual_ops.append(cmap_op)
            # Concept-layer injection: synthetic clauses produced by
            # ``_apply_concept_layer_rewrite`` carry an ``_inject_
            # formula_ids`` list.  These are the formulas the dropped
            # PDF clauses would have emitted — re-attach them here as
            # visual ops staggered across this clause's audio so all
            # cards still land despite the script swap.
            inject_ids = getattr(clause, "_inject_formula_ids", None)
            inject_pos = getattr(clause, "_inject_position", "spread")
            # Forward-reference pre-emit: scan the clause's spoken
            # text for "Equation N.M" mentions.  For any cite whose
            # canonical formula isn't yet on the chalkboard (and that
            # this same clause isn't already injecting), emit the
            # formula card BEFORE the narrator names it.  When the
            # cite's own per-formula clause runs later, the existing
            # card dedups the add — only the highlight + narration
            # fire then.
            attached_cites: set[str] = set()
            if inject_ids and self.math_graph is not None:
                for fid in inject_ids:
                    f0 = self.math_graph.formulas.get(fid)
                    if f0 and f0.cite_labels:
                        for cl in f0.cite_labels:
                            attached_cites.add(_norm_cite(cl))
            pre_emit_formulas: list = []
            if not _mode_zoom and self.math_graph is not None:
                import re as _re_pre
                seen_cite_in_clause: set[str] = set()
                for m in _re_pre.finditer(
                        r"\bEquation\s+(\d+\.\d+)|\[FORMULA:(\d+(?:\.\d+)*)\]",
                        spoken_text or ""):
                    cite = (m.group(1) or m.group(2) or "").strip()
                    if not cite or cite in seen_cite_in_clause:
                        continue
                    seen_cite_in_clause.add(cite)
                    if cite in attached_cites:
                        continue   # this clause already injects it
                    if cite in self._pre_emitted_cites:
                        continue
                    if not self._canonical_by_cite:
                        continue
                    canonical = self._canonical_by_cite.get(cite)
                    if canonical is None:
                        continue
                    if self._chalkboard_has(canonical.id):
                        # Already on the board (some other path
                        # emitted it) — record so we don't try again.
                        self._pre_emitted_cites.add(cite)
                        continue
                    pre_emit_formulas.append(canonical)
                    self._pre_emitted_cites.add(cite)
            if pre_emit_formulas:
                visual_ops.extend(self._emit_formulas_as_ops(
                    pre_emit_formulas,
                    home_nid=clause.home_nid,
                    audio_dur=audio.duration,
                    base_t=0.0,
                    position="start",   # land immediately, ahead of
                                        # any text mentioning them
                ))
            if inject_ids and self.math_graph is not None:
                formulas = [
                    self.math_graph.formulas[fid]
                    for fid in inject_ids
                    if fid in self.math_graph.formulas
                ]
                if formulas:
                    visual_ops.extend(self._emit_formulas_as_ops(
                        formulas,
                        home_nid=clause.home_nid,
                        audio_dur=audio.duration,
                        position=inject_pos,
                    ))
            # First time we see a home_nid, drop a passage card + the top
            # concepts mentioned in that passage's full body_text.  This
            # gives the board real shape content (curves / sets / matrices)
            # even when the spoken sentence itself names no indexed concept.
            # Suppressed in chapter-zoom mode: every section is already a
            # cell in the treemap, so a banner card would just duplicate it.
            if (not _mode_zoom
                    and clause.home_nid
                    and clause.home_nid not in self.seen_home_nids):
                self.seen_home_nids.add(clause.home_nid)
                visual_ops.extend(
                    self._passage_visual_ops(
                        clause.home_nid, seq,
                        audio_dur=audio.duration,
                    )
                )
            seen_clause_cids: set[str] = set()
            emitted_clause_shapes = 0
            CLAUSE_SHAPE_CAP = 4
            # Chapter-zoom mode owns the canvas (the treemap is THE
            # visual): skip every per-clause shape emission below.
            # The clause's ``home_nid`` still matches a nested cell so
            # the rAF sync engine highlights the right subtree.
            if _mode_zoom:
                event_times = []
            for cid, t in event_times:
                # Dedup within a single clause: the same word repeated
                # produces one shape, not many.
                if cid in seen_clause_cids:
                    continue
                seen_clause_cids.add(cid)
                if emitted_clause_shapes >= CLAUSE_SHAPE_CAP:
                    continue
                rs = resolve(self.book, cid, current_nid=clause.home_nid)
                if _is_meaningless_shape(rs):
                    continue
                emitted_clause_shapes += 1
                shape_svg = render_resolved_g(rs)
                nid = f"n_{cid}_{seq}_{int(t * 1000)}"
                # Append to chalkboard.
                self.chalkboard.add(
                    nid=nid, svg_body=shape_svg,
                    primitive=rs.primitive, label=rs.label,
                    w=_estimate_w(rs), h=_estimate_h(rs),
                    meta={"cid": cid, "from_corpus": rs.from_corpus,
                          "home_nid": rs.home_nid},
                )
                visual_ops.append({
                    "t": t, "kind": "add", "nid": nid,
                    "svg": shape_svg, "label": rs.label,
                    "primitive": rs.primitive, "cid": cid,
                    "from_corpus": rs.from_corpus,
                })
                # Highlight at the same moment.
                visual_ops.append({
                    "t": t, "kind": "highlight", "nid": nid, "on": True,
                })
                self.seen_nids.add(nid)

            # Inline formula detection runs FIRST so that any equation
            # citations + variable definitions in the same clause get
            # attached *to* the formula card.  Each formula card we
            # emit also marks its citations as already-seen, so the
            # reference scanner below skips them — no duplicate
            # ``Equation 5.42`` placeholder card next to the formula.
            if not _mode_zoom:
                visual_ops.extend(self._inline_formula_visual_ops(
                    clause_text=clause.text, home_nid=clause.home_nid,
                    seq=seq, audio_dur=audio.duration,
                    word_timestamps=audio.word_timestamps,
                ))

            # Surface every remaining Figure / Table / Theorem reference
            # the clause makes — citations already attached to a
            # formula card are skipped via ``seen_refs``.
            #
            # Suppress duplicate reference_card emission for citations
            # whose formula card was already attached to this clause
            # via the concept-layer narrative builder.  Without this,
            # the user sees TWO cards for "Equation 5.48": a purple
            # spoken-formula card (with role + meaning) AND a green
            # reference_card (without explanation).  Pre-seeding
            # seen_refs keeps the purple one only.
            inject_ids = getattr(clause, "_inject_formula_ids", None)
            if inject_ids and self.math_graph is not None:
                for fid in inject_ids:
                    fobj = self.math_graph.formulas.get(fid)
                    if fobj is None:
                        continue
                    for lab in (fobj.cite_labels or []):
                        norm = (lab.replace("Equation ", "")
                                  .strip())
                        # _reference_visual_ops uses ``Equation::N.M``.
                        self.seen_refs.add(f"Equation::{norm}")
            if not _mode_zoom:
                visual_ops.extend(self._reference_visual_ops(
                    clause.text, clause.home_nid, seq,
                    word_timestamps=audio.word_timestamps,
                    audio_dur=audio.duration,
                ))

                # Per-clause canonical trigger — when the spoken sentence
                # introduces a new canonical topic (e.g. the narrator says
                # "sigmoid" mid-clause), drop the matching diagram on the
                # board, deduped via seen_canonical_topics.
                mid_op = self._clause_canonical_visual_op(
                    clause_text=clause.text, home_nid=clause.home_nid,
                    seq=seq, audio_dur=audio.duration,
                    word_timestamps=audio.word_timestamps,
                )
                if mid_op is not None:
                    visual_ops.append(mid_op)

                # Operation phrase detection — when the narrator says
                # "dot product", "transpose", "gradient of a function", …
                # emit a formula card showing the canonical LaTeX for that
                # operation, timed to when the phrase is spoken.
                visual_ops.extend(self._operation_visual_ops(
                    clause_text=clause.text, home_nid=clause.home_nid,
                    seq=seq, audio_dur=audio.duration,
                    word_timestamps=audio.word_timestamps,
                ))

            # Per-clause Tier-3 deterministic semantic pipeline — runs
            # on every clause so prose without a curated topic match
            # (notably GPT-intro answers when the book has no good
            # passage on the question) still produces diagrams and
            # canonical LaTeX side-cards.  Deduped via
            # ``seen_semantic_keys`` so repeated mentions don't double-
            # render the same diagram.
            if not _mode_zoom:
                visual_ops.extend(self._clause_semantic_visual_ops(
                    clause_text=clause.text, home_nid=clause.home_nid,
                    seq=seq, audio_dur=audio.duration,
                    word_timestamps=audio.word_timestamps,
                ))

            # Cross-clause annotation attachment.  If this clause has
            # ``where x is …`` declarations or equation citations that
            # weren't already folded onto a function card emitted in
            # this clause, attach them to the most-recent function card
            # on the chalkboard via an ``update`` op — so the user
            # always sees variable definitions and equation IDs *on
            # the same box as the function they describe*.  When no
            # function card exists yet, emit a small math-note so the
            # declaration isn't lost.
            if not _mode_zoom:
                visual_ops.extend(
                    self._apply_orphan_annotations(
                        clause_text=clause.text,
                        home_nid=clause.home_nid, seq=seq,
                        audio_dur=audio.duration,
                        word_timestamps=audio.word_timestamps,
                        current_ops=visual_ops,
                    )
                )

                # Mention-based highlighting: if this clause names an
                # artifact already on the board, glow it so the user sees
                # exactly where the narrator's focus is.  Use the SPOKEN
                # form of the clause text — that's what the narrator
                # actually says, so phrases like "f of x" / "K of f g"
                # registered as artifact keys can match.  Searching the
                # raw form for "f of x" never matches because the raw
                # form contains "f(x)".
                visual_ops.extend(self._mention_highlight_visual_ops(
                    clause_text=spoken_text,
                    audio_dur=audio.duration,
                    word_timestamps=audio.word_timestamps,
                    current_ops=visual_ops,
                ))

            # Pair every newly-added card with a co-timed highlight so
            # the user sees a visible pulse the moment the narrator
            # introduces it — not just on cross-clause re-mention.  The
            # cid-resolver path already emits these natively (add + on
            # at the same t), so we only inject for nids that don't
            # already have a highlight at the same time.
            paired = {(o.get("nid"), o.get("t"))
                      for o in visual_ops if o.get("kind") == "highlight"}
            extra_highlights: list[dict] = []
            for o in visual_ops:
                if o.get("kind") != "add":
                    continue
                nid = o.get("nid")
                t = o.get("t", 0.0)
                if not nid:
                    continue
                if (nid, t) in paired:
                    continue
                paired.add((nid, t))
                extra_highlights.append({
                    "t": t, "kind": "highlight", "nid": nid, "on": True,
                })
            visual_ops.extend(extra_highlights)

            visual_ops = self._enforce_caps(visual_ops)
            # Compute a quick stats snapshot for the frontend coverage
            # badge.  Cheap (O(edges)) so we don't worry about doing
            # it every clause.
            graph_stats: dict = {}
            # Phase-0 graph: record this clause as a Passage node and
            # wire ``about(Passage, Formula)`` + ``paired_in_clause``
            # edges for every formula card emitted in it.  The list
            # was accumulated across the per-emission ``_graph_ingest_
            # formula`` calls above.
            if self.math_graph is not None:
                self.math_graph.ingest_passage(
                    seq=seq, text=spoken_text,
                    home_nid=clause.home_nid,
                    formula_ids_in_clause=tuple(self._clause_formula_nids),
                )
                # Phase-1 enrichment — derived_from / specializes /
                # related_to / instance_of from prose patterns.
                # Failures don't block the clause; just log.
                try:
                    from sevim.math_graph_phase1 import enrich_clause
                    enrich_clause(
                        self.math_graph,
                        seq=seq, text=spoken_text,
                        home_nid=clause.home_nid,
                        formula_ids_in_clause=tuple(self._clause_formula_nids),
                    )
                except Exception as e:
                    print(f"[orchestrator] phase1 enrich_clause "
                          f"failed: {e}")
                # Snapshot for the coverage badge.
                try:
                    rep = self.math_graph.coverage_report()
                    graph_stats = {
                        "n_formulas": rep["n_formulas"],
                        "n_vars": rep["n_vars"],
                        "n_passages": len(self.math_graph.passages),
                        "covered": rep["covered_passages"],
                        "uncovered": len(rep["uncovered_passages"]),
                    }
                except Exception:
                    graph_stats = {}
                # Visual connectors — for each semantic edge
                # (derived_from / specializes / related_to /
                # references) that landed because of THIS clause's
                # enrichment, emit an "edge" visual op so the
                # frontend draws a connector between the two cards.
                # Stable end-of-clause time so the arrow appears
                # together with the formula being explained.
                if not _mode_zoom:
                    try:
                        edge_ops = self._collect_edge_visual_ops(
                            clause_formula_nids=self._clause_formula_nids,
                            audio_dur=audio.duration,
                        )
                        visual_ops.extend(edge_ops)
                    except Exception as e:
                        print(f"[orchestrator] edge ops failed: {e}")
            self._clause_formula_nids = []
            # Reading pause after a new formula lands.  Mayer's
            # segmenting principle: when a complex artifact appears,
            # give the learner a beat to parse it before the next
            # sentence runs.  We append a silent PCM chunk + bump the
            # clause's reported duration so the browser's audio queue
            # holds the next clause for ~1.2 s.
            new_formula_count = sum(
                1 for o in visual_ops
                if o.get("kind") == "add"
                and o.get("primitive") == "formula_card"
            )
            if new_formula_count > 0 and getattr(audio, "streamed", False):
                pause_s = min(2.5, 0.6 + 0.4 * new_formula_count)
                rate = audio.rate or 24000
                silent = b"\x00\x00" * int(rate * pause_s)
                audio.chunks = (audio.chunks or []) + [
                    AudioChunkEvent(
                        seq=seq,
                        chunk_idx=len(audio.chunks or []),
                        pcm_b64=base64.b64encode(silent).decode("ascii"),
                        rate=rate,
                        text="",
                    )
                ]
                audio.duration = (audio.duration or 0.0) + pause_s
            yield StreamEvent(
                seq=seq,
                # Send the SPOKEN form to the frontend so the running
                # transcript pane shows exactly what the user hears
                # (no raw LaTeX symbols leaking through).
                clause_text=spoken_text,
                home_nid=clause.home_nid,
                audio_b64=("" if getattr(audio, "streamed", False)
                           else base64.b64encode(audio.wav_bytes).decode("ascii")),
                audio_dur=audio.duration,
                rate=audio.rate,
                voice=audio.voice,
                word_timestamps=audio.word_timestamps,
                visual_ops=visual_ops,
                streaming=getattr(audio, "streamed", False),
                graph_stats=graph_stats,
            )
            # Hand off the streamed PCM chunks now that the clause
            # event has been dispatched.  ``audio.chunks`` is set by
            # the synth-via-stream branch below.
            for ch in getattr(audio, "chunks", []) or []:
                yield ch
            if getattr(audio, "streamed", False):
                yield AudioCompleteEvent(
                    seq=seq, duration=audio.duration,
                    n_chunks=len(getattr(audio, "chunks", []) or []),
                )
            # Per-formula audible explanation: for each formula card
            # this clause just emitted that has a pre-computed
            # F1_meaning the learner hasn't heard yet, yield a
            # synthetic "explanation" StreamEvent.  The frontend
            # treats it as just another clause, so the running
            # transcript shows it and the rAF sync loop schedules its
            # audio in the natural queue.  No new visuals, just speech.
            #
            # Skipped at L0_gist (terse mode), when the formula
            # layer doesn't have an entry for the formula id, OR
            # when this clause's section threads formulas into its
            # narrative prose (the prose already names each
            # formula's role at the right moment, so a post-clause
            # interjection would be redundant).
            in_narrative = (clause.home_nid in self._narrative_sections)
            if (not _mode_zoom
                    and self.detail_level != "L0_gist"
                    and not in_narrative):
                for explain_ev in self._yield_formula_explanations(
                        visual_ops):
                    yield explain_ev

    def _emit_chapter_map_op(self, payload) -> Optional[dict]:
        """Build the single ``chapter_map`` visual op that owns the
        canvas in chapter-zoom mode.

        ``payload`` is the parsed ``chapter_map.<root>.json`` sidecar.
        We render the squarified treemap once and persist it on the
        chalkboard so the snapshot / restore machinery treats it like
        any other shape.  Returns ``None`` for malformed input so the
        orchestrator degrades to plain narration.
        """
        if not isinstance(payload, dict):
            return None
        root_node = payload.get("root") or {}
        root_nid = (
            root_node.get("nid")
            or payload.get("root_nid")
            or ""
        )
        if not root_nid or not root_node.get("children"):
            return None
        try:
            from viz.treemap import render_chapter_map
        except Exception as e:
            print(f"[orchestrator] viz.treemap import failed: {e}")
            return None
        # Stack layout fills the board's full width; height is whatever
        # the cell-stack ends up needing.  ``render_chapter_map`` returns
        # the actual rendered height so we size the chalkboard shape to
        # the real bounding box (the canvas auto-grows to fit) and the
        # frontend ``scrollIntoView`` hook can scroll the active cell
        # into the viewport centre as the narration moves down the
        # stack.
        cb = self.chalkboard
        w = max(720.0, float(cb.canvas_w) - 60.0)
        h_hint = max(440.0, float(cb.canvas_h) - 80.0)
        # Load the per-section SeVim concept diagrams (built offline
        # by ``tools/build_sevim_diagrams``) so the renderer can
        # embed each cell's diagram inline next to its text and
        # formula.  Looked up from the same directory the chapter-map
        # sidecar lives in: ``<book_stem>.sevim_diagrams.<root>.json``.
        sevim_diagrams: dict[str, str] = {}
        try:
            import os as _os, json as _json
            book_path = self.book.source or ""
            if book_path:
                stem, _ = _os.path.splitext(book_path)
                flat = root_nid.replace("/", "_")
                diag_path = f"{stem}.sevim_diagrams.{flat}.json"
                if _os.path.isfile(diag_path):
                    with open(diag_path) as f:
                        loaded = _json.load(f)
                    if isinstance(loaded, dict):
                        sevim_diagrams = {
                            k: v for k, v in loaded.items()
                            if isinstance(k, str) and isinstance(v, str)
                        }
        except Exception as e:
            print(f"[orchestrator] sevim_diagrams load failed: {e}")
        svg_body, h = render_chapter_map(
            root_node, w=w, h=h_hint,
            sevim_diagrams=sevim_diagrams,
        )
        if not svg_body:
            return None
        flat_root = root_nid.replace("/", "_")
        nid = f"chapter_map_{flat_root}"
        label = (
            f"Chapter {root_node.get('number') or '?'}: "
            f"{root_node.get('title') or ''}"
        )
        self.chalkboard.add(
            nid=nid, svg_body=svg_body,
            primitive="chapter_map", label=label,
            w=w, h=h,
            meta={"home_nid": root_nid, "kind": "chapter_map"},
        )
        self.seen_nids.add(nid)
        return {
            "t": 0.0, "kind": "add", "nid": nid,
            "svg": svg_body, "label": label,
            "primitive": "chapter_map", "cid": "",
            "from_corpus": True, "w": w, "h": h,
        }

    def _passage_visual_ops(
        self, home_nid: str, seq: int, *, audio_dur: float,
    ) -> list[dict]:
        """Build the per-passage visuals: a source-banner + concept shapes
        scanned from the full passage body_text.

        Anchors the narration on the board even when the spoken sentence
        names no indexed concept — important for QA answers, where
        retrieved sentences rarely mention canonical mathematical terms.
        """
        ops: list[dict] = []
        node = self.book.find(home_nid) if home_nid else None
        if node is None or node.nid in ("b", ""):
            return ops

        # ---- 1. Source banner ------------------------------------------------
        number = (node.number or "").strip()
        title = (node.title or "").strip()
        kind = (node.kind or "passage").strip()
        if number and title:
            label = f"§{number} {title}"
        elif title:
            label = title
        elif number:
            label = f"§{number}"
        else:
            label = kind.replace("_", " ")
        w, h = 280.0, 64.0
        svg_body = _passage_card_svg(label, kind, w, h)
        banner_nid = f"n_passage_{home_nid.replace('/', '_')}_{seq}"
        self.chalkboard.add(
            nid=banner_nid, svg_body=svg_body,
            primitive="passage_card", label=label,
            w=w, h=h,
            meta={"home_nid": home_nid, "kind": kind},
        )
        self.seen_nids.add(banner_nid)
        # Register the section title + number so a later clause that
        # says "as we saw in Section 5.4" highlights this banner.
        passage_keys = [title]
        if number:
            passage_keys.append(f"Section {number}")
            passage_keys.append(f"§{number}")
        self._register_artifact(banner_nid, passage_keys)
        ops.append({
            "t": 0.0, "kind": "add", "nid": banner_nid,
            "svg": svg_body, "label": label,
            "primitive": "passage_card", "cid": "",
            "from_corpus": True, "w": w, "h": h,
        })

        # ---- 1.5. LLM-spec → renderer (tangent answers only) -----------------
        # The retrieved passage is sometimes a poor proxy for the
        # question (BM25 misses).  For Q&A flows, fire the LLM-supplied
        # SemanticGraph spec card up-front so the answer ALWAYS gets a
        # visual that's about the question, not the home passage.
        mode = (self.plan.meta or {}).get("mode") if self.plan else None
        emitted_spec_op = False
        if mode in ("tangent", "streaming_tutor"):
            spec_op = self._canonical_visual_op(
                home_nid=home_nid, node=node, seq=seq, audio_dur=audio_dur,
            )
            if spec_op is not None:
                ops.append(spec_op)
                emitted_spec_op = True

        # ---- 2. Real book figures within this passage's chapter scope --------
        # Prefer the actual book artwork over speculative concept shapes.
        # Search outward from the home_nid: same node → ancestor section →
        # ancestor chapter, picking up to 2 unseen figures.
        figs = _figures_for_scope(self.book, home_nid)
        # Q&A: filter book figures by relevance to the user's question.
        # Without this, empty-caption images extracted from the PDF
        # (covers, decorations, off-topic embeds) get pulled in as long
        # as their home_nid is in scope — see the §5.8.1 "Spaces of
        # Functions Generated by Kernels" case where a yellow Scream
        # painting was surfaced for the question "what is a functional".
        mode = (self.plan.meta or {}).get("mode") if self.plan else None
        if mode in ("tangent", "streaming_tutor") and figs:
            figs = self._filter_figures_by_relevance(figs)
        emitted = 0
        for i, fig in enumerate(figs):
            if fig.fid in self.seen_figure_fids:
                continue
            if emitted >= 2:
                break
            self.seen_figure_fids.add(fig.fid)
            t = max(0.5, audio_dur * (i + 1) / (len(figs) + 1))
            fig_w, fig_h = 720.0, 460.0
            fig_svg = _book_figure_card_svg(fig, node, fig_w, fig_h)
            fig_nid = f"n_fig_{fig.fid}_{seq}"
            self.chalkboard.add(
                nid=fig_nid, svg_body=fig_svg,
                primitive="book_figure", label=f"Figure on p.{fig.page}",
                w=fig_w, h=fig_h,
                meta={"home_nid": home_nid, "fid": fig.fid,
                      "fig_home_nid": fig.home_nid, "via": "passage_figure"},
            )
            self.seen_nids.add(fig_nid)
            # Register the figure's caption / number so re-mentions
            # ("see Figure 7.8") highlight this card.
            fig_keys: list[str] = []
            fig_number = (getattr(fig, "number", "") or "").strip()
            if fig_number:
                fig_keys.append(f"Figure {fig_number}")
            fig_caption = (getattr(fig, "caption", "") or "").strip()
            if fig_caption:
                fig_keys.append(fig_caption[:60])
            self._register_artifact(fig_nid, fig_keys)
            ops.append({
                "t": t, "kind": "add", "nid": fig_nid,
                "svg": fig_svg,
                "label": f"Figure on p.{fig.page}",
                "primitive": "book_figure", "cid": "",
                "from_corpus": True, "w": fig_w, "h": fig_h,
            })
            emitted += 1
        # If the chapter has actual figures, stop here — those carry more
        # signal than abstract concept shapes.
        if emitted > 0:
            return ops

        # ---- 3. Canonical generated visualisation ----------------------------
        # When the book has no figure and the passage is *about* a known
        # canonical concept (overfitting, bias-variance, ROC, k-fold, …),
        # synthesise a real chart locally (no API).  Pass it through the
        # inspector and regenerate up to 2× if rejected.
        # Skip when tangent mode already emitted a spec_op above — we
        # don't want two canonical cards (Tier-2 + Tier-3) for the same
        # question.
        if emitted_spec_op:
            return ops
        gen_op = self._canonical_visual_op(
            home_nid=home_nid, node=node, seq=seq, audio_dur=audio_dur,
        )
        if gen_op is not None:
            ops.append(gen_op)
            return ops

        # ---- 4. Fallback: concept shapes from passage body_text --------------
        # Used only when the book has no figure for this scope.  We still
        # filter out generic placeholder primitives.
        body = node.body_text or ""
        if not body:
            return ops
        from narrator.planner import _build_surface_regex, _tag_clause
        if not hasattr(self, "_surface_re"):
            self._surface_re, self._alias_to_cid = _build_surface_regex(self.book)
        tags = _tag_clause(body, self._surface_re, self._alias_to_cid)
        if not tags:
            return ops
        seen_cids: set[str] = set()
        ordered: list[str] = []
        for cid, _off in tags:
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            ordered.append(cid)
            if len(ordered) >= 3:
                break
        resolved: list[tuple[str, ResolvedShape]] = []
        for cid in ordered:
            rs = resolve(self.book, cid, current_nid=home_nid)
            if _is_meaningless_shape(rs) or _is_generic_concept_shape(rs):
                continue
            resolved.append((cid, rs))
        n = len(resolved)
        for i, (cid, rs) in enumerate(resolved):
            t = max(0.4, audio_dur * (i + 1) / (n + 1))
            shape_svg = render_resolved_g(rs)
            shape_nid = f"n_{cid}_passage_{seq}_{i}"
            self.chalkboard.add(
                nid=shape_nid, svg_body=shape_svg,
                primitive=rs.primitive, label=rs.label,
                w=_estimate_w(rs), h=_estimate_h(rs),
                meta={"cid": cid, "from_corpus": rs.from_corpus,
                      "home_nid": rs.home_nid, "via": "passage_scan"},
            )
            self.seen_nids.add(shape_nid)
            ops.append({
                "t": t, "kind": "add", "nid": shape_nid,
                "svg": shape_svg, "label": rs.label,
                "primitive": rs.primitive, "cid": cid,
                "from_corpus": rs.from_corpus,
            })
        return ops

    def _canonical_visual_op(
        self, *, home_nid: str, node, seq: int, audio_dur: float,
    ) -> Optional[dict]:
        """Synthesise a topic-relevant chart for *node*.

        Tier-1: hand-rolled curated generator (sub-100 ms).
        Tier-2 (LLM spec → renderer): the LLM supplies a structured
                :class:`SemanticGraph` JSON spec via
                :mod:`viz.llm_spec`; our own renderer turns it into
                SVG.  The LLM never produces SVG markup directly.
        Tier-3 (deterministic): clause/title text is parsed into a
                :class:`SemanticGraph` and rendered to SVG + LaTeX with
                fixed templates — no LLM, no randomness, target <30 ms.

        All paths run through the structural inspector + (when the
        local Qwen-VL endpoint is up) visual inspection, with regen
        up to 2×.  Local-only — never calls Anthropic / OpenAI.
        """
        from viz import find_visualization, inspect_svg
        from viz import semantic_parser, semantic_to_latex, semantic_to_svg
        from viz.semantic_ir import SemanticEdge, SemanticGraph, SemanticNode

        title = (node.title or "").strip()
        body_preview = (node.body_text or "")[:1500]
        question = self.plan.meta.get("question", "") if self.plan.meta else ""
        if not question:
            raw_topic = (self.plan.topic or "").strip()
            # Strip the synthetic placeholders the planner uses for
            # non-Q&A modes (``<outline:nid>``, ``<full-book>``,
            # ``Q: …``) so the canonical card's title falls back to
            # the actual section title rather than showing the
            # internal placeholder string.
            if (raw_topic.startswith("<outline:") and raw_topic.endswith(">")) \
                    or raw_topic == "<full-book>":
                question = ""
            elif raw_topic.startswith("Q: "):
                question = raw_topic[3:].strip()
            else:
                question = raw_topic

        # When the plan came from a low-similarity GPT intro, the book
        # is a poor proxy for the topic — the curated topic registry
        # then frequently misfires (e.g. "what is an autoencoder" was
        # matching "regularization path" because their descriptions
        # share a few embedding tokens).  Skip Tier-1 entirely in
        # that case and trust the LLM directly: it already wrote the
        # spoken intro and knows the topic.
        low_sim = bool(self.plan.meta.get("low_similarity")) \
            if self.plan.meta else False

        # ---- Tier 1: curated generator (only when book matched) ---------
        if not low_sim:
            match = find_visualization(
                title=title, question=question, body_preview=body_preview,
            )
            if match is not None:
                topic, generator = match
                if topic in self.seen_canonical_topics:
                    return None
                for attempt in range(3):
                    gen = generator(seed=attempt)
                    result = inspect_svg(
                        gen.svg_body, topic=topic,
                        width=gen.width, height=gen.height,
                    )
                    if result.accepted:
                        self.seen_canonical_topics.add(topic)
                        return self._wrap_canonical(
                            gen=gen, topic=topic, seq=seq,
                            home_nid=home_nid, result=result, tier=1,
                        )
                # All curated regen attempts failed — fall through to Tier-2.

        spec = question or title or ""

        # ---- Tier 2: LLM-supplied semantic-graph spec → our renderer ---
        # ``qa.answer()`` runs the spec call in parallel with the prose
        # intro and stashes the result on plan.meta.  We re-inflate it
        # to a SemanticGraph and feed it to the same ``semantic_to_svg``
        # renderer the deterministic parser uses, so the SVG style is
        # uniform with the rest of the chalkboard.
        spec_dict = (self.plan.meta or {}).get("semantic_spec") or None
        if spec_dict and spec_dict.get("nodes"):
            llm_graph = SemanticGraph()
            for n in spec_dict.get("nodes") or []:
                llm_graph.add_node(SemanticNode(
                    id=str(n.get("id") or ""),
                    type=str(n.get("type") or ""),
                    label=str(n.get("label") or ""),
                    params=dict(n.get("params") or {}),
                ))
            for e in spec_dict.get("edges") or []:
                llm_graph.add_edge(SemanticEdge(
                    source=str(e.get("source") or ""),
                    target=str(e.get("target") or ""),
                    relation=str(e.get("relation") or ""),
                    params=dict(e.get("params") or {}),
                ))
            llm_key = f"llm_spec:{spec[:80].lower()}"
            if llm_key not in self.seen_canonical_topics and not llm_graph.is_empty():
                svg_body = semantic_to_svg.render_svg(llm_graph)
                if svg_body:
                    sem_w, sem_h = semantic_to_svg.canvas_size(llm_graph)
                    result = inspect_svg(
                        svg_body, topic=spec or "diagram",
                        width=sem_w, height=sem_h,
                        use_vlm=False,
                    )
                    if result.accepted or result.structural_ok:
                        self.seen_canonical_topics.add(llm_key)
                        from viz.generators import GenResult
                        gen = GenResult(
                            svg_body=svg_body,
                            width=sem_w, height=sem_h,
                            params={
                                "source": "llm_spec",
                                "graph": llm_graph.to_dict(),
                                "latex": semantic_to_latex.render_latex(
                                    llm_graph,
                                ),
                            },
                            title=(spec[:60] or "diagram"),
                        )
                        return self._wrap_canonical(
                            gen=gen, topic=spec or "diagram", seq=seq,
                            home_nid=home_nid, result=result, tier=2,
                        )

        # ---- Tier 3 (deterministic): semantic graph → SVG + LaTeX -------
        # Parse the most informative text we have.  Question is the
        # user's words; title is a fallback when no question; body_preview
        # supplies extra signal when both above are short.
        parse_text = " ".join(s for s in (spec, body_preview[:400]) if s)
        graph = semantic_parser.parse(parse_text)
        if not graph.is_empty():
            sem_key = f"semantic:{spec[:80].lower()}"
            if sem_key in self.seen_canonical_topics:
                return None
            svg_body = semantic_to_svg.render_svg(graph)
            sem_w, sem_h = semantic_to_svg.canvas_size(graph)
            if svg_body:
                result = inspect_svg(
                    svg_body, topic=spec or "diagram",
                    width=sem_w, height=sem_h,
                    use_vlm=False,
                )
                if result.accepted or result.structural_ok:
                    # Structural-only — the deterministic templates always
                    # have axes/labels/curves, but the inspector may
                    # tighten thresholds; accept structural_ok too.
                    self.seen_canonical_topics.add(sem_key)
                    from viz.generators import GenResult
                    gen = GenResult(
                        svg_body=svg_body,
                        width=sem_w, height=sem_h,
                        params={
                            "source": "semantic",
                            "graph": graph.to_dict(),
                            "latex": semantic_to_latex.render_latex(graph),
                            "parse_ms": graph.meta.get("parse_ms", 0.0),
                        },
                        title=(spec[:60] or "diagram"),
                    )
                    return self._wrap_canonical(
                        gen=gen, topic=spec or "diagram", seq=seq,
                        home_nid=home_nid, result=result, tier=3,
                    )

        # No further fallback: the system's own pipeline (curated
        # registry → semantic graph) is the only canonical-card path.
        # We don't pull pre-baked SVG markup from the LLM — diagrams
        # have to be produced by code that lives in this repo so they
        # stay consistent in style and inspectable.  When neither tier
        # matches, return None and let formula cards / operation cards
        # carry the visualization weight on their own.
        return None

    def _emit_graph_passage_ops(
        self, *, passage, home_nid: str, seq: int, audio_dur: float,
    ) -> list[dict]:
        """Emit add+highlight ops for every Formula attached to
        *passage* in the offline-prebuilt math graph.

        Each Formula's nid is stable across runs, so subsequent
        graph queries (``best_anchor_for``, ``out_edges`` for
        ``contains``/``references``/...) keep working without
        re-extraction.
        """
        if self.math_graph is None:
            return []
        formulas = self.math_graph.formulas_for_passage(passage.id)
        if not formulas:
            return []
        return self._emit_formulas_as_ops(
            formulas, home_nid=home_nid, audio_dur=audio_dur,
        )

    def _emit_formulas_as_ops(
        self, formulas: list, *, home_nid: str, audio_dur: float,
        base_t: float = 0.0, position: str = "spread",
    ) -> list[dict]:
        """Emit add+highlight visual ops for an arbitrary list of
        ``Formula`` objects, staggered across *audio_dur* seconds
        starting at *base_t*.

        Used by both ``_emit_graph_passage_ops`` (passage-driven) and
        the concept-layer rewriter's section-level formula injection
        (so all section formulas appear even when the spoken script
        no longer reads the PDF text that originally cited them).

        Per-session ``seen_formulas`` dedup prevents re-emission.
        """
        if self.math_graph is None or not formulas:
            return []
        ops: list[dict] = []
        # Spread cards across the audio so they appear progressively
        # rather than all at t=0.
        n = max(1, len(formulas))
        for i, f in enumerate(formulas):
            # Per-session dedup: a formula already emitted once on
            # this chalkboard shouldn't re-emit.
            key = _normalize_formula_key(f.latex)
            if key in self.seen_formulas:
                continue
            # Cite-label dedup: one card per equation citation,
            # period.  Stops the OCR-truncated ``J(f) = Z`` form
            # from landing next to the clean ``J(f) = ∫…`` form
            # because both share cite "5.43".
            cite_n = (_norm_cite(f.cite_labels[0])
                      if f.cite_labels else "")
            if cite_n and cite_n in self.seen_cites:
                continue
            self.seen_formulas.add(key)
            if cite_n:
                self.seen_cites.add(cite_n)
            # Card emit time:
            #  * ``position="start"`` → card appears almost
            #    immediately so it lands as the narrator names it
            #    ("Equation 5.42 — …" + card on screen).  Used by
            #    per-formula intro clauses where the formula's name
            #    is the first thing said.
            #  * ``position="spread"`` (default) → staggered across
            #    audio so multiple formulas don't all pop at once.
            if position == "start":
                t = base_t + 0.3 + i * 0.5
            else:
                t = base_t + max(0.4, audio_dur * (i + 1) / (n + 1))
            cite_labels = list(f.cite_labels)
            # Look up the per-formula explanation from the offline
            # formula layer (F0_role / F1_meaning) so the card carries
            # its own "what this formula is" caption.  Falls back to
            # citation-label lookup when the runtime nid doesn't match
            # the offline graph's id.
            role, meaning = "", ""
            if self.formula_layer is not None:
                fe = self.formula_layer.lookup(
                    formula_id=f.id,
                    cite_label=(cite_labels[0] if cite_labels else ""),
                )
                if fe is not None:
                    role = fe.F0_role
                    meaning = fe.F1_meaning
            w, h = _formula_card_size(f.surface or f.latex,
                                      cite_labels, [], meaning=meaning)
            svg_body = _formula_card_svg(
                f.surface or f.latex, f.latex, w, h,
                cite_labels=cite_labels, var_defs=[],
                role=role, meaning=meaning,
            )
            self.chalkboard.add(
                nid=f.id, svg_body=svg_body,
                primitive="formula_card",
                label=(f.surface or f.latex)[:40],
                w=w, h=h,
                meta={"home_nid": home_nid, "fragment": f.surface,
                      "latex": f.latex, "via": "graph",
                      "cites": cite_labels},
            )
            self.seen_nids.add(f.id)
            self._last_function_nid = f.id
            self._function_card_state[f.id] = {
                "kind": "formula", "fragment": f.surface,
                "latex": f.latex, "cites": cite_labels,
                "var_defs": [],
            }
            # Keep mention scanning consistent — register both the
            # cite labels and verbalized function-call phrases of the
            # graph-stored LaTeX.
            artifact_keys: list[str] = []
            for lab in cite_labels:
                artifact_keys.append(f"Equation {lab}")
                artifact_keys.append(f"({lab})")
            artifact_keys.extend(_verbalized_formula_keys(f.latex))
            self._register_artifact(f.id, artifact_keys)
            # Track in the per-clause list so ``paired_in_clause``,
            # ``about(P, F)`` and the visual edge collector all see it.
            if f.id not in self._clause_formula_nids:
                self._clause_formula_nids.append(f.id)
            anchor_nid = self.math_graph.best_anchor_for(f.id) or ""
            ops.append({
                "t": t, "kind": "add", "nid": f.id,
                "svg": svg_body, "label": (f.surface or f.latex)[:40],
                "primitive": "formula_card", "cid": "",
                "from_corpus": True, "w": w, "h": h,
                "anchor_nid": anchor_nid,
                "source": "graph",
            })
        return ops

    def _inline_formula_visual_ops(
        self, *, clause_text: str, home_nid: str, seq: int,
        audio_dur: float, word_timestamps,
    ) -> list[dict]:
        """Detect math fragments in *clause_text* and emit KaTeX cards.

        Fast path (Phase-0+ persistent graph): if the offline-built
        math_graph already contains a Passage matching this clause,
        emit cards from the graph's Formula nodes — same stable
        nids → all the offline-precomputed contains / references /
        derived_from edges line up at runtime.  Falls back to live
        detection only when the graph has nothing for this clause.

        A fragment qualifies as math if it contains an ``=`` (or
        equivalent comparator) with at least one Greek letter, operator,
        or sub/super-script signal; or if it contains ≥3 mathematical
        characters in a short span.  Each unique formula renders once
        per session (dedup via ``seen_formulas``).
        """
        from .refcontent import to_latex
        ops: list[dict] = []
        if not clause_text:
            return ops

        # === GRAPH-DRIVEN FAST PATH ============================================
        # Look up the offline Passage by (home_nid, text).  If found,
        # iterate its formulas via the ``about`` edges and emit add ops
        # from the persisted Formula records.
        if self.math_graph is not None:
            p = self.math_graph.passage_by_home_text(home_nid, clause_text)
            if p is not None:
                graph_ops = self._emit_graph_passage_ops(
                    passage=p, home_nid=home_nid, seq=seq,
                    audio_dur=audio_dur,
                )
                if graph_ops:
                    return graph_ops

        # === LEGACY LIVE-DETECTION PATH (fallback) =============================
        fragments = _detect_math_fragments(clause_text)
        if not fragments:
            return ops

        spans = _word_spans(clause_text)

        def _time_at_offset(offset: int) -> float:
            if not word_timestamps or not spans:
                return min(audio_dur, max(0.4, audio_dur * 0.5))
            n = min(len(spans), len(word_timestamps))
            for i in range(n):
                ws, we, _ = spans[i]
                if ws <= offset < we or offset < ws:
                    return word_timestamps[i][1]
            return word_timestamps[n - 1][1]

        # Companion context for the whole clause — applied to every
        # formula card emitted from this clause.  Stops the orchestrator
        # from spawning a separate reference_card for "(5.42)" when a
        # corresponding formula is already on the board, and makes
        # "where y is the loss function" land *on* the formula card.
        cite_labels = _equation_citations_in(clause_text)
        var_defs = _variable_definitions_in(clause_text)

        for frag, frag_offset in fragments:
            latex = to_latex(frag).strip()
            if not latex:
                continue
            # Prefer the offline-graph's clean LaTeX whenever the
            # detected fragment cites a known equation.  The runtime
            # extractor turns ``\sum_{i=1}^N \alpha_i`` from the PDF
            # back into garbled strings like ``X = NXi = 1alpha`` —
            # readable for nobody.  When a cited Formula exists in the
            # offline graph, swap in its source LaTeX instead so the
            # learner sees ``\sum_{i=1}^N \alpha_i K(x, x_i)`` rendered
            # properly.
            if cite_labels and self.math_graph is not None:
                for lab in cite_labels:
                    canonical = None
                    for f in self.math_graph.formulas.values():
                        if lab in f.cite_labels and f.latex:
                            canonical = f
                            break
                    if canonical is not None:
                        latex = canonical.latex
                        frag = canonical.surface or canonical.latex
                        break
            key = _normalize_formula_key(latex)
            if key in self.seen_formulas:
                continue
            # Containment fold: if this fragment is a sub-expression
            # of an existing formula on the board (e.g. ``J(f)`` inside
            # ``L(yi, f(xi)) + λJ(f)``), don't emit a duplicate card.
            # Instead, mark contains(parent, child-key) on the graph
            # and fire a highlight on the parent at this clause's
            # mention moment — the user gets a visual cue without a
            # board cluttered with redundant cards.
            if self.math_graph is not None:
                parent_nid = find_subexpression_parent(
                    self.math_graph, latex,
                )
                if parent_nid:
                    self.seen_formulas.add(key)
                    t_parent = _time_at_offset(frag_offset)
                    self.math_graph.add_edge(
                        parent_nid, "contains", f"_subexpr:{key}",
                        meta={"latex": latex},
                    )
                    if self._chalkboard_has(parent_nid):
                        ops.append({
                            "t": t_parent, "kind": "highlight",
                            "nid": parent_nid, "on": True,
                        })
                    continue
            self.seen_formulas.add(key)
            t = _time_at_offset(frag_offset)
            role, meaning = "", ""
            if self.formula_layer is not None and cite_labels:
                fe = self.formula_layer.by_cite(cite_labels[0])
                if fe is not None:
                    role = fe.F0_role
                    meaning = fe.F1_meaning
            # Show var_defs only if the formula layer has no F1_meaning
            # — otherwise the meaning sentence wins (it's higher-level
            # and usually subsumes per-symbol notes).
            shown_var_defs = [] if meaning else var_defs
            w, h = _formula_card_size(frag, cite_labels, shown_var_defs,
                                      meaning=meaning)
            svg_body = _formula_card_svg(
                frag, latex, w, h,
                cite_labels=cite_labels, var_defs=shown_var_defs,
                role=role, meaning=meaning,
            )
            nid = f"n_formula_{seq}_{abs(hash(key)) % 1_000_000}"
            self.chalkboard.add(
                nid=nid, svg_body=svg_body,
                primitive="formula_card", label=frag[:40],
                w=w, h=h,
                meta={"home_nid": home_nid, "fragment": frag,
                      "latex": latex, "via": "inline",
                      "cites": list(cite_labels),
                      "var_defs": [list(t) for t in var_defs]},
            )
            # Mark these citations as already attached so the
            # reference scanner won't emit a duplicate reference_card.
            for label in cite_labels:
                self.seen_refs.add(label.replace(" ", "::", 1))
            self.seen_nids.add(nid)
            # Remember this card for cross-clause annotation attachment.
            self._last_function_nid = nid
            self._function_card_state[nid] = {
                "kind": "formula",
                "fragment": frag,
                "latex": latex,
                "cites": list(cite_labels),
                "var_defs": [tuple(t) for t in var_defs],
            }
            # Register every spoken-form way the narrator might
            # reference this card:
            #   * Citation labels ("Equation 5.42", "(5.42)")
            #   * **Verbalized function calls** from the LaTeX —
            #     ``f(x)`` → "f of x", ``K(f,g)`` → "K of f g".  This
            #     is the critical piece that lets the card light up
            #     when the narrator speaks the formula's content
            #     instead of its citation tag.
            artifact_keys: list[str] = []
            for lab in cite_labels:
                artifact_keys.append(f"Equation {lab}")
                artifact_keys.append(f"({lab})")
            artifact_keys.extend(_verbalized_formula_keys(latex))
            self._register_artifact(nid, artifact_keys)
            # Phase-0 graph ingest — Formula node + uses/binds/defines.
            # Returns the best layout anchor (most-shared-vars formula
            # already on the board) so we can pass it to the frontend.
            anchor_nid = self._graph_ingest_formula(
                nid=nid, latex=latex,
                cite_labels=list(cite_labels),
                home_nid=home_nid,
            )
            ops.append({
                "t": t, "kind": "add", "nid": nid,
                "svg": svg_body, "label": frag[:40],
                "primitive": "formula_card", "cid": "",
                "from_corpus": False, "w": w, "h": h,
                # Layout hint for the frontend: place near this
                # existing card (the one with biggest shared-variables
                # overlap) so related formulas cluster.
                "anchor_nid": anchor_nid or "",
            })
        return ops

    # ------------------------------------------------------------------
    # Cross-clause annotation attachment
    # ------------------------------------------------------------------

    def _apply_orphan_annotations(
        self, *, clause_text: str, home_nid: str, seq: int,
        audio_dur: float, word_timestamps,
        current_ops: list[dict],
    ) -> list[dict]:
        """Attach annotations the user spoke this clause to the right
        function card — so ``where y is the loss function`` and
        ``Equation 5.42`` always land *on the same box* as the function
        they describe, even when the formula was emitted in an earlier
        clause.

        Strategy:
          1. Extract this clause's ``cite_labels`` and ``var_defs``.
          2. If a function card was emitted in *this* clause's ops
             (formula_card or Equation reference_card), the inline
             paths already attached the annotations — nothing to do.
          3. Else if a previous-clause function card is still tracked,
             append the new annotations to its state, regenerate its
             SVG, and emit an ``update`` op so the existing box grows
             to include them.
          4. Else (no function card on the board yet), emit a small
             ``math_note`` card carrying the declarations so they
             aren't dropped.
        """
        cite_labels = _equation_citations_in(clause_text)
        var_defs = _variable_definitions_in(clause_text)
        if not cite_labels and not var_defs:
            return []

        # Did this clause already emit a function-bearing card?
        for op in current_ops:
            if op.get("kind") != "add":
                continue
            nid = op.get("nid", "")
            if nid in self._function_card_state:
                # Inline-formula path already wrote annotations onto
                # the card during initial render.  Nothing more to do.
                return []

        # Find a previous-clause function card to attach to.
        target_nid = self._last_function_nid
        target_state = self._function_card_state.get(target_nid)
        if not target_state or not self._chalkboard_has(target_nid):
            return self._emit_math_note(
                cite_labels=cite_labels, var_defs=var_defs,
                home_nid=home_nid, seq=seq,
                audio_dur=audio_dur, word_timestamps=word_timestamps,
                clause_text=clause_text,
            )

        # Append annotations to the target's state, deduped.
        added = False
        for label in cite_labels:
            if label not in target_state["cites"]:
                target_state["cites"].append(label)
                self.seen_refs.add(label.replace(" ", "::", 1))
                added = True
        for sym, defn in var_defs:
            if not any(s == sym for s, _ in target_state["var_defs"]):
                target_state["var_defs"].append((sym, defn))
                added = True
        if not added:
            return []

        # Regenerate the SVG with the merged annotation set.
        regenerated = self._regenerate_function_card(target_state)
        if regenerated is None:
            return []
        new_svg, new_w, new_h = regenerated
        # Push the new SVG / size into the chalkboard's shape so
        # snapshot() reflects reality.
        for s in self.chalkboard.shapes:
            if s.nid == target_nid:
                s.svg_body = new_svg
                s.w = new_w
                s.h = new_h
                s.meta["cites"] = list(target_state["cites"])
                s.meta["var_defs"] = [list(t) for t in target_state["var_defs"]]
                break

        # Time the update to early in the clause's audio so the user
        # sees the annotation appear roughly when it's spoken.
        spans = _word_spans(clause_text)
        t = (word_timestamps[0][1]
             if word_timestamps else min(0.4, audio_dur * 0.4))
        primitive = ("formula_card"
                     if target_state["kind"] == "formula"
                     else "reference_card")
        return [{
            "t": t, "kind": "update", "nid": target_nid,
            "svg": new_svg, "w": new_w, "h": new_h,
            "primitive": primitive,
            "label": (target_state.get("fragment")
                      or target_state.get("ref_label", ""))[:40],
        }]

    def _chalkboard_has(self, nid: str) -> bool:
        if not nid:
            return False
        return any(s.nid == nid for s in self.chalkboard.shapes)

    def _register_artifact(self, nid: str, keys) -> None:
        """Register *keys* (spoken-form strings) as ways to refer to *nid*.

        Future clauses that mention any of these keys get a
        ``highlight`` op so the existing card glows the moment the
        narrator names it.  Keys shorter than 3 characters or that
        reduce to a single common word are skipped to avoid spurious
        matches (we don't want every "the" or "is" highlighting a card).

        Multiple cards may register the *same* spoken-form key (e.g.
        several formulas all contain ``f(x)`` → "f of x").  Each is
        appended; the mention scanner highlights all of them when the
        key is uttered, so the user sees every relevant card glow.
        """
        if not nid:
            return
        for raw in keys:
            if not raw:
                continue
            key = str(raw).strip().lower()
            # Underscored canonical topic names ("linear_regression") are
            # spoken with spaces.
            key = key.replace("_", " ")
            # Collapse runs of whitespace.
            key = re.sub(r"\s+", " ", key)
            if len(key) < 3:
                continue
            # Drop pure-symbol keys — math fragments like "= 0" are too
            # short / ambiguous to use as highlight triggers.  Require at
            # least one alphabetic character.
            if not re.search(r"[A-Za-z]", key):
                continue
            bucket = self._artifact_index.get(key)
            if bucket is None:
                self._artifact_index[key] = [nid]
            elif isinstance(bucket, list):
                if nid not in bucket:
                    bucket.append(nid)
            else:
                # Legacy single-nid value — promote to list.
                if bucket != nid:
                    self._artifact_index[key] = [bucket, nid]
                else:
                    self._artifact_index[key] = [bucket]

    def _mention_highlight_visual_ops(
        self, *, clause_text: str, audio_dur: float,
        word_timestamps, current_ops: list[dict],
    ) -> list[dict]:
        """Emit highlight ops for artifacts already on the board that
        the *current* clause mentions by name.

        Skips nids that are being added in *current_ops* — those
        already get a natural appearance moment.  Caps at 3 highlights
        per clause to avoid the chalkboard strobing.
        """
        if not clause_text or not self._artifact_index:
            return []
        # nids born this clause shouldn't be re-highlighted.
        born_now: set[str] = {
            op.get("nid", "") for op in current_ops
            if op.get("kind") == "add"
        }
        spans = _word_spans(clause_text)

        def _time_at(offset: int) -> float:
            if not word_timestamps or not spans:
                return min(audio_dur, max(0.4, audio_dur * 0.5))
            n = min(len(spans), len(word_timestamps))
            for i in range(n):
                ws, we, _ = spans[i]
                if ws <= offset < we or offset < ws:
                    return word_timestamps[i][1]
            return word_timestamps[n - 1][1]

        # Punctuation-tolerant matching.  Verbalized speech keeps commas
        # between function arguments ("L of yi, f of xi"), but we
        # register keys in canonical form ("L of yi f of xi") so a
        # rigid regex match fails on the comma.  Normalize both sides:
        # collapse every non-alphanumeric run to a single space, then
        # search.  Also build an offset map so highlight ``t`` lookups
        # still resolve in the *original* text.
        text_lc = clause_text.lower()
        norm_chars: list[str] = []
        norm_to_orig: list[int] = []
        prev_space = True
        for i, ch in enumerate(text_lc):
            if ch.isalnum():
                norm_chars.append(ch)
                norm_to_orig.append(i)
                prev_space = False
            else:
                if not prev_space:
                    norm_chars.append(" ")
                    norm_to_orig.append(i)
                    prev_space = True
        norm_text = "".join(norm_chars)

        def _norm_key(s: str) -> str:
            out = []
            ps = True
            for ch in s.lower():
                if ch.isalnum():
                    out.append(ch)
                    ps = False
                else:
                    if not ps:
                        out.append(" ")
                        ps = True
            return "".join(out).strip()

        # Sort keys longest-first so "Equation 5.42" wins over "Equation".
        keys = sorted(self._artifact_index.keys(), key=len, reverse=True)
        emitted_nids: set[str] = set()
        ops: list[dict] = []
        used_offsets: list[tuple[int, int]] = []
        # Cap total highlights per clause so the chalkboard doesn't
        # strobe when one phrase ("f of x") matches several cards.
        HIGHLIGHTS_PER_CLAUSE = 4
        for key in keys:
            bucket = self._artifact_index.get(key)
            if bucket is None:
                continue
            # Normalise to a list (legacy values may be a bare nid str).
            nids = bucket if isinstance(bucket, list) else [bucket]
            # Drop nids that are already born this clause / already
            # emitted / no longer on the board.
            live_nids = []
            for n in nids:
                if not n or n in born_now or n in emitted_nids:
                    continue
                if not self._chalkboard_has(n):
                    continue
                live_nids.append(n)
            if not live_nids:
                continue
            norm_key = _norm_key(key)
            if not norm_key:
                continue
            # Word-boundary match against the normalized text.
            try:
                pat = re.compile(
                    r"(?<![A-Za-z0-9])" + re.escape(norm_key)
                    + r"(?![A-Za-z0-9])", re.IGNORECASE,
                )
            except re.error:
                continue
            m = pat.search(norm_text)
            if not m:
                continue
            # Map the normalised match offset back into the original
            # clause_text so timestamp lookup uses the right word.
            offset = (norm_to_orig[m.start()] if m.start() < len(norm_to_orig)
                      else m.start())
            # Skip if a longer key already covered this offset region.
            if any(a <= offset < b for a, b in used_offsets):
                continue
            used_offsets.append((m.start(), m.end()))
            t = _time_at(offset)
            for n in live_nids:
                ops.append({
                    "t": t, "kind": "highlight", "nid": n, "on": True,
                })
                emitted_nids.add(n)
                if len(ops) >= HIGHLIGHTS_PER_CLAUSE:
                    break
            if len(ops) >= HIGHLIGHTS_PER_CLAUSE:
                break
        # ----- Variable-level mention highlighting --------------------
        # Phase-0 graph extension: when the narrator names a single
        # variable ("K", "lambda", "phi"), light up *every* Formula
        # card that uses that variable.  This is what catches the
        # cross-card connections users expect when the narrator says
        # "the kernel K" without naming a specific formula.
        if (self.math_graph is not None
                and len(ops) < HIGHLIGHTS_PER_CLAUSE):
            from sevim.math_graph import _name_mentions
            seen_vars: set[str] = set()
            for var_name in _name_mentions(clause_text):
                if var_name in seen_vars:
                    continue
                seen_vars.add(var_name)
                vid = f"v:{var_name}"
                if vid not in self.math_graph.vars:
                    continue
                fids = self.math_graph.formulas_using(vid)
                if not fids:
                    continue
                # Find the offset of the variable mention so the
                # highlight fires at the right moment in the audio.
                # Use a permissive boundary match so "K" doesn't
                # trip on "Kernel" / "Kahn".
                pat = re.compile(
                    r"(?<![A-Za-z0-9])" + re.escape(var_name)
                    + r"(?:\s+of\s+|\s*\(|(?![A-Za-z0-9]))",
                    re.IGNORECASE,
                )
                m2 = pat.search(clause_text)
                if not m2:
                    continue
                t = _time_at(m2.start())
                # Stable iteration so the cap drops the same nids
                # across runs.
                for fid in fids:
                    if fid in born_now or fid in emitted_nids:
                        continue
                    if not self._chalkboard_has(fid):
                        continue
                    ops.append({
                        "t": t, "kind": "highlight", "nid": fid,
                        "on": True,
                    })
                    emitted_nids.add(fid)
                    if len(ops) >= HIGHLIGHTS_PER_CLAUSE:
                        break
                if len(ops) >= HIGHLIGHTS_PER_CLAUSE:
                    break
        return ops

    def _regenerate_function_card(
        self, state: dict,
    ) -> Optional[tuple[str, float, float]]:
        """Re-render the function card SVG for an updated annotation set."""
        kind = state.get("kind", "formula")
        cites = state.get("cites", [])
        var_defs = [tuple(t) for t in state.get("var_defs", [])]
        if kind == "formula":
            frag = state.get("fragment", "")
            latex = state.get("latex", "")
            if not frag or not latex:
                return None
            w, h = _formula_card_size(frag, cites, var_defs)
            svg = _formula_card_svg(
                frag, latex, w, h,
                cite_labels=cites, var_defs=var_defs,
            )
            return svg, w, h
        if kind == "equation_ref":
            ref_label = state.get("ref_label", "")
            text = state.get("text", "")
            latex = state.get("latex", "")
            return _render_equation_ref_card_with_annotations(
                ref_label=ref_label, text=text, latex=latex,
                cite_labels=[c for c in cites
                             if c != f"Equation {ref_label}"],
                var_defs=var_defs,
            )
        if kind == "math_note":
            return _render_math_note_card(cites, var_defs)
        return None

    def _emit_math_note(
        self, *, cite_labels: list[str],
        var_defs: list[tuple[str, str]],
        home_nid: str, seq: int,
        audio_dur: float, word_timestamps,
        clause_text: str,
    ) -> list[dict]:
        """Emit a small note card carrying citations + variable
        definitions when there's no prior function card to attach to.

        The user said *every* math notation should appear — this is
        the safety net for clauses like ``Note that L denotes the
        loss function.`` spoken before any formula has hit the board.
        """
        if not cite_labels and not var_defs:
            return []
        # Skip cite-only notes when the reference scanner is also going
        # to emit a card for it — the reference card already shows the
        # equation.  We only need the safety-net for var_defs (or
        # cite-labels that the reference scanner can't resolve).
        if not var_defs:
            return []
        nid = f"n_mathnote_{seq}_{abs(hash(tuple(var_defs))) % 1_000_000}"
        if nid in self.seen_nids:
            return []
        self.seen_nids.add(nid)
        svg, w, h = _render_math_note_card(cite_labels, var_defs)
        self.chalkboard.add(
            nid=nid, svg_body=svg,
            primitive="math_note", label="math notation",
            w=w, h=h,
            meta={"home_nid": home_nid,
                  "cites": list(cite_labels),
                  "var_defs": [list(t) for t in var_defs]},
        )
        # Math notes are a function-bearing card too — future
        # annotations should attach to them.
        self._last_function_nid = nid
        self._function_card_state[nid] = {
            "kind": "math_note",
            "cites": list(cite_labels),
            "var_defs": [tuple(t) for t in var_defs],
        }
        t = (word_timestamps[0][1]
             if word_timestamps else min(0.4, audio_dur * 0.4))
        return [{
            "t": t, "kind": "add", "nid": nid,
            "svg": svg, "label": "math notation",
            "primitive": "math_note", "cid": "",
            "from_corpus": False, "w": w, "h": h,
        }]

    def _operation_visual_ops(
        self, *, clause_text: str, home_nid: str, seq: int,
        audio_dur: float, word_timestamps,
    ) -> list[dict]:
        """Detect natural-language operation phrases and emit formula cards.

        Driven by ``viz.operations.find_operations``.  Deduped via the
        same ``seen_formulas`` set used by inline-formula detection so
        the same operation does not double-render across clauses.
        """
        from viz.operations import find_operations
        ops: list[dict] = []
        if not clause_text:
            return ops
        spans = _word_spans(clause_text)

        def _time_at_offset(offset: int) -> float:
            if not word_timestamps or not spans:
                return min(audio_dur, max(0.4, audio_dur * 0.5))
            n = min(len(spans), len(word_timestamps))
            for i in range(n):
                ws, we, _ = spans[i]
                if ws <= offset < we or offset < ws:
                    return word_timestamps[i][1]
            return word_timestamps[n - 1][1]

        for label, latex, start, _end in find_operations(clause_text):
            key = _normalize_formula_key(latex)
            if key in self.seen_formulas:
                continue
            self.seen_formulas.add(key)
            t = _time_at_offset(start)
            w, h = _operation_card_size(latex)
            svg_body = _operation_card_svg(label, latex, w, h)
            nid = f"n_op_{seq}_{abs(hash(key)) % 1_000_000}"
            self.chalkboard.add(
                nid=nid, svg_body=svg_body,
                primitive="operation_card", label=label,
                w=w, h=h,
                meta={"home_nid": home_nid, "operation": label,
                      "latex": latex, "via": "operation_phrase"},
            )
            self.seen_nids.add(nid)
            # Operation phrases are spoken verbatim ("dot product",
            # "transpose") — register so re-mentions glow this card.
            # Also register the LaTeX's verbalized function calls in
            # case the narrator says the formula's content rather than
            # the operation's name.
            verb_keys = _verbalized_formula_keys(latex)
            self._register_artifact(nid, [label, *verb_keys])
            op_anchor = self._graph_ingest_formula(
                nid=nid, latex=latex, home_nid=home_nid,
            )
            ops.append({
                "t": t, "kind": "add", "nid": nid,
                "svg": svg_body, "label": label,
                "primitive": "operation_card", "cid": "",
                "from_corpus": False, "w": w, "h": h,
                "anchor_nid": op_anchor or "",
            })
        return ops

    def _clause_canonical_visual_op(
        self, *, clause_text: str, home_nid: str, seq: int,
        audio_dur: float, word_timestamps,
    ) -> Optional[dict]:
        """Trigger a curated canonical generator from the spoken clause text.

        Runs only the Tier-1 (curated) registry — Tier-3 LLM-driven
        synthesis stays gated to the per-passage path because it costs
        ~1.5 s and shouldn't fire mid-clause.  Deduped via
        ``seen_canonical_topics`` so a topic only renders once per session.
        """
        from viz import find_visualization, inspect_svg

        # Strict mode at clause level: only fire when the clause text
        # *literally* names a curated topic (regex unique hit).
        # Without this, NN-adjacent prose like "A restricted Boltzmann
        # machine has no within-layer connections" gets mapped to
        # ``activation_functions`` purely on embedding cosine — the
        # whiteboard then shows an Activation Function chart for an
        # RBM clause, which is what the user reported.
        match = find_visualization(question=clause_text, strict=True)
        if match is None:
            return None
        topic, generator = match
        if topic in self.seen_canonical_topics:
            return None

        for attempt in range(3):
            gen = generator(seed=attempt)
            result = inspect_svg(
                gen.svg_body, topic=topic,
                width=gen.width, height=gen.height,
                use_vlm=False,   # mid-clause: stay structural-only for speed
            )
            if result.accepted:
                self.seen_canonical_topics.add(topic)
                op = self._wrap_canonical(
                    gen=gen, topic=topic, seq=seq,
                    home_nid=home_nid, result=result, tier=1,
                )
                # Time the appearance to coincide with the trigger word.
                t = _trigger_time(clause_text, topic, word_timestamps,
                                  audio_dur)
                op["t"] = t
                return op
        return None

    def _filter_figures_by_relevance(self, figs: list) -> list:
        """Drop book figures that don't look relevant to the question.

        Uses the local embedding server (Qwen3-Embedding) to score each
        figure's caption (or owning-passage body if caption is empty)
        against the user's question.  Figures below
        :data:`FIGURE_RELEVANCE_THRESHOLD` are filtered out.

        Order is preserved so passage-figure preference (closer scope
        first) is honored.  Adds a ``relevance`` attribute on each
        passing figure for downstream meta.

        Falls back to NO filtering when:
          * no question is recorded on the plan,
          * the embedding server is unreachable,
          * embedding the question itself fails.
        """
        question = ""
        if self.plan and self.plan.meta:
            question = (self.plan.meta.get("question") or "").strip()
        if not question:
            return figs
        if not self._query_vec_loaded:
            self._query_vec_loaded = True
            try:
                from book import embeddings as _emb
                if _emb.is_available():
                    qv = _emb.embed_text(question)
                    self._query_vec = tuple(qv) if qv else None
            except Exception:
                self._query_vec = None
        if not self._query_vec:
            return figs
        kept: list = []
        for f in figs:
            score, anchor = _figure_relevance(self.book, f, self._query_vec)
            if score >= FIGURE_RELEVANCE_THRESHOLD:
                # Annotate so downstream emit can carry the score in meta.
                try:
                    setattr(f, "_relevance_score", score)
                    setattr(f, "_relevance_anchor", anchor)
                except Exception:
                    pass
                kept.append(f)
        return kept


    def _clause_semantic_visual_ops(
        self, *, clause_text: str, home_nid: str, seq: int,
        audio_dur: float, word_timestamps,
    ) -> list[dict]:
        """Run the deterministic semantic pipeline on the spoken clause.

        Produces a Tier-3 card per *renderable* clause — the same
        engine the orchestrator uses at passage-level, but called on
        every clause so that GPT-intro paragraphs (and any other
        prose without curated topic matches) still get diagrams and
        canonical LaTeX side-cards on the board.

        Dedupe is keyed on the *content* of the parsed graph (sorted
        node IDs), so the same set of concepts only renders once per
        session even when the narrator repeats the phrase.
        """
        from viz import semantic_parser, semantic_to_latex, semantic_to_svg

        ops: list[dict] = []
        if not clause_text:
            return ops
        # When a passage-level canonical card (Tier-1 curated, Tier-2
        # LLM-spec, or Tier-3 semantic) has already landed for this
        # Q&A session, the per-clause Tier-3 path would just emit a
        # duplicate / off-topic card (e.g. parsing "we see that depends
        # on an existing model" produces a "that → an" flow diagram
        # alongside the real bias-variance plot).  Skip clause-level
        # Tier-3 in that case — the headline card already carries the
        # answer's visual and per-clause stragglers add only clutter.
        mode = (self.plan.meta or {}).get("mode") if self.plan else None
        if (mode in ("tangent", "streaming_tutor")
                and self.seen_canonical_topics):
            return ops
        graph = semantic_parser.parse(clause_text)
        if graph.is_empty():
            return ops

        # Need at least one *renderable* node — pure relationship-only
        # graphs ("y depends on x") still render a flow diagram, so we
        # only skip when nothing concrete is in the graph.
        renderable_types = {
            "function", "matrix", "vector", "shape", "operation", "equation",
        }
        renderable = [n for n in graph.nodes if n.type in renderable_types]
        flow_edges = [
            e for e in graph.edges
            if not e.source.startswith("_") and not e.target.startswith("_")
        ]
        if not renderable and not flow_edges:
            return ops
        # Content-quality gate: a card whose only renderable content is a
        # bare ``vector v`` / ``shape circle`` / ``label foo`` adds no
        # information beyond the spoken word.  Skip the card unless the
        # graph carries an equation / function / matrix / operation, or
        # has labelled flow edges that would draw a real diagram.
        rich_types = {"function", "matrix", "operation", "equation"}
        has_rich = any(n.type in rich_types for n in graph.nodes)
        if not has_rich and not flow_edges:
            return ops

        sig = "|".join(sorted(n.id for n in graph.nodes))
        if sig in self.seen_semantic_keys:
            return ops
        self.seen_semantic_keys.add(sig)

        svg_body = semantic_to_svg.render_svg(graph)
        if not svg_body:
            return ops
        sem_w, sem_h = semantic_to_svg.canvas_size(graph)
        latex_items = semantic_to_latex.render_latex(graph)

        # Anchor the card on the first renderable node's keyword if we
        # can find one in the spoken text — keeps the card timed to the
        # narrator's mention rather than appearing at t=0.
        t = self._semantic_trigger_time(
            clause_text, graph, word_timestamps, audio_dur,
        )

        # Wrap the SVG in the same labelled card the passage path uses
        # so the frontend renders it identically.
        topic = (renderable[0].label if renderable
                 else "diagram").strip() or "diagram"
        outer_w, outer_h = sem_w + 24.0, sem_h + 56.0
        card_svg = _canonical_card_svg(
            inner_svg=svg_body,
            inner_w=sem_w, inner_h=sem_h,
            topic=topic, title=topic[:60],
            outer_w=outer_w, outer_h=outer_h,
            inspection_note="semantic", from_vlm=False,
            tier=3, source="semantic",
        )
        nid = f"n_sem_{seq}_{abs(hash(sig)) % 1_000_000}"
        self.chalkboard.add(
            nid=nid, svg_body=card_svg,
            primitive="canonical_figure", label=topic[:60],
            w=outer_w, h=outer_h,
            meta={"home_nid": home_nid, "topic": topic,
                  "tier": 3, "source": "semantic",
                  "graph": graph.to_dict(),
                  "latex": latex_items,
                  "via": "clause_semantic"},
        )
        self.seen_nids.add(nid)
        self._register_artifact(nid, [topic])
        ops.append({
            "t": t, "kind": "add", "nid": nid,
            "svg": card_svg, "label": topic[:60],
            "primitive": "canonical_figure", "cid": "",
            "from_corpus": False, "w": outer_w, "h": outer_h,
            "synthesised": True, "topic": topic,
            "tier": 3, "source": "semantic",
            "latex": latex_items,
        })

        # Emit each LaTeX item as a formula_card alongside the diagram,
        # deduped against the existing seen_formulas set so per-session
        # formulas only appear once even when several clauses produce
        # the same canonical form.
        for i, latex in enumerate(latex_items):
            key = _normalize_formula_key(latex)
            if key in self.seen_formulas:
                continue
            self.seen_formulas.add(key)
            label = (renderable[i].label if i < len(renderable)
                     else topic)[:40]
            fw, fh = _formula_card_size(latex)
            f_svg = _formula_card_svg(label, latex, fw, fh)
            f_nid = f"n_sem_eq_{seq}_{abs(hash(key)) % 1_000_000}"
            self.chalkboard.add(
                nid=f_nid, svg_body=f_svg,
                primitive="formula_card", label=label,
                w=fw, h=fh,
                meta={"home_nid": home_nid, "fragment": label,
                      "latex": latex, "via": "clause_semantic"},
            )
            self.seen_nids.add(f_nid)
            # Register the spoken-form label (e.g. "quadratic",
            # "sigmoid") AND the verbalized function-call phrases from
            # the LaTeX so the card can light up on either channel.
            verb_keys = _verbalized_formula_keys(latex)
            self._register_artifact(f_nid, [label, *verb_keys])
            anchor_nid = self._graph_ingest_formula(
                nid=f_nid, latex=latex, home_nid=home_nid,
            )
            ops.append({
                "t": t, "kind": "add", "nid": f_nid,
                "svg": f_svg, "label": label,
                "primitive": "formula_card", "cid": "",
                "from_corpus": False, "w": fw, "h": fh,
                "source": "semantic", "latex": [latex],
                "anchor_nid": anchor_nid or "",
            })
        return ops

    def _semantic_trigger_time(
        self, clause_text: str, graph, word_timestamps, audio_dur: float,
    ) -> float:
        """Map the most identifying graph keyword to its TTS timestamp.

        Falls back to mid-clause when no keyword matches — a stable
        offset is still better than t=0, which would stack the card on
        top of the passage banner.
        """
        if not word_timestamps:
            return max(0.5, audio_dur * 0.5)
        # Prefer matrix names ("A"), then function forms ("quadratic"),
        # then operation labels ("dot product").
        candidates: list[str] = []
        for n in graph.nodes:
            if n.type == "matrix" and n.params.get("name"):
                candidates.append(n.params["name"])
            if n.type == "function" and n.params.get("form"):
                candidates.append(n.params["form"])
            if n.type == "operation" and n.label:
                candidates.append(n.label.split()[0])
            if n.type == "shape" and n.params.get("kind"):
                candidates.append(n.params["kind"])
        spans = _word_spans(clause_text)
        n_pairs = min(len(spans), len(word_timestamps))
        text_lower = clause_text.lower()
        for kw in candidates:
            kw_lower = kw.lower()
            idx = text_lower.find(kw_lower)
            if idx < 0:
                continue
            for i in range(n_pairs):
                ws, we, _ = spans[i]
                if ws <= idx < we or idx < ws:
                    return word_timestamps[i][1]
        return max(0.5, audio_dur * 0.5)

    def _wrap_canonical(
        self, *, gen, topic: str, seq: int, home_nid: str,
        result, tier: int,
    ) -> dict:
        """Wrap a synthesised SVG in a labelled card and place on board."""
        w, h = gen.width + 24.0, gen.height + 56.0
        source = (getattr(gen, "params", {}) or {}).get("source", "")
        svg_body = _canonical_card_svg(
            inner_svg=gen.svg_body,
            inner_w=gen.width, inner_h=gen.height,
            topic=topic, title=gen.title,
            outer_w=w, outer_h=h,
            inspection_note=result.reason,
            from_vlm=result.vlm_used,
            tier=tier,
            source=source,
        )
        canon_nid = f"n_canon_{tier}_{source or 'gen'}_{abs(hash(topic)) % 1_000_000}_{seq}"
        self.chalkboard.add(
            nid=canon_nid, svg_body=svg_body,
            primitive="canonical_figure", label=gen.title,
            w=w, h=h,
            meta={"home_nid": home_nid, "topic": topic,
                  "tier": tier, "source": source, "params": gen.params,
                  "inspection": {
                      "accepted": result.accepted,
                      "reason": result.reason,
                      "vlm_used": result.vlm_used,
                      "elapsed_ms": result.elapsed_ms,
                  }},
        )
        self.seen_nids.add(canon_nid)
        # Register topic + title as highlight keys so re-mentions glow
        # this canonical card instead of synthesising a duplicate.
        self._register_artifact(canon_nid, [topic, gen.title])
        op = {
            "t": 0.5, "kind": "add", "nid": canon_nid,
            "svg": svg_body, "label": gen.title,
            "primitive": "canonical_figure", "cid": "",
            "from_corpus": False, "w": w, "h": h,
            "synthesised": True, "topic": topic, "tier": tier,
            "source": source,
        }
        # Tier-3 deterministic path also carries the LaTeX list so the
        # frontend can render formula side-cards without re-parsing.
        latex_items = (getattr(gen, "params", {}) or {}).get("latex")
        if latex_items:
            op["latex"] = list(latex_items)
        return op

    def _reference_visual_ops(
        self, clause_text: str, home_nid: str, seq: int,
        *, word_timestamps: list, audio_dur: float,
    ) -> list[dict]:
        """Detect Figure/Table/Equation/Theorem mentions in *clause_text*
        and emit a card for each, timed to when its trigger word is spoken.

        Cards are deduped per session via ``self.seen_refs`` so a passage
        that mentions Figure 7.8 three times only renders one Figure-7.8
        card on the board.
        """
        ops: list[dict] = []
        if not clause_text:
            return ops
        spans = _word_spans(clause_text)

        def _time_at(offset: int) -> float:
            if not word_timestamps or not spans:
                return min(audio_dur, max(0.4, audio_dur * 0.5))
            n = min(len(spans), len(word_timestamps))
            for i in range(n):
                ws, we, _ = spans[i]
                if ws <= offset < we or offset < ws:
                    return word_timestamps[i][1]
            return word_timestamps[n - 1][1]

        for kind, ref_label, offset, target_nid in _scan_references(
            clause_text, self.book, home_nid,
        ):
            dedup_key = f"{kind}::{ref_label}"
            if dedup_key in self.seen_refs:
                continue
            self.seen_refs.add(dedup_key)
            t = _time_at(offset)
            content = resolve_reference(
                self.book, kind, ref_label, hint_nid=home_nid,
            )
            # Hot path: substitute the LLM-recovered clean LaTeX when
            # we have one for this equation label.  Falls through to
            # the OCR-text monospace render when the cache missed
            # (LLM was down, deadline blew, or text wasn't garbled).
            if (kind == "Equation"
                    and not (content.latex and content.latex.strip())
                    and ref_label in self._eq_latex_cache):
                cached = self._eq_latex_cache[ref_label]
                if cached:
                    from dataclasses import replace as _replace
                    content = _replace(content, latex=cached)
            # Body-cleanup cache hit: the LLM has reformatted this
            # reference's prose+math so the math markers are wrapped in
            # ``\(..\)`` / ``\[..\]``.  Render via foreignObject so
            # KaTeX auto-renders math inline with the surrounding prose.
            prose_html = self._ref_body_cache.get((kind, ref_label), "")
            rendered = _render_reference_card(content, prose_html=prose_html)
            if rendered is None:
                # Render only returns None when there is genuinely no
                # content to show for this reference (kind we don't
                # know how to handle).  Skip silently.
                continue
            svg_body, w, h = rendered
            nid_suffix = re.sub(r"[^A-Za-z0-9]+", "_", ref_label).strip("_")
            ref_nid = f"n_ref_{kind}_{nid_suffix}_{seq}"
            self.chalkboard.add(
                nid=ref_nid, svg_body=svg_body,
                primitive="reference_card", label=f"{kind} {ref_label}",
                w=w, h=h,
                meta={"kind": kind, "ref_label": ref_label,
                      "from_nid": home_nid,
                      "to_nid": content.target_nid or target_nid or "",
                      "has_content": bool(content.text or content.figure)},
            )
            self.seen_nids.add(ref_nid)
            # Equation reference cards with body content count as the
            # current "function on the board" — later clauses that
            # introduce a ``where x is …`` declaration without a fresh
            # formula will attach the definition here.
            if kind == "Equation" and (content.text or content.latex):
                self._last_function_nid = ref_nid
                self._function_card_state[ref_nid] = {
                    "kind": "equation_ref",
                    "ref_label": ref_label,
                    "text": content.text or "",
                    "latex": (content.latex or "").strip(),
                    "cites": [f"Equation {ref_label}"],
                    "var_defs": [],
                }
            # Register spoken-form keys (e.g., "Figure 7.8", "Equation 5.42")
            # so subsequent clauses that re-mention this reference glow
            # the existing card.  For Equation refs that carry the
            # actual LaTeX, also register verbalized function-call
            # phrases so the card lights up when the narrator says the
            # formula's *content*.
            ref_keys = [f"{kind} {ref_label}"]
            if kind == "Equation":
                ref_keys.append(f"({ref_label})")
                ref_keys.append(f"Eq. {ref_label}")
                if content.latex:
                    ref_keys.extend(_verbalized_formula_keys(content.latex))
            self._register_artifact(ref_nid, ref_keys)
            # Phase-0 graph ingest for Equation references that carry
            # body LaTeX — gives the graph a Formula node for cited
            # equations even when the narrator never opened the
            # original.
            if kind == "Equation" and content.latex:
                self._graph_ingest_formula(
                    nid=ref_nid, latex=content.latex,
                    cite_labels=[ref_label], home_nid=home_nid,
                )
            ops.append({
                "t": t, "kind": "add", "nid": ref_nid,
                "svg": svg_body, "label": f"{kind} {ref_label}",
                "primitive": "reference_card", "cid": "",
                "from_corpus": True,
                "to_nid": content.target_nid or target_nid or "",
                "ref_kind": kind,
                "w": w, "h": h,
            })
        return ops


# ---------------------------------------------------------------------------
# Reference detection
# ---------------------------------------------------------------------------

# Patterns we recognise in clause text.  Order matters: longer / more specific
# patterns come first so "Figure 7.8" wins over a bare "(7.8)".
_REF_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Algorithm",  re.compile(r"\bAlgorithm\s+(\d+(?:\.\d+){0,2})\b")),
    ("Figure",     re.compile(r"\bFigure\s+(\d+(?:\.\d+){0,2})\b")),
    ("Table",      re.compile(r"\bTable\s+(\d+(?:\.\d+){0,2})\b")),
    ("Theorem",    re.compile(r"\bTheorem\s+(\d+(?:\.\d+){0,2})\b")),
    ("Lemma",      re.compile(r"\bLemma\s+(\d+(?:\.\d+){0,2})\b")),
    ("Proposition",re.compile(r"\bProposition\s+(\d+(?:\.\d+){0,2})\b")),
    ("Corollary",  re.compile(r"\bCorollary\s+(\d+(?:\.\d+){0,2})\b")),
    ("Definition", re.compile(r"\bDefinition\s+(\d+(?:\.\d+){0,2})\b")),
    ("Example",    re.compile(r"\bExample\s+(\d+(?:\.\d+){0,2})\b")),
    ("Exercise",   re.compile(r"\bExercise\s+(\d+(?:\.\d+){0,2})\b")),
    ("Section",    re.compile(r"\bSection\s+(\d+(?:\.\d+){0,2})\b")),
    ("Chapter",    re.compile(r"\bChapter\s+(\d+)\b")),
    # Equation references: "equation (7.24)", "Eq. 7.24", or bare "(7.24)".
    ("Equation",   re.compile(
        r"\b(?:[Ee]quations?|Eqs?\.?)\s*\(?(\d+\.\d+)\)?"
    )),
    ("Equation",   re.compile(r"\((\d+\.\d+)\)")),
]


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0))
            for m in re.finditer(r"\S+", text)]


_MATH_CHARS_RE = re.compile(
    "[α-ωΑ-Ω=+·×÷≈≠≤≥∑∏∫∂∇√∞±^]"
)
# Boundary on sentence-final punctuation OR blank lines.  Single
# newlines are intentionally NOT a boundary: OCR'd PDFs split a single
# displayed equation across many lines (``min\nf∈HK\nN\nX\ni=1\n
# L(yi, f(xi)) + λ||f||²``) and treating each row as its own chunk
# loses the formula entirely (each row has too few math tokens to
# qualify).  Blank-line gaps still act as boundaries so we don't fuse
# unrelated paragraphs.
_FRAG_BOUNDARY_RE = re.compile(r"[.;]\s+|(?<=\?)\s+|(?<=\!)\s+|\n\s*\n+")


# PyMuPDF renders the ``\sum`` / ``\prod`` / ``\int`` glyphs as
# capital ``X`` (or ``Y`` / ``R``), with the upper bound on the line
# above and the lower bound on the line below.  After single-newline
# flattening the body_text reads ``... = − N X i=1 K X k=1 yik …`` —
# the bounds (``N``, ``K``) and the glyph (``X``) are plain capitals
# that ``_MATH_CHARS_RE`` does not recognise, so the math-cluster
# detector breaks at ``=`` and the rest of the equation is dropped.
# This pre-processor folds the vertical-stack representation back
# into a one-line LaTeX form, so the cluster detector keeps reading
# through the equation and ``to_latex`` produces a faithful
# transcript.
_PYMUPDF_SUM_GLYPHS = {
    "X": r"\sum",     # capital sigma
    "Y": r"\prod",    # capital pi (occasionally)
    "R": r"\int",     # integral, when between bounds with ``d<var>``
}

# ``<upper> X <lower>`` where upper is a single capital letter or a
# small number and lower is ``<var>=<expr>``.  The lower bound is
# required to disambiguate prose like "set X" from a Σ glyph.
_VSTACK_RE = re.compile(
    r"\b([A-Z]|\d{1,3})\s+([XYR])\s+([a-zA-Z]\w*\s*=\s*[^\s,;]{1,12})"
)

# ``<X-glyph> <indices>`` with a multi-index list (``k,m``) or a single
# index (``k``) and NO upper bound.  Common in penalty-style notation
# ``\sum_{k,m} \beta^2_{km}``.  We require:
#   * a comma-separated list of two or more single letters, OR
#   * a single letter followed immediately by a math/Greek-bearing
#     continuation (the integrand) so we don't repair prose like
#     ``the set X k holds``.
# Greek lowercase letters (α-ω, including ``ℓ``) are accepted as
# index names.  ``\b`` before X / Y / R prevents matching inside a
# longer token like ``XYZ``.
_INDEX_CHAR = r"(?:[a-zA-Z]|[α-ω]|ℓ)"
_VSTACK_NOUPPER_RE = re.compile(
    rf"\b([XYR])\s+({_INDEX_CHAR}(?:,{_INDEX_CHAR})+)"
    r"(?=\s+(?:\\?[a-zA-Zα-ω]|\d|\\))"
)

# Transpose-with-subscript: PyMuPDF lays out ``α^T_m X`` as
#   αT
#   mX
# which after newline-flatten reads ``αT mX``.  The ``T`` is the
# superscript, the lowercase letter just after the whitespace is the
# subscript, and the immediately following uppercase letter is the
# matrix/vector the transposed object multiplies.  Convert to
# ``\alpha_m^T X`` (LaTeX-style; KaTeX renders fine).
_GREEK_LOWER_CHARS = "αβγδεζηθικλμνξοπρςστυφχψω"
_TRANSPOSE_SUB_RE = re.compile(
    r"(\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|"
    r"lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)"
    rf"|[{_GREEK_LOWER_CHARS}]|[A-Za-z])T\s+([a-z])\s*([A-Z])\b"
)

# Subscript-T-without-space: ``β T\nk Z`` → ``βT k Z`` (no space
# between glyph and T).  Same fix shape but with the variable possibly
# also broken across the next line, fused on flatten as a single
# token like ``mX`` or ``kZ``.  Already covered by _TRANSPOSE_SUB_RE
# above when there's whitespace between T and the subscript.


def _repair_pymupdf_vstack(text: str) -> str:
    """Fold PyMuPDF's vertical \\sum/\\prod/\\int glyph layout into
    inline LaTeX so the math-fragment detector keeps reading through
    a multi-line equation instead of stopping at the first plain-
    capital bound.  Also repairs the transpose-with-subscript glyph
    layout PyMuPDF emits for ``\\alpha^T_m`` and friends."""
    def _sub_full(m: "re.Match[str]") -> str:
        upper, glyph, lower = m.group(1), m.group(2), m.group(3)
        cmd = _PYMUPDF_SUM_GLYPHS.get(glyph, r"\sum")
        return f"{cmd}_{{{lower}}}^{{{upper}}}"

    def _sub_noupper(m: "re.Match[str]") -> str:
        glyph, indices = m.group(1), m.group(2)
        cmd = _PYMUPDF_SUM_GLYPHS.get(glyph, r"\sum")
        return f"{cmd}_{{{indices}}}"

    def _sub_transpose(m: "re.Match[str]") -> str:
        head, sub, var = m.group(1), m.group(2), m.group(3)
        return f"{head}_{{{sub}}}^{{T}} {var}"

    # Pass 1: ``<upper> X <lower=expr>`` (the bounded form).
    out = _VSTACK_RE.sub(_sub_full, text)
    out = _VSTACK_RE.sub(_sub_full, out)
    # Pass 2: ``X <indices>`` with no upper bound.  Run after the
    # bounded form so we don't match the lower-bound's leading char.
    out = _VSTACK_NOUPPER_RE.sub(_sub_noupper, out)
    out = _VSTACK_NOUPPER_RE.sub(_sub_noupper, out)
    # Pass 3: transpose with subscript.
    out = _TRANSPOSE_SUB_RE.sub(_sub_transpose, out)
    return out


_FUNC_CALL_RE = re.compile(
    # ``f(x)`` / ``L(y, f(x))`` / ``J(f)`` / ``\phi(x)`` / etc.
    # One letter (or LaTeX command) followed by balanced single-level
    # parens.  Used as a *secondary* signal so cards land for clauses
    # like "L(y, f(x)) is the loss" that have no ``=``.
    r"(?:\\[A-Za-z]+|[A-Za-z])\([^()]+(?:\([^()]*\)[^()]*)*\)"
)


def _detect_math_fragments(text: str) -> list[tuple[str, int]]:
    """Return ``[(fragment, char_offset), …]`` for math-rich substrings.

    Strategy: split on sentence-final punctuation; in each chunk,
    tokenise on whitespace, find the tokens that carry math characters,
    and keep that contiguous run plus *one* neighbour-token on each side
    (so ``z`` survives in ``z = W x + b``).  Then drop leading / trailing
    common-English-word tokens.

    Secondary pass: when no math-char-driven fragment is found in a
    chunk, also accept *function-call notation* (``L(y, f(x))``,
    ``J(f)``, …) so formula cards land for clauses where the math is
    pure function-call shape with no ``=`` operator.  This is what the
    user actually says in re-mention clauses ("J(f) is a penalty
    functional"), so without this the card never appears.
    """
    out: list[tuple[str, int]] = []
    pos = 0
    # Flatten single newlines into spaces so that line-broken OCR'd
    # equations (typical in PDFs) survive as a single chunk.  The
    # boundary regex already handles blank-line breaks, so paragraphs
    # don't fuse.
    text = re.sub(r"\n(?!\s*\n)", " ", text)
    # Repair PyMuPDF's vertical-stack \sum/\prod/\int glyphs so the
    # cluster detector reads through a multi-line equation instead
    # of stopping at the first plain-capital bound.
    text = _repair_pymupdf_vstack(text)
    for chunk in _FRAG_BOUNDARY_RE.split(text):
        idx = text.find(chunk, pos)
        if idx < 0:
            idx = pos
        pos = idx + len(chunk)
        if len(chunk) < 4:
            continue
        # Token list with their offsets in chunk.
        token_spans: list[tuple[int, int, str]] = [
            (m.start(), m.end(), m.group(0))
            for m in re.finditer(r"\S+", chunk)
        ]
        if not token_spans:
            continue
        math_idx = [
            i for i, (_s, _e, t) in enumerate(token_spans)
            if _MATH_CHARS_RE.search(t)
        ]
        if not math_idx:
            # Secondary signal: standalone function-call notation
            # — ``L(y, f(x))``, ``J(f)`` — surfaces the formula even
            # when the sentence has no ``=`` / Greek / sum / integral.
            # Only emit when the call has an arg (``f(x)`` not ``f()``)
            # and there's at least one parenthesised content character
            # that looks math-y (a comma, single-letter, or sub/super).
            for fcm in _FUNC_CALL_RE.finditer(chunk):
                cand = fcm.group(0)
                if len(cand) < 4 or len(cand) > 60:
                    continue
                inner = cand[cand.index("(") + 1:cand.rindex(")")]
                if not inner.strip():
                    continue
                # Reject obvious prose like ``According to (Smith)``.
                if re.match(r"^[A-Z][a-z]+(?:\s+[A-Za-z]+)*$", inner):
                    continue
                # Skip when the function head matches a common English
                # word — ``Note(that)`` / ``See(below)`` shouldn't
                # become formula cards.
                head_word = re.match(r"\\?[A-Za-z]+", cand)
                if head_word and head_word.group(0).lower() in {
                    "the", "this", "that", "and", "for", "see", "let",
                    "note", "fig", "table", "eq",
                }:
                    continue
                out.append((cand, idx + fcm.start()))
            continue
        # Cluster math tokens into runs: tokens are in the same cluster
        # if they're within MAX_GAP non-math tokens of each other.
        MAX_GAP = 2
        clusters: list[list[int]] = []
        cur: list[int] = []
        for mi in math_idx:
            if not cur or mi - cur[-1] <= MAX_GAP + 1:
                cur.append(mi)
            else:
                clusters.append(cur)
                cur = [mi]
        if cur:
            clusters.append(cur)
        for cluster in clusters:
            has_eq = any("=" in token_spans[i][2] for i in cluster)
            has_greek = any(
                re.search(r"[α-ωΑ-Ω]", token_spans[i][2])
                for i in cluster
            )
            if not (has_eq or len(cluster) >= 3 or
                    (has_greek and len(cluster) >= 2)):
                continue
            lo = max(0, cluster[0] - 1)
            hi = min(len(token_spans), cluster[-1] + 2)
            # When the cluster ends in a big-operator command (\sum,
            # \prod, \int, \oint), extend the window forward to capture
            # the integrand body — typical patterns are
            # ``\sum_{...}^{...} f(x_i)`` or ``\sum yik log fk(xi),``,
            # where the integrand sits 1–6 tokens past the operator.
            # Without this, ``\sum_{i=1}^{N} \sum_{k=1}^{K} yik`` cuts
            # off ``log fk(xi)`` and the captured LaTeX has no body.
            if cluster:
                last_tok = token_spans[cluster[-1]][2]
                # ``\b`` doesn't fire after ``\sum`` because ``_`` is a
                # word-char in regex — match by next-glyph instead.
                if re.search(
                    r"\\(?:sum|prod|int|oint|bigcup|bigcap)"
                    r"(?:[_^\s\{\(]|$)",
                    last_tok,
                ):
                    j = cluster[-1] + 1
                    extended_to = j
                    while j < len(token_spans) and j < cluster[-1] + 8:
                        t = token_spans[j][2]
                        # Stop at sentence-ending punctuation tokens.
                        if t in {",", ";", ".", "(11.10)"} \
                                or t.startswith("(") and t.rstrip(",").endswith(")") \
                                and re.match(r"^\(\d+(?:\.\d+)?\)[,.;]?$", t):
                            extended_to = j
                            break
                        extended_to = j + 1
                        j += 1
                    hi = max(hi, min(len(token_spans), extended_to))
            win_tokens = token_spans[lo:hi]
            while win_tokens and _is_prose_token(win_tokens[0][2]):
                win_tokens.pop(0)
            while win_tokens and _is_prose_token(win_tokens[-1][2]):
                win_tokens.pop()
            if not win_tokens:
                continue
            fragment = " ".join(t for _s, _e, t in win_tokens).rstrip(",;:.")
            # Multi-sum / multi-integral equations easily run past 100
            # characters once the integrand body is included; the upper
            # cap exists to reject prose runs, not real equations.
            if len(fragment) < 3 or len(fragment) > 220:
                continue
            if not _MATH_CHARS_RE.search(fragment):
                continue
            # Reject prose / pseudocode middles.  KaTeX renders any text
            # with whitespace stripped (math mode), so a fragment like
            # "b = 1 to B: (a) Draw a bootstrap sample Z* of size N"
            # would render as the unreadable run-on
            # "b=1toB:(a)Drawabootstrapsample…".  Same heuristic as the
            # parser's equation gate.
            if _looks_like_pseudocode(fragment):
                continue
            char_offset = idx + win_tokens[0][0]
            out.append((fragment, char_offset))
    return out


# Words that indicate algorithm/pseudocode prose rather than math.
# Includes the short connectors ("to", "do") that appear in steps like
# "for i = 1 to N do" but never in real math expressions.
_PSEUDOCODE_WORDS = frozenset({
    "draw", "drawn", "iterate", "compute", "computed", "sample", "samples",
    "size", "training", "test", "data", "step", "from", "input", "output",
    "where", "let", "such", "that", "this", "these", "those", "for",
    "each", "with", "via", "using", "defined", "denote", "denoted",
    "given", "obtain", "obtained", "return", "yields", "produces",
    "consider", "perform", "until", "while", "loop", "repeat", "of",
    "the", "and", "or", "model", "set",
    "to", "do", "is", "be", "as", "if", "in", "on", "at", "by",
})


def _looks_like_pseudocode(fragment: str) -> bool:
    """True when *fragment* looks like algorithm prose, not math.

    Triggers on:
      * a parenthesised single lower-case letter NOT directly preceded
        by another letter — catches step labels like ``(a)`` while
        leaving function arguments like ``p(x)`` alone.
      * ≥ 2 word-like tokens of length ≥ 2 *outside* parens — content
        inside ``(...)`` is typically a function argument list
        (``L(yi, xi)``, ``f(x)``) and its short alpha runs are math
        subscripts, not prose words.
      * any prose word (≥ 2 letters) outside parens drawn from the
        pseudocode stop list.
      * unbalanced parens / brackets / braces — when the PDF text
        extractor splits an equation mid-bracket we get fragments
        like ``i = 1 L(yi`` that KaTeX renders as italic gibberish.

    Note: a bare colon is no longer an automatic reject.  Many
    legitimate sentences place a formula after a colon ("the kernel
    form: K(f,g) = …"); rejecting them silenced cards the user
    expected to see.  We still catch true pseudocode via the
    word-count + stop-list heuristics below.
    """
    if re.search(r"(?<![A-Za-z])\([a-z]\)", fragment):
        return True
    if not _balanced_brackets(fragment):
        return True
    # Prose-word counting ignores tokens *inside* parens, which are
    # function-arg subscripts in math (``L(yi, f(xi))`` is math, not
    # prose, even though ``yi`` / ``xi`` look like word tokens).
    outside = re.sub(r"\([^()]*(?:\([^()]*\)[^()]*)*\)", "", fragment)
    # LaTeX commands (``\sum``, ``\log``, ``\theta`` …) are math, not
    # prose — strip them before the prose-word count.  Without this,
    # a multi-sum equation like ``R(θ) = − \sum_{i=1}^{N} \sum_{k=1}^{K}
    # yik log fk`` was rejected because ``sum``, ``log``, ``yik`` got
    # counted as prose words.
    outside = re.sub(r"\\[A-Za-z]+", "", outside)
    # Math operator names that PyMuPDF emits unescaped — also drop
    # before counting.  These appear inside formulas, not in algorithm
    # prose.
    outside = re.sub(
        r"\b(?:log|ln|exp|sin|cos|tan|cot|sec|csc|min|max|arg|argmax|"
        r"argmin|sup|inf|lim|det|tr|diag|var|cov|rank|sgn|sign)\b",
        "", outside,
    )
    word_tokens = re.findall(r"[A-Za-z]{2,}", outside)
    # A fragment with a clear math backbone (``\sum``/``\prod``/``\int``
    # or ``=``) is allowed up to two stray word-like tokens, which lets
    # short labels like ``yik`` or ``Var`` ride alongside the formula.
    has_strong_math = bool(re.search(
        r"\\(?:sum|prod|int|partial|nabla|frac)\b|=", fragment
    ))
    word_cap = 2 if has_strong_math else 1
    if len(word_tokens) > word_cap:
        return True
    for w in word_tokens:
        if w.lower() in _PSEUDOCODE_WORDS:
            return True
    return False


# ---------------------------------------------------------------------------
# Companion-context helpers — turn ambient prose around a formula into
# annotations attached to the formula's card.  Two flavours:
#
#   * Equation citations ("(5.42)", "Equation 5.42", "Theorem 3.2"…)
#     present in the same clause → emitted as a small tag *on* the
#     formula card instead of as a separate reference card.
#   * Variable definitions ("where y is the loss function", "where Σ
#     denotes the covariance matrix") → emitted as a sub-line under
#     the rendered formula.
# ---------------------------------------------------------------------------

_EQ_CITE_RE = re.compile(
    r"\b(?:Equation|Eq\.?|Theorem|Lemma|Definition|Algorithm|Table|Figure)\s+"
    r"(\d+(?:\.\d+){0,2})|\((\d+\.\d+)\)"
)


def _equation_citations_in(clause_text: str) -> list[str]:
    """Return human-readable citation labels found in *clause_text*.

    e.g. ``["Equation 5.42", "Theorem 3.2"]``.  Bare ``(5.42)``
    becomes ``Equation 5.42`` for consistency.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _EQ_CITE_RE.finditer(clause_text):
        labelled, bare = m.group(1), m.group(2)
        if labelled:
            kind = m.group(0).split()[0].rstrip(".")
            label = f"{kind} {labelled}"
        elif bare:
            label = f"Equation {bare}"
        else:
            continue
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


# Symbol must look like a math variable: a short identifier
# (≤ 6 letters, covers "sigma", "lambda", "theta"…), a Greek letter,
# a LaTeX command, OR a function-call form like ``f(x)``,
# ``L(y, f(x))``, ``J(f)`` — accepting up to one level of nested
# parens so prose like ``where L(y, f(x)) is the loss`` still
# captures the full ``L(y, f(x))`` symbol with its definition.
_SYM_RE = (
    r"\\[A-Za-z]+"
    r"|[α-ωΑ-Ω]"
    r"|[A-Za-z][A-Za-z0-9_]{0,5}"
    r"(?:\((?:[^()]|\([^()]*\))*\))?"
)

_WHERE_HEAD_RE = re.compile(
    rf"\bwhere\s+(?P<sym>{_SYM_RE})\s+"
    r"(?:is|are|denotes|denote)\s+"
    r"(?P<def>[^.;\n]{3,80}?)(?=[.;\n,]|\s+(?:and\b|where\b)|$)",
    re.I,
)
# Continuations of the "where" clause — accept comma-separated AND
# "and"-separated forms so a sentence like
# ``where L(y, f(x)) is a loss function, J(f) is a penalty
# functional, and H is a space of functions on which J(f) is defined``
# captures all three (L, J, H) — not just L and the trailing H.
_AND_DEF_RE = re.compile(
    rf"(?:,\s*(?:and\s+)?|\band\s+)(?P<sym>{_SYM_RE})\s+"
    r"(?:is|are|denotes|denote)\s+"
    r"(?P<def>[^.;\n]{3,80}?)(?=[.;\n,]|\s+(?:and\b|where\b)|$)",
    re.I,
)


# Common spoken Greek-letter names that pass the symbol filter even
# though they have ≥3 chars.  Plus a few math standbys.
_ALLOWED_LONG_SYMS = frozenset({
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon",
    "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "pi", "rho", "sigma", "tau", "phi", "varphi", "chi",
    "psi", "omega",
    "vec", "bar", "hat", "tilde",
})


def _is_math_symbol(sym: str) -> bool:
    """Conservative check that *sym* looks like a math symbol rather
    than an English noun pulled out of a 'where ...' clause.

    Accepts function-call forms like ``f(x)`` / ``L(y, f(x))`` / ``J(f)``
    so a clause like ``where L(y, f(x)) is the loss function`` keeps
    its full math identifier in the variable-definition.
    """
    if not sym:
        return False
    # Function-call form — ``f(x)``, ``L(y, f(x))`` etc.
    if "(" in sym and sym.count("(") == sym.count(")"):
        head = sym.split("(", 1)[0]
        # Head before the opening paren must look like a short identifier
        # (1-6 chars) so prose-noun heads like ``where parameters(...)``
        # don't sneak through.
        if 1 <= len(head) <= 6 and head[0].isalpha():
            return True
    if len(sym) <= 2:
        return True
    if sym.startswith("\\"):
        return True
    if any(c in "αβγδεζηθικλμνξπρστυφχψω" for c in sym):
        return True
    if sym.lower() in _ALLOWED_LONG_SYMS:
        return True
    return False


def _variable_definitions_in(clause_text: str) -> list[tuple[str, str]]:
    """Return ``[(symbol, definition), …]`` extracted from "where X is Y"
    patterns in *clause_text*.

    Only continues collecting after a ``where`` head; bare ``and X is
    Y`` outside a where-clause is treated as ordinary prose.  Symbol
    must pass ``_is_math_symbol`` so English nouns like "parameters"
    or "models" don't slip through.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _WHERE_HEAD_RE.finditer(clause_text):
        sym = m.group("sym").strip()
        defn = m.group("def").strip().rstrip(",;.")
        if defn and sym and _is_math_symbol(sym) and sym not in seen:
            seen.add(sym)
            out.append((sym, defn))
        # Then look for "and X is Y" continuations *after* this match.
        tail = clause_text[m.end():]
        for am in _AND_DEF_RE.finditer(tail):
            asym = am.group("sym").strip()
            adef = am.group("def").strip().rstrip(",;.")
            if adef and asym and _is_math_symbol(asym) and asym not in seen:
                seen.add(asym)
                out.append((asym, adef))
            # Stop if a sentence boundary appears between matches.
            if re.search(r"[.;]", tail[:am.start()]):
                break
    return out


def _smooth_citation_speech(text: str) -> str:
    """Rewrite citation labels so the TTS reads them as a single
    smooth phrase.

    The narrator's chunker (Kokoro) treats the period inside ``5.42``
    as a sentence terminator and inserts an audible pause, so
    "Equation 5.42 is the regularization functional" comes out as
    "Equation five.  Forty-two is the…".  Substituting ``N.M`` with
    ``N point M`` puts a real word between the digits, the chunker
    stays mid-phrase, and the audio reads as one unit.

    Only triggers inside common citation contexts (``Equation``,
    ``Eq.``, ``(N.M)``, ``Theorem N.M``, …) to avoid mangling
    decimals that should be read normally.
    """
    import re as _re
    if not text:
        return text
    pat = _re.compile(
        r"(\b(?:Equation|Eq\.?|Theorem|Section|Chapter|Figure|Fig\.?)\s+"
        r"|\()(\d+)\.(\d+)",
    )
    return pat.sub(lambda m:
                   f"{m.group(1)}{m.group(2)} point {m.group(3)}",
                   text)


def _norm_cite(s: str) -> str:
    """Strip an ``Equation N.M`` / ``Eq. N.M`` prefix to leave just
    the numeric label.  Used by every code path that needs a single
    canonical key for a citation regardless of which extractor
    produced it (the runtime extractor stores ``"Equation 5.42"``,
    the offline build stores ``"5.42"``)."""
    s = (s or "").strip()
    for p in ("Equation ", "Eq. ", "Eq "):
        if s.lower().startswith(p.lower()):
            return s[len(p):]
    return s


def _split_into_clauses(text: str) -> list[str]:
    """Split a paragraph of prose into spoken clauses.

    Used by the concept-layer rewriter to break L1/L2/L3/L4 prose
    into clause-sized units for the orchestrator's per-clause loop.

    The splitter is sentence-boundary-aware but careful around:
      * Citation labels (``Equation 5.42``, ``(5.42)``) — the dot
        inside a citation should never start a new clause.
      * Common math/scientific abbreviations (``i.e.``, ``e.g.``,
        ``et al.``, ``Eq.``) — same reason.

    Returns a list of clause strings (no empty entries).
    """
    import re as _re
    if not text or not text.strip():
        return []
    # Mask citation-label dots and abbreviation dots so the simple
    # sentence-end regex doesn't fire inside them.
    masked = text
    # ``(5.42)`` / ``(5.43, 5.45)`` — preserve whole token.
    masked = _re.sub(r"\((\d+)\.(\d+)([,\s\d.]*)\)",
                     lambda m: m.group(0).replace(".", ""),
                     masked)
    # ``Equation 5.42`` / ``Eq 5.42`` / ``Eq. 5.42``
    masked = _re.sub(
        r"\b(Equation|Eq\.?|Theorem|Section|Chapter|Figure|Fig\.?)\s*"
        r"(\d+)\.(\d+)",
        lambda m: m.group(0).replace(".", ""),
        masked,
    )
    # Common abbreviations.
    for abbr in ("i.e.", "e.g.", "et al.", "etc.",
                 "vs.", "cf.", "Mr.", "Dr.", "Prof.", "Fig."):
        masked = masked.replace(abbr, abbr.replace(".", ""))
    # Now split on real sentence terminators followed by whitespace +
    # capital or end-of-string.
    parts = _re.split(r"(?<=[.!?])\s+(?=[A-Z(])", masked)
    # Restore the masked dots.
    out = [p.replace("", ".").strip() for p in parts]
    return [p for p in out if p]


def _normalize_formula_key(s: str) -> str:
    """Normalise a LaTeX / formula string to a dedup key.

    Strips all whitespace and surrounding ``\\(``/``\\[`` delimiters,
    so e.g. ``f(x) = a x + b`` and ``f(x)=ax+b`` collapse to the same
    key.  This is for cross-source dedup (intro formulas from the LLM
    vs spoken-formula extraction vs operation phrases).
    """
    if not s:
        return ""
    s = s.strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    return re.sub(r"\s+", "", s)


def _verbalized_formula_keys(latex: str) -> list[str]:
    """Extract distinctive *spoken-form* phrases from a formula's LaTeX
    so the mention scanner can light the card up when the narrator says
    its content (not just its citation label).

    The narrator speaks math by verbalizing it: ``f(x) = \\int k(x,y)
    \\phi(y) dy`` is read as "f of x equals the integral of k of x y
    phi of y dy".  The literal LaTeX never appears in the spoken
    stream, so registering ``"f(x) = \\int…"`` as an artifact key
    matches nothing.

    What *does* appear is **function-call notation**: ``f(x)`` →
    "f of x", ``K(f,g)`` → "K of f g", ``L(y_i, f(x_i))`` →
    "L of y sub i f of x sub i".  These phrases are short, distinctive,
    and the verbalizer is deterministic — so they make excellent
    artifact keys.

    Returns a deduplicated list of phrases.  The orchestrator's
    ``_register_artifact`` filters anything < 3 characters or with no
    letters, so we don't need to be defensive here.
    """
    if not latex:
        return []
    # Lazy import — narrator.qa imports from this package and we want
    # to avoid a circular at module load.
    try:
        from narrator.qa import _sanitize_for_narration
    except Exception:
        _sanitize_for_narration = None  # type: ignore[assignment]
    keys: list[str] = []
    # Function calls: ``name(args)`` where ``name`` is one or more
    # letters (optionally with digits / Greek-via-LaTeX-cmd ``\alpha``).
    # Skip LaTeX commands themselves (``\int(``, ``\sum(``).
    func_pat = re.compile(
        r"(?<!\\)([A-Za-z][A-Za-z0-9]{0,8})\s*\(([^()]+)\)"
    )
    for m in func_pat.finditer(latex):
        name = m.group(1).strip()
        args = m.group(2).strip()
        if not name or not args:
            continue
        # Skip very long arg lists — those usually aren't said verbatim.
        if len(args) > 40:
            continue
        # Verbalize args via the narrator's sanitizer when available.
        if _sanitize_for_narration is not None:
            verb_args = _sanitize_for_narration(args)
        else:
            verb_args = args
        # Strip commas / awkward punctuation; the mention scanner
        # word-boundary regex doesn't include commas anyway.
        verb_args = verb_args.replace(",", " ")
        verb_args = re.sub(r"\s+", " ", verb_args).strip()
        if not verb_args:
            continue
        key = f"{name} of {verb_args}"
        # Don't register pathologically short keys (``f of x`` is OK
        # at 6 chars; anything shorter is over-eager).
        if len(key) < 6:
            continue
        keys.append(key)
    # Dedupe case-insensitively while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        kl = k.lower()
        if kl in seen:
            continue
        seen.add(kl)
        out.append(k)
    return out


def _balanced_brackets(s: str) -> bool:
    """Return True iff every opener has a matching closer in *s*.

    Used to drop math fragments like ``L(yi`` or ``f(x] + b`` that the
    PDF extractor split mid-bracket.  We allow any of ``()``, ``[]``,
    ``{}``.
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


_PROSE_STOP = frozenset({
    "a", "an", "the", "of", "is", "are", "in", "on", "by", "for", "to",
    "and", "or", "with", "where", "such", "that", "this", "these",
    "those", "as", "from", "into", "onto", "at", "be", "if", "then",
    "applied", "before", "after", "represent", "represents", "contains",
    "containing", "denote", "denotes", "denoted", "given", "have", "has",
    "let", "we", "you", "they", "it", "its", "their", "our", "his", "her",
    "instance", "example", "called", "known", "while", "but", "so", "also",
    "use", "using", "used", "implies", "yields", "gives", "produces",
    "via", "thus", "hence", "therefore", "i", "ii", "iii",
})


def _is_prose_token(tok: str) -> bool:
    bare = tok.lower().strip(",.;:()[]{}'\"")
    if not bare:
        return False
    return bare in _PROSE_STOP


def _trim_prose_prefix(s: str) -> str:
    """Drop leading / trailing English-word tokens around the formula,
    so the rendered card holds only the math-bearing slice.
    """
    tokens = s.split()
    while tokens and _is_prose_token(tokens[0]):
        tokens.pop(0)
    while tokens and _is_prose_token(tokens[-1]):
        tokens.pop()
    return " ".join(tokens).strip()


# Card-typography constants — single source of truth so the size
# function and the SVG renderer agree on every offset.  Changing these
# here updates both layout reservation and the rendered glyphs.
#
# The math glyph size (``_MATH_FONT_PX``) is **uniform across every
# card type** — spoken-formula, operation, canonical.  A single math
# size is what makes the chalkboard read as one document.
_MATH_FONT_PX = 36
_FORMULA_HEAD_FONT_PX = 22
_FORMULA_MATH_FONT_PX = _MATH_FONT_PX
_FORMULA_MEANING_FONT_PX = 22
_FORMULA_VARDEF_FONT_PX = 20
_FORMULA_PAD_X = 22.0       # horizontal padding inside the card
_FORMULA_HEAD_BAND = 56.0   # space reserved above the math (header + gap)
_FORMULA_MATH_GAP = 14.0    # gap between math foreignObject and what follows
_FORMULA_BOTTOM_PAD = 16.0  # padding below the last text block
_FORMULA_MEANING_LINE_H = 30.0
_FORMULA_VARDEF_LINE_H = 30.0
_FORMULA_MEANING_CHAR_W = 11.0   # avg glyph width at 22 px italic
_FORMULA_W_MIN = 640.0
_FORMULA_W_MAX = 1240.0


def _math_render_height(latex: str, font_px: float) -> float:
    """Conservative deterministic estimate of KaTeX display-math height.

    KaTeX renders symbols at runtime in the browser; we don't have a
    layout engine here, so we estimate from the LaTeX source by counting
    features that grow vertically (display fractions, big operators with
    limits, multi-row environments).  Returning *too tall* leaves a
    little dead space; returning *too short* clips glyphs — so we err
    on the conservative side.

    The estimator is **deterministic**: same input → same height.  This
    is what lets ``_formula_card_size`` and ``_formula_card_svg``
    reserve exactly the box the math will occupy with no overlap.
    """
    s = latex or ""
    # Base box for the main glyph row, plus generous KaTeX padding.
    # Display math at font_px adds ~0.4× line-height padding above and
    # below; we add a bit more so stretched delimiters never clip.
    h = font_px * 2.0
    # Big operators that *display* their sub/superscript above and below
    # in display style — \sum_{i=1}^N stacks both an upper and a lower
    # limit at ~0.9× font each, plus the operator glyph itself stretches
    # to ~1.5× font in display mode.  Empirically allowing 2.0× covers
    # KaTeX's actual rendered extent with margin.
    two_sided = (r"\sum", r"\int", r"\iint", r"\iiint", r"\oint",
                 r"\prod", r"\coprod", r"\bigcup", r"\bigcap",
                 r"\bigoplus", r"\bigotimes", r"\biguplus",
                 r"\bigvee", r"\bigwedge", r"\bigodot", r"\bigsqcup")
    for op in two_sided:
        n = s.count(op)
        if n:
            h += n * font_px * 2.0
    # \min, \max, \lim, \sup, \inf — limit appears below the operator.
    one_sided = (r"\min", r"\max", r"\lim", r"\sup", r"\inf",
                 r"\argmin", r"\argmax", r"\liminf", r"\limsup")
    for op in one_sided:
        n = s.count(op)
        if n:
            h += n * font_px * 1.0
    # Display fractions — each adds a numerator/denominator stack.
    n_dfrac = (s.count(r"\dfrac") + s.count(r"\cfrac"))
    n_frac_total = s.count(r"\frac")  # includes \dfrac/\cfrac false-positive
    n_frac_only = max(0, n_frac_total - n_dfrac)
    h += n_dfrac * font_px * 1.6
    h += n_frac_only * font_px * 1.4
    # Smaller text-style fractions still add a bit.
    h += s.count(r"\tfrac") * font_px * 0.7
    # Square roots and accents (modest extra height).
    h += s.count(r"\sqrt") * font_px * 0.3
    h += s.count(r"\overline") * font_px * 0.2
    h += s.count(r"\underline") * font_px * 0.2
    h += (s.count(r"\overbrace") + s.count(r"\underbrace")) * font_px * 0.6
    # Multi-row environments (matrices, aligned, cases) — each ``\\``
    # adds a row.  Beware: ``\\`` also appears outside arrays in rare
    # cases, but treating each as a row over-estimates safely.
    n_rows = s.count(r"\\")
    if n_rows:
        h += n_rows * font_px * 1.5
    # Cap pathological inputs (deeply nested fractions / huge matrices).
    return min(h, font_px * 16.0)


def _formula_meaning_lines(meaning: str, w: float) -> int:
    """How many wrapped lines a meaning string occupies inside a card
    of width *w*.  Deterministic — uses the same constants as the
    renderer so reserved height matches rendered height."""
    if not meaning:
        return 0
    inner = max(1.0, w - 2 * _FORMULA_PAD_X)
    chars_per_line = max(20, int(inner / _FORMULA_MEANING_CHAR_W))
    # Honor explicit newlines; rewrap each paragraph.
    total = 0
    for para in meaning.splitlines() or [meaning]:
        if not para:
            total += 1
            continue
        n = (len(para) + chars_per_line - 1) // chars_per_line
        total += max(1, n)
    return total


def _formula_card_size(
    fragment: str,
    cite_labels: Optional[list[str]] = None,
    var_defs: Optional[list[tuple[str, str]]] = None,
    meaning: str = "",
) -> tuple[float, float]:
    """Deterministically compute card (w, h) so that nothing inside
    will overlap.

    Width comes from the longest line of *fragment* (or the longest
    var-def line) and is clamped to a sensible band; height is the sum
    of every reserved sub-block (header band, math box, var-defs, and
    wrapped meaning) plus interior padding.  This function and
    :func:`_formula_card_svg` consult the same module-level constants,
    so the renderer will always fit inside the box the layout reserved.
    """
    fragment = fragment or ""
    cite_labels = cite_labels or []
    var_defs = var_defs or []
    longest = max((len(line) for line in fragment.splitlines()),
                  default=len(fragment))
    longest_def = max(
        (len(f"{sym} = {defn}") for sym, defn in var_defs),
        default=0,
    )
    longest = max(longest, longest_def)
    # Width based on math glyph budget; widen further if the meaning
    # would otherwise wrap to many lines, so the card uses the board's
    # horizontal space instead of stacking text vertically.
    w = max(_FORMULA_W_MIN,
            min(_FORMULA_W_MAX, 28.0 * longest + 140.0))
    if meaning and _formula_meaning_lines(meaning, w) > 2:
        # Try a wider card and re-measure; pick the narrowest width
        # that fits the meaning in <= 3 lines, capped at the max.
        for candidate in (900.0, 1040.0, 1160.0, _FORMULA_W_MAX):
            if candidate <= w:
                continue
            if _formula_meaning_lines(meaning, candidate) <= 3:
                w = candidate
                break
        else:
            w = _FORMULA_W_MAX
    # Reserved vertical sub-blocks.
    math_h = _math_render_height(fragment, _FORMULA_MATH_FONT_PX) + 24.0
    h = _FORMULA_HEAD_BAND + math_h
    if var_defs:
        h += _FORMULA_VARDEF_LINE_H * len(var_defs) + _FORMULA_MATH_GAP
    if meaning:
        h += (_formula_meaning_lines(meaning, w)
              * _FORMULA_MEANING_LINE_H) + _FORMULA_MATH_GAP
    h += _FORMULA_BOTTOM_PAD
    return w, h


_OPERATION_HEAD_BAND = 76.0  # "operation" tag + label sit above the math


def _operation_card_size(latex: str) -> tuple[float, float]:
    """Deterministic size for an operation card.  Same math font and
    estimator as spoken cards so the chalkboard reads at one visual
    weight."""
    longest = max(len(line) for line in latex.split(r"\\")) if latex else 24
    w = max(_FORMULA_W_MIN,
            min(_FORMULA_W_MAX, 22.0 * longest + 110.0))
    math_h = _math_render_height(latex, _MATH_FONT_PX) + 24.0
    h = _OPERATION_HEAD_BAND + math_h + _FORMULA_BOTTOM_PAD
    return w, h


def _operation_card_svg(label: str, latex: str, w: float, h: float) -> str:
    """Card with KaTeX-rendered LaTeX for a named operation."""
    safe_label = (label.replace("&", "&amp;").replace("<", "&lt;")
                       .replace(">", "&gt;").replace('"', "&quot;"))
    safe_latex = (latex.replace("&", "&amp;").replace("<", "&lt;")
                       .replace(">", "&gt;").replace('"', "&quot;"))
    pad = _FORMULA_PAD_X
    math_h = _math_render_height(latex, _MATH_FONT_PX) + 24.0
    return (
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="#e0f2f1" stroke="#00897b" stroke-width="1.4"/>'
        f'<text x="{pad:.1f}" y="32" font-size="18" fill="#00695c" '
        f'font-family="ui-sans-serif,sans-serif">operation</text>'
        f'<text x="{pad:.1f}" y="62" font-size="22" fill="#004d40" '
        f'font-weight="700" '
        f'font-family="ui-sans-serif,sans-serif">{safe_label}</text>'
        f'<foreignObject x="{pad:.1f}" y="{_OPERATION_HEAD_BAND:.1f}" '
        f'width="{w - 2 * pad:.1f}" height="{math_h:.1f}" '
        f'overflow="visible">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" class="math-card" '
        f'data-latex="{safe_latex}" '
        f'style="font-family:KaTeX_Main, ui-serif, serif; '
        f'color:#004d40; padding:0; overflow:visible; '
        f'font-size:{_MATH_FONT_PX}px;">'
        f'\\[{latex}\\]'
        f'</div>'
        f'</foreignObject>'
    )


# --- repaired-latex registry --------------------------------------------
# The math graph stores whatever PyMuPDF extracted; for a non-trivial
# fraction of equations that ends up as a bare ``x =`` or ``i = 1``.
# The chapter-map builder + ``tools.repair_formula_latex`` produce
# correct LaTeX keyed by citation label (``"Equation 5.9"`` →
# full reconstructed string).  We index every ``*.chapter_map.*.json``
# sidecar found in the books directories at startup once and consult
# the index as a fallback when the formula-card's latex looks broken.
_REPAIRED_LATEX_BY_LABEL: dict[str, str] = {}
_REPAIRED_LATEX_LOADED: bool = False


def _is_truncated_latex(s: str) -> bool:
    """Heuristic: does this latex look like an OCR artefact rather
    than a real equation?  Mirrors the rule in
    ``tools.repair_formula_latex._is_broken`` so we trigger the
    fallback on the same shapes the offline repair tool would have
    rewritten."""
    import re as _re_t
    s = (s or "").strip()
    if not s:
        return True
    if "=" in s:
        rhs = s.split("=", 1)[1].strip()
        rhs = _re_t.sub(r",?\s*\(\d+\.\d+\)\s*$", "", rhs).strip()
        # bare capital letter / placeholder
        if _re_t.fullmatch(r"[A-Z]", rhs):
            return True
        if _re_t.fullmatch(r"[A-Z](\s*[+\-]\s*[A-Z])?", rhs):
            return True
        # no math operators / subscripts / Greek on the RHS and the
        # whole thing is short
        if (not _re_t.search(r"[\\_^+\-*/]", rhs)
                and len(rhs) <= 4):
            return True
    elif len(s) <= 6 and not _re_t.search(r"[\\_^+\-*/]", s):
        return True
    return False


def _load_repaired_latex_index() -> None:
    """Walk every ``books/*.chapter_map.*.json`` once and populate
    ``_REPAIRED_LATEX_BY_LABEL`` from the canonical formulas inside.
    Idempotent — protected by ``_REPAIRED_LATEX_LOADED`` so we don't
    re-scan on every formula card."""
    global _REPAIRED_LATEX_LOADED
    if _REPAIRED_LATEX_LOADED:
        return
    _REPAIRED_LATEX_LOADED = True
    import glob as _glob
    import json as _json_t
    import os as _os_t
    candidate_dirs = ["books", _os_t.path.join(_os_t.getcwd(), "books")]
    seen_dirs: set[str] = set()
    for d in candidate_dirs:
        try:
            real = _os_t.path.abspath(d)
        except Exception:
            continue
        if real in seen_dirs or not _os_t.path.isdir(real):
            continue
        seen_dirs.add(real)
        for sidecar in _glob.glob(_os_t.path.join(real,
                                                    "*.chapter_map.*.json")):
            try:
                with open(sidecar) as f:
                    payload = _json_t.load(f)
            except Exception:
                continue
            root = (payload or {}).get("root") or {}

            def _walk(n: dict):
                cf_label = (n.get("canonical_formula_label") or "").strip()
                cf_latex = (n.get("canonical_formula_latex") or "").strip()
                if cf_label and cf_latex and cf_label not in _REPAIRED_LATEX_BY_LABEL:
                    _REPAIRED_LATEX_BY_LABEL[cf_label] = cf_latex
                for c in n.get("children", []) or []:
                    _walk(c)

            if root:
                _walk(root)


def _repaired_latex_for_label(label: str) -> str:
    """Lookup helper: returns the repaired LaTeX for *label* if we
    have one cached, else empty string.  Triggers the one-time
    sidecar scan on first use."""
    if not label:
        return ""
    _load_repaired_latex_index()
    return _REPAIRED_LATEX_BY_LABEL.get(label.strip(), "")


def _formula_card_svg(
    fragment: str, latex: str, w: float, h: float,
    cite_labels: Optional[list[str]] = None,
    var_defs: Optional[list[tuple[str, str]]] = None,
    role: str = "",
    meaning: str = "",
) -> str:
    """Card with KaTeX-rendered LaTeX, plus optional citation chips,
    variable-definition lines, and per-formula explanations.

    Layered annotations (each optional, folded into the card so the
    learner doesn't have to look elsewhere):

      * ``cite_labels`` — top-row chip ("Equation 5.42").
      * ``role``        — short noun phrase from the formula layer's
                          ``F0_role`` (e.g. "regularization functional").
                          Promoted next to the citation chip so the
                          learner names what kind of object this is at
                          a glance.
      * ``meaning``     — one-sentence plain-English description from
                          ``F1_meaning``.  Wraps below the formula in
                          a smaller italic line — that's the "explain
                          each formula" surface the learner reads
                          alongside the symbols.
      * ``var_defs``    — per-symbol definitions when present.
    """
    cite_labels = cite_labels or []
    var_defs = var_defs or []
    # Clean PyMuPDF OCR artefacts ("M X m=1 \beta mhm" → "\sum…",
    # `\hat{\alpha}i` → `\hat{\alpha}_i`, etc.) before KaTeX sees the
    # latex.  When the math graph's stored latex is OBVIOUSLY
    # truncated (a bare RHS like "x =" or "i = 1") the cleaner can't
    # reconstruct anything — fall back to the repaired LaTeX from
    # the chapter-map sidecars (built by ``tools.repair_formula_latex``)
    # by cite-label lookup, so the tangent's spoken-formula card
    # shows the same full equation the narrator pronounces.
    try:
        from viz.treemap import _clean_ocr_latex
        cleaned = _clean_ocr_latex(latex) or latex
    except Exception:
        cleaned = latex
    latex = cleaned
    if cite_labels and _is_truncated_latex(latex):
        repaired = _repaired_latex_for_label(cite_labels[0])
        if repaired:
            latex = repaired
    safe_latex = (latex.replace("&", "&amp;").replace("<", "&lt;")
                       .replace(">", "&gt;").replace('"', "&quot;"))
    parts: list[str] = []
    parts.append(
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="#f3e5f5" stroke="#8e24aa" stroke-width="1.4"/>'
    )
    # Top tag — combines "spoken formula" with citation chips and
    # the formula's role label when one is available.
    head_left = "spoken formula"
    if cite_labels:
        head_left += "  ·  " + ", ".join(cite_labels)
    if role:
        head_left += "  ·  " + role
    safe_head = (head_left.replace("&", "&amp;").replace("<", "&lt;")
                          .replace(">", "&gt;"))
    pad = _FORMULA_PAD_X
    parts.append(
        f'<text x="{pad:.1f}" y="42" font-size="{_FORMULA_HEAD_FONT_PX}" '
        f'fill="#8e24aa" font-weight="600" '
        f'font-family="ui-sans-serif,sans-serif">{safe_head}</text>'
    )
    # Foreign object holds the KaTeX render.  We size the box from the
    # *same* deterministic estimator that ``_formula_card_size`` used
    # to allocate vertical space, so the math draws without clipping
    # and the text below it never collides.
    math_h = _math_render_height(fragment or latex,
                                 _FORMULA_MATH_FONT_PX) + 24.0
    math_y = _FORMULA_HEAD_BAND
    # ``overflow="visible"`` on the foreignObject lets KaTeX glyphs
    # spill past the reserved box if the estimator under-shoots; the
    # SVG group is still anchored at math_y, so glyphs spill *below*
    # the reserved space rather than getting clipped.  The estimator
    # is generous, so spill should be rare — but it's a safety net.
    parts.append(
        f'<foreignObject x="{pad:.1f}" y="{math_y:.1f}" '
        f'width="{w - 2 * pad:.1f}" height="{math_h:.1f}" '
        f'overflow="visible">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" class="math-card" '
        f'data-latex="{safe_latex}" '
        f'style="font-family:KaTeX_Main, ui-serif, serif; '
        f'color:#4a148c; padding:0; overflow:visible; '
        f'font-size:{_FORMULA_MATH_FONT_PX}px;">'
        f'\\[{latex}\\]'
        f'</div>'
        f'</foreignObject>'
    )
    cursor_y = math_y + math_h + _FORMULA_MATH_GAP
    # Plain-English meaning — the "explain this formula" line.  Uses
    # a foreignObject so HTML wrapping handles long text gracefully.
    if meaning:
        n_lines = _formula_meaning_lines(meaning, w)
        meaning_h = n_lines * _FORMULA_MEANING_LINE_H
        safe_meaning = (meaning.replace("&", "&amp;")
                               .replace("<", "&lt;").replace(">", "&gt;"))
        parts.append(
            f'<foreignObject x="{pad:.1f}" y="{cursor_y:.1f}" '
            f'width="{w - 2 * pad:.1f}" height="{meaning_h:.1f}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" '
            f'style="font-family:ui-sans-serif,sans-serif; '
            f'font-size:{_FORMULA_MEANING_FONT_PX}px; line-height:1.35; '
            f'color:#6a1b9a; font-style:italic; '
            f'padding-top:2px;">{safe_meaning}</div>'
            f'</foreignObject>'
        )
    # Variable definitions (legacy fallback when no F1_meaning).
    elif var_defs:
        # baseline of first line sits inside the reserved row.
        y = cursor_y + _FORMULA_VARDEF_FONT_PX
        for sym, defn in var_defs:
            line = f"{sym}: {defn}"
            safe_line = (line.replace("&", "&amp;").replace("<", "&lt;")
                              .replace(">", "&gt;"))
            parts.append(
                f'<text x="{pad:.1f}" y="{y:.1f}" '
                f'font-size="{_FORMULA_VARDEF_FONT_PX}" fill="#4a148c" '
                f'font-family="ui-sans-serif,sans-serif">{safe_line}</text>'
            )
            y += _FORMULA_VARDEF_LINE_H
    return "".join(parts)


def _trigger_time(
    clause_text: str, topic: str, word_timestamps, audio_dur: float,
) -> float:
    """When should the canonical for *topic* appear within this clause?

    We map the topic's most-discriminating keyword to its occurrence in
    *clause_text*, then look up the time of that word from the TTS
    word_timestamps.  Falls back to mid-clause when no match is found.
    """
    from viz.registry import _TOPIC_REGISTRY
    keyword_offset = -1
    for t, _gen, patterns in _TOPIC_REGISTRY:
        if t != topic:
            continue
        for p in patterns:
            m = p.search(clause_text)
            if m:
                keyword_offset = m.start()
                break
        break
    if keyword_offset < 0 or not word_timestamps:
        return max(0.4, audio_dur * 0.4)
    spans = _word_spans(clause_text)
    n = min(len(spans), len(word_timestamps))
    for i in range(n):
        ws, we, _ = spans[i]
        if ws <= keyword_offset < we or keyword_offset < ws:
            return word_timestamps[i][1]
    return max(0.4, audio_dur * 0.4)


# Primitives that are just a labelled rect with no actual geometry.  When
# the resolver falls back to one of these, the resulting card is a empty
# box with text — we'd rather skip it than crowd the board with placeholders.
_PLACEHOLDER_PRIMITIVES = {
    "rect", "node", "label", "box",
    # Bare-concept words that the body-text resolver maps to a generic
    # primitive without meta — these render through ``_render_shape``'s
    # fallback rect (an empty box with the word "vector" / "function" /
    # "scalar" inside).  They add no information beyond the spoken word
    # and clutter the chalkboard alongside the real LLM-spec Tier-2
    # diagram.  The Tier-2 path renders these concepts properly (vector
    # → arrow with arrowhead, function → plotted curve, etc.) so the
    # concept-overlay path is redundant.
    "vector", "function", "scalar", "operation", "set",
    "matrix",  # bare-concept "matrix" without rows/ncols meta
}

# Generic shape primitives that, in absence of meaningful meta (cells,
# labels, nrows/ncols, …), become uninformative scribbles.  E.g. a bare
# `curve` or `segment` with no axes / data is just a colored squiggle.
_GENERIC_VISUAL_PRIMITIVES = {
    "curve", "segment", "arrow", "line",
    "polygon", "circle", "ellipse",
}


def _is_meaningless_shape(rs: ResolvedShape) -> bool:
    """True if this resolution is a glorified text label, not real geometry."""
    if rs.primitive in _PLACEHOLDER_PRIMITIVES:
        return True
    # Empty or 1×1 matrix_bracket — just two brackets with a label, no data.
    if rs.primitive == "matrix_bracket":
        cells = rs.meta.get("cells")
        nrows = rs.meta.get("nrows", 0)
        ncols = rs.meta.get("ncols", 0)
        if not cells and nrows * ncols < 4:
            return True
    # set_blob / category / axes that carry zero label / member info.
    if rs.primitive in ("set_blob", "category", "tensor_box"):
        useful = (rs.meta.get("members") or rs.meta.get("labels")
                  or rs.meta.get("legs") or rs.meta.get("cells"))
        if not useful:
            return True
    return False


def _is_generic_concept_shape(rs: ResolvedShape) -> bool:
    """True for primitives whose meta is too sparse to convey content."""
    if rs.primitive in _GENERIC_VISUAL_PRIMITIVES:
        # Treat as informative only if the template carries non-trivial
        # meta — labels/anchors/cells beyond just a 'kind' marker.
        meta_keys = set(rs.meta.keys()) - {"kind"}
        return len(meta_keys) == 0
    return False


FIGURE_RELEVANCE_THRESHOLD = float(
    os.environ.get("FIGURE_RELEVANCE_THRESHOLD", "0.30")
)

# Per-session caps (env-overridable).  Lifted from 3→8 / 2→3 after the
# user reported "still math concepts told but not shown" — the cap
# was hiding genuinely-different formulas late in the section.  The
# chalkboard's own ``max_content`` FIFO eviction takes over once the
# board is full.
FORMULA_CARDS_MAX = int(os.environ.get("FORMULA_CARDS_MAX", "32"))
CANONICAL_FIGURES_MAX = int(os.environ.get("CANONICAL_FIGURES_MAX", "8"))

# Reference kinds whose body text is long enough to warrant LLM-driven
# math reflow.  Section / Chapter / Algorithm / Table either don't
# carry math at all, or already render cleanly as monospace.
_BODY_LLM_KINDS = frozenset({
    "Exercise", "Theorem", "Lemma", "Proposition", "Corollary",
    "Definition", "Example",
})

# Minimum bbox area (in PDF points squared) for a "figure" to be
# treated as a real diagram.  ESLII (and similar Springer textbooks)
# embed tiny decoration / margin-mark / equation-icon images at
# ~18×25 pt.  Below this threshold we treat them as page furniture
# rather than figures, regardless of caption.
FIGURE_MIN_BBOX_AREA = float(
    os.environ.get("FIGURE_MIN_BBOX_AREA", "5000")
)


def _figure_bbox_area(fig) -> float:
    """Bounding-box area in PDF point² for a :class:`FigureRef`.

    Returns 0 for malformed bboxes so the area check rejects them.
    """
    bb = getattr(fig, "bbox", None)
    if not bb or len(bb) < 4:
        return 0.0
    try:
        w = float(bb[2]) - float(bb[0])
        h = float(bb[3]) - float(bb[1])
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, w) * max(0.0, h)


def _figure_relevance(
    book: Book, fig, query_vec: tuple,
) -> tuple[float, str]:
    """Cosine relevance between a book figure and the question.

    Strategy:
      * If the figure has a non-empty caption, embed it and compare.
      * Otherwise embed the figure's owning passage body_text snippet.
      * If neither is available, return (0.0, "no anchor text").

    Returns ``(score, anchor_kind)``.  ``score`` is cosine in [0, 1];
    ``anchor_kind`` says what we compared the question against (for
    diagnostics + the meta payload).
    """
    try:
        from book import embeddings as _emb
    except Exception:
        return 0.0, "no embeddings module"
    if not _emb.is_available():
        return 0.0, "embedding server unreachable"
    cap = (fig.caption or "").strip()
    if cap:
        v = _emb.embed_text(cap[:600])
        if v:
            return float(_emb.cosine(query_vec, v)), "caption"
    # Fall back to the owning passage body.
    body = ""
    for n in book.root.walk():
        if n.nid == fig.home_nid:
            body = (n.body_text or "")
            break
    if not body:
        return 0.0, "no caption, no body"
    v = _emb.embed_text(body[:1500])
    if not v:
        return 0.0, "embedding failed"
    return float(_emb.cosine(query_vec, v)), "passage_body"


def _figures_for_scope(book: Book, home_nid: str) -> list:
    """Return book figures attached to the *exact* passage subtree.

    Strict-match policy:

      * If ``home_nid`` is at chapter level (``b/chN``) we accept ONLY
        figures whose home_nid equals the chapter exactly.  Descendant
        section figures are excluded — a chapter intro should not pull
        in MNIST examples that happen to live in §11.7.
      * If ``home_nid`` is a section / subsection / deeper, we accept
        the exact node plus its strict descendants.
      * **Sibling fallback**: when a *deep* subsection has no figures
        attached at all, broaden to the parent section's descendants.
        Ingestion sometimes attaches a figure to a sibling subsection
        (e.g. ESLII Figure 17.6 lives under ``ss17_4_2`` while the
        section ``ss17_4_4`` covering RBMs has none).  This recovers
        the right diagram without flooding chapter-intro queries.
    """
    if not home_nid:
        return []
    parts = [p for p in home_nid.split("/") if p]
    chapter_level = len(parts) <= 2  # "b/chN"
    strict: list = []
    for f in book.figures:
        if chapter_level:
            if f.home_nid == home_nid:
                strict.append(f)
        else:
            if f.home_nid == home_nid or f.home_nid.startswith(home_nid + "/"):
                strict.append(f)
    # Drop tiny-bbox icons before considering sibling fallback.
    # ESLII has ~99 such 18×25 pt embeddings (the yellow-Scream meme
    # being one of them).  They are page furniture, not diagrams.
    strict = [f for f in strict
              if _figure_bbox_area(f) >= FIGURE_MIN_BBOX_AREA]
    # Sibling fallback only for *subsection*-or-deeper nids whose
    # parent is itself a section (depth ≥ 4 like ``b/chN/sX/ssY``).
    # At section level (``b/chN/sX``, depth 3) the parent would be the
    # chapter and we'd flood the board with off-topic figures.
    if strict or chapter_level or len(parts) < 4:
        return strict
    parent = "/".join(parts[:-1])
    siblings: list = []
    for f in book.figures:
        if ((f.home_nid == parent or f.home_nid.startswith(parent + "/"))
                and _figure_bbox_area(f) >= FIGURE_MIN_BBOX_AREA):
            siblings.append(f)
    return siblings


def _canonical_card_svg(
    *, inner_svg: str, inner_w: float, inner_h: float,
    topic: str, title: str,
    outer_w: float, outer_h: float,
    inspection_note: str = "", from_vlm: bool = False,
    tier: int = 1, source: str = "",
) -> str:
    """Wrap a synthesised generator SVG in a labelled card with a
    'generated' badge so the user can tell book figures from synthetics.
    """
    if tier == 3 and source == "semantic":
        badge = "deterministic · vision-checked" if from_vlm else "deterministic"
    elif tier == 3:
        badge = "live-LLM · vision-checked" if from_vlm else "live-LLM"
    else:
        badge = "generated · vision-checked" if from_vlm else "generated"
    safe_title = (title.replace("&", "&amp;").replace("<", "&lt;")
                       .replace(">", "&gt;"))
    safe_topic = topic.replace("_", " ")
    return (
        f'<rect x="0" y="0" width="{outer_w:.1f}" height="{outer_h:.1f}" '
        f'rx="6" fill="#fff" stroke="#7e57c2" stroke-width="1.4"/>'
        f'<text x="12" y="20" font-size="11" fill="#5e35b1" '
        f'font-family="ui-sans-serif,sans-serif">{badge} · {safe_topic}</text>'
        f'<text x="12" y="40" font-size="14" fill="#212121" font-weight="600" '
        f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
        f'<svg x="12" y="48" width="{inner_w:.1f}" height="{inner_h:.1f}" '
        f'viewBox="0 0 {inner_w:.0f} {inner_h:.0f}">{inner_svg}</svg>'
    )


def _book_figure_card_svg(fig, owning_node, w: float, h: float) -> str:
    """Card with the actual page bitmap embedded as <image href>."""
    href = f"/api/figure/{fig.fid}"
    pg = f"p.{fig.page}" if fig.page else ""
    section_label = ""
    if owning_node is not None:
        num = (owning_node.number or "").strip()
        title = (owning_node.title or "").strip()
        if num and title:
            section_label = f"§{num} {title}"
        elif title:
            section_label = title
    cap = (fig.caption or "").strip()
    if not cap and section_label:
        cap = f"figure from {section_label}"
    if not cap:
        cap = pg or "figure"
    safe_cap = cap.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if len(safe_cap) > 64:
        safe_cap = safe_cap[:62] + "…"
    return (
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="#fff" stroke="#90a4ae" stroke-width="1.4"/>'
        f'<text x="12" y="20" font-size="11" fill="#546e7a" '
        f'font-family="ui-sans-serif,sans-serif">book figure · {pg}</text>'
        f'<image href="{href}" x="12" y="28" width="{w - 24:.1f}" height="{h - 56:.1f}" '
        f'preserveAspectRatio="xMidYMid meet"/>'
        f'<text x="12" y="{h - 10:.1f}" font-size="11" fill="#37474f" '
        f'font-family="ui-sans-serif,sans-serif">{safe_cap}</text>'
    )


def _scan_references(
    text: str, book: Book, home_nid: str,
) -> list[tuple[str, str, int, Optional[str]]]:
    """Find every ``(kind, ref_label, offset, target_nid_or_None)``
    in *text*, in source order.  Looks up target nids in ``book.cross_refs``
    when one originates near the same home_nid + label.
    """
    out: list[tuple[str, str, int, Optional[str]]] = []
    consumed: list[tuple[int, int]] = []  # spans already claimed

    def _overlaps(a: tuple[int, int]) -> bool:
        for b in consumed:
            if a[0] < b[1] and b[0] < a[1]:
                return True
        return False

    # Build a label→to_nid lookup biased to the current home_nid.
    if home_nid:
        local_refs = {
            cr.label: cr.to_nid for cr in book.cross_refs
            if cr.from_nid == home_nid
        }
    else:
        local_refs = {}

    for kind, pat in _REF_PATTERNS:
        for m in pat.finditer(text):
            span = (m.start(), m.end())
            if _overlaps(span):
                continue
            consumed.append(span)
            ref_label = m.group(1)
            full_label = f"{kind} {ref_label}"
            target = local_refs.get(full_label)
            out.append((kind, ref_label, m.start(), target))
    out.sort(key=lambda r: r[2])
    return out


_REF_PALETTE = {
    "Algorithm":   ("#e0f7fa", "#00838f", "#006064"),
    "Figure":      ("#e3f2fd", "#1976d2", "#0d47a1"),
    "Table":       ("#f3e5f5", "#8e24aa", "#4a148c"),
    "Equation":    ("#e8f5e9", "#388e3c", "#1b5e20"),
    "Theorem":     ("#fff3e0", "#f57c00", "#e65100"),
    "Lemma":       ("#fff3e0", "#f57c00", "#e65100"),
    "Proposition": ("#fff3e0", "#f57c00", "#e65100"),
    "Corollary":   ("#fff3e0", "#f57c00", "#e65100"),
    "Definition":  ("#ede7f6", "#5e35b1", "#311b92"),
    "Example":     ("#fce4ec", "#d81b60", "#880e4f"),
    "Exercise":    ("#f1f8e9", "#689f38", "#33691e"),
    "Section":     ("#eceff1", "#546e7a", "#263238"),
    "Chapter":     ("#eceff1", "#37474f", "#263238"),
}


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _wrap_lines(text: str, max_chars: int) -> list[str]:
    """Soft-wrap *text* preserving original line breaks, splitting overlong
    lines on whitespace.  No tabs, no fancy hyphenation."""
    out: list[str] = []
    for raw in text.splitlines():
        if len(raw) <= max_chars:
            out.append(raw)
            continue
        words = raw.split(" ")
        cur = ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                out.append(cur)
                cur = w
        if cur:
            out.append(cur)
    return out


def _looks_garbled_equation(text: str) -> bool:
    """True when the OCR-extracted equation looks like fragmented PDF
    text rather than a coherent equation.

    Triggered shapes:

    * Many short lines (β = Σ uⱼ (dⱼ²/(dⱼ²+λ)) uⱼᵀ y becomes
      ``j=1 / uj / d2 / j / d2 / j + λuT / j y,``).
    * Single-line equations with no LHS — e.g. ``i/γi < ∞,`` —
      where rendering as math shows half an inequality with no
      definition.  We require ``=``, ``\\to``, or two operands
      around the comparator before accepting a one-line eq.
    * Trailing-comma-only fragments that are clearly continuations
      of a larger equation cut by the OCR pass.
    """
    if not text or not text.strip():
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    short = sum(1 for ln in lines if len(ln) <= 3)
    if short >= max(2, len(lines) // 2):
        return True
    if len(lines) >= 4:
        avg = sum(len(ln) for ln in lines) / len(lines)
        if avg <= 5.0:
            return True
    # Single-line fragment heuristics.
    if len(lines) == 1:
        ln = lines[0]
        # No comparator at all → not a real equation/inequality.
        has_cmp = bool(re.search(
            r"=|<|>|≤|≥|\\(?:to|rightarrow|mapsto|le|ge|ne)\b", ln,
        ))
        if not has_cmp:
            return True
        # Trailing-comma after a comparator is the signature of a
        # fragment cut from a longer equation (``... < ∞,``).
        if re.search(r"[<>≤≥]\s*\S+,\s*$", ln):
            return True
    return False


def _render_uncertain_equation_card(
    kind: str, ref_label: str, *,
    fill: str, stroke: str, ink: str,
    xml_escape,
) -> tuple[str, float, float]:
    """Plain-text card for an Equation reference whose extracted body
    is too garbled to KaTeX-render.

    Renders just the equation tag and a one-liner pointing to the book
    so the user knows the reference exists without seeing scrambled
    italicised glyph soup on the board.
    """
    title = f"{kind} {ref_label}"
    safe_title = xml_escape(title)
    note = f"see book — {title}"
    safe_note = xml_escape(note)
    w, h = 320.0, 110.0
    body = (
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        f'<text x="12" y="20" font-size="11" fill="{stroke}" '
        f'font-family="ui-sans-serif,sans-serif">{xml_escape(kind.lower())}</text>'
        f'<text x="12" y="44" font-size="14" fill="{ink}" font-weight="600" '
        f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
        f'<text x="12" y="76" font-size="12" fill="{ink}" '
        f'font-family="ui-sans-serif,sans-serif">{safe_note}</text>'
    )
    return body, w, h


def _render_text_equation_card(
    kind: str, ref_label: str, raw_text: str, *,
    fill: str, stroke: str, ink: str,
) -> tuple[str, float, float]:
    """Render an Equation reference's extracted body as plain monospace
    text — used when KaTeX can't make sense of the OCR (multi-line
    fragments, single-symbol lines).

    The user asked to see *the actual function from the book* even
    when the OCR is messy.  We strip the trailing ``(N.M)`` marker
    and the cosmetic ``"`` / ``#`` artefacts the PDF text extractor
    leaves behind, then lay each remaining line out monospaced so
    operators and Greek letters survive readable.
    """
    title = f"{kind} {ref_label}"
    safe_title = _xml_escape(title)
    # Clean up obvious extraction artefacts: leading/trailing markers,
    # the literal ``"`` pseudo-summation sign and the ``#`` placeholder
    # that some PDFs use for the (N.M) anchor.  We keep Σ / λ / etc.
    lines: list[str] = []
    for ln in raw_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s in ('"', "#") or re.fullmatch(r"\(\d+\.\d+\)", s):
            continue
        s = s.rstrip(",")
        lines.append(s)
    if not lines:
        # No usable text — fall back to the see-book stub.
        return _render_uncertain_equation_card(
            kind, ref_label, fill=fill, stroke=stroke, ink=ink,
            xml_escape=_xml_escape,
        )
    longest = max(len(ln) for ln in lines)
    w = max(280.0, min(560.0, 9.0 * longest + 40.0))
    line_h = 18.0
    h = 56.0 + line_h * len(lines) + 12.0
    parts: list[str] = []
    parts.append(
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        f'<text x="12" y="20" font-size="11" fill="{stroke}" '
        f'font-family="ui-sans-serif,sans-serif">{_xml_escape(kind.lower())}</text>'
        f'<text x="12" y="40" font-size="14" fill="{ink}" font-weight="600" '
        f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
    )
    y = 56.0 + line_h
    for ln in lines:
        safe = _xml_escape(ln)
        parts.append(
            f'<text x="12" y="{y:.1f}" font-size="13" fill="{ink}" '
            f'font-family="ui-monospace,monospace">{safe}</text>'
        )
        y += line_h
    return "".join(parts), w, h


def _render_equation_ref_card_with_annotations(
    *, ref_label: str, text: str, latex: str,
    cite_labels: list[str], var_defs: list[tuple[str, str]],
) -> tuple[str, float, float]:
    """Re-render an Equation reference card with extra citations and
    variable definitions folded in.  Used by the cross-clause
    annotation attachment when a later clause's
    ``where x is …`` declarations belong on this card.
    """
    fill, stroke, ink = _REF_PALETTE["Equation"]
    title = f"Equation {ref_label}"
    safe_title = _xml_escape(title)

    # Choose the body rendering: KaTeX foreignObject when we have any
    # latex (sidecar OR LLM-recovered), monospace lines when neither.
    use_katex = bool(latex.strip())
    body_lines: list[str] = []
    if not use_katex:
        for ln in (text or "").splitlines():
            s = ln.strip()
            if not s or s in ('"', "#"):
                continue
            if re.fullmatch(r"\(\d+\.\d+\)", s):
                continue
            body_lines.append(s.rstrip(","))

    # Width: based on the longest content line (latex or text or var_def).
    longest_chars = len(safe_title)
    longest_chars = max(longest_chars,
                        max((len(ln) for ln in body_lines), default=0))
    longest_chars = max(longest_chars,
                        max((len(f"{s}: {d}") for s, d in var_defs),
                            default=0))
    if cite_labels:
        longest_chars = max(longest_chars, len(", ".join(cite_labels)) + 8)
    w = max(320.0, min(720.0, 9.0 * longest_chars + 40.0))

    # Height: header + body + var_defs + extra cites.
    head_h = 48.0
    body_h = (38.0 * max(len(latex.split(r'\\')) if use_katex else 1,
                          len(body_lines), 1) + 16.0
              if use_katex else 18.0 * len(body_lines) + 12.0)
    cite_h = 16.0 if cite_labels else 0.0
    var_h = (16.0 * len(var_defs) + 8.0) if var_defs else 0.0
    h = max(110.0, head_h + body_h + cite_h + var_h)

    parts: list[str] = []
    parts.append(
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
    )
    head_left = "equation"
    if cite_labels:
        head_left += "  ·  " + ", ".join(cite_labels)
    parts.append(
        f'<text x="12" y="20" font-size="11" fill="{stroke}" '
        f'font-family="ui-sans-serif,sans-serif">'
        f'{_xml_escape(head_left)}</text>'
    )
    parts.append(
        f'<text x="12" y="40" font-size="14" fill="{ink}" font-weight="600" '
        f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
    )

    if use_katex:
        safe_latex = _xml_escape(latex)
        fo_w = w - 24.0
        fo_h = body_h
        parts.append(
            f'<foreignObject x="12" y="48" width="{fo_w:.1f}" height="{fo_h:.1f}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="math-card" '
            f'data-latex="{safe_latex}" '
            f'style="font-family:KaTeX_Main, ui-serif, serif; '
            f'color:{ink}; padding:4px 0; overflow:hidden; '
            f'max-width:{fo_w:.1f}px; max-height:{fo_h:.1f}px; '
            f'font-size:14px; line-height:1.25;">'
            f'\\[{latex}\\]'
            f'</div>'
            f'</foreignObject>'
        )
        y = head_h + body_h + 16.0
    else:
        y = 56.0 + 18.0
        for ln in body_lines:
            parts.append(
                f'<text x="12" y="{y:.1f}" font-size="13" fill="{ink}" '
                f'font-family="ui-monospace,monospace">{_xml_escape(ln)}</text>'
            )
            y += 18.0
        y += 4.0

    for sym, defn in var_defs:
        parts.append(
            f'<text x="12" y="{y:.1f}" font-size="11" fill="{ink}" '
            f'font-family="ui-sans-serif,sans-serif">'
            f'{_xml_escape(f"{sym}: {defn}")}</text>'
        )
        y += 16.0

    return "".join(parts), w, h


def _render_math_note_card(
    cite_labels: list[str], var_defs: list[tuple[str, str]],
) -> tuple[str, float, float]:
    """Standalone card listing variable definitions / equation IDs
    when no function card exists yet.  Acts as a safety net so a
    declaration spoken before any formula has surfaced isn't lost.
    """
    fill, stroke, ink = "#fffde7", "#fbc02d", "#5d4037"
    lines: list[str] = []
    if cite_labels:
        lines.append("cited: " + ", ".join(cite_labels))
    for sym, defn in var_defs:
        lines.append(f"{sym}: {defn}")
    if not lines:
        lines.append("(empty)")
    longest = max(len(ln) for ln in lines)
    w = max(280.0, min(560.0, 9.0 * longest + 40.0))
    h = 44.0 + 18.0 * len(lines) + 12.0
    parts: list[str] = []
    parts.append(
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="12" y="20" font-size="11" fill="{stroke}" '
        f'font-family="ui-sans-serif,sans-serif">math notation</text>'
    )
    y = 40.0
    for ln in lines:
        parts.append(
            f'<text x="12" y="{y:.1f}" font-size="13" fill="{ink}" '
            f'font-family="ui-sans-serif,sans-serif">'
            f'{_xml_escape(ln)}</text>'
        )
        y += 18.0
    return "".join(parts), w, h


def _render_prose_math_card(
    *, kind: str, ref_label: str, prose_html: str,
    fill: str, stroke: str, ink: str,
) -> tuple[str, float, float]:
    """Render a reference-card body that contains LLM-cleaned prose
    with math wrapped in ``\\(..\\)`` / ``\\[..\\]`` delimiters.

    The body lives inside an ``<foreignObject>`` carrying the
    ``math-prose`` class — the frontend's ``runMathAutoRender`` walks
    every ``.math-prose`` element after insertion and asks KaTeX to
    compile any math markers in place.  Result: the user sees real
    rendered math (Σ, fractions, ⟨ ⟩) inline with the surrounding
    English, instead of a wall of monospace OCR fragments.
    """
    title = f"{kind} {ref_label}"
    safe_title = _xml_escape(title)
    # Width: fixed wide reading column — KaTeX rendering is wider than
    # monospace so we err on the side of generous horizontal space.
    w = 560.0
    # Estimate height from the cleaned text: count actual paragraph
    # breaks (each \[..\] block adds ~40 px; each prose line ~22 px).
    n_display = prose_html.count(r"\[")
    n_lines = max(1, prose_html.count("\n") + 1)
    base_h = 56.0
    body_h = max(80.0, 22.0 * n_lines + 40.0 * n_display + 24.0)
    h = min(540.0, base_h + body_h)
    fo_w = w - 24.0
    fo_h = h - base_h - 8.0
    # Escape HTML metacharacters only — preserve LaTeX delimiters.
    safe_body = (prose_html.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;"))
    parts: list[str] = []
    parts.append(
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        f'<text x="12" y="20" font-size="11" fill="{stroke}" '
        f'font-family="ui-sans-serif,sans-serif">'
        f'{_xml_escape(kind.lower())}</text>'
        f'<text x="12" y="40" font-size="14" fill="{ink}" font-weight="600" '
        f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
    )
    parts.append(
        f'<foreignObject x="12" y="{base_h - 8:.1f}" '
        f'width="{fo_w:.1f}" height="{fo_h:.1f}">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" class="math-prose" '
        f'style="font-family:ui-serif, Georgia, serif; '
        f'color:{ink}; font-size:13px; line-height:1.45; '
        f'overflow:hidden; white-space:pre-wrap; '
        f'word-wrap:break-word;">'
        f'{safe_body}'
        f'</div>'
        f'</foreignObject>'
    )
    return "".join(parts), w, h


def _render_reference_card(
    content: RefContent,
    *,
    prose_html: str = "",
) -> Optional[tuple[str, float, float]]:
    """Render a reference card SVG body, sized to its content.

    When ``prose_html`` is non-empty (LLM-cleaned reference body with
    math wrapped in ``\\(..\\)`` / ``\\[..\\]``), the text-content
    branch renders the body inside a foreignObject so the frontend's
    KaTeX auto-render compiles the math inline with the prose.
    """
    fill, stroke, ink = _REF_PALETTE.get(
        content.kind, ("#fafafa", "#9e9e9e", "#212121"),
    )
    title = f"{content.kind} {content.ref_label}"
    safe_title = _xml_escape(title)

    # ---- Figure with image ------------------------------------------------
    if content.figure is not None:
        w, h = 720.0, 480.0
        href = f"/api/figure/{_xml_escape(content.figure.fid)}"
        cap = _xml_escape((content.figure.caption or "").strip())
        if len(cap) > 60:
            cap = cap[:57] + "…"
        body = (
            f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
            f'<text x="12" y="20" font-size="11" fill="{stroke}" '
            f'font-family="ui-sans-serif,sans-serif">{_xml_escape(content.kind.lower())}</text>'
            f'<text x="12" y="40" font-size="14" fill="{ink}" font-weight="600" '
            f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
            f'<image href="{href}" x="12" y="50" width="{w - 24:.1f}" height="{h - 80:.1f}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
        if cap:
            body += (
                f'<text x="12" y="{h - 10:.1f}" font-size="11" fill="{ink}" '
                f'font-family="ui-sans-serif,sans-serif">{cap}</text>'
            )
        return body, w, h

    # ---- Equation card: render real math via KaTeX in a foreignObject ----
    if (content.text or content.latex) and content.kind == "Equation":
        # If the OCR sidecar (Phase 3) gave us proper LaTeX, use that;
        # otherwise fall back to the heuristic Unicode→LaTeX conversion.
        latex = content.latex.strip() if content.latex else to_latex(content.text)
        # When the heuristic-converted LaTeX looks like fragmented PDF
        # text (multi-line single-character tokens) AND we have no
        # clean sidecar LaTeX, KaTeX would render gibberish.  Skip
        # KaTeX and render the extracted text as monospace lines so
        # the user still sees the actual equation body — earlier
        # versions returned a "see book" placeholder, but the user
        # asked for "the actual functions from the book," so we
        # surface whatever the OCR gave us.
        if not (content.latex and content.latex.strip()) \
                and _looks_garbled_equation(content.text or ""):
            return _render_text_equation_card(
                content.kind, content.ref_label, content.text,
                fill=fill, stroke=stroke, ink=ink,
            )
        # Width: based on the longest *line* (not total length) so multi-
        # line LaTeX doesn't blow up width.
        text_lines = [ln for ln in content.text.splitlines() if ln]
        latex_segs = [s.strip() for s in latex.split(r"\\") if s.strip()]
        max_chars = max(
            (len(ln) for ln in text_lines + latex_segs),
            default=20,
        )
        w = max(320.0, min(720.0, 9.0 * max_chars + 40.0))
        # Height: grows with line count so KaTeX has room to render.
        # Each \\ separator in LaTeX becomes a new display line; KaTeX
        # display mode renders ~36px per line at ~17px font.
        n_lines = max(len(latex_segs), len(text_lines), 1)
        # Headers + padding take ~64px; each line ~38px.
        h = max(110.0, min(560.0, 64.0 + 38.0 * n_lines + 16.0))
        # Cap KaTeX font-size so a wide formula stays inside the foreignObject
        # even when n_lines is small.  Display mode at 14px keeps room for
        # subscripts/superscripts.
        font_px = 14
        # The foreignObject holds an HTML <div> with the LaTeX source as
        # text content (delimited by \(...\)) plus a data-latex fallback.
        # The frontend's auto-render extension picks this up after insert.
        safe_latex = _xml_escape(latex)
        fo_w = w - 24.0
        fo_h = h - 56.0
        body = (
            f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
            f'<text x="12" y="20" font-size="11" fill="{stroke}" '
            f'font-family="ui-sans-serif,sans-serif">{_xml_escape(content.kind.lower())}</text>'
            f'<text x="12" y="40" font-size="14" fill="{ink}" font-weight="600" '
            f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
            f'<foreignObject x="12" y="48" width="{fo_w:.1f}" height="{fo_h:.1f}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="math-card" '
            f'data-latex="{safe_latex}" '
            f'style="font-family:KaTeX_Main, ui-serif, serif; '
            f'color:{ink}; padding:4px 0; overflow:hidden; '
            f'max-width:{fo_w:.1f}px; max-height:{fo_h:.1f}px; '
            f'font-size:{font_px}px; line-height:1.25;">'
            f'\\[{latex}\\]'
            f'</div>'
            f'</foreignObject>'
        )
        return body, w, h

    # ---- Text-content card -----------------------------------------------
    if content.text:
        # KaTeX-aware rendering when an LLM-cleaned body is available —
        # math markers (``\(..\)`` / ``\[..\]``) flow through the
        # frontend's ``runMathAutoRender`` so prose and compiled math
        # share a single foreignObject.
        if prose_html and prose_html.strip():
            return _render_prose_math_card(
                kind=content.kind, ref_label=content.ref_label,
                prose_html=prose_html,
                fill=fill, stroke=stroke, ink=ink,
            )
        # Pick width/height by longest line.
        wrapped = _wrap_lines(content.text, max_chars=64)
        # Cap rendering to avoid huge cards.
        if len(wrapped) > 18:
            wrapped = wrapped[:17] + ["…"]
        line_h = 14.5
        max_chars = max((len(line) for line in wrapped), default=20)
        w = max(280.0, min(680.0, 8.0 * max_chars + 28.0))
        h = 56.0 + line_h * len(wrapped) + 8.0
        body = (
            f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
            f'<text x="12" y="20" font-size="11" fill="{stroke}" '
            f'font-family="ui-sans-serif,sans-serif">{_xml_escape(content.kind.lower())}</text>'
            f'<text x="12" y="40" font-size="14" fill="{ink}" font-weight="600" '
            f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
        )
        y = 56.0
        for line in wrapped:
            body += (
                f'<text x="12" y="{y:.1f}" font-size="12" fill="#222" '
                f'font-family="ui-monospace,monospace">{_xml_escape(line)}</text>'
            )
            y += line_h
        return body, w, h

    # ---- Label-only fallback ---------------------------------------------
    w, h = 220.0, 64.0
    body = (
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        f'<text x="12" y="20" font-size="11" fill="{stroke}" '
        f'font-family="ui-sans-serif,sans-serif">reference · {_xml_escape(content.kind.lower())}</text>'
        f'<text x="12" y="44" font-size="14" fill="{ink}" font-weight="600" '
        f'font-family="ui-sans-serif,sans-serif">{safe_title}</text>'
    )
    return body, w, h


def _passage_card_svg(label: str, kind: str, w: float, h: float) -> str:
    """Render the passage-card body (no outer <svg>; the board wraps it)."""
    safe = (label.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;"))
    if len(safe) > 38:
        safe = safe[:36] + "…"
    return (
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="#fff8e1" stroke="#fbc02d" stroke-width="1.4"/>'
        f'<text x="12" y="20" font-size="11" fill="#827717" '
        f'font-family="ui-sans-serif,sans-serif">passage · {kind}</text>'
        f'<text x="12" y="44" font-size="14" fill="#222" font-weight="600" '
        f'font-family="ui-sans-serif,sans-serif">{safe}</text>'
    )


def render_resolved_g(rs: ResolvedShape) -> str:
    """Render a ResolvedShape and strip the outer <svg> wrapper.

    The chalkboard composes shapes inside a single <svg>, so each shape
    body is wrapped in <g transform="translate(...)"> instead of having
    its own document.
    """
    full = render_resolved(rs)
    # Find the inner content between the outer <svg ...> and </svg>.
    start = full.index(">", full.index("<svg")) + 1
    end = full.rindex("</svg>")
    inner = full[start:end]
    return inner


def _estimate_w(rs: ResolvedShape) -> float:
    # Match the size overrides in s3_map._math_size_override approximately.
    p = rs.primitive
    if p in ("point", "decoration_mark"):
        return 60.0
    if p == "matrix_bracket":
        ncols = rs.meta.get("ncols", 2)
        return 36.0 * ncols + 28.0 + 40.0
    if p == "axes" or p == "axes_3d":
        return 240.0
    if p == "set_blob":
        return 180.0
    if p == "equation_block":
        return 240.0
    return 160.0


def _estimate_h(rs: ResolvedShape) -> float:
    p = rs.primitive
    if p in ("point", "decoration_mark"):
        return 60.0
    if p == "matrix_bracket":
        nrows = rs.meta.get("nrows", 2)
        return 28.0 * nrows + 8.0 + 20.0
    if p in ("axes", "axes_3d", "set_blob"):
        return 160.0
    if p == "equation_block":
        return 90.0
    return 90.0
