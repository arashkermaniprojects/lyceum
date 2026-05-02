"""Session — main narration + tangent panel + pause/resume control.

A session wraps two narration streams that can run independently:

  * **main**     — the primary narration (topic-driven or read-the-book).
  * **tangent**  — an optional Q&A diversion that pre-empts main.

The session exposes a single ``next_event()`` generator that the HTTP/SSE
layer consumes.  Internally:

  * ``pause()``         — halts dispatch; ``next_event()`` blocks/returns None.
  * ``resume()``        — re-enables dispatch.
  * ``ask(question)``   — synthesises a tangent NarrationPlan via
                          ``narrator.qa.answer`` and queues it as the
                          active stream.  Tangent events come out tagged
                          with ``panel="tangent"`` so the frontend routes
                          them to the side panel.  When the tangent ends,
                          dispatch returns to the main iterator.
  * ``cancel()``        — kills both streams.

Each event carries a ``panel`` tag so the frontend can route it to the
correct chalkboard.

Determinism note: pause/resume changes *what* events are dispatched and
when, but not *what data* each event contains.  The session does not
introduce any wall-clock dependency in event payloads.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from book.ir import Book
from chalkboard import Chalkboard, ReadingOrderPolicy
from narrator import NarrationPlan, qa as qa_mod
from narrator import router as router_mod
from narrator.tts import _BackendBase, NullTTS

from .orchestrator import Orchestrator, StreamEvent


# ---------------------------------------------------------------------------
# PanelEvent — wraps StreamEvent with routing metadata
# ---------------------------------------------------------------------------

@dataclass
class PanelEvent:
    """A StreamEvent + the panel it should be displayed in."""
    panel: str   # "main" | "tangent"
    event: StreamEvent
    is_tangent_start: bool = False
    is_tangent_end: bool = False
    tangent_id: str = ""


@dataclass
class DialogueTurn:
    """One user turn in the session's dialogue history.

    Used by ``Session.ask`` to expand follow-ups against the most
    recent focus, and by future planners that need to know what
    concepts have already been narrated.
    """
    user_text: str = ""
    intent: str = ""           # one of router.INTENT_* constants
    focus_nid: str = ""        # last focused BookNode (chapter / section)
    focus_topic: str = ""      # last topic phrase
    plan_id: str = ""          # tangent_id this turn produced (if any)
    t: float = 0.0             # session-relative timestamp


@dataclass
class SessionKnowledge:
    """What the user has been shown so far, across every tangent.

    The Session creates one of these and shares the underlying set
    objects with every Orchestrator it instantiates.  Both the main
    orchestrator and any tangent orchestrators mutate the *same*
    sets, so a primitive (function plot, formula card, canonical
    figure, equation reference) emitted in one tangent won't be
    re-emitted in the next.

    The reverse — "show me X again" — is handled by
    ``Session.snapshot_then_clear`` / ``restore`` so the orchestrator
    can temporarily ignore a subset of these for a single re-show
    turn without forgetting everything.
    """
    seen_nids: set[str] = field(default_factory=set)
    seen_home_nids: set[str] = field(default_factory=set)
    seen_refs: set[str] = field(default_factory=set)
    seen_figure_fids: set[str] = field(default_factory=set)
    seen_canonical_topics: set[str] = field(default_factory=set)
    seen_formulas: set[str] = field(default_factory=set)
    seen_semantic_keys: set[str] = field(default_factory=set)
    # Topics announced in dialogue (router-resolved) — so a recap can
    # cite the actual phrases the user asked about, not just the
    # concepts the orchestrator emitted as cards.
    spoken_topics: list[str] = field(default_factory=list)

    def clear(self) -> None:
        for s in (self.seen_nids, self.seen_home_nids, self.seen_refs,
                  self.seen_figure_fids, self.seen_canonical_topics,
                  self.seen_formulas, self.seen_semantic_keys):
            s.clear()
        self.spoken_topics.clear()

    def snapshot(self) -> dict:
        """Return a shallow copy of every set, for ``restore``."""
        return {
            "seen_nids": set(self.seen_nids),
            "seen_home_nids": set(self.seen_home_nids),
            "seen_refs": set(self.seen_refs),
            "seen_figure_fids": set(self.seen_figure_fids),
            "seen_canonical_topics": set(self.seen_canonical_topics),
            "seen_formulas": set(self.seen_formulas),
            "seen_semantic_keys": set(self.seen_semantic_keys),
            "spoken_topics": list(self.spoken_topics),
        }

    def restore(self, snap: dict) -> None:
        """Replace contents from a previous ``snapshot``.  In-place
        so existing aliases (the Orchestrator's set fields) keep
        seeing the right contents."""
        for name in ("seen_nids", "seen_home_nids", "seen_refs",
                     "seen_figure_fids", "seen_canonical_topics",
                     "seen_formulas", "seen_semantic_keys"):
            cur = getattr(self, name)
            cur.clear()
            cur.update(snap.get(name, set()))
        self.spoken_topics[:] = list(snap.get("spoken_topics", []))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """One narration session: main + optional tangent.

    Attributes
    ----------
    plan_id:
        Externally-visible identifier (set by the server).
    book:
        Loaded corpus.
    main_orch:
        Orchestrator driving the primary narration.
    main_board, tangent_board:
        Separate Chalkboards.  The tangent_board resets between tangents.
    tts_factory:
        Callable returning a fresh TTS backend per orchestrator
        (each Orchestrator instance owns its own, since some backends are
        not thread-safe).
    """
    plan_id: str
    book: Book
    # Optional main orchestrator: a freshly-built session always has one;
    # a session resumed from disk (after a server restart or page reload)
    # has ``main_orch=None`` because the original main narration is gone.
    # Tangents work either way — they're built on-demand inside ``ask``.
    main_orch: Optional[Orchestrator] = None
    main_board: Chalkboard = field(default_factory=Chalkboard)
    tangent_board: Chalkboard = field(default_factory=Chalkboard)
    tts_factory: callable = field(default=lambda: NullTTS())

    # Internal state.
    _main_iter: Optional[Iterator[StreamEvent]] = None
    _tangent_iter: Optional[Iterator[StreamEvent]] = None
    _tangent_id: str = ""
    _paused: bool = False
    _cancelled: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _resume_signal: threading.Event = field(default_factory=threading.Event)

    # Dialogue memory — bounded to the last ``_history_cap`` turns so
    # the prompt context never grows unbounded.  Inspected by ``ask``
    # when the user's utterance is a follow-up like "tell me more".
    dialogue_history: list[DialogueTurn] = field(default_factory=list)
    _history_cap: int = 8
    # Last focus carried across turns — populated from the most recent
    # non-control / non-followup turn so a follow-up always anchors
    # somewhere even after several control commands in a row.
    _last_focus_nid: str = ""
    _last_focus_topic: str = ""

    # Persistent knowledge state — shared with every Orchestrator the
    # session creates so a concept emitted in one tangent is treated
    # as already-shown in the next.
    knowledge: SessionKnowledge = field(default_factory=SessionKnowledge)

    # Callback fired after every recorded turn so persistence layers
    # can snapshot to disk.  Server wires this in ``_register``; tests
    # leave it None.  Signature: ``callback(session)``.
    _save_callback: Optional[callable] = None

    def __post_init__(self) -> None:
        # Resumable sessions arrive with main_orch=None — they're
        # rebuilt from a snapshot and only handle tangents going
        # forward.  Skip the orchestrator-bound init in that case;
        # ``knowledge`` will be loaded by ``restore_from_dict``.
        if self.main_orch is not None:
            # Adopt the main orchestrator's seen_* sets as our shared
            # knowledge — keep the same set objects so existing
            # references in main_orch keep working, and tangent_orch
            # will be created against these same sets.
            self.knowledge = SessionKnowledge(
                seen_nids=self.main_orch.seen_nids,
                seen_home_nids=self.main_orch.seen_home_nids,
                seen_refs=self.main_orch.seen_refs,
                seen_figure_fids=self.main_orch.seen_figure_fids,
                seen_canonical_topics=self.main_orch.seen_canonical_topics,
                seen_formulas=self.main_orch.seen_formulas,
                seen_semantic_keys=self.main_orch.seen_semantic_keys,
            )
            self._main_iter = iter(self.main_orch.stream())
        self._resume_signal.set()  # not paused initially

    # ----- control ----------------------------------------------------------

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._resume_signal.clear()

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._resume_signal.set()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            self._resume_signal.set()
        # Persist the math graph so enrichment from this session
        # survives even when the user hits Stop mid-flight.
        try:
            self.main_orch.save_math_graph()
        except Exception:
            pass

    def is_active(self) -> bool:
        return not self._cancelled

    def ask(
        self, question: str, *,
        top_k: int = 3,
        voice: Optional[str] = None,
        speed: float = 1.0,
        detail_level: str = "L1_story",
    ) -> str:
        """Inject a tangent answering *question*.

        Routes the utterance through ``narrator.router`` so the right
        planner runs:
          * ``book_overview``    → ``qa.book_overview(book)``
          * ``chapter_overview`` → ``planner.plan_outline(root_nid=...)``
          * ``section_overview`` → ``planner.plan_outline(root_nid=...)``
          * ``follow_up``        → ``qa.answer`` re-anchored on last focus
          * ``control``          → no plan; control action applied to
                                   the running session and ``""`` returned
          * ``topic_qa``         → existing ``qa.answer`` retrieval

        Returns the new tangent_id, or ``""`` for a control intent.
        """
        intent = router_mod.route(
            question, book=self.book,
            last_focus_nid=self._last_focus_nid,
            last_focus_topic=self._last_focus_topic,
        )

        # Control intents short-circuit — never run a planner / LLM.
        if intent.intent == router_mod.INTENT_CONTROL:
            self._apply_control(intent.control_action)
            self._record_turn(intent, plan_id="")
            return ""

        plan = self._plan_for_intent(intent, top_k=top_k)
        if plan is None:
            self._record_turn(intent, plan_id="")
            return ""

        # Re-show: hand the tangent its OWN fresh seen_* sets so it
        # re-emits previously-deduped primitives without polluting the
        # session's persistent knowledge.  Other intents share the
        # session-wide knowledge so dedup applies.
        is_reshow = intent.intent == router_mod.INTENT_RESHOW
        if is_reshow:
            tangent_seen_kwargs = {}  # all default-factory empty sets
        else:
            tangent_seen_kwargs = {
                "seen_nids": self.knowledge.seen_nids,
                "seen_home_nids": self.knowledge.seen_home_nids,
                "seen_refs": self.knowledge.seen_refs,
                "seen_figure_fids": self.knowledge.seen_figure_fids,
                "seen_canonical_topics": self.knowledge.seen_canonical_topics,
                "seen_formulas": self.knowledge.seen_formulas,
                "seen_semantic_keys": self.knowledge.seen_semantic_keys,
            }

        # Reset the tangent board so old tangent shapes don't accumulate.
        self.tangent_board.shapes = []
        self.tangent_board.ops = []
        self.tangent_board.started_at = time.monotonic()
        tangent_orch = Orchestrator(
            book=self.book, plan=plan,
            chalkboard=self.tangent_board,
            tts=self.tts_factory(),
            voice=voice, speed=speed,
            detail_level=detail_level,
            **tangent_seen_kwargs,
        )
        with self._lock:
            self._tangent_iter = iter(tangent_orch.stream())
            self._tangent_id = (
                f"tangent-{int(time.monotonic() * 1000) % 1_000_000_000}"
            )
            # Auto-resume so the tangent flows immediately.
            self._paused = False
            self._resume_signal.set()
        self._record_turn(intent, plan_id=self._tangent_id)
        return self._tangent_id

    # ----- routing helpers --------------------------------------------------

    def _plan_for_intent(
        self, intent: "router_mod.RoutedIntent", *, top_k: int,
    ) -> Optional[NarrationPlan]:
        """Build the right NarrationPlan for *intent*.  Returns ``None``
        when the intent shouldn't produce a plan (e.g., the resolver
        couldn't find a matching nid)."""
        from narrator import planner as plan_mod

        kind = intent.intent
        if kind == router_mod.INTENT_BOOK_OVERVIEW:
            return qa_mod.book_overview(self.book, depth=intent.depth)
        if kind == router_mod.INTENT_RECAP:
            return qa_mod.recap(
                self.book,
                history=list(self.dialogue_history),
                knowledge=self.knowledge,
            )
        if kind == router_mod.INTENT_RESHOW:
            # If we have a topic, hand it to the QA pipeline; otherwise
            # there's nothing to re-show, fall back to follow-up.
            topic = intent.topic or self._last_focus_topic
            if topic:
                return qa_mod.answer_streaming(
                    self.book, topic, top_k=top_k,
                    history=list(self.dialogue_history),
                )
            return None
        if kind == router_mod.INTENT_XREF_EXPLORE:
            # Walk the citation neighborhood of the current focus.
            focus = intent.target_nid or self._last_focus_nid
            if not focus:
                # Nothing to explore — fall back to a friendly nudge
                # via topic_qa on the raw text so the user gets a
                # response instead of silence.
                return qa_mod.answer(self.book, intent.raw, top_k=top_k)
            return qa_mod.xref_explore(self.book, focus)
        if kind == router_mod.INTENT_DEPENDENCIES:
            focus = intent.target_nid or self._last_focus_nid
            return qa_mod.dependencies(self.book, focus)
        if kind in (router_mod.INTENT_CHAPTER_OVERVIEW,
                    router_mod.INTENT_SECTION_OVERVIEW):
            target = intent.target_nid
            if not target or self.book.find(target) is None:
                # Unrecognised chapter/section number → degrade to QA.
                return qa_mod.answer(self.book, intent.raw, top_k=top_k)
            # Tighter caps for section-level outlines so they don't run on.
            if kind == router_mod.INTENT_SECTION_OVERVIEW:
                sentences = (-1, 3, 1, 0, 0)
            else:
                sentences = (1, 3, 2, 1, 0)
            return plan_mod.plan_outline(
                self.book, root_nid=target,
                sentences_at_depth=sentences,
            )
        if kind == router_mod.INTENT_FOLLOW_UP:
            # If we have a focus, answer "more about <focus>".  Otherwise
            # there's nothing meaningful to expand on; degrade to QA on
            # the raw utterance so the user gets *something*.
            base = intent.topic or self._last_focus_topic
            if base:
                expanded = (f"{intent.raw}, focusing on {base}"
                            if intent.raw and intent.raw.lower() != base.lower()
                            else f"more on {base}")
                return qa_mod.answer_streaming(
                    self.book, expanded, top_k=top_k,
                    history=list(self.dialogue_history),
                )
            return qa_mod.answer_streaming(
                self.book, intent.raw, top_k=top_k,
                history=list(self.dialogue_history),
            )
        # topic_qa default.
        return qa_mod.answer_streaming(
            self.book, intent.raw, top_k=top_k,
            history=list(self.dialogue_history),
        )

    def _apply_control(self, action: str) -> None:
        if action == "pause":
            self.pause()
        elif action == "resume":
            self.resume()
        elif action == "stop":
            self.cancel()
        elif action == "forget":
            # Wipe the shared knowledge so the next tangent treats every
            # concept as new again.  Useful for "start fresh".
            self.knowledge.clear()
        # ``skip``, ``louder``, ``quieter``, ``slow``, ``fast`` —
        # acknowledged but no concrete effect here yet; they're hooks
        # for future audio-pipeline plumbing.

    def _record_turn(
        self, intent: "router_mod.RoutedIntent", *, plan_id: str,
    ) -> None:
        """Append a turn to dialogue_history and refresh focus pointers."""
        turn = DialogueTurn(
            user_text=intent.raw,
            intent=intent.intent,
            focus_nid=intent.target_nid,
            focus_topic=intent.topic,
            plan_id=plan_id,
            t=time.monotonic(),
        )
        self.dialogue_history.append(turn)
        if len(self.dialogue_history) > self._history_cap:
            self.dialogue_history = self.dialogue_history[-self._history_cap:]
        # Update focus pointers for future follow-ups.  Control turns
        # don't move the focus; follow-ups inherit the current focus.
        if intent.intent in (router_mod.INTENT_BOOK_OVERVIEW,
                             router_mod.INTENT_CHAPTER_OVERVIEW,
                             router_mod.INTENT_SECTION_OVERVIEW,
                             router_mod.INTENT_TOPIC_QA):
            if intent.target_nid:
                self._last_focus_nid = intent.target_nid
            if intent.topic:
                self._last_focus_topic = intent.topic
                self.knowledge.spoken_topics.append(intent.topic)
        # Auto-save the session after every turn so a server crash or
        # page reload can recover the dialogue + chalkboard.  The
        # callback is wired by the server in ``_register``; tests
        # don't set it and therefore see a no-op.
        if self._save_callback is not None:
            try:
                self._save_callback(self)
            except Exception as e:
                # Persistence failures must never crash a turn — log
                # and move on, the user still gets their answer.
                print(f"[session] auto-save failed: {e}")

    # ----- snapshot / restore ----------------------------------------------

    def snapshot_dict(self) -> dict:
        """Serialise the session to a JSON-friendly dict.

        Captures dialogue history, knowledge state, focus pointers, and
        the main chalkboard's current SVG snapshot + per-shape metadata.
        Restoring from this dict yields a *resumable* session — no
        main_orch (the original main narration is gone) but every
        tangent and every memory the user already accumulated stays.

        Also captures ``book_name`` so a multi-book server can
        re-activate the right corpus before resuming the session.
        """
        # Resolve the session's book name from the active SERVER_STATE
        # registry — the session itself only holds the Book object,
        # not its registered key.  Falls back to "" when running
        # outside the server (tests).
        book_name = ""
        try:
            from .server import SERVER_STATE
            srv = SERVER_STATE.get("server")
            if srv is not None:
                for name, bk in srv.books_by_name.items():
                    if bk is self.book:
                        book_name = name
                        break
        except Exception:
            pass
        return {
            "version": 1,
            "plan_id": self.plan_id,
            "book_title": (self.book.title or "").strip() if self.book else "",
            "book_name": book_name,
            "updated_at": time.time(),
            "dialogue_history": [
                {
                    "user_text": t.user_text,
                    "intent": t.intent,
                    "focus_nid": t.focus_nid,
                    "focus_topic": t.focus_topic,
                    "plan_id": t.plan_id,
                    "t": t.t,
                }
                for t in self.dialogue_history
            ],
            "knowledge": {
                "seen_nids": sorted(self.knowledge.seen_nids),
                "seen_home_nids": sorted(self.knowledge.seen_home_nids),
                "seen_refs": sorted(self.knowledge.seen_refs),
                "seen_figure_fids": sorted(self.knowledge.seen_figure_fids),
                "seen_canonical_topics": sorted(
                    self.knowledge.seen_canonical_topics,
                ),
                "seen_formulas": sorted(self.knowledge.seen_formulas),
                "seen_semantic_keys": sorted(self.knowledge.seen_semantic_keys),
                "spoken_topics": list(self.knowledge.spoken_topics),
            },
            "last_focus_nid": self._last_focus_nid,
            "last_focus_topic": self._last_focus_topic,
            "chalkboard": {
                "canvas_w": self.main_board.canvas_w,
                "canvas_h": self.main_board.canvas_h,
                "max_content": self.main_board.max_content,
                "shapes": [
                    {
                        "nid": s.nid,
                        # ``chapter_map`` is rendered deterministically
                        # by ``viz/treemap.py`` from the on-disk sidecar
                        # JSON; persisting its svg_body would freeze the
                        # rendered layout against the *renderer code at
                        # save time*, so a renderer change (e.g. treemap
                        # → vertical stack) wouldn't take effect on
                        # reload until the user manually wiped the
                        # session.  Drop the cached body for chapter_map
                        # so it re-renders fresh on every reload.
                        "svg_body": (
                            "" if s.primitive == "chapter_map"
                            else s.svg_body
                        ),
                        "primitive": s.primitive,
                        "label": s.label,
                        "x": s.x, "y": s.y, "w": s.w, "h": s.h,
                        "meta": s.meta,
                    }
                    for s in self.main_board.shapes
                ],
                "snapshot_svg": self.main_board.snapshot(),
            },
        }

    def restore_from_dict(self, snap: dict) -> None:
        """Apply state from a previously-saved snapshot in place.

        Used by ``Session.from_snapshot`` (factory) and by tests.
        Doesn't touch the orchestrators — the caller is responsible
        for keeping ``main_orch=None`` when restoring.
        """
        # Dialogue history.
        self.dialogue_history = []
        for h in snap.get("dialogue_history", []):
            self.dialogue_history.append(DialogueTurn(
                user_text=h.get("user_text", ""),
                intent=h.get("intent", ""),
                focus_nid=h.get("focus_nid", ""),
                focus_topic=h.get("focus_topic", ""),
                plan_id=h.get("plan_id", ""),
                t=h.get("t", 0.0),
            ))
        # Knowledge.
        k = snap.get("knowledge", {})
        self.knowledge = SessionKnowledge(
            seen_nids=set(k.get("seen_nids", [])),
            seen_home_nids=set(k.get("seen_home_nids", [])),
            seen_refs=set(k.get("seen_refs", [])),
            seen_figure_fids=set(k.get("seen_figure_fids", [])),
            seen_canonical_topics=set(k.get("seen_canonical_topics", [])),
            seen_formulas=set(k.get("seen_formulas", [])),
            seen_semantic_keys=set(k.get("seen_semantic_keys", [])),
            spoken_topics=list(k.get("spoken_topics", [])),
        )
        # Focus pointers.
        self._last_focus_nid = snap.get("last_focus_nid", "")
        self._last_focus_topic = snap.get("last_focus_topic", "")
        # Chalkboard shapes — rehydrate so future tangents can dedup
        # against the visual history.
        cb = snap.get("chalkboard", {})
        from chalkboard.state import ChalkShape
        self.main_board.shapes = [
            ChalkShape(
                nid=s.get("nid", ""),
                svg_body=s.get("svg_body", ""),
                primitive=s.get("primitive", "rect"),
                label=s.get("label", ""),
                x=s.get("x", 0.0), y=s.get("y", 0.0),
                w=s.get("w", 160.0), h=s.get("h", 90.0),
                meta=dict(s.get("meta") or {}),
            )
            for s in cb.get("shapes", [])
        ]

    # ----- dispatch ---------------------------------------------------------

    def next_event(self) -> Optional[PanelEvent]:
        """Return the next event to dispatch, blocking on pause if needed.

        Returns None when the session is fully complete or cancelled.
        """
        # Block while paused, but wake on resume/cancel.
        self._resume_signal.wait()
        with self._lock:
            if self._cancelled:
                return None
            tangent_iter = self._tangent_iter
            tangent_id = self._tangent_id

        # Tangent has priority.
        if tangent_iter is not None:
            try:
                ev = next(tangent_iter)
                # First tangent event flagged as start.
                with self._lock:
                    is_start = (
                        getattr(self, "_tangent_announced", False) is False
                    )
                    self._tangent_announced = True
                return PanelEvent(
                    panel="tangent", event=ev,
                    is_tangent_start=is_start,
                    tangent_id=tangent_id,
                )
            except StopIteration:
                with self._lock:
                    self._tangent_iter = None
                    self._tangent_announced = False
                print(f"[session {self.plan_id}] tangent ended; "
                      f"main_iter={'present' if self._main_iter else 'None'}")
                # Emit a synthetic end-of-tangent marker so the frontend
                # closes the panel cleanly.
                return PanelEvent(
                    panel="tangent",
                    event=StreamEvent(
                        seq=-1, clause_text="", home_nid="",
                        audio_b64="", audio_dur=0.0, rate=0,
                        voice="", word_timestamps=[],
                        visual_ops=[],
                    ),
                    is_tangent_end=True,
                    tangent_id=tangent_id,
                )

        # Main iterator — None for resumed sessions whose original
        # main narration is gone.
        if self._main_iter is None:
            return None
        try:
            ev = next(self._main_iter)
            return PanelEvent(panel="main", event=ev)
        except StopIteration:
            print(f"[session {self.plan_id}] main_iter exhausted")
            return None


# ---------------------------------------------------------------------------
# Builder helpers — construct a Session for a given plan or a topic.
# ---------------------------------------------------------------------------

def build_session(
    *,
    plan_id: str,
    book: Book,
    plan: NarrationPlan,
    tts_factory: callable,
    canvas_w: float = 1280.0,
    canvas_h: float = 720.0,
    max_content: int = 8,
    voice: Optional[str] = None,
    speed: float = 1.0,
    book_path: str = "",
    detail_level: str = "L1_story",
) -> Session:
    main_board = Chalkboard(
        canvas_w=canvas_w, canvas_h=canvas_h,
        max_content=max_content,
        policy=ReadingOrderPolicy(),
    )
    tangent_board = Chalkboard(
        canvas_w=canvas_w * 0.4, canvas_h=canvas_h,
        max_content=max(4, max_content // 2),
        policy=ReadingOrderPolicy(),
    )
    main_orch = Orchestrator(
        book=book, plan=plan, chalkboard=main_board,
        tts=tts_factory(), voice=voice, speed=speed,
        book_path=book_path,
        detail_level=detail_level,
    )
    return Session(
        plan_id=plan_id, book=book,
        main_orch=main_orch,
        main_board=main_board,
        tangent_board=tangent_board,
        tts_factory=tts_factory,
    )
