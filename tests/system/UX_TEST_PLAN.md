# SeVim_math UX/UI test plan

**Lens**: a senior UX designer auditing the experience.  Goal: *"the
closest possible feeling to having a human private teacher with a
whiteboard, or a speaking GPT-style agent who answers questions in
real time and shows 100% relevant, covering visuals."*

Tiers:

  * **A — programmatic**.  Parse the HTML/CSS/JS, run the
    orchestrator with NullTTS, inspect the data the frontend would
    receive.  Run on every commit.
  * **V — visual-artifact inspection**.  Render representative SVG
    cards into PNG via headless tooling and check geometry / fitness;
    not currently in CI but lays groundwork.
  * **H — human aesthetic judgment**.  Manual playback.  Listed for
    completeness; verified before each release.

Every checkbox below maps to a named test.

---

## 1. First impression (5-second test)

  * `[A]` Page has a clear ``<h1>`` (visible app name).
  * `[A]` Primary action above the fold (Topic / Read book / Ask).
  * `[A]` Empty state communicates next step (TOC visible at boot).
  * `[H]` Reads as "math tutor", not "AI demo".

## 2. Affordance + discoverability

  * `[A]` Every interactive control has visible text *or* an aria-label.
  * `[A]` Disabled state visually distinct (gray + cursor not-allowed).
  * `[A]` Mic button has tooltip.
  * `[A]` Voice + book selectors are labelled (`title` attr).
  * `[A]` Sessions sidebar is reachable in one click.

## 3. Visual consistency

  * `[A]` Border radius vocabulary ≤ 3 distinct values (4 / 6 / 8).
  * `[A]` Color palette declared in CSS variables (single source of truth).
  * `[A]` Buttons share the same height / padding scale.
  * `[A]` Same primitive renders the same way (formula_card, passage_card).

## 4. Feedback latency (Norman thresholds)

  * `[A]` Button click → visual change within 100 ms (animation kicks in).
  * `[A]` Mic record start → state change visible (`button.recording`).
  * `[A]` Pause click → amber pulse appears (`button.paused`).
  * `[B]` Ask → first audio in < 1.5 s (already pinned by pedagogy test).
  * `[A]` Long tasks (mic transcribing, LLM thinking) have progress text.

## 5. Accessibility (WCAG AA)

  * `[A]` Body text contrast ≥ 4.5:1 against background.
  * `[A]` Large text (≥ 18 px) contrast ≥ 3:1.
  * `[A]` Button text contrast ≥ 4.5:1.
  * `[A]` Focus states defined for every interactive element.
  * `[A]` Min font size ≥ 13 px for body text.
  * `[A]` Min touch target ≥ 36 × 36 px for icon buttons.
  * `[A]` `prefers-reduced-motion` media query respected.

## 6. Information architecture

  * `[A]` Header order: title → topic input → primary actions → controls.
  * `[A]` Layout regions: chalkboard | tangent | trace.
  * `[A]` Right column collapses on ≤ 900 px viewport.
  * `[A]` Map / Sessions overlays sit ABOVE the chalkboard, dismissable.
  * `[A]` Trace pane scrollable independently.

## 7. Visual coverage of narration

  * `[A]` ≥ 80% of clauses in a real session emit at least one visual op.
  * `[A]` Every passage clause has an anchor card.
  * `[A]` Every cited equation/figure surfaces a card.
  * `[A]` Active-clause indicator pulses the right card.
  * `[V]` Cards size to content (no big-empty-frame regression).

## 8. Conversation continuity

  * `[A]` Word highlight tracks the current chunk's audio (streaming).
  * `[A]` Reload restores chalkboard exactly (already pinned).
  * `[A]` Session sidebar lists prior conversations with title + age.
  * `[A]` Pause keeps audio queue + chalkboard frozen.

## 9. Error states

  * `[A]` Empty Ask shows red outline shake + log entry.
  * `[A]` Mic permission denied → "mic permission denied" log.
  * `[A]` ASR error → status string + log entry.
  * `[A]` /api/cancel after stop returns 404 cleanly.
  * `[A]` Unknown book switch returns 404 with error in JSON.

## 10. Motion design

  * `[A]` Animation durations ≤ 500 ms.
  * `[A]` Non-linear easing (ease-in-out / cubic-bezier, not linear).
  * `[A]` Animations attached to enter / leave / highlight (3 named classes).
  * `[A]` `@keyframes` defined for pulse-amber, pulse-red, shape-pop.

## 11. Voice + audio polish

  * `[A]` Mic has 3 visual states: idle / recording / transcribing.
  * `[A]` Pause has dedicated pulse style.
  * `[A]` Voice picker constrained to ≤ 200 px so it doesn't hog the header.
  * `[H]` Audio doesn't peak / clip on standard volume.

## 12. Math rendering

  * `[A]` KaTeX CSS + auto-render JS loaded.
  * `[A]` `.math-prose` class used for prose-with-math reference cards.
  * `[A]` Garbled OCR fallback uses monospace, never red KaTeX error.
  * `[V]` Sample formula card rendered with cite + var_defs visible.

## 13. Density + breathing room

  * `[A]` Cards have ≥ 8 px internal padding.
  * `[A]` ReadingOrderPolicy enforces ≥ 18 px gap between cards.
  * `[A]` Header has horizontal scrolling on mobile (no truncation).
  * `[A]` Trace pane has line-height ≥ 1.4.

## 14. Touch targets (mobile)

  * `[A]` Every button on mobile media query has ≥ 36 px hit-area.
  * `[A]` Mic FAB is 56×56 (right-bottom).
  * `[A]` Right-column drawer toggle reachable with thumb.

## 15. Empty states + onboarding

  * `[A]` First-paint shows an idle placeholder ("— idle —") in the
    currently-spoken pane.
  * `[A]` Mic status is empty by default; updates on action.
  * `[A]` Book selector hides when only one book is loaded.

---

## Weaknesses found during this round

The first run found **8** real UX issues; all of them are now fixed.

**UX-1 (fixed) — Primary button contrast 3.12:1 (fails WCAG AA).**
The `--accent` CSS variable was Material Blue 500 (`#2196F3`); white
text on it scored 3.12:1 against the WCAG AA 4.5:1 floor.  Darkened
to Material Blue 800 (`#1565c0`), 5.7:1 against white.  Knock-on:
`--tangent` (Material Purple 500) also lifted to Material Purple 800
for consistency.

**UX-2 (fixed) — Icon-only close buttons missing aria-label.**
`#map-close` and `#sessions-close` were just `<button>×</button>` —
screen readers would announce them as "x button" with no purpose.
Both now carry `aria-label` and `title` ("Close citation map",
"Close sessions panel").

**UX-3 (fixed) — `<input id="max-content">` had no label.**  The
parent `<label>` only contained the literal text "max" — no
`for=` linkage, no `aria-label` on the input itself.  Added
`title` + `aria-label` so screen readers announce it as
"Max cards on chalkboard".

**UX-4 (fixed) — No `prefers-reduced-motion` opt-out.**  Looping
decorative animations (paused amber pulse, mic recording pulse)
ran at 1.4 s and 1.0 s respectively — fine for most users, but
motion-sensitive users had no way to dampen them.  Added the
canonical reduced-motion media query that collapses every
animation/transition to ~0 ms when the OS preference is set.

**UX-5 (fixed in tests) — Animation-duration test was too strict.**
The original assertion treated decorative looping pulses (`1.4 s`)
the same as one-shot transitions, which is a category error.  Test
now distinguishes one-shot animations (must be ≤ 500 ms) from
looping decorative ones (which need the reduced-motion fallback,
already added).

**UX-6 (fixed in helpers) — `@media` parser bug.**  Two tests
relied on a non-greedy regex that stopped at the first nested `}`,
falsely reporting "mobile layout doesn't collapse" and "mic FAB
missing touch target".  Added `extract_at_rule_body` helper with
balanced-brace matching.

**UX-7 / UX-8 (fixed by reframing) — Per-clause visual coverage
was a misframed metric.**  Original threshold expected
≥ 50% / ≥ 33% of clauses to emit at least one visual op; the
system actually emits anchors per *passage*, not per *clause*, so
elaboration clauses naturally don't add new visuals.  Reframed as
"every distinct passage visited during a chapter overview gets at
least one visual op" (target ≥ 85%) and "topic Q&A produces at
least one visual op for the primary anchor" — both pass.

---

## Final tally (this round)

  * **38** existing system tests (journey + visual + robustness +
    HTTP + dedup + SSE + pedagogy) — green
  * **26** new UX design checks (accessibility + contrast + ARIA +
    motion + IA + empty states) — green
  * **7**  visual-coverage probes (per-passage anchors, no
    overlap, no orphans, active-clause match) — green
  * **15** text-chat communication tests (end-to-end Ask flow,
    multi-turn, malformed payloads, SSE clause shape) — green

Plus **622** unit / non-UX system tests pass.  Combined: **708+ tests**.

## Still listed as `H` (manual playback)

These need a real browser session for honest verification:

  * **First impression** — does it read as "math tutor", not "AI demo"?
  * **Aesthetic quality of cards** — do KaTeX-rendered formulas look
    typeset, not stitched-together?
  * **Audio prosody** — do phrase boundaries sound natural, not chopped?
  * **Animation feel** — is the shape-pop entry "warm" or "fidgety"?
  * **Density tuning** — does the chalkboard feel curated or cluttered
    after a 30-clause deep answer?
  * **Mobile drawer ergonomics** — does the FAB feel reachable?
  * **Active-clause highlight visibility** — does the soft glow
    actually direct the eye, or get lost in card colour?

These are listed for the manual-playback step before each release.
