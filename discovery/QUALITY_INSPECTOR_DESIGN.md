# Lyceum quality inspector design — what we want guaranteed

The ambition: a user uploads a textbook PDF and starts learning. They never
need to file a bug. To get there we need an inspecting-agent suite that
catches every regression class we have already hit, plus the structural
completeness gaps that have not yet shown up.

This document lists, in one place, **every quality concern we have
encountered**, **what we did about it**, and **what the inspector agents
must verify**. Each item names a specific failure mode, the file/section
where it lives, and the check that would catch it.

---

## I. Bugs already fixed (must regress-test)

Each line: brief description, where it lives, the check that would fire
if it regresses.

### Layout / rendering
1. **Chapter-zoom SVG overflows the middle column** — right figures-panel
   covered the right edge of the board. (`serve/static/index.html`
   `panelInfo()` + `narrateNode` `canvas_w`; commit `df62fe0`.)
   *Check:* at every supported viewport (1280 / 1366 / 1920),
   `#board.scrollWidth - #board-wrap.clientWidth == 0` once chapter-zoom
   has rendered.

2. **Figures-panel scope too narrow when narration descends** — at
   `b/ch5/s5_1` the chapter prefix was the section itself, missing
   sibling-section figures. (`serve/server.py` `_figures_for_nid`;
   commit `55d0fe6`.)
   *Check:* `/api/figures/<nid>` returns the same set for the chapter
   root and any descendant of that chapter.

3. **CSS transform on SVG keyframe** clobbers the SVG `translate`
   attribute and teleports cards to (0,0). (Memory:
   `feedback_no_scale_in_svg_keyframe.md`.)
   *Check:* highlight-pulse animations only modify `filter` /
   attribute swaps; never `transform: scale()` / `translate()` on
   shapes that already carry SVG `transform`.

### Math-text extraction
4. **PyMuPDF vertical-stack `\sum`/`\prod`/`\int` truncated equations**.
   (`serve/orchestrator.py:_repair_pymupdf_vstack` +
   `tools/build_math_graph.py`; commits `f5f3608`, `309f750`.)
   *Check:* every formula in `<book>.math_graph.json` whose `cite_label`
   matches an equation number that appears in the original PDF text
   has at least one `\sum` / `\prod` / `\int` if the PDF text contains
   that operator's stack signature within ±20 lines.

5. **Multi-index sums** `\sum_{k,m}` (no upper bound) — captured as
   plain `X k,m`. (`_VSTACK_NOUPPER_RE`.)
   *Check:* no formula latex contains the regex
   `\b[XYR]\s+[a-zA-Zα-ω](?:,[a-zA-Zα-ω])+\b` followed by math.

6. **Transpose-with-subscript** `αT mX` should be `\alpha_{m}^{T} X`.
   (`_TRANSPOSE_SUB_RE`.)
   *Check:* no formula latex matches `(?:[α-ω]|\\alpha|...)T\s+[a-z]\s*[A-Z]`.

7. **`_looks_like_pseudocode` false-positive on multi-sum equations**
   — `\sum`, `\log` were counted as prose words.
   *Check:* unit test on a known list of valid math fragments
   (deviance, regularization functional, RKHS) — none should be rejected.

8. **Cluster window stops one token past last math token** —
   `\sum_{...} f(x_i)` cut off `f(x_i)`.
   *Check:* unit test on synthetic
   `\sum_{i=1}^{N} f(x_i, y_i)` — captured fragment must include all
   3 trailing tokens.

9. **OCR'd LaTeX from PyMuPDF needs targeted cleanup** —
   `M X m=1 \beta mhm` → `\sum_{m=1}^{M} \beta_m h_m`. (Memory:
   `feedback_ocr_latex_pitfalls.md`.)
   *Check:* a regression suite of (raw_pdf_text, expected_latex)
   pairs — all must round-trip.

### Audio / visual sync
10. **Per-seq audio tail** — `handleAudioComplete` must pace clause
    finish off *this clause's* `audioChunkLog`, not the global
    AudioContext queue tail. (Memory: `feedback_per_seq_audio_tail.md`.)
    *Check:* multi-clause SSE harness — transcript pane must advance
    one clause per audio_complete event, not stay stuck on clause 0.

11. **Audio-clock rAF sync engine** — replaced timer-based UI scheduling
    with a single rAF loop polling `audioCtx.currentTime`. (Memory:
    `feedback_audio_clock_sync.md`.)
    *Check:* word-marker drift = `|expected_word_at(audio_t) -
    rendered_word_at(audio_t)| < 100ms` for the entire narration.

12. **Streaming clauses bypass `highlightActiveClause`** — visual
    side-effects must wire into `syncFrame`'s `activeSeq` transition
    for streaming TTS. (Memory: `feedback_streaming_highlight.md`.)
    *Check:* both streaming and non-streaming TTS produce the same
    sequence of (seq → highlighted-clause-text) transitions.

13. **Verbalized artifact keys** — every formula card registers
    spoken-form keys (`f(x)` → "f of x") so the mention scanner
    matches when the narrator speaks. (Memory:
    `feedback_verbalized_artifact_keys.md`.)
    *Check:* every formula on the board has at least one entry in
    `_artifact_index`; `_artifact_index` is a `dict[str, list[str]]`
    (multi-nid, not single-nid).

### Q&A / tangent panel
14. **Narration-after-question** — main-panel rAF freeze gated on
    `state.tangentLive` (cleared at `tangent_end`), not on
    `tangentCard.classList.contains("open")`. (Memory:
    `feedback_narration_after_question.md`.)
    *Check:* end-to-end SSE harness — after `tangent_end` event, the
    next main clause's audio starts playing within 0.5s.

15. **Tangent panel main-board snapshot/restore** — `previously said
    things must never be gone` after the answer ends. (Paper §10.)
    *Check:* harness invariant `pre_question_shapes ⊆
    post_question_main_board_shapes`.

### Figure / artifact wiring
16. **Missing `<book>.figures.json` sidecar** for ESLII — server
    silently cached an empty result. Run
    `tools/extract_book_figures.py` to produce it. (Diagnosed this
    session.)
    *Check:* every active book has `<stem>.figures.json` on disk and
    the loader reports `>0 owning sections`.

17. **Figures sidecar schema mismatch** — `<book>_figures_v2.json`
    (flat list) is NOT what the server reads; it reads
    `<book>.figures.json` (`{by_nid, image_dir}`).
    *Check:* sidecar present in the right name with the right schema.

### Chapter-zoom narrative
18. **Long chapters overflow vLLM context** — Ch.14 / Ch.18 of ESLII
    returned HTTP 400. (`tools/build_chapter_map.py` `_call_chunked`;
    commit `309f750`.)
    *Check:* every chapter map has `story_paragraph_coverage ≥ 75%`.

19. **LLM silently truncates JSON output** even when no error fires —
    proactive batching for chapters > 12 sections. (Same commit.)
    *Check:* same coverage metric.

20. **LLM drops "boring" sections** (intros, bibliographic notes,
    exercises) — fill-missing retry pass.
    *Check:* same.

21. **Chapter-wide narrative didn't mention figures** — added rule 4
    to `_STORY_SYSTEM_PROMPT`, `_collect_chapter_figures`. (Same
    commit.)
    *Check:* for every figure label in `<book>.figures.json`, the
    `story_paragraph` of its owning section contains the figure's
    spelled-out citation (`figure five point one`).

### Multi-book server
22. **Active book switch must reload figure cache** — class-level
    `_figures_cache` keyed by book_path; first call after restart
    populates it. (Diagnosed this session.)
    *Check:* after `/api/active_book`, `/api/figures/<chapter_root>`
    returns the new book's figures, not the previous book's.

23. **Pause / Stop disabled mid-narration** — `narrateNode` runs
    `_hardStopCurrentSession` (which flips `setRunningUI(false)`)
    before the POST to `/api/narrate_node` returns; for the ~1-2 s
    LLM warmup the controls are dead.  (`serve/static/index.html`,
    commit `d928bfd`.)
    *Check:* within 200 ms of clicking a chapter row, both
    `#pause-btn` and `#stop-btn` have `disabled === false`.

24. **PDF-extracted formulas missing subscripts** — `Zm`, `yik`,
    `fk`, `α0m`, `\log fk(xi)` rendered as run-on text because
    PyMuPDF strips the typographic subscripts.  Two regex passes
    in `serve/refcontent.py:to_latex` join `\command 0xx` →
    `\command_{0xx}` (digit-prefix tail only, so `\alpha x` stays
    as two independents) and glued `Hh` / `H012` / `yik` →
    `H_{tail}` (letter tail capped at 2 chars to skip 4+ letter
    English words).  Operator names (`log`, `min`, `arg`, …) are
    converted to LaTeX commands, not subscripted.  (Same commit.)
    *Check:* every formula latex in `<book>.math_graph.json` is
    free of glued tokens matching `[A-Za-z][a-z]{1,2}` or
    `\\command\s+\d` outside an existing subscript.

25. **Active-figure highlight regression** — the right panel's
    `.figure-card.is-active` flash fires from
    `_setActiveFigureLabelFromClause` → `_highlightMentionedFigure`
    inside `highlightActiveClause`; the streaming-TTS path reaches
    this through `syncFrame`'s `activeSeq` transition.  Regression
    risk every time we touch `panelInfo`, the SSE clause schema, or
    the figure-card data attributes.  Also: the highlight must
    PERSIST until a different figure-class label is mentioned (a
    figure typically gets named once and discussed for several
    clauses; the highlight should not flicker off on the next
    non-figure clause).  Fixed in commit `bded974`.
    *Check:* end-to-end: trigger a chapter narration that mentions
    a figure label by spoken citation; assert
    `document.querySelectorAll('.figure-card.is-active').length === 1`
    within 1 second of the relevant clause becoming active and for
    the next 3 clauses thereafter (persistence).

26. **Mentioned equations rendered only by their canonical-cell
    formula** — chapter-zoom cells display ONE canonical formula
    (the one ``_pick_canonical_formula`` chose during chapter_map
    build), but the cell's story_paragraph routinely names two or
    three other cited equations the user is told to "see".  The
    user hears "equation 11.2 represents the cost function" but the
    cell only shows Equation 11.1.  This contradicts the paper's
    promise that "every formula the spoken text names is on screen."
    Fix not yet shipped — the architecturally correct solution is
    to surface the mentioned-equation as a transient card on the
    right panel (analogous to the figure-card highlight) or to
    weaken the chapter-zoom-suppresses-everything rule for cited
    equations.
    *Check:* parse every chapter-zoom cell's story_paragraph for
    spoken equation labels (``equation N.M``); assert each label is
    either the cell's canonical formula OR has a corresponding
    visual artifact rendered when that clause is the active one.
    Open A-class issue, not a regression.  Tracked in inspector
    section A as **A11_mentioned_equation_visible** (new).

27. **SeVim-diagram node labels carry raw OCR'd LaTeX** — the
    per-section concept diagrams generated by
    ``tools/build_sevim_diagrams.py`` (sidecar files
    ``books/<book>.sevim_diagrams.<chapter>.json``) bake node
    labels directly into the rendered SVG at build time.  These
    labels still show ``α0m``, ``αTmX``, ``Zm`` etc. — the
    PyMuPDF subscript-stripping artefacts the math-graph repair
    catches at extraction time but the SeVim diagram pipeline
    does not.  Result: the user sees correctly-typeset
    ``\alpha_{0m}`` in the cell's canonical formula card AND
    raw ``α0m`` in the diagram immediately below it.
    Fix path:
      * ``sevim/s2_extract.py`` and / or ``sevim/s2b_improve.py``
        must run the same ``_repair_pymupdf_vstack`` +
        subscript-joining pass on every node label *before* the
        label is committed to the graph.
      * Re-run ``tools/build_sevim_diagrams.py`` for every chapter
        whose sevim_diagrams sidecar predates the fix.
    *Check:* parse every SeVim diagram SVG, extract every
    ``<text>`` node body, and assert no body matches the
    PyMuPDF-artefact regex
    ``\b[A-Za-z][a-z]{1,2}(?![a-zA-Z])`` outside an existing
    LaTeX subscript / KaTeX-rendered span.  Open A-class issue
    (data-build).  Tracked in inspector section A as
    **A12_sevim_diagram_label_quality** (new).

28. **Story-paragraph prose contains raw LaTeX-like notation** —
    rule 5 of ``_STORY_SYSTEM_PROMPT`` says "NO RAW LATEX —
    variables get spoken names (``the smoothing parameter lambda``
    not ``λ``)" but the LLM regularly violates it.  Symptom: a
    cell's narrated prose contains ``Vm = ω_m^T X`` or
    ``Z_m = σ(α_{0m} + α^T_m X)`` rendered as plain text inside
    the chapter-zoom cell, alongside the SAME formula correctly
    KaTeX-typeset in the canonical-formula card just below.  The
    user sees broken-looking math TWICE per cell (once in prose,
    once typeset).
    Possible fixes:
      * Tighten the prompt + re-prompt failed cells until the
        violation rate is below threshold.
      * Post-process: detect raw-LaTeX spans and either strip them
        ("Vm = ω_m^T X" → "Vm") or wrap them in a KaTeX inline
        renderer span.
      * Or: render ALL prose math through KaTeX at the chalkboard,
        treating ``$...$`` and Unicode-math runs as inline math.
    *Check:* regex-scan every ``story_paragraph`` for any of:
    ``\w+\^[A-Za-z0-9_{}]``, ``\w+_\{?[A-Za-z0-9]``,
    ``[α-ωΑ-Ω∑∏∫∂∇√≤≥≈]`` outside a quoted citation, ``=``
    surrounded by symbol-bearing tokens.  Flag ALL chapters where
    >5 % of cells trip the regex.  Tracked in inspector section
    A as **A13_story_paragraph_clean_prose** (new).

---

## II. Quality concerns we considered but kept as-is

These are design choices, not bugs. The inspector should NOT flag
them, but should know they are intentional.

A. **Chapter-zoom mode suppresses every other visual primitive**
   (passage cards, formula cards, reference cards, …). The
   chapter-map cell already names every section + canonical formula,
   so additional cards would crowd the canvas.

B. **Figures-panel `_chapterPrefixFor` uses up to 3 segments** for
   compatibility with both `b/ch5` (ESLII, 2-seg) and
   `b/p1/s_ch_1_regular_languages` (Sipser, 3-seg). The server-side
   chapter walk uses the heuristic "first segment containing both
   `ch` and a digit" instead.

C. **books/ is .gitignored** in the lyceum code repo — copyrighted
   PDFs and per-user generated sidecars never enter the repo.
   Inspector should never try to git-add anything under `books/`.

D. **paper/ is .gitignored** in the lyceum code repo — the paper has
   its own private repo (`narrated-visualisation-of-mathematical-textbooks`).
   Inspector running in lyceum must not modify `paper/`.

---

## III. What the inspector-agent suite must guarantee

Each item: a property the inspector verifies, where it runs, what
to do on failure.

### A. Data-layer (post-upload, per book — runs once)

A1. **Sidecar completeness**: every active book has, on disk, in
    `books/<stem>.*`:
      * `<stem>.json` (corpus)
      * `<stem>.pdf` (source)
      * `<stem>.concepts.json`
      * `<stem>.formulas.json`
      * `<stem>.math_graph.json`
      * `<stem>.figures.json`
      * `<stem>.chapter_map.<chapter_nid>.json` for every real chapter
      * `<stem>.sevim_diagrams.<chapter_nid>.json` for every chapter
        with a chapter_map
      * `<stem>_figures_v2/<fid>.png` for every figure entry
    On failure: print missing list; offer to regenerate via the
    matching `tools/build_*.py`.

A2. **Math-graph integrity**:
      * Every `Formula.home_nid` resolves to a real `BookNode`.
      * Every `cite_label` matches at least one `(N.M)` regex hit in
        the home node's `body_text`.
      * No formula has `latex == ""`.
      * No formula's latex matches the truncation signatures
        (` = -$`, `^X $`, `^M X $`).
      * No formula's latex contains an unrepaired `X k,m` /
        `αT mX` pattern.

A3. **Figures sidecar integrity**:
      * Every `home_nid` resolves to a real `BookNode`.
      * Every `image_path` PNG exists on disk.
      * Every label parses as `Figure N.M` / `Table N.M` /
        `Theorem N.M` / `Definition N.M` / `Algorithm N.M`.

A4. **Chapter-map coverage**: for every chapter,
    `populated_sections / total_sections ≥ 0.80` (we currently sit
    at ≈0.92 average for ESLII).

A5. **Concept-layer coverage**: every section that owns a Formula
    has a corresponding `ConceptEntry` with at least an L1 narrative.

A6. **Formula-explanation coverage**: every Formula in the math
    graph has F0/F1/F2 entries in `<stem>.formulas.json`.

A7. **Theorem / Definition / Algorithm coverage**: for every
    ` THEOREM N.M`, ` DEFINITION N.M`, ` ALGORITHM N.M` marker in
    the corpus, a corresponding entry exists in the figures sidecar
    OR in a dedicated theorems sidecar (currently rolled into
    figures).

A8. **Cross-reference resolution**: every `(N.M)` citation in the
    corpus points to either a real equation, a real figure, a real
    table, or a real theorem.

A9. **KaTeX validity**: every `latex` field in math_graph and
    chapter_map parses without error through KaTeX (server-side
    headless `katex.renderToString`).

A10. **Local-only invariant**: no sidecar, no .py, no JS file
     references `api.openai.com`, `api.anthropic.com`, or any other
     external LLM endpoint. (Memory: `feedback_local_only.md`.)

A11. **Mentioned-equation visibility**: every spoken equation label
     (`equation N.M`) inside a chapter-zoom cell's `story_paragraph`
     either matches that cell's `canonical_formula_label` OR has an
     entry in the math graph that the runtime narrator surfaces as a
     visible card when the host clause is active.  In practice this
     means: parse the story_paragraph, extract `equation \d+\.\d+`
     matches, and for each, confirm `cite_label == canonical_label`
     OR plan a follow-up surface (transient card / ghost card / or
     widen the chapter-zoom suppression rule).  Initial enforcement
     is offline (data-quality agent flags chapters where N>1
     equation labels are spoken but only 1 is the canonical
     formula); the runtime fix is on the roadmap.

A12. **SeVim diagram node-label quality**: every `<text>` node body
     inside `<book>.sevim_diagrams.<chapter>.json` SVG payloads
     must be free of PyMuPDF subscript-stripping artefacts.  The
     SeVim diagram pipeline (`sevim/s2_extract.py` →
     `sevim/s2b_improve.py` → `sevim/s5_render.py`) bakes node
     labels into the rendered SVG at build time and bypasses the
     `serve.refcontent.to_latex` repair that the math-graph
     extraction uses.  Symptom: cell shows correct
     `Z_m = \sigma(\alpha_{0m} + \alpha_m^T X)` formula card AND a
     diagram below it with raw `α0m` / `αTmX` labels.
     Check: regex-scan every text body for
     `\b[A-Za-z][a-z]{1,2}(?![a-zA-Z])` glued-token patterns and
     `\bX\s+[a-zA-Z](?:,[a-zA-Z])+\b` multi-index sums; flag any
     hit.  Fix path: route node labels through the same repair
     stack before rendering, then re-run
     `tools/build_sevim_diagrams.py` for every chapter whose
     sidecar predates the patch.

A13. **Story-paragraph clean prose** (no raw LaTeX-like notation):
     rule 5 of `_STORY_SYSTEM_PROMPT` mandates that variable names
     be spoken in plain English ("the parameter lambda", not "λ";
     "V sub m equals omega sub m transpose X", not "Vm = ω_m^T X").
     The LLM regularly violates it.  Check: for every
     `story_paragraph` in every `<book>.chapter_map.<root>.json`,
     regex-scan for raw-LaTeX-like signals — `\w+\^[A-Za-z0-9_{}]`,
     `\w+_[A-Za-z0-9{]`, lone Greek letters
     `[α-ωΑ-Ω∑∏∫∂∇√≤≥≈]`, equality signs surrounded by
     symbol-bearing tokens.  Fail when more than 5 % of cells in a
     chapter trip the regex.  Surfaces the cells where the prose
     reads like broken LaTeX — fix is either a stronger prompt
     pass, a post-strip step, or an inline-KaTeX renderer for the
     prose pane.

### B. Render-layer (per visual op, online)

B1. **Card width fits in column**: any emitted shape has
    `shape.w ≤ panelInfo.w` (the board column's width).

B2. **Card font legible**: every text element has computed font-size
    ≥ 12px at viewport widths ≥ 1280; ≥ 14px at ≥ 1920.

B3. **No card overlap**: at the moment a new card is placed,
    `_aabbCollides(...) === null`.

B4. **Figure image fetches successfully**: when a `book_figure` op
    fires, `GET /api/figure_image/<fid>` returns 200 with
    `content-type: image/*`, non-empty body.

B5. **Tier-3 LLM SVG passes structural inspection** (≥3 polylines,
    correct viewBox, no `<script>`, no `<foreignObject>`).

B6. **Tier-3 LLM SVG passes vision-language inspection** when the
    VLM endpoint is reachable.

### C. Sync-layer (per clause, online)

C1. **Word-marker drift** ≤ 100ms throughout the clause.

C2. **Highlighted clause matches active audio chunk**: the clause
    text in the transcript pane equals the spoken text of the
    currently-playing audio.

C3. **Pause/resume preserves visual queue**: after pause/resume,
    `pending_visual_ops` count is unchanged.

C4. **Per-seq audio tail**: `handleAudioComplete` advances the
    transcript pane by exactly one clause per audio_complete event.

### D. Q&A / tangent (per ask, online)

D1. **`openTangent` succeeds**: after `Ask` click, `<main>` has the
    `with-tangent` class and `#tangent-card` is visible within 1s.

D2. **Main board frozen during tangent**: `pre_question_shapes ⊆
    post_question_shapes` (no shape disappears).

D3. **Audio handoff at `tangent_end`**: next main clause's audio
    starts within 0.5s of the tangent's last queued chunk's end.

D4. **`state.tangentLive` invariant**: flag is `true` from
    `tangent_start` to `tangent_end`, regardless of `closeTangent`
    timer.

D5. **Tangent panel snapshot/restore**: every `nid` present in the
    main panel before `tangent_start` is also present after
    `tangent_end`.

### E. Coverage (per chapter, post-build)

E1. **Story-paragraph coverage** ≥ 80% of sections have non-empty
    `story_paragraph`.

E2. **Equation-mention coverage**: ≥ 80% of cited equations are
    named (by spelled-out citation) in the prose of their owning
    section.

E3. **Figure-mention coverage**: ≥ 80% of cited figures are named
    (by spelled-out citation) in the prose of their owning section.

E4. **Theorem-mention coverage**: every cited theorem/definition is
    named in the prose of its owning section.

E5. **No "as introduced in section X" artefacts** survive the
    `_clean` post-processor.

### F. UX-flow (end-to-end harness, periodic)

F1. **First clause within 5s** of clicking a chapter that has a
    chapter_map sidecar (excluding cold-start LLM warmup).

F2. **Chapter narration completes** without unhandled JS errors,
    SSE disconnects, or audio drops.

F3. **Question answered within 10s** for a topic that retrieves
    high-similarity content.

F4. **Voice input round-trip** (mic-on → text in Ask box) ≤ 3s for
    a 5-second utterance.

F5. **Resumable session**: kill the server, restart, page reload —
    the chalkboard's prior shapes re-mount and the next Ask flows as
    a tangent into the persisted plan.

### G. Performance (per request, online)

G1. **Per-clause latency** ≤ 2s (paper Table 14 budgets).

G2. **TTS clause synth** ≤ 2s.

G3. **VLM inspection** ≤ 2s when fired.

G4. **Tier-3 LLM SVG** ≤ 1.6s.

### H. Safety / locality (always)

H1. No outbound HTTPS connections except to `127.0.0.1`.

H2. No external API key in env or settings.

H3. Quantised models on the user's hardware only.

---

## IV. Suggested inspector-agent architecture

Three tiers of agents, each with a clear lifecycle:

### Tier 1 — `data-quality-agent` (offline, per book upload)

* Runs every check in **section A** (data-layer).
* Triggered by `/api/upload_book` once the ingest pipeline finishes.
* Output: a JSON health report at
  `books/<stem>.health.json` with `{check_id: pass|fail, detail}`.
* On any A-class failure, the book is marked `phase: degraded` in
  `/api/books` and the user sees a "this book is partly built" banner.
* Recovery: an "Auto-repair" button re-runs the missing
  `tools/build_*.py` step.

### Tier 2 — `runtime-quality-agent` (online, per request)

* Lives inside the orchestrator; checks **B**, **C**, **D**, **G**.
* Each check that fails emits a structured event to
  `serve/_health_log.jsonl` with `{ts, plan_id, check_id, severity,
  payload}`.
* `severity = "fatal"` halts narration with a user-visible banner
  ("audio-visual sync lost — restarting clause").
* `severity = "warn"` is logged silently; agent surfaces a daily
  summary on the operator console.

### Tier 3 — `e2e-smoke-agent` (offline, per build / nightly)

* Headless Playwright harness; runs **F** end-to-end.
* Drives a fresh Chromium against a known book (ESLII Ch.5 + Sipser
  Ch.1 are the canonical fixtures), records:
    * Wall-clock time to first clause.
    * Total clause count.
    * Console errors / warnings.
    * Word-marker drift histogram.
    * Screenshot diff vs. last green build.
* On any failure → opens a GitHub issue automatically (or, in a
  developer-mode workflow, prints the failing check + reproduction
  command).

### Cross-cutting: a `coverage-agent` (offline, per chapter rebuild)

* Runs section **E** against every freshly-built chapter map.
* When `equation/figure/theorem mention coverage < 80 %`, calls back
  into `tools/build_chapter_map.py` with a smaller, focused batch
  (single section + just its citations) until coverage is met.
* Persists per-chapter coverage to
  `books/<stem>.chapter_map.<root>.coverage.json` so the operator
  can audit which chapters need re-running.

---

## V. Things we have NOT yet designed for (gaps for future work)

These are real concerns that this session did not address. The
inspector should at minimum LOG when it sees them, even if it
cannot self-heal yet.

V1. **Books in non-English** — every regex assumes Latin script + ASCII
    punctuation. PyMuPDF on a Spanish or Chinese textbook would
    silently produce nonsense.

V2. **Books with handwritten or scanned-image equations** — PyMuPDF's
    text extractor returns nothing usable. Need a vision-LLM pass.

V3. **Books with very long algorithms** (`Algorithm 14.x` in ESLII has
    pseudocode that the formula extractor mistakes for math).

V4. **Multi-page figures** spread across page boundaries — the
    on-demand cropper assumes a single page.

V5. **Two narrators in different languages on the same book** — the
    Kokoro voice list is fixed per session; switching mid-narration is
    untested.

V6. **Overflow / long-form questions** — the Q&A path's
    `max_content` cap might silently drop a long answer.

V7. **Figure mentions that use Examples instead of Figures** —
    Sipser's `Example 1.7` references `Figure 1.8`; the figures-list
    matcher only scans for the literal label.

V8. **Book uploads that exceed the offline build's time budget** —
    a 1500-page graduate textbook could take hours; the user has no
    progress indicator past the "ingesting…" status.

V9. **Concurrent users on the same in-process server** — the
    `serve.session` registry assumes a single learner; two browsers
    would fight over the AudioContext clock.

V10. **Books with zero figures, zero theorems, or zero equations** —
     the chapter-map narrative agent does not currently degrade
     gracefully when the input lists are empty.

---

## VI. The "happy path" we want to guarantee

The end-state we want, expressed as a single test the smoke agent
runs every night:

> Given a fresh PDF the system has never seen, after the upload
> finishes, a learner clicks Chapter 1 and within 5 seconds hears the
> first sentence of a coherent chapter-wide narrative; every cell in
> the chapter map renders with its own paragraph, canonical formula,
> and either a book-figure or a curated diagram; figures and equations
> are named by their spoken citations as they come up; clicking Ask
> opens a side panel with an answer in under 10 seconds; pause /
> resume / stop / replay all behave as labelled; the learner can run
> the full chapter without seeing a JS error in the console, an
> overflowing card on the board, an audio drop, or a missing figure.

If that test passes for a never-seen book, the user does not file a
bug. Every check in this document is a clause of that test.
