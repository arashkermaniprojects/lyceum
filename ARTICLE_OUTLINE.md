# SeVim v2 — Article Outline

**Status:** v0.1, 2026-04-19. Derived from `ARCHITECTURE.md` v0.2. Replaces the superseded TVCG draft in `old/`.

**Target venue:** IEEE TVCG (default, matches prior submission format). Alternatives: UIST / IUI for the interactive-tool framing, VIS for the visualization framing. Decision deferred — outline is venue-agnostic; formatting comes after content converges.

**Total budget:** ~6 pages two-column + refs. Per-section page estimates in brackets.

---

## Title + Abstract [0.1 pp]

**Working title:** "SeVim: Real-Time Deterministic Semantic-to-Visual Mapping for Live Whiteboarding"

**Abstract (~200 words):**
- Problem: live visual explanation during teaching is either pre-authored (slow, rigid) or generative-AI (non-deterministic, opaque).
- Approach: semantic graph + hybrid symbolic/numeric mapping + deterministic layout + minimal shape grammar.
- Result (headline): sub-50 ms incremental updates, byte-identical SVG for identical input, every visual element back-traceable to an input span.
- Claim: we occupy an empty region in the design space — fast, deterministic, interpretable visualization without a generative image model.

---

## 1. Introduction [0.7 pp]

**Goal:** motivate the problem + state contributions.

**Content:**
- Whiteboard teaching anecdote → humans use a tiny vocabulary (rect, arrow, text) but express unbounded meaning via composition. The computational question: can a system mirror this?
- Two common approaches and their failure modes:
  - Generative image/SVG models: high variance, opaque, slow, hallucinated content
  - Authored tooling (PowerPoint, draw.io): fast per-frame but no automation; ill-suited to live speech-driven explanation
- Our position: the right target is **semantic-to-visual mapping**, not image generation. Make language the input and a limited shape grammar the output.
- Contributions (numbered, concrete):
  1. A five-stage deterministic pipeline with four non-negotiable invariants (I1–I4).
  2. A closed 8-relation visual ontology sufficient for educational whiteboarding.
  3. A rule-tree layout algorithm that sidesteps optimization while preserving incremental stability.
  4. An implementation + evaluation showing sub-50 ms incremental update latency with byte-level determinism.

---

## 2. Related work [0.8 pp]

**Organize by the failure mode each line of work addresses and what it fails to deliver.**

- **Generative SVG / diagram models:** DiagrammerGPT, IconShop, SVGDreamer, Reason-SVG. Strength: open-ended expressivity. Weakness: non-determinism, path-token outputs that aren't semantically editable.
- **Image diffusion:** Stable Diffusion derivatives for figures. Weakness: raster, opaque, slow.
- **Structured diagram tools:** Graphviz, Mermaid, PlantUML. Strength: determinism, speed. Weakness: no semantic frontend — user must already have the graph.
- **Concept maps + live sketch:** classic concept mapping (Novak), real-time sketch recognition systems. Closest in spirit; we differ by driving from language rather than pen strokes.
- **Neuro-symbolic semantic parsing:** LLM-assisted triple extraction with verifiable backends. Our S2 borrows here.
- **Scene graph generation in vision:** inverse direction (image→graph); we run graph→image.

**Framing sentence:** "We combine the determinism of structured tools with the language-driven frontend of semantic parsing, in the empty region left by generative approaches."

---

## 3. Problem formulation and invariants [0.5 pp]

**Content:**
- Formal statement: the system is a pipeline `P: X → Y` where `X` is a text/speech stream and `Y` is an evolving SVG DOM. We require `P` be a pure function of `(x, θ)`, given frozen parameters `θ`.
- The four invariants (reproduced from ARCHITECTURE.md §1):
  - **I1 Determinism:** `P(x; θ)` byte-identical for identical `x`
  - **I2 Real-time:** ≤ 50 ms p95 per incremental update
  - **I3 Interpretability:** every visual element traces to a semantic node/edge; every node traces to input span(s)
  - **I4 Minimal grammar:** six primitives, composition-based expressivity
- Each invariant is stated as a testable property, not an aspiration. Evaluation §8 tests them directly.

---

## 4. Architecture overview [0.4 pp]

**Content:**
- Five-stage pipeline figure (pipeline.pdf — same structure as the diagram in ARCHITECTURE.md §2, redrawn for the paper).
- Typed I/O per stage listed in a small table. Key point: no back-edges, all stages pure functions.
- Design contract: adding capability means extending the grammar (new node type, new relation, new container layout), **not** adding an optimizer or a learned black box.

---

## 5. Semantic IR [0.5 pp]

**Content:**
- Scene graph schema (nodes, edges, provenance spans).
- Closed NodeType set (5) and closed RelationType set (8) — tabulated with their visual-pattern counterparts.
- Rationale for closed-at-v0.1: the cost of openness (ontology drift, inconsistent visuals) outweighs the benefit (coverage of rare relations) for educational whiteboarding.
- Deterministic dedup rule (cosine ≥ 0.85 + label/type match), with a brief sensitivity discussion.

---

## 6. Hybrid mapping [0.6 pp]

**Content:**
- Symbolic side: per-relation visual pattern dictionary (fixed table).
- Numeric side: linear projection `φ(embedding, salience) → (w, h, fs, sw, fill)` with frozen `W`, clipped to compact ranges.
- Why linear: determinism + inspectability + sufficient dynamic range for educational input. We position kernel / manifold / OT methods as out-of-scope overkill for v1, with the hook for future ablation.
- Provenance propagation: every `VisualShape` carries `nid` = source `SceneNode.id`; every `VisualConn` carries `eid` = source `SceneEdge.id`.

---

## 7. Deterministic layout [0.7 pp] ★ **core contribution**

**Content:**
- Critique of optimization-based layout: constraint solvers and force-directed methods are either non-deterministic across library versions or sensitive to init. Neither meets I1.
- Our rule-tree: container-recursive, with per-container layout rule derived from the dominant child relation (sequence → strip, causes → Sugiyama-layered, similar_to → grid, mixed → salience-sorted grid).
- Deterministic Sugiyama for `causes` DAGs: fixed tie-breakers at each step (node id asc for level assignment, barycenter-with-id-tiebreak for crossing reduction, fixed x-coordinate assignment). Cite Sugiyama et al. original + determinism caveats.
- **Anchor-then-extend incrementality:** on `SceneGraph.revision` increase, existing positions are held fixed; new shapes placed in first-available slots; overlap resolved by container-padding expansion, not child-nudging. Proof sketch: same input → same placement order → same output.
- Counter-case: what if rule-tree can't express a target layout? Answer: extend the grammar (new container type), not the optimizer. This is a core design commitment.

---

## 8. Streaming and incrementality [0.3 pp]

**Content:**
- Parse-unit = clause (punctuation + conjunction boundaries), rationale: smallest unit producing meaningful IR delta.
- Per-clause pipeline pass from S2 downward; S1 streams continuously.
- Backpressure: coalesce pending clauses; never drop, never speculate past input.
- Latency budget table (from ARCHITECTURE.md §8.2).

---

## 9. Implementation [0.4 pp]

**Content:**
- Python reference implementation (research/offline): S1–S5 as pure functions, test suite for I1/I3.
- TypeScript browser app (interactive target): shared JSON IR contract, incremental SVG DOM patching with `data-nid` / `data-eid` diffing.
- Frozen models: MiniLM-L6-v2 sentence encoder (int8 quantized for browser WASM), linear φ weights baked into the build artifact.
- Code + artifacts release plan (repository URL, tag at publication).

---

## 10. Evaluation [0.8 pp]

**Split into four property-matched studies, one per invariant.**

- **I1 Determinism:** property tests over 10⁴ randomly sampled inputs × 2 runs → byte-identical SVG rate. Headline metric: 100% or bug-at-first-failure.
- **I2 Real-time:** per-stage latency distributions (p50/p95/p99) on (a) synthetic scaling benchmark (|nodes| ∈ {10, 50, 100, 500}), (b) real lecture transcripts from the student corpus (`reference_paths.md`). Plot: latency vs graph size.
- **I3 Interpretability:** forward + reverse provenance coverage (% of SVG elements with valid `data-nid` or `data-eid`; % of nodes with at least one `SpanRef`). Expected: 100%.
- **I4 Minimal grammar:** coverage study — % of test inputs rendered with only the 6 primitives + 8 relations. If <100%, enumerate gaps.

**Qualitative comparison:** side-by-side on 5–10 representative educational prompts vs. (a) Stable Diffusion, (b) DiagrammerGPT, (c) raw GPT-4 SVG emission. Criteria: determinism (run twice, diff), interpretability (can reader identify which input span produced which visual?), faithfulness (does visual match claim?).

**User study (optional, Phase 3):** classroom deployment with two instructors over one semester; exit survey on perceived explanation quality. Note as future work if not done by submission.

---

## 11. Discussion and limitations [0.4 pp]

**Content:**
- Closed ontology is a feature, not a bug, for the educational whiteboarding use case. What breaks if we lift it: drift, inconsistency, visual noise. When to lift it: adaptive-ontology research, Phase 4.
- Rule-tree layout can't handle certain dense network cases gracefully (empirical limitation).
- Speech input assumes reasonable ASR; domain-specific jargon degrades triple extraction. Deferred to Phase 3.
- Not a replacement for authored diagrams; complement.

---

## 12. Conclusion [0.1 pp]

One paragraph: recap thesis, invariants, reproducible implementation. Frame as an invitation for the community to test and extend the closed ontology.

---

## References [~0.8 pp]

Keep the 23 verified citations from the prior draft where still relevant:
- **Still relevant:** DiagrammerGPT, IconShop, SVGDreamer, Reason-SVG, Graphviz, Mermaid, concept-map foundational work.
- **Add for v2:** sentence-transformers / MiniLM origin, Sugiyama hierarchical layout, relevant neuro-symbolic parsing refs, scene-graph-generation surveys (briefly, to differentiate direction).
- **Drop for v2:** anything tied to the prior 9-kind SEPM thesis (none of the new claims rest on that design).

Action item: audit `old/refs.bib` against the v2 claim list when prose drafting begins.

---

## Appendix plan (online-supplement, not in page count)

- A — Full relation → pattern table with rendering examples
- B — Property-test harness + reproducibility (commit hash, seed not applicable — determinism invariant)
- C — Sugiyama tie-breaker specification
- D — Example session traces (input → IR → SVG)

---

## Writing order

1. §3 (invariants) — anchor everything else.
2. §7 (layout) — the core contribution; shape the prose before the easy sections.
3. §4–§6 (pipeline bulk) — once layout is clear, these fall out.
4. §8–§9 (streaming + impl) — concrete, short.
5. §10 (evaluation) — last, driven by actual measurement.
6. §1–§2 (intro + related work) — rewrite after the body stabilizes; avoids redrafting when claims shift.
7. Abstract + title — after everything else converges.

This order prevents rewriting the intro three times.
