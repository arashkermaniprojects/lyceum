# Math Semantic Network — Research, Design, Plan

> Goal: build a **persistent semantic graph of mathematical
> notations, formulas, and concepts** so that every UX decision
> (which card lights up, which formulas are placed adjacent, which
> math is missing) follows from one source of truth instead of
> ad-hoc heuristics.

---

## 1. State of the art

### 1.1 Symbolic / standards-based
| System | What it gives us | Why we care |
|---|---|---|
| **MathML 4 Content** | XML tree of `apply / ci / cn / bvar / fn / lambda / …`. Each operator has a semantic identity. | Canonical operator/argument tree for every expression. |
| **OpenMath + Content Dictionaries (CDs)** | Same as Content MathML but with replaceable, versioned CDs for vocabularies. | Lets us pick the *meaning* of a symbol (e.g. `int` ∈ `calculus.cd`) instead of guessing. |
| **OntoMathPRO 2.0** ([arXiv 2303.13542](https://arxiv.org/pdf/2303.13542)) | LOD ontology with **3 layers** — foundational, domain, linguistic — connecting math entities to natural-language surface forms. ~8000 classes, ~4000 relations. | Off-the-shelf concept inventory for "kernel function", "loss function", "regularization", etc. + linguistic surface forms. |
| **LaTeXML** ([NIST manual](https://math.nist.gov/~BMiller/LaTeXML/manual/math/math.details.html)) | LaTeX → presentation MathML + OPT (Operator Tree). Produces XMDual nodes pairing presentation with semantics. | Production-grade parser for our OCR'd LaTeX. |
| **sTeX / MathHub** | Semantic LaTeX dialect with declared symbol bindings. | Reference for how to declare variable scope. |
| **Lean / Mathlib** | Fully formal theorem / definition graph. | Far overkill for pedagogy; useful as a "north star" for what *complete* would mean. |

### 1.2 Rule-based pipelines
- **Tangent (Symbol Layout Tree / Operator Tree)**: pure-rule expression-tree extraction, used by Tangent-CFT below.
- **OntoMathPRO publishing pipeline**: paragraph-level pattern matchers for definitions, theorems, statements; matches surface forms to ontology concepts.
- Co-occurrence / shared-symbol graph (the cheapest possible): build edges between formulas that share at least one variable name.

### 1.3 ML / LLM
| Model | Idea | Use case |
|---|---|---|
| **MathBERT** ([arXiv 2105.00377](https://ar5iv.labs.arxiv.org/html/2105.00377)) | BERT pre-trained jointly on formula + surrounding text; predicts masked OPT substructures. | Formula embedding for **similarity** / **retrieval**. |
| **Tangent-CFT** ([ACM 10.1145/3341981.3344235](https://dl.acm.org/doi/10.1145/3341981.3344235)) | n-grams over SLT + OPT paths, FastText embeddings. | Strongest *structure-only* baseline, no training data needed beyond the formulas themselves. |
| **ColBERT-MM** | Late-interaction retrieval that captures *contextual* similarity (formula in different prose still matches). | Linking formulas to clause text. |
| **Graph Contrastive Learning over formulas** ([arXiv 2603.08012](https://arxiv.org/html/2603.08012)) | Treat each formula as a graph; learn embeddings via contrastive augmentations. | Robust formula relatedness when surface forms differ. |
| **LLM as triple extractor** ([Enhancing Math KGs with LLMs](https://www.mdpi.com/2673-3951/6/3/53)) | Feed LaTeX + prose to a local LLM, ask for `(entity, relation, entity)` triples, merge into graph. | Flexible relation inference (derivation, equivalence, motivation) without rule fatigue. |

### 1.4 Project assets we already have
(per `memory/MEMORY.md`)
- `book/concept_graph.py` — concept extraction layer.
- 13 math primitives + 18 math relations in `sevim/`.
- Penrose Domain/Substance/Style triple is the architectural analogue.
- Local-only constraint: no Anthropic / OpenAI in production paths (Qwen / local Kokoro / local embeddings).

---

## 2. What the graph should hold

```
Node types
==========
  Var          name, type-hint?          e.g. f, x, K, λ, J
  Formula      latex, opt, surface       e.g. K(x,y) = Σᵢαᵢk(xᵢ,yᵢ)
  Concept      ontomathpro_iri?          e.g. "kernel function", "loss"
  Passage      nid, text, home_nid       narrator clauses

Edge types
==========
  uses(Formula, Var)           formula's free variables
  binds(Formula, Var)           formula's bound variables (Σᵢ binds i)
  defines(Formula, Var)         formula's LHS = X kind
  derived_from(Formula, Formula)  "rewriting (5.42) we have …"
  specializes(Formula, Formula)  substitution: J(f) = ‖f‖² is a special J
  references(Formula, Formula)   citation label match (Eq 5.48 cites 5.42)
  about(Passage, Formula)        passage clause mentions formula
  about(Passage, Concept)        passage clause names concept
  instance_of(Var | Formula, Concept)  symbol grounded in OntoMathPRO IRI
  shared_vars(Formula, Formula)  derived; weight = |intersection|
  paired_in_clause(Formula, Formula)  emitted in the same StreamEvent
```

Persistence: in-memory `dict` per session, serialised to the existing chalkboard snapshot JSON so reload restores it.

---

## 3. Phased implementation

### Phase 0 — **deterministic, in-process** (~1 day)
*No new dependencies. Reuses what we already have.*

1. **Parse every formula's LaTeX into an OPT** using a small visitor over KaTeX's parser (already bundled on the frontend; we mirror just the parts we need in Python via a regex-driven approximation, or call `latex2mathml`).
2. **Extract Vars / Binders / Function calls / LHS** from the OPT. This is purely structural.
3. **Build the cheap edges**:
   - `uses`, `binds`, `defines` from the OPT.
   - `references` from `(5.42)` etc. citation labels (already in `_equation_citations_in`).
   - `about`, `paired_in_clause` from existing orchestrator emission.
   - `shared_vars` derived from `uses` overlap.
4. **Wire to UX**:
   - Layout: `placeShape` clusters by `shared_vars` weight (replace the simple "right of last formula" with "right of formula with biggest var-overlap").
   - Highlight: when narrator names a Var, highlight every Formula that `uses` it.
   - Coverage audit: every Formula that has at least one `about(Passage, Formula)` edge is "covered"; report uncovered formulas.

### Phase 1 — **rule-based enrichment** (~1–2 days)
*Still no ML.  Just better pattern matchers.*

1. **Definition patterns**: `"where x is the loss function"` / `"let f denote …"` → `defines(Formula, Concept)` + `instance_of(Var, Concept)`.
2. **Derivation patterns**: `"rewriting (5.42)"`, `"from (5.42), we get"`, `"this leads to"` → `derived_from`.
3. **Equivalence patterns**: `"or equivalently"`, `"this is the same as"` → bidirectional `derived_from`.
4. **Specialization patterns**: `"in the special case where g = f"` → `specializes`.

### Phase 2 — **OntoMathPRO grounding** (~2 days)
*Adds a knowledge backbone.*

1. Pull the OntoMathPRO 2.0 RDF/Turtle (CC-BY).
2. Build a local index: `{surface_form → IRI}` (~8000 entries).
3. For each Concept node already in our graph, attach an `instance_of(Concept, ontomath_iri)` whenever a surface form matches.
4. Read sibling/parent IRIs from OntoMathPRO to enrich `related_to` edges (e.g. "kernel function" → parent "function" + sibling "loss function").

### Phase 3 — **embedding-based similarity** (~2–3 days)
*Optional. Only when rules + ontology miss obvious connections.*

1. Compute Tangent-CFT embeddings for every Formula (no training needed; n-grams + FastText).
2. Add `similar_to(F, F', score)` edges for top-k pairs above a threshold.
3. Surface "you might also be interested in …" suggestions when the narrator mentions a Formula.

### Phase 4 — **LLM-assisted triples** (~2 days, local Qwen only)
*Highest risk; lowest priority.  Use only for relations rules don't catch.*

1. Per clause, prompt local Qwen with `(prose, latex_block)` and ask for triples:
   - `(formula_id, derived_from, formula_id)`
   - `(formula_id, motivates, formula_id)`
2. Merge into graph behind a confidence gate.
3. Verify: cross-check each LLM-emitted edge against the rule-based graph; reject contradictions.

### Phase 5 — **interactive UX surfaces**
1. **Card adjacency**: layout uses `shared_vars` weights.
2. **Group highlight**: hover/mention a Var → all Formulas that `use` it pulse.
3. **Trace**: clicking a Formula opens a side panel listing its edges.
4. **Coverage warning**: a small badge on the chalkboard if any Passage's math is missing a Formula card.

---

## 4. Recommended starting point

**Phase 0 (deterministic) is enough to fix today's complaints**:
- "related formulas should be physically close" → cluster by `shared_vars` overlap.
- "every math notation must be on the panel" → coverage audit emits warnings for any Formula without an `about` edge.
- "highlight related formulas" → mention scanner widens to Var-level (highlight all formulas that use the named Var).

It also lays the data layer that Phases 1-4 plug into without re-architecting.

**Defer Phases 2-4** until Phase 0 + 1 are stable. They add power but rule-based extraction handles ~80% of textbook math out of the box.

---

## 5. Open questions for you

1. **Local-only constraint** — is Phase 2 (OntoMathPRO RDF) acceptable? It's CC-BY data, no network calls at runtime.
2. **Phase 4** — happy with local Qwen as the only LLM, or skip Phase 4 entirely?
3. **Scope of "Concept"** — do we want to ground concepts in OntoMathPRO IRIs, or keep them as free-form strings with our own IDs?
4. **Persistence** — should the graph be per-session (rebuilt every Read) or per-book (cached between sessions)?

---

## 6. What I'm asking you to greenlight

If this matches your intent, I'll:

1. Build the Phase-0 graph extractor (`sevim/math_graph.py`) with `Var / Formula / Passage / Concept` nodes and the 7 deterministic edge types.
2. Wire it into `Orchestrator` so every emitted card carries its graph node, every clause's mentions are recorded as `about` edges.
3. Replace the current `_clusterAnchor` heuristic in the frontend with a `shared_vars`-weighted nearest-anchor.
4. Extend `_mention_highlight_visual_ops` to highlight every Formula that uses a mentioned Var.
5. Emit a coverage warning when any Passage names math that has no corresponding Formula.

Phases 1–4 follow as separate PRs once Phase 0 is observed working.

---

## Sources

- [OntoMathPRO 2.0 Ontology — arXiv 2303.13542](https://arxiv.org/pdf/2303.13542)
- [OntoMathPRO Linked Data Hub for Mathematics](https://link.springer.com/chapter/10.1007/978-3-319-11716-4_9)
- [OpenMath and MathML semantic markup — ACM](https://dl.acm.org/doi/10.1145/333104.333110)
- [OpenMath ↔ MathML integration](https://openmath.org/om-mml/)
- [LaTeXML math details](https://math.nist.gov/~BMiller/LaTeXML/manual/math/math.details.html)
- [MathBERT — arXiv 2105.00377](https://ar5iv.labs.arxiv.org/html/2105.00377)
- [Tangent-CFT — ACM 10.1145/3341981.3344235](https://dl.acm.org/doi/10.1145/3341981.3344235)
- [Mathematical Information Retrieval review — ACM Computing Surveys 2024](https://dl.acm.org/doi/10.1145/3699953)
- [Structure-Preserving Graph Contrastive Learning for MIR — arXiv 2603.08012](https://arxiv.org/html/2603.08012)
- [Enhancing Mathematical Knowledge Graphs with LLMs — MDPI 2025](https://www.mdpi.com/2673-3951/6/3/53)
- [Extracting Mathematical Concepts with LLMs — arXiv 2309.00642](https://arxiv.org/pdf/2309.00642)
