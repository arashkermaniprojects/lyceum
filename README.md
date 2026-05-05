# Lyceum

**A fully-local multimodal system that turns a static mathematics or
statistics PDF textbook into an interactive, narrated whiteboard.**

Lyceum ingests a textbook, builds a per-book *math semantic graph*
(formulas, variables, concepts and their dependencies) before any
narration starts, and at run time emits a synchronised audio stream
plus a stream of typed visual primitives (passage cards, book figures,
curated canonical diagrams, formula cards, reference cards,
cross-section connectors).  Every visual emission is timed against the
audio clock by a single animation-frame loop, so the spoken track and
the whiteboard cannot drift apart.

No external API is reachable from any production process — the entire
runtime stack (text LLM, vision-language inspector, sentence
embedder, text-to-speech) is served on the user's own machine.

> **Status:** the accompanying paper is currently under review.  This
> repository ships the implementation, evaluation harness, and per-book
> sidecars used to produce every numerical claim in the paper.

---

## Citation

If you use Lyceum, the math semantic graph design, the evaluation
harness, or any of the material in this repository in your research,
**please cite the paper**:

```bibtex
@article{lyceum2026,
  title    = {{Lyceum}: a fully-local multimodal system for narrated
              visualisation of mathematical textbooks},
  author   = {Kermani Kolankeh, Arash and Zgheib, Rita},
  journal  = {<TO BE FILLED IN ON ACCEPTANCE>},
  year     = {<YEAR>},
  volume   = {<VOLUME>},
  number   = {<ISSUE>},
  pages    = {<PAGES>},
  doi      = {<DOI>},
  url      = {<URL>},
  note     = {Preprint / under review at the time of writing}
}
```

The bundled SeVim diagram engine has its own citation under
[`sevim/README.md`](sevim/README.md); please cite both when you use
the diagram-rendering pipeline directly.

A `CITATION.cff` file at the repository root resolves to the same
record so GitHub's "Cite this repository" button works end-to-end.

---

## Highlights

- **Fully local.** vLLM serves a quantised text LLM, an embedding
  model and a vision-language inspector; Kokoro is the text-to-speech
  subprocess.  No request leaves the user's machine.
- **Math semantic graph per book.** Four node types (variable,
  formula, passage, concept), eleven edge types (uses, binds, defines,
  contains, references, derived\_from, specializes, related\_to,
  instance\_of, about, paired\_in\_clause), persisted as a per-book
  JSON sidecar that is monotone across rebuilds.
- **Eight typed visual primitives** emitted per spoken clause —
  passage card, book figure, canonical diagram from a 12-entry curated
  library, fall-back local-LLM SVG, formula card with sub-expression
  containment folded, reference card for in-text mentions, math note,
  and chapter-zoom map.
- **Three-tier quality assurance** — a data-quality inspector audits
  every ingested book offline, a runtime SVG inspector accepts or
  rejects every emitted figure, and an end-to-end smoke harness
  exercises the live server against scripted scenarios.  A
  cross-cutting math-fidelity agent crops the source PDF region around
  every cite marker, asks the local vision model whether the parsed
  LaTeX matches the rendered image, and asks the text LLM to repair on
  mismatch.
- **Reproducible evaluation.** Six metrics (M1 – M6) under
  `bench/eval/` produce one JSON record per metric and one auto-built
  LaTeX fragment per table.  The paper's per-method comparison tables
  are generated mechanically from those JSON records.

---

## Repository layout

```
.
├── book/             Book intermediate representation (parser, corpus,
│                     concept index, alias map, embeddings hooks)
├── narrator/         Per-clause narrator: ten-intent router, planner,
│                     concept and formula explanation layers, Q&A
│                     synthesis, narration sanitiser, Kokoro TTS worker
├── serve/            HTTP / SSE server, orchestrator, session manager,
│                     ASR endpoint, on-demand figure cropping,
│                     persistence
├── viz/              Visualisation: 12 curated SVG generators,
│                     structural / VLM inspector, operation vocabulary,
│                     chapter-map renderer, edge styles
├── chalkboard/       Browser-side chalkboard module (audio-clock loop,
│                     per-clause timeline, rAF sync engine)
├── sevim/            SeVim diagram engine — the math semantic graph
│                     builder, deterministic SVG layout, and ESLII
│                     formula-fidelity tooling.  Has its own README
│                     under sevim/README.md
├── tools/            Offline builders (chapter map, concept layer,
│                     formula explanations, math graph, SeVim
│                     diagrams), the data-quality inspector
│                     (data_quality_agent), the smoke harness
│                     (smoke_agent), the math-fidelity agent
│                     (fidelity_agent), and the figure / equation
│                     re-ingestion tools
├── bench/eval/       Journal-grade evaluation harness — six driver
│                     scripts (M1 – M6), raw JSON results, generated
│                     LaTeX fragments, the verbatim Sonnet-judge rubric,
│                     and a per-run cost ledger
├── service/          Service entry points (e.g. multi-book server)
├── tests/            705 deterministic unit / system tests
├── discovery/        Design notes (math-graph plan, inspector design,
│                     visual relation QA gallery)
├── paper/            Manuscript source (LaTeX) + figures + bibliography
└── books/            Per-book artefacts (corpus JSON, math graph, figures
                      sidecar, chapter-map sidecars, …)  --- gitignored
                      because the source PDFs are copyrighted
```

---

## Quick start

### Prerequisites

- Python 3.10+
- `pip install -e .` to install the runtime in development mode
- A Linux-style box with reasonable RAM
- A workstation GPU is recommended for the full local model fabric;
  the runtime degrades gracefully when individual model servers are
  unreachable
- Optional: vLLM endpoints for the text LLM, the vision-language
  inspector, and the embedding model; Kokoro for TTS

### Run an end-to-end ingestion + serve

```bash
# Install
pip install -e .

# Drop a copyrighted PDF into books/ — say, ESLII.pdf
cp /path/to/ESLII.pdf books/

# Run the full offline pipeline (parse, concept layer, math graph,
# per-chapter sidecars, data-quality inspector, math-fidelity agent)
python -m tools.ingest_agent books/ESLII.pdf

# Start the multi-book server
python -m serve.server books/ESLII.json
# Browse to http://127.0.0.1:8001
```

The server tolerates missing model servers — it falls back to
retrieval-only Q&A when the text LLM endpoint is unreachable, and to
the heuristic Unicode→LaTeX path when the vision-language equation OCR
sidecar is absent.

### Reproduce the paper's evaluation

```bash
# Run every objective metric (M1 – M5)
.venv/bin/python bench/eval/drivers/m1_routing.py
.venv/bin/python bench/eval/drivers/m2_retrieval.py
.venv/bin/python bench/eval/drivers/m3_viz_ops.py
.venv/bin/python bench/eval/drivers/m4_graph_coverage.py
.venv/bin/python bench/eval/drivers/m5_inline_math.py

# Optional: M6 narration-quality LLM judge (Claude Sonnet 4.6).
# Requires an Anthropic API key in the environment; capped at USD 20
# by the cost ledger.
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic
.venv/bin/python bench/eval/drivers/m6_narration_judge.py

# Regenerate the paper's LaTeX table fragments from the JSON results
.venv/bin/python bench/eval/build_tables.py
```

Each driver writes one JSON record into `bench/eval/results/`, and
`build_tables.py` rebuilds the paper-side `paper/tables/*.tex`
fragments mechanically.  The paper's headline numbers (Table 16 in
the manuscript) are produced by `build_summary` from those JSON
records.

### Run the test suite

```bash
pytest -q
```

There are currently 705 deterministic tests covering the router, the
math-graph builders, the cross-tangent dedup contract, the
narration-after-question SSE harness, the per-clause primitive
detectors, and a property-based determinism suite.

---

## Architecture in one paragraph

A click on a topic, chapter or section in the live UI is classified
by a regex-cascade router into one of ten teaching intents.  The
router resolves the intent to a `NarrationPlan` — a deterministic,
ordered list of clauses anchored at book nodes.  An orchestrator
streams clauses one at a time: each clause is sent to Kokoro for TTS,
to a per-clause primitive-detection cascade (inline-math, operation
vocabulary, reference patterns), and to the math semantic graph for
formula / variable / concept lookup.  Every detected primitive becomes
a typed visual op the frontend renders against the audio clock; ghost
cards are dropped for cross-section formulas the current passage
references, with semantic connectors drawn between endpoints.  When
the corpus has no figure for a topic, the visualisation registry
selects a curated SVG generator by sentence-embedding similarity and
falls through to a local-LLM-synthesised SVG inspected for structural
and visual fidelity.

---

## Reading order for the code base

If you want to understand the system end-to-end, read the modules in
this order:

1. `book/ir.py` — the data model
2. `book/parse.py` and `book/corpus.py` — how a PDF becomes the IR
3. `narrator/router.py` — the ten-intent classifier (rule-based)
4. `narrator/planner.py` — BM25 + dense + RRF retrieval and the
   depth-decay outline planner
5. `serve/orchestrator.py` — the per-clause emission pipeline (the
   long file at the heart of the system)
6. `viz/operations.py` and `viz/generators.py` — the operation
   vocabulary and the 12-entry curated registry
7. `sevim/math_graph.py` and `sevim/math_graph_phase1.py` — the
   offline graph builder
8. `tools/ingest_agent.py` and the auto-pipeline phases under
   `serve/ingest_pipeline.py` — what runs when a new book is uploaded
9. `bench/eval/` — the journal-grade evaluation harness

---

## Authors

- **Arash Kermani Kolankeh** (corresponding author),
  School of Engineering, Applied Science, and Technology (SEAST),
  Canadian University Dubai.
  ORCID [0009-0003-6494-414X](https://orcid.org/0009-0003-6494-414X).
- **Rita Zgheib**, Canadian University Dubai.
  ORCID [0000-0001-6301-8783](https://orcid.org/0000-0001-6301-8783).

---

## License

This repository is licensed under the
[Creative Commons Attribution–NonCommercial 4.0 International
license (CC BY-NC 4.0)](LICENSE) while the accompanying paper is
under review.  Upon paper acceptance the license will be updated to
MIT.

You are free to share and adapt the material for non-commercial
purposes with appropriate attribution.  **Please cite the paper**
(see the [Citation](#citation) section above) in any work that uses
this code, the math semantic graph design, the evaluation harness,
or any of the per-book artefacts.

---

## Acknowledgements

This work uses *The Elements of Statistical Learning* (Hastie,
Tibshirani, Friedman, 2nd ed.) as the working corpus.  The corpus
itself is copyrighted and is not redistributed in this repository;
the publisher's freely available author-hosted copy is referenced in
the paper's bibliography.

The bundled SeVim diagram engine is a separate scientific
contribution; see `sevim/README.md` and the SeVim Zenodo preprint
([10.5281/zenodo.20011107](https://doi.org/10.5281/zenodo.20011107))
for the standalone description.

The implementation was prepared with the help of generative AI
writing assistants (Anthropic Claude) for prose editing,
copy-editing, and reconciling numerical claims against the working
tree.  All scientific contributions, design decisions and reported
results are the authors' own.
