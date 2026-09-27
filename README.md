# Lyceum

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20043457.svg)](https://doi.org/10.5281/zenodo.20043457)
[![Licence: CC BY-NC 4.0](https://img.shields.io/badge/licence-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

> **If you use this software, the math-semantic-graph design, the
> evaluation harness, or any of the per-book artefacts in your
> research, you MUST cite the accompanying paper
> (DOI: [10.5281/zenodo.20043457](https://doi.org/10.5281/zenodo.20043457)).**
> See the [Citation](#citation) section below and the
> [`NOTICE`](NOTICE) file at the repository root.  This is a
> condition of the licence (see [`LICENSE`](LICENSE); CC BY-NC 4.0
> requires attribution).

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

All model inference in the runtime stack (text LLM, vision-language
inspector, sentence embedder, text-to-speech, speech recognition) is
served on the user's own machine; no book content is sent to an
external model API by default.  Three exceptions are worth knowing:

- the browser UI loads KaTeX (CSS/JS) from the jsDelivr CDN;
- the SeVim graph-improvement step (`sevim/s2b_improve.py`) calls the
  Anthropic API, but only when explicitly enabled with
  `SEVIM_IMPROVE=1` and an `ANTHROPIC_API_KEY`;
- the optional M6 narration-quality judge in `bench/eval/` uses the
  Anthropic API; M1 – M5 do not.

> **Status:** the accompanying paper is available as a preprint on
> Zenodo (DOI [10.5281/zenodo.20043457](https://doi.org/10.5281/zenodo.20043457),
> 2026-05-05).  This repository ships the implementation, the
> evaluation harness and the raw JSON results behind the reported
> measurements.  Per-book artefacts (`books/`) are not included
> because the source textbooks are copyrighted: M1 reproduces from
> this repository alone, while M2 – M5 require you to ingest your own
> copy of the book first (see below).

---

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20043457.svg)](https://doi.org/10.5281/zenodo.20043457)

If you use Lyceum, the math semantic graph design, the evaluation
harness, or any of the material in this repository in your research,
**please cite the paper**:

```bibtex
@misc{lyceum2026,
  title     = {{Lyceum}: a fully-local multimodal system for narrated
               visualisation of mathematical textbooks},
  author    = {Kermani Kolankeh, Arash and Zgheib, Rita},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20043457},
  url       = {https://doi.org/10.5281/zenodo.20043457},
  note      = {Preprint.  Version 1 archived as
               doi:10.5281/zenodo.20043458}
}
```

The DOI above is the Zenodo *concept DOI* — it always resolves to
the latest version of the preprint.  Use the version-pinned DOI
`10.5281/zenodo.20043458` only when you need to refer to v1
(2026-05-05) specifically.

The bundled SeVim diagram engine has its own citation under
[`sevim/README.md`](sevim/README.md); please cite both when you use
the diagram-rendering pipeline directly.

A `CITATION.cff` file at the repository root resolves to the same
record so GitHub's "Cite this repository" button works end-to-end.

---

## Highlights

- **Fully local.** vLLM serves a quantised text LLM, an embedding
  model and a vision-language inspector; Kokoro is the text-to-speech
  subprocess.  By default no model request leaves the user's machine
  (see the exceptions listed above).
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
├── chalkboard/       Python whiteboard-state model (accumulating board,
│                     placement / eviction policies); the browser-side
│                     audio-clock rAF sync loop lives in
│                     serve/static/index.html
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
├── tests/            708 deterministic unit / system tests
├── discovery/        Design notes (math-graph plan, inspector design,
│                     visual relation QA gallery)
└── books/            Per-book artefacts (corpus JSON, math graph,
                      figures sidecar, chapter-map sidecars, …) ---
                      gitignored because the source PDFs are
                      copyrighted
```

---

## Quick start

### Prerequisites

- Python 3.10+
- `pip install -e ".[dev,book]" numpy` from the repository root (run
  all commands from the repository root; `narrator/`, `serve/`,
  `tools/` and `viz/` are used in place, not installed as packages)
- A Linux-style box with reasonable RAM
- A workstation GPU is recommended for the full local model fabric;
  the runtime degrades gracefully when individual model servers are
  unreachable
- Optional: vLLM endpoints for the text LLM, the vision-language
  inspector, and the embedding model; Kokoro for TTS

### Run an end-to-end ingestion + serve

```bash
# Install
pip install -e ".[dev,book]" numpy

# Put your own copy of the PDF into books/ — say, ESLII.pdf
mkdir -p books
cp /path/to/ESLII.pdf books/

# 1. Parse the PDF into the book corpus JSON (writes books/ESLII.json)
python -m book.cli books/ESLII.pdf

# 2. Build the sidecars (concept layer, math graph, per-chapter maps,
#    data-quality inspector, math-fidelity agent).  The ingest agent
#    takes the corpus JSON, not the PDF, and uses the local text LLM
#    at 127.0.0.1:8000 to decide which builder to run next.
python -m tools.ingest_agent books/ESLII.json

# 3. Start the multi-book server
python -m serve.server books/ESLII.json
# Browse to http://127.0.0.1:8001
```

Alternatively, uploading a PDF through the running web UI triggers
the same phases automatically (`serve/ingest_pipeline.py`).

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
`build_tables.py` rebuilds the LaTeX fragments in
`bench/eval/tables/*.tex` mechanically; the combined
`measurement_summary.tex` digest is produced by `build_summary` from
the same JSON records.  M1 runs from the committed files alone;
M2 – M5 read `books/ESLII.json` and `books/ESLII.math_graph.json`,
which you must build from your own copy of the book (see above).
`build_tables.py` itself only needs the committed JSON results.

### Run the test suite

```bash
pip install -e ".[dev,book]" numpy   # pytest, PyMuPDF, NumPy
pytest -q
```

`pyproject.toml` puts the repository root on the test path, so plain
`pytest` works from the root.  There are currently 708 collected
deterministic tests; tests that need a locally ingested book
(`books/ESLII.json`) or a live model server skip when those are
absent.  The suite covers the router, the
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
license (CC BY-NC 4.0)](LICENSE), copyright (c) 2026 Arash Kermani
Kolankeh and Rita Zgheib.  The citation requirement carried by the
[`NOTICE`](NOTICE) file is part of the licence terms.

You are free to share and adapt the material for non-commercial
purposes **with appropriate attribution**.  CC BY-NC 4.0 §3(a)
makes preserving the attribution notice in [`NOTICE`](NOTICE) part
of the licence terms; the academic citation requirement (see the
[Citation](#citation) section above) is therefore not optional.

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

Parts of the code and documentation were developed with the
assistance of AI coding tools (Anthropic Claude); all content was
reviewed by the authors.
