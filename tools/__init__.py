"""Offline tooling for Lyceum.

This package collects the command-line and library entry points that
run *outside* the live narration loop.  Conceptually they break into
four families:

* **Sidecar builders** --- produce the per-book artefacts the runtime
  reads at session start.  Each builder is idempotent and re-runnable;
  re-running on an existing artefact monotonically enriches it.

  - ``build_math_graph``        -- Phase-0 + Phase-1 math semantic graph
  - ``build_concept_layer``     -- five-level concept rewrites
  - ``build_formula_layer``     -- three-level per-formula explanations
  - ``build_formula_explanations`` -- kind-aware section explanations
  - ``build_chapter_map``       -- top-down chapter-zoom narration tree
  - ``build_chapter_visualization`` -- per-chapter visualisation prep
  - ``build_sevim_diagrams``    -- SeVim concept diagrams per section
  - ``extract_book_figures``    -- figure / caption extraction
  - ``extract_inline_formulas`` -- inline-formula recovery pass
  - ``reingest_figures``        -- post-hoc PyMuPDF figure recovery
  - ``reingest_equations``      -- VLM-driven equation OCR
  - ``repair_formula_latex``    -- OCR / ligature repair for LaTeX

* **Quality-assurance agents** --- three tiers of audits.

  - ``data_quality_agent``      -- Tier-1 offline data audit
  - ``smoke_agent``             -- Tier-3 end-to-end SLA harness
  - ``fidelity_agent``          -- math-fidelity self-healing loop

* **Pipeline driver** --- the local-LLM ingest agent that decides
  which sidecars need (re)building given the current health report.

  - ``ingest_agent``            -- closed-loop "observe then act" driver
  - ``ingest_existing_chapters`` -- bulk refresh helper

* **Capture and render helpers** --- used by the paper's figure-capture
  step (``capture_paper_screenshots``, ``record_narration``) and by
  the OCR fallback (``ocr_engines``, ``vlm_engine``).

Every script under this package is safe to run standalone:

    python -m tools.build_math_graph books/ESLII.json
    python -m tools.fidelity_agent books/ESLII.json --apply
    python -m tools.smoke_agent --book ESLII

See ``serve/ingest_pipeline.py`` for the exact phase ordering applied
to a freshly uploaded PDF.

Citation
--------
If you use any of these tools in your research, please cite the
Lyceum paper.  See ``CITATION.cff`` and ``NOTICE`` at the repository
root.
"""
