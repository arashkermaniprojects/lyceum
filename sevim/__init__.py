"""SeVim --- semantic-to-visual mapping pipeline.

A five-stage neuro-symbolic pipeline that converts natural-language
educational text into deterministic, byte-reproducible SVG diagrams
without hallucination and without external API calls.

Stages
------
S1 ``s1_parse``     clause splitter + span-token tagger
S2 ``s2_extract``   dependency-parse + regex semantic-triple extractor
S3 ``s3_map``       symbolic + frozen-projection visual-graph mapper
S4 ``s4_layout``    deterministic Sugiyama / grid layout
S5 ``s5_render``    SVG serialiser with shape grammar

Inside Lyceum, SeVim is invoked by ``tools/build_sevim_diagrams`` to
produce one concept diagram per textbook section; it also exposes the
``math_graph`` / ``math_graph_phase1`` modules used by the offline
graph builder (``tools/build_math_graph``).

See ``sevim/README.md`` for the standalone story and the SeVim Zenodo
preprint (``10.5281/zenodo.20011107``) for the citation form.

Citation
--------
Work that uses the SeVim pipeline directly should cite the SeVim
preprint above.  Work that uses the bundled Lyceum runtime
(orchestrator, math semantic graph, evaluation harness) should ALSO
cite the Lyceum paper --- see ``CITATION.cff`` and ``NOTICE`` at the
repository root.
"""

__version__ = "0.1.0"
