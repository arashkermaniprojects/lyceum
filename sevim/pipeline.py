"""Main pipeline orchestrator — S1 → S5 end-to-end.

Call run_pipeline() with a string of natural language.  It returns a
PipelineResult containing the SVG, the final scene graph, the placed
geometry, and a per-stage trace log for debugging.

Multi-turn / streaming usage
----------------------------
Pass ``graph=<prev_result.graph>`` to extend an existing diagram instead
of starting from scratch.  Each new clause is extracted into the same
SceneGraph, which then continues through S3–S5.

Overlap handling
----------------
After layout (S4), a geometry check (S4.5) runs unconditionally.
In the default mode, any overlapping shapes are recorded in the trace log
and the pipeline continues to render.  Set SEVIM_STRICT_OVERLAPS=1 to
raise OverlapError instead and produce no SVG.  Overlaps are never
auto-corrected — they are a cosmetic issue that does not violate any
invariant (I1–I4).
"""
from __future__ import annotations

import os

from .ir import PipelineResult, SceneGraph, TraceEvent
from .overlap import detect_overlaps, assert_no_overlaps
from .s1_parse import parse_text
from .s2_extract import extract
from .s2b_improve import improve as improve_graph
from .s3_map import map_visual
from .s4_layout import layout
from .s5_render import render
from .strict_layout import resolve_overlaps, violations


def run_pipeline(
    text: str,
    utterance_id: str = "u0",
    graph: SceneGraph | None = None,
) -> PipelineResult:
    """Run the full S1→S5 pipeline on *text*.

    Parameters
    ----------
    text:
        Natural-language input (one or more sentences / clauses).
    utterance_id:
        Identifier for this turn; increment across turns in streaming mode
        so provenance spans can be attributed to specific utterances.
    graph:
        Pass a previous PipelineResult.graph to extend an existing diagram
        (streaming / multi-turn).  Pass None (default) for a fresh render.

    Returns
    -------
    PipelineResult
        .svg        — canonical SVG string (byte-identical for identical inputs)
        .graph      — final SceneGraph (after optional S2b improvement)
        .placed     — PlacedGraph with all x/y coordinates
        .trace      — list of TraceEvents, one per stage, for debugging
    """
    trace: list[TraceEvent] = []

    # S1 — split text into clause tokens.
    tokens = parse_text(text, utterance_id=utterance_id)
    trace.append(TraceEvent("S1", f"parsed {len(tokens)} spans",
                            {"token_count": len(tokens)}))

    # S2 — extract semantic triples and build/extend the SceneGraph.
    graph = extract(tokens, graph=graph)
    trace.append(TraceEvent(
        "S2",
        f"graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges",
        {
            "node_ids": [n.id for n in graph.nodes],
            "edge_ids": [e.id for e in graph.edges],
        },
    ))

    # S2b (optional) — LLM-based graph inspection and improvement.
    # Disabled unless SEVIM_IMPROVE=1 and ANTHROPIC_API_KEY are set.
    graph = improve_graph(graph, text)
    trace.append(TraceEvent(
        "S2b",
        f"after improve: {len(graph.nodes)} nodes, {len(graph.edges)} edges",
        {},
    ))

    # S3 — map semantic graph to visual shapes and connectors.
    vg = map_visual(graph)
    trace.append(TraceEvent(
        "S3",
        f"mapped {len(vg.shapes)} shapes, {len(vg.connectors)} conns",
        {},
    ))

    # S4 — assign absolute x/y coordinates with the rule-tree layout engine.
    pg = layout(vg)
    trace.append(TraceEvent("S4", "layout applied",
                            {"canvas": (pg.canvas_w, pg.canvas_h)}))

    # S4.6 — strict-layout post-pass: provable non-overlap + min_gap spacing
    # between top-level groups.  Opt-in because it's slower (1–3 ms) and
    # changes coordinates; useful for offline figures where correctness
    # matters more than turn-by-turn determinism with previous renderings.
    if os.environ.get("SEVIM_STRICT_LAYOUT") == "1":
        # Reconstruct parent_of from the visual graph's container hierarchy.
        parent_of: dict[str, str] = {}
        for parent, children in vg.containers:
            for child in children:
                parent_of.setdefault(child, parent)
        before = len(violations(pg, parent_of))
        pg = resolve_overlaps(pg, parent_of)
        after = len(violations(pg, parent_of))
        trace.append(TraceEvent(
            "S4.6", f"strict layout: {before} → {after} violations",
            {"before": before, "after": after,
             "canvas": (pg.canvas_w, pg.canvas_h)},
        ))

    # S4.5 — geometry check.  Strict mode raises; default mode logs to trace.
    if os.environ.get("SEVIM_STRICT_OVERLAPS") == "1":
        assert_no_overlaps(pg)
        trace.append(TraceEvent("S4.5", "overlap check clean", {}))
    else:
        issues = detect_overlaps(pg)
        trace.append(TraceEvent("S4.5", f"overlap findings: {len(issues)}",
                                {"findings": issues[:5]}))

    # S5 — serialise PlacedGraph to canonical SVG.
    svg = render(pg)
    trace.append(TraceEvent("S5", f"svg {len(svg)} bytes", {}))

    return PipelineResult(svg=svg, graph=graph, placed=pg, trace=trace)
