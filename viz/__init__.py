"""Canonical visualisation generators for the SeVim_math teaching board.

When a passage's narration mentions a canonical concept (overfitting,
bias-variance, ROC, k-fold cross-validation, gradient descent, …) and the
book corpus has no figure attached for that scope, the orchestrator asks
this package to *synthesise* a relevant SVG illustration.

All generators are
  * deterministic — same args → byte-identical SVG;
  * pure-Python / hand-rolled SVG — no matplotlib, no native deps;
  * sub-100 ms — fits comfortably inside the 2 s realtime budget;
  * topic-aware — they don't draw "a generic curve", they draw THE curve
    that defines the concept (e.g. overfitting renders training loss
    monotonically falling while test loss u-shapes upward).

Local-only by policy: no Anthropic / OpenAI / external API is used here.
Optional VLM inspection talks to a *local* Qwen2.5-VL vLLM endpoint.
"""
from .registry import find_visualization, list_topics
from .inspector import inspect_svg, InspectionResult
from .semantic_ir import SemanticEdge, SemanticGraph, SemanticNode
from . import semantic_parser, semantic_to_latex, semantic_to_svg

__all__ = [
    "find_visualization",
    "list_topics",
    "inspect_svg",
    "InspectionResult",
    "SemanticGraph",
    "SemanticNode",
    "SemanticEdge",
    "semantic_parser",
    "semantic_to_svg",
    "semantic_to_latex",
]
