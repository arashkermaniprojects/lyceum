"""Loader + lookup for the pre-computed multi-level narrative
explanations produced by ``tools.build_concept_layer``.

Schema on disk (``<book>.concepts.json``)::

    {
      "book": "...",
      "model": "...",
      "by_home_nid": {
        "<nid>": {
          "home_nid": "<nid>",
          "title": "...",
          "L0_gist": "...",
          "L1_story": "...",
          "L2_with_formulas": "...",
          "L3_connections": "...",
          "L4_anchor": "...",
          "metaphor": "...",
          "prerequisites": [...],
          "key_formula_labels": [...]
        }
      }
    }
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional


# Default detail level: L1 (story without formulas).  Picked because
# it sets up the *why* without overwhelming the learner with notation
# right out of the gate.  L0 is too terse to be a stand-alone preamble;
# L2+ is for users who want depth.
DEFAULT_LEVEL = "L1_story"

# Allowed levels in user-facing detail order: L0 (punchline) → L4
# (worked example).  Validated by the orchestrator before use.
LEVELS = (
    "L0_gist",
    "L1_story",
    "L2_with_formulas",
    "L3_connections",
    "L4_anchor",
)


@dataclass
class SectionConcept:
    home_nid: str
    title: str
    L0_gist: str = ""
    L1_story: str = ""
    L2_with_formulas: str = ""
    L3_connections: str = ""
    L4_anchor: str = ""
    metaphor: str = ""
    prerequisites: list[str] = field(default_factory=list)
    key_formula_labels: list[str] = field(default_factory=list)

    def text_at(self, level: str) -> str:
        """Return the narration text for *level*, falling back to the
        nearest non-empty level if the requested one is missing.

        The fallback order favours keeping the same *modality*: a
        request for L1 falls back to L0 (still story-mode) rather than
        L2 (formula-mode).  Returns ``""`` when nothing usable exists.
        """
        if level in LEVELS and getattr(self, level, ""):
            return getattr(self, level)
        # Walk outward from the requested level: same-direction
        # neighbours first, then opposite-direction.
        try:
            idx = LEVELS.index(level)
        except ValueError:
            idx = 1
        order = []
        for d in range(1, len(LEVELS)):
            for cand in (idx - d, idx + d):
                if 0 <= cand < len(LEVELS):
                    order.append(LEVELS[cand])
        for cand in order:
            if getattr(self, cand, ""):
                return getattr(self, cand)
        return ""


class ConceptLayer:
    """In-memory index keyed by ``home_nid``.

    Construction is cheap (parse one JSON); lookup is O(1).  Designed
    to live on the orchestrator so every clause-emission can ask
    "do I have a story for this section?" without disk I/O.
    """

    def __init__(self,
                 sections: Optional[dict[str, SectionConcept]] = None,
                 *, source: str = "") -> None:
        self.sections: dict[str, SectionConcept] = sections or {}
        self.source = source

    # ---- factories --------------------------------------------------------

    @classmethod
    def empty(cls) -> "ConceptLayer":
        return cls(sections={})

    @classmethod
    def load(cls, path: str) -> "ConceptLayer":
        if not os.path.exists(path):
            return cls.empty()
        with open(path, "r") as f:
            raw = json.load(f) or {}
        sections: dict[str, SectionConcept] = {}
        for nid, d in (raw.get("by_home_nid") or {}).items():
            sections[nid] = SectionConcept(
                home_nid=d.get("home_nid", nid),
                title=d.get("title", ""),
                L0_gist=d.get("L0_gist", ""),
                L1_story=d.get("L1_story", ""),
                L2_with_formulas=d.get("L2_with_formulas", ""),
                L3_connections=d.get("L3_connections", ""),
                L4_anchor=d.get("L4_anchor", ""),
                metaphor=d.get("metaphor", ""),
                prerequisites=list(d.get("prerequisites") or []),
                key_formula_labels=list(d.get("key_formula_labels") or []),
            )
        return cls(sections=sections, source=path)

    # ---- queries ----------------------------------------------------------

    def has(self, home_nid: str) -> bool:
        return bool(home_nid) and home_nid in self.sections

    def get(self, home_nid: str) -> Optional[SectionConcept]:
        return self.sections.get(home_nid)

    def text_for(self, home_nid: str, level: str = DEFAULT_LEVEL) -> str:
        """Return the narration text for *home_nid* at *level*, or ""."""
        sc = self.sections.get(home_nid)
        if sc is None:
            return ""
        return sc.text_at(level)

    def __len__(self) -> int:
        return len(self.sections)

    def __repr__(self) -> str:
        return (f"ConceptLayer(sections={len(self.sections)}, "
                f"source={self.source!r})")
