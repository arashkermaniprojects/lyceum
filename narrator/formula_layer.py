"""Loader + lookup for the pre-computed multi-level formula explanations
produced by ``tools.build_formula_layer``.

Each Formula in the math graph gets three explanation tiers:

    F0_role     short noun phrase ("regularization functional")
    F1_meaning  one-sentence plain-English description
    F2_walk     2-4 sentence walk for users who want to dive in

The orchestrator looks these up by ``formula_id`` whenever it emits a
formula card; the F0/F1 text is rendered onto the card itself so the
learner *reads* the meaning beside the symbols, not just hears the
narrator name a citation label.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


LEVELS = ("F0_role", "F1_meaning", "F2_walk")
DEFAULT_LEVEL = "F1_meaning"


@dataclass
class FormulaExplanation:
    formula_id: str
    cite_label: str = ""
    latex: str = ""
    home_nid: str = ""
    F0_role: str = ""
    F1_meaning: str = ""
    F2_walk: str = ""

    def text_at(self, level: str) -> str:
        if level in LEVELS and getattr(self, level, ""):
            return getattr(self, level)
        # Fall through to nearest-non-empty level.
        try:
            idx = LEVELS.index(level)
        except ValueError:
            idx = 1
        for d in range(1, len(LEVELS)):
            for cand in (idx - d, idx + d):
                if 0 <= cand < len(LEVELS):
                    val = getattr(self, LEVELS[cand], "")
                    if val:
                        return val
        return ""


class FormulaLayer:
    """In-memory index keyed by formula_id with cite-label fallback.

    Two lookups:
      * ``get(formula_id)``   — primary key, exact match
      * ``by_cite(label)``    — for runtime-emitted formulas whose nid
                                doesn't match the offline graph's id
                                but whose cite_label does (e.g. an
                                inline-detected ``Equation 5.42``).
    """

    def __init__(self,
                 entries: Optional[dict[str, FormulaExplanation]] = None,
                 *, source: str = "") -> None:
        self.entries: dict[str, FormulaExplanation] = entries or {}
        # cite-label index, normalised so "Equation 5.43" and "5.43"
        # both map to the same entry.  The runtime extractor stores
        # one form, the offline build stores another; this index
        # papers over both.  Prefer entries whose F0/F1 are populated
        # (some references cards were emitted at runtime without a
        # human-facing role / meaning).
        self._by_cite: dict[str, FormulaExplanation] = {}
        for fe in self.entries.values():
            if not fe.cite_label:
                continue
            for key in self._cite_keys(fe.cite_label):
                cur = self._by_cite.get(key)
                if cur is None or (
                        not cur.F0_role and not cur.F1_meaning
                        and (fe.F0_role or fe.F1_meaning)):
                    self._by_cite[key] = fe
        self.source = source

    @staticmethod
    def _cite_keys(cite_label: str) -> list[str]:
        """Both raw and prefix-stripped forms of a citation label."""
        s = (cite_label or "").strip()
        if not s:
            return []
        keys = [s]
        for p in ("Equation ", "Eq. ", "Eq "):
            if s.lower().startswith(p.lower()):
                keys.append(s[len(p):])
                break
        return keys

    @classmethod
    def empty(cls) -> "FormulaLayer":
        return cls(entries={})

    @classmethod
    def load(cls, path: str) -> "FormulaLayer":
        if not os.path.exists(path):
            return cls.empty()
        with open(path, "r") as f:
            raw = json.load(f) or {}
        entries: dict[str, FormulaExplanation] = {}
        for fid, d in (raw.get("by_formula_id") or {}).items():
            entries[fid] = FormulaExplanation(
                formula_id=d.get("formula_id", fid),
                cite_label=d.get("cite_label", ""),
                latex=d.get("latex", ""),
                home_nid=d.get("home_nid", ""),
                F0_role=d.get("F0_role", ""),
                F1_meaning=d.get("F1_meaning", ""),
                F2_walk=d.get("F2_walk", ""),
            )
        return cls(entries=entries, source=path)

    def get(self, formula_id: str) -> Optional[FormulaExplanation]:
        return self.entries.get(formula_id)

    def by_cite(self, cite_label: str) -> Optional[FormulaExplanation]:
        if not cite_label:
            return None
        # Try every alias of the citation label (raw + prefix-stripped).
        for key in self._cite_keys(cite_label):
            fe = self._by_cite.get(key)
            if fe is not None:
                return fe
        return None

    def lookup(self, *, formula_id: str = "",
               cite_label: str = "") -> Optional[FormulaExplanation]:
        """Combined lookup: try formula_id first, then cite_label."""
        if formula_id:
            fe = self.get(formula_id)
            if fe is not None:
                return fe
        return self.by_cite(cite_label)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (f"FormulaLayer(entries={len(self.entries)}, "
                f"source={self.source!r})")
