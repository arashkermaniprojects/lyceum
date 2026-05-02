"""Chalkboard state — accumulating whiteboard with adjustable max-content cap.

The chalkboard is a *stateful* surface, distinct from SeVim's stateless
SVG renderer.  As the narrator speaks, new shapes are *appended* to the
chalkboard; when the cap is reached, oldest shapes start fading out (or,
in compact mode, a layout shift moves the new shapes to a fresh column).

Operations
----------
  ``add(shape)``      — append a new ResolvedShape with its SVG fragment.
  ``connect(a, b, …)`` — add an edge between two existing shapes.
  ``highlight(nid)``  — mark a shape as currently-spoken.
  ``erase(nid)``      — explicitly remove a shape.
  ``clear()``         — wipe the board.
  ``snapshot()``      — full SVG of the current board state.
  ``deltas_since(t)`` — just the operations since timestamp *t* (for
                        WebSocket streaming).

Determinism
-----------
- Shape positions are computed by the chalkboard's layout policy (default
  left-to-right reading order with wrap-around).
- The same op sequence always produces the same SVG bytes.
- Time stamps are stored as monotonic floats relative to session start.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .policy import LayoutPolicy, ReadingOrderPolicy


# ---------------------------------------------------------------------------
# Operation log
# ---------------------------------------------------------------------------

@dataclass
class ChalkOp:
    """One operation in the chalkboard's history."""
    t: float                  # seconds since session start
    kind: str                 # "add" | "connect" | "highlight" | "erase" | "clear"
    nid: str
    payload: dict             # kind-specific data:
                              #   add: {"svg": str, "label": str, "primitive": str}
                              #   connect: {"to": str, "relation": str}
                              #   highlight: {"on": bool}
                              #   erase: {}
                              #   clear: {}


@dataclass
class ChalkShape:
    """One shape currently visible on the chalkboard."""
    nid: str
    svg_body: str             # SVG fragment ready to embed in <svg>
    primitive: str
    label: str
    x: float = 0.0
    y: float = 0.0
    w: float = 160.0
    h: float = 90.0
    age: int = 0              # ops applied since this shape was added
    fading: bool = False
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chalkboard
# ---------------------------------------------------------------------------

@dataclass
class Chalkboard:
    """Stateful accumulating whiteboard.

    Attributes
    ----------
    canvas_w / canvas_h:
        Visible canvas size; oversized content fades or scrolls per policy.
    max_content:
        Maximum number of shapes simultaneously visible.  When exceeded,
        ``policy.evict`` decides what happens (default: oldest fades out).
    policy:
        :class:`LayoutPolicy` instance controlling shape placement and
        eviction.  Default: ``ReadingOrderPolicy`` (left-to-right wrap).
    shapes:
        Currently-visible shapes, in insertion order.
    ops:
        Append-only log of every ChalkOp ever applied.  Used for replay
        and for ``deltas_since``.
    started_at:
        Monotonic timestamp of session start.  All ChalkOp.t values are
        relative to this.
    """
    canvas_w: float = 1280.0
    canvas_h: float = 720.0
    max_content: int = 8
    policy: LayoutPolicy = field(default_factory=ReadingOrderPolicy)
    shapes: list[ChalkShape] = field(default_factory=list)
    ops: list[ChalkOp] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)

    # ----- core mutators ----------------------------------------------------

    def _now(self) -> float:
        return time.monotonic() - self.started_at

    def add(
        self,
        nid: str,
        svg_body: str,
        primitive: str,
        label: str,
        *,
        w: float = 160.0,
        h: float = 90.0,
        meta: Optional[dict] = None,
    ) -> ChalkShape:
        """Append a new shape; evict per policy when over cap."""
        shape = ChalkShape(
            nid=nid, svg_body=svg_body, primitive=primitive,
            label=label, w=w, h=h, meta=dict(meta or {}),
        )
        # Place it via the layout policy.
        self.policy.place(self, shape)
        self.shapes.append(shape)
        # Age existing shapes; the new shape stays at age 0.
        for s in self.shapes[:-1]:
            s.age += 1
        # Evict if over cap.
        evicted = self.policy.evict(self)
        for ev in evicted:
            self._record("erase", ev.nid, {"reason": "cap"})
            self.shapes.remove(ev)
        self._record("add", nid, {
            "svg": svg_body, "label": label, "primitive": primitive,
            "x": shape.x, "y": shape.y, "w": shape.w, "h": shape.h,
        })
        return shape

    def connect(
        self,
        from_nid: str, to_nid: str, relation: str,
    ) -> None:
        """Record an edge between two existing shapes.

        The actual SVG line is computed by the rendering layer at snapshot
        time (so it reflects any post-add layout shifts).
        """
        if not any(s.nid == from_nid for s in self.shapes):
            return
        if not any(s.nid == to_nid for s in self.shapes):
            return
        self._record("connect", from_nid, {
            "to": to_nid, "relation": relation,
        })

    def highlight(self, nid: str, on: bool = True) -> None:
        """Toggle the 'currently spoken' marker on a shape."""
        if not any(s.nid == nid for s in self.shapes):
            return
        self._record("highlight", nid, {"on": on})

    def erase(self, nid: str) -> None:
        """Remove a shape by id."""
        before = len(self.shapes)
        self.shapes = [s for s in self.shapes if s.nid != nid]
        if len(self.shapes) < before:
            self._record("erase", nid, {"reason": "explicit"})

    def clear(self) -> None:
        """Wipe the board."""
        self.shapes = []
        self._record("clear", "", {})

    def _record(self, kind: str, nid: str, payload: dict) -> None:
        self.ops.append(ChalkOp(t=self._now(), kind=kind, nid=nid, payload=payload))

    # ----- queries ----------------------------------------------------------

    def deltas_since(self, t: float) -> list[ChalkOp]:
        """Return every operation with op.t > t — for WebSocket streaming."""
        return [op for op in self.ops if op.t > t]

    def snapshot(self) -> str:
        """Render the current state as a complete SVG document."""
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.canvas_w:g}" height="{self.canvas_h:g}" '
            f'viewBox="0 0 {self.canvas_w:g} {self.canvas_h:g}">',
        ]
        # Connectors (from ops) drawn first, then shapes.
        existing = {s.nid for s in self.shapes}
        connect_ops = [op for op in self.ops
                       if op.kind == "connect"
                       and op.nid in existing
                       and op.payload.get("to") in existing]
        nid_to_shape = {s.nid: s for s in self.shapes}
        for op in connect_ops:
            a = nid_to_shape[op.nid]
            b = nid_to_shape[op.payload["to"]]
            ax, ay = a.x + a.w / 2, a.y + a.h / 2
            bx, by = b.x + b.w / 2, b.y + b.h / 2
            parts.append(
                f'<line x1="{ax:g}" y1="{ay:g}" x2="{bx:g}" y2="{by:g}" '
                f'stroke="#666" stroke-width="1.2" stroke-dasharray="3,3"/>'
            )
        for s in self.shapes:
            opacity = 0.4 if s.fading else 1.0
            parts.append(
                f'<g transform="translate({s.x:g},{s.y:g})" '
                f'opacity="{opacity:g}">{s.svg_body}</g>'
            )
        parts.append("</svg>")
        return "".join(parts)

    def replay_to(self, ops: list[ChalkOp]) -> None:
        """Reset state and apply *ops* in order.  Used for deterministic
        playback / debugging."""
        self.shapes = []
        self.ops = []
        self.started_at = time.monotonic()
        for op in ops:
            if op.kind == "add":
                self.add(op.nid, op.payload.get("svg", ""),
                         op.payload.get("primitive", "rect"),
                         op.payload.get("label", ""),
                         w=op.payload.get("w", 160.0),
                         h=op.payload.get("h", 90.0))
            elif op.kind == "connect":
                self.connect(op.nid, op.payload["to"], op.payload["relation"])
            elif op.kind == "highlight":
                self.highlight(op.nid, op.payload.get("on", True))
            elif op.kind == "erase":
                self.erase(op.nid)
            elif op.kind == "clear":
                self.clear()
