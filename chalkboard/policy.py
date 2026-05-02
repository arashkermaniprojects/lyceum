"""Layout & eviction policies for the chalkboard.

Two pluggable policies decide:

  * **Placement** — where a newly-added shape goes on the canvas.
  * **Eviction** — which shape leaves when the cap is reached.

Default: ``ReadingOrderPolicy`` — left-to-right, top-to-bottom flow with
wrap-around at the canvas right edge.  Eviction is FIFO with fade-out:
the oldest shape is marked ``fading=True`` for one cycle, then removed.

Policies are stateless; they receive the chalkboard and the new shape and
return either a placement (mutating the shape's x/y in place) or a list
of shapes to evict.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Chalkboard, ChalkShape


# ---------------------------------------------------------------------------
# Base interface (duck-typed)
# ---------------------------------------------------------------------------

class LayoutPolicy:
    name: str = ""

    def place(self, board: "Chalkboard", shape: "ChalkShape") -> None:
        """Set shape.x and shape.y based on board state."""
        raise NotImplementedError

    def evict(self, board: "Chalkboard") -> list["ChalkShape"]:
        """Return the shapes that should be removed now."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Reading-order policy — left to right, top to bottom, wrap at canvas edge.
# ---------------------------------------------------------------------------

class ReadingOrderPolicy(LayoutPolicy):
    """Default policy: places shapes in reading order."""
    name = "reading_order"

    def __init__(self, gap: float = 18.0, padding: float = 24.0,
                 row_advance: float = 18.0) -> None:
        self.gap = gap
        self.padding = padding
        self.row_advance = row_advance

    def place(self, board: "Chalkboard", shape: "ChalkShape") -> None:
        if not board.shapes:
            shape.x = self.padding
            shape.y = self.padding
            return
        # Reading-order placement WITH overlap avoidance.  We try each
        # candidate position in scan order (right of last, then wrap)
        # and keep advancing until the shape's AABB clears every
        # existing shape's AABB.  This handles tall cards (canonical
        # diagrams ~360 px) sitting next to short cards (formula cards
        # ~96 px) on the previous row — without this, the tall card
        # would overlap the next-row shape that was placed under the
        # short one.
        last = board.shapes[-1]
        # First candidate: right of the last shape on its row.
        cand_x = last.x + last.w + self.gap
        cand_y = last.y
        if cand_x + shape.w + self.padding > board.canvas_w:
            # Doesn't fit in the row → wrap.
            cand_x = self.padding
            cand_y = self._next_row_y(board, after=last.y)

        # If the candidate position overlaps any existing shape, drop
        # below the deepest overlapping shape and try again from the
        # left margin.  Loop is bounded by the number of existing rows.
        for _ in range(len(board.shapes) + 4):
            colliders = self._colliders(board, cand_x, cand_y, shape)
            if not colliders:
                shape.x = cand_x
                shape.y = cand_y
                return
            # Push below the lowest collider's bottom edge.
            new_y = max(c.y + c.h for c in colliders) + self.row_advance
            cand_x = self.padding
            cand_y = new_y
        # Pathological fallback — place at the bottom regardless.
        shape.x = self.padding
        shape.y = max(s.y + s.h for s in board.shapes) + self.row_advance

    def _next_row_y(self, board: "Chalkboard", *, after: float) -> float:
        """Y of the row immediately below ``after``."""
        row_h = max(
            (s.h for s in board.shapes if abs(s.y - after) < 1.0),
            default=0.0,
        )
        return after + row_h + self.row_advance

    def _colliders(
        self, board: "Chalkboard", x: float, y: float, shape: "ChalkShape",
    ) -> list["ChalkShape"]:
        """Existing shapes whose AABB intersects (x, y, shape.w, shape.h)."""
        x2 = x + shape.w
        y2 = y + shape.h
        out: list["ChalkShape"] = []
        for s in board.shapes:
            sx2, sy2 = s.x + s.w, s.y + s.h
            # Two AABBs intersect when neither is fully on one side.
            if x < sx2 and s.x < x2 and y < sy2 and s.y < y2:
                out.append(s)
        return out

    def evict(self, board: "Chalkboard") -> list["ChalkShape"]:
        if len(board.shapes) <= board.max_content:
            return []
        # FIFO: evict the oldest until under cap.  Implement a one-cycle
        # fade by first marking, then removing on the next overflow.
        excess = len(board.shapes) - board.max_content
        candidates = sorted(board.shapes, key=lambda s: -s.age)[:excess + 1]
        # Mark not-yet-fading as fading; return only those already fading.
        fading_now: list["ChalkShape"] = []
        already_fading: list["ChalkShape"] = []
        for s in candidates:
            if s.fading:
                already_fading.append(s)
            elif len(fading_now) < excess:
                s.fading = True
                fading_now.append(s)
        # Remove only those that have been faded in a previous cycle.
        return already_fading


# ---------------------------------------------------------------------------
# Compact policy — replace the oldest shape in-place.
# ---------------------------------------------------------------------------

class CompactPolicy(LayoutPolicy):
    """Reuses the oldest shape's slot for the newest shape (in-place
    replacement).  Good for fast-moving narrations where the layout
    shouldn't grow."""
    name = "compact"

    def __init__(self, gap: float = 18.0, padding: float = 24.0) -> None:
        self.gap = gap
        self.padding = padding

    def place(self, board: "Chalkboard", shape: "ChalkShape") -> None:
        if not board.shapes:
            shape.x = self.padding
            shape.y = self.padding
            return
        # Find an "open" slot — the position of the oldest shape is the
        # natural target if we're at cap.  Otherwise extend.
        if len(board.shapes) >= board.max_content:
            oldest = max(board.shapes, key=lambda s: s.age)
            shape.x, shape.y = oldest.x, oldest.y
            return
        last = board.shapes[-1]
        candidate_x = last.x + last.w + self.gap
        if candidate_x + shape.w + self.padding > board.canvas_w:
            shape.x = self.padding
            shape.y = last.y + last.h + self.gap
        else:
            shape.x = candidate_x
            shape.y = last.y

    def evict(self, board: "Chalkboard") -> list["ChalkShape"]:
        if len(board.shapes) <= board.max_content:
            return []
        oldest = max(board.shapes, key=lambda s: s.age)
        return [oldest]
