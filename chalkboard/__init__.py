"""Chalkboard package — accumulating whiteboard state.

    from chalkboard import Chalkboard, ReadingOrderPolicy, CompactPolicy

    board = Chalkboard(max_content=10, policy=ReadingOrderPolicy())
    board.add("n_matrix", svg_body, "matrix_bracket", "matrix")
    board.connect("n_matrix", "n_vector", "transforms")
    svg = board.snapshot()

Citation
--------
If you use this package in your research, please cite the Lyceum
paper.  See ``CITATION.cff`` and ``NOTICE`` at the repository root.
"""
from .state import Chalkboard, ChalkOp, ChalkShape
from .policy import LayoutPolicy, ReadingOrderPolicy, CompactPolicy

__all__ = [
    "Chalkboard", "ChalkOp", "ChalkShape",
    "LayoutPolicy", "ReadingOrderPolicy", "CompactPolicy",
]
