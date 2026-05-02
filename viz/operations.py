"""Natural-language operation phrases → canonical LaTeX renderings.

When the narrator says "their dot product is zero" the user expects to
*see* ``\\vec{u} \\cdot \\vec{v} = 0`` on the board, not just hear the
words.  This module owns the phrase-to-formula table and the matcher.

Layered match (cheap-first):

  1. **Variant patterns**: each operation may carry several regexes so
     a contextualised mention (e.g. "dot product is zero") renders the
     specific form ``\\vec u \\cdot \\vec v = 0`` rather than the
     generic dot-product expansion.
  2. **Bare patterns**: the operation's primary aliases ("dot product",
     "inner product", …) match anywhere and render the canonical form.

Each matched phrase yields:

    (label, latex, char_offset, end_offset)

so the orchestrator can build a card and time it to TTS word stamps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class _Op:
    key: str
    label: str
    latex: str
    aliases: list[str]
    variants: list[tuple[str, str, str]] = None  # (regex, label, latex)

    def __post_init__(self):
        if self.variants is None:
            self.variants = []


OPERATIONS: list[_Op] = [
    # ---- Vector products ---------------------------------------------------
    _Op(
        key="dot_product",
        label="dot product",
        latex=(r"\vec{u}\cdot\vec{v}"
               r"= u_1 v_1 + u_2 v_2 + \cdots + u_n v_n"),
        aliases=[
            r"\bdot product\b",
            r"\binner product\b",
            r"\bscalar product\b",
        ],
        variants=[
            # "dot product is zero" / "their dot product is 0" → orthogonality.
            (r"\b(?:dot|inner|scalar) product\b[^.]{0,32}\bis\b\s*(?:equal to\s+)?(?:zero|0)\b",
             "dot product = 0 (orthogonal)",
             r"\vec{u}\cdot\vec{v} = 0"),
        ],
    ),
    _Op(
        key="cross_product",
        label="cross product",
        latex=r"\vec{u}\times\vec{v}",
        aliases=[r"\bcross product\b", r"\bvector product\b"],
    ),
    _Op(
        key="outer_product",
        label="outer product",
        latex=r"\vec{u}\,\vec{v}^{\top}",
        aliases=[r"\bouter product\b"],
    ),

    # ---- Scalar arithmetic -------------------------------------------------
    _Op(
        key="multiply_two",
        label="multiplication of two numbers",
        latex=r"x \cdot y",
        aliases=[
            r"\bmultiplication of two numbers\b",
            r"\bproduct of two (?:numbers|scalars|values)\b",
            r"\bmultiplying two (?:numbers|scalars|values)\b",
        ],
    ),
    _Op(
        key="add_two",
        label="addition of two numbers",
        latex=r"x + y",
        aliases=[
            r"\baddition of two numbers\b",
            r"\bsum of two (?:numbers|scalars|values)\b",
            r"\badding two (?:numbers|scalars|values)\b",
        ],
    ),

    # ---- Matrix operations -------------------------------------------------
    _Op(
        key="matrix_mul",
        label="matrix multiplication",
        latex=r"(AB)_{ij} = \sum_k A_{ik}\, B_{kj}",
        aliases=[
            r"\bmatrix multiplication\b",
            r"\bmatrix[\s-]?matrix product\b",
            r"\bmultiplying (?:two )?matrices\b",
            r"\bproduct of two matrices\b",
        ],
    ),
    _Op(
        key="matrix_vector",
        label="matrix–vector product",
        latex=r"y = A\, x, \quad y_i = \sum_j A_{ij} x_j",
        aliases=[
            r"\bmatrix[- ]?vector product\b",
            r"\bmultiplying a matrix (?:by|with|and) a vector\b",
        ],
    ),
    _Op(
        key="transpose",
        label="transpose",
        latex=r"A^{\top}",
        aliases=[
            r"\btranspose(?:s|d)?\b",
            r"\bA transpose\b",
        ],
    ),
    _Op(
        key="inverse",
        label="matrix inverse",
        latex=r"A^{-1}\, A = A\, A^{-1} = I",
        aliases=[
            r"\bmatrix inverse\b",
            r"\binverse of a matrix\b",
            r"\bA inverse\b",
        ],
    ),
    _Op(
        key="determinant",
        label="determinant",
        latex=r"\det(A)",
        aliases=[r"\bdeterminant\b"],
    ),
    _Op(
        key="eigen",
        label="eigen-equation",
        latex=r"A\, v = \lambda\, v",
        aliases=[
            r"\beigen[\s-]?value\s+equation\b",
            r"\beigen[\s-]?vector\s+equation\b",
            r"\beigendecomposition\b",
        ],
    ),

    # ---- Geometry ----------------------------------------------------------
    _Op(
        key="orthogonal_vectors",
        label="orthogonal vectors",
        latex=r"\vec{u}\cdot\vec{v} = 0",
        aliases=[
            r"\borthogonal vectors\b",
            r"\bvectors? (?:are|is) (?:perpendicular|orthogonal)\b",
            r"\bperpendicular vectors?\b",
        ],
    ),
    _Op(
        key="norm",
        label="vector norm",
        latex=r"\|x\| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}",
        aliases=[
            r"\b(?:Euclidean )?norm of (?:a |the )?vector\b",
            r"\b(?:vector )?(?:length|magnitude)\b",
            r"\bL2 norm\b",
        ],
    ),
    _Op(
        key="distance",
        label="Euclidean distance",
        latex=r"d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}",
        aliases=[
            r"\bEuclidean distance\b",
            r"\bdistance between (?:two )?points\b",
        ],
    ),

    # ---- Calculus ----------------------------------------------------------
    _Op(
        key="gradient",
        label="gradient",
        latex=r"\nabla f(x) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)",
        aliases=[
            r"\bgradient of (?:a |the )?function\b",
            r"\bgradient vector\b",
        ],
    ),
    _Op(
        key="derivative",
        label="derivative",
        latex=r"\frac{df}{dx}",
        aliases=[
            r"\bderivative of (?:a |the )?function\b",
            r"\bfirst derivative\b",
        ],
    ),
    _Op(
        key="partial_derivative",
        label="partial derivative",
        latex=r"\frac{\partial f}{\partial x_i}",
        aliases=[
            r"\bpartial derivative\b",
            r"\bpartial of\b",
        ],
    ),
    _Op(
        key="chain_rule",
        label="chain rule",
        latex=r"\frac{d}{dx}\, f(g(x)) = f'(g(x))\, g'(x)",
        aliases=[r"\bchain rule\b"],
    ),
    _Op(
        key="integral",
        label="integral",
        latex=r"\int_a^b f(x)\, dx",
        aliases=[
            r"\bintegral of (?:a |the )?function\b",
            r"\bdefinite integral\b",
        ],
    ),

    # ---- Probability / statistics -----------------------------------------
    _Op(
        key="expectation",
        label="expectation",
        latex=r"\mathbb{E}[X] = \sum_x x\, P(X=x)",
        aliases=[
            r"\bexpected value\b",
            r"\bexpectation of (?:a )?random variable\b",
        ],
    ),
    _Op(
        key="variance",
        label="variance",
        latex=r"\mathrm{Var}(X) = \mathbb{E}\!\left[(X - \mu)^2\right]",
        aliases=[
            r"\bvariance of (?:a )?random variable\b",
            r"\bvariance of X\b",
        ],
    ),
    _Op(
        key="covariance",
        label="covariance",
        latex=r"\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]",
        aliases=[r"\bcovariance\b"],
    ),
    _Op(
        key="correlation",
        label="Pearson correlation",
        latex=r"\rho_{XY} = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}",
        aliases=[
            r"\bPearson correlation\b",
            r"\bcorrelation coefficient\b",
        ],
    ),
    _Op(
        key="bayes",
        label="Bayes' rule",
        latex=r"P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}",
        aliases=[r"\bBayes(?:'s|')? (?:rule|theorem|formula)\b"],
    ),

    # ---- Loss / optimisation ----------------------------------------------
    _Op(
        key="mse",
        label="mean squared error",
        latex=r"\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2",
        aliases=[
            r"\bmean squared error\b",
            r"\bMSE\b",
            r"\bsquared loss\b",
        ],
    ),
    _Op(
        key="cross_entropy",
        label="cross-entropy",
        latex=r"H(p, q) = -\sum_x p(x)\, \log q(x)",
        aliases=[r"\bcross[\s-]?entropy\b"],
    ),
    _Op(
        key="softmax",
        label="softmax",
        latex=r"\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}",
        aliases=[r"\bsoftmax\b"],
    ),
    _Op(
        key="sigmoid",
        label="sigmoid",
        latex=r"\sigma(z) = \frac{1}{1 + e^{-z}}",
        aliases=[r"\b(?:sigmoid|logistic)\s+function\b",
                 r"\blogistic activation\b"],
    ),

    # ---- Linear algebra (scalar concepts) ---------------------------------
    _Op(
        key="affine",
        label="affine transform",
        latex=r"z = W x + b",
        aliases=[
            r"\baffine (?:transform(?:ation)?|map)\b",
            r"\blinear transformation followed by a bias\b",
        ],
    ),
    _Op(
        key="linear_combo",
        label="linear combination",
        latex=r"y = c_1 v_1 + c_2 v_2 + \cdots + c_k v_k",
        aliases=[
            r"\blinear combination\b",
            r"\bweighted sum of vectors\b",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Compilation + matcher
# ---------------------------------------------------------------------------

def _compiled_variants(op: _Op) -> list[tuple[re.Pattern, str, str]]:
    cache_key = f"_var_{op.key}"
    cache = getattr(_compiled_variants, cache_key, None)
    if cache is None:
        cache = [(re.compile(r, re.I), lab, lat)
                 for r, lab, lat in op.variants]
        setattr(_compiled_variants, cache_key, cache)
    return cache


def _compiled_aliases(op: _Op) -> list[re.Pattern]:
    cache_key = f"_ali_{op.key}"
    cache = getattr(_compiled_aliases, cache_key, None)
    if cache is None:
        cache = [re.compile(p, re.I) for p in op.aliases]
        setattr(_compiled_aliases, cache_key, cache)
    return cache


def find_operations(text: str) -> list[tuple[str, str, int, int]]:
    """Return ``[(label, latex, start, end), …]`` for every operation
    phrase mention in *text*.

    Variants are tried first (so contextualised mentions render the
    contextual formula).  When two patterns overlap, the longer match
    wins; identical operations are deduped within this single text.
    """
    if not text:
        return []
    hits: list[tuple[int, int, str, str]] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(a: tuple[int, int]) -> bool:
        for b in consumed:
            if a[0] < b[1] and b[0] < a[1]:
                return True
        return False

    # Pass 1 — variants (more specific).
    for op in OPERATIONS:
        for pat, lab, lat in _compiled_variants(op):
            for m in pat.finditer(text):
                span = (m.start(), m.end())
                if overlaps(span):
                    continue
                consumed.append(span)
                hits.append((m.start(), m.end(), lab, lat))

    # Pass 2 — aliases.
    for op in OPERATIONS:
        for pat in _compiled_aliases(op):
            for m in pat.finditer(text):
                span = (m.start(), m.end())
                if overlaps(span):
                    continue
                consumed.append(span)
                hits.append((m.start(), m.end(), op.label, op.latex))

    hits.sort(key=lambda h: h[0])
    return [(label, latex, start, end) for start, end, label, latex in hits]


def list_operation_keys() -> list[str]:
    return [op.key for op in OPERATIONS]
