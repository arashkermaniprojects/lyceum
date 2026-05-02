"""Topic→generator routing — embedding-based, with a small regex fast-path.

Each curated topic carries:

  * a ``description`` — a one-line definition we embed once at startup
    against the existing Qwen3-Embedding endpoint at ``:8003``;
  * an optional ``regex_aliases`` list — used as a fast confirmatory
    check after the embedding shortlist (lets unambiguous textual hits
    like ``"sigmoid"`` skip the embedding round-trip altogether).

Selection algorithm (``find_visualization``):

  1. Build the query string from question + title (heaviest signals).
  2. Try each topic's regex aliases against the query.  If exactly one
     topic matches with high specificity, return it immediately
     (sub-millisecond, doesn't depend on the embedding server).
  3. Otherwise embed the query and cosine-rank against the pre-computed
     topic embeddings.  Accept the top topic only if:
        - cosine ≥ ``ACCEPT_THRESHOLD`` (0.55 by default), AND
        - top - second ≥ ``MARGIN`` (0.04 by default).
     This is what stops "synaptic weight" from grabbing
     "back-propagation" just because the section discusses both.
  4. If the embedding server is unreachable, fall through to a
     loose-regex backup (the previous behaviour, kept for resilience).

The thresholds are conservative on purpose — better to render no
canonical figure than to surface an off-topic one.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Callable, Optional

from . import generators as G

try:
    from book import embeddings as _emb  # type: ignore
except Exception:  # pragma: no cover — running without book ingestion installed
    _emb = None  # type: ignore


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Tighter than the historical 0.55 / 0.04 to keep curated diagrams
# pinned to questions that *are* about the curated topic rather than
# adjacent ones.  Calibration on Qwen3-Embedding-0.6B against ESLII:
#   * legitimate queries (overfitting / ROC / k-fold / bias-variance /
#     gradient descent / lasso path) score 0.78–0.89 cosine with
#     0.10–0.26 margin against the runner-up.
#   * adjacent-but-wrong queries (forward-stagewise / deep Boltzmann /
#     hopf algebra) score 0.39–0.62 with margin ≤ 0.04.
# 0.65 / 0.07 leaves a clean gap, with both still env-overridable
# for future re-calibration.
ACCEPT_THRESHOLD = float(os.environ.get("VIZ_TOPIC_THRESHOLD", "0.65"))
MARGIN = float(os.environ.get("VIZ_TOPIC_MARGIN", "0.07"))


# ---------------------------------------------------------------------------
# Curated topic catalog — generator + semantic description + tight aliases
# ---------------------------------------------------------------------------

_TOPIC_CATALOG: list[dict] = [
    {
        "key": "overfitting",
        "generator": G.overfitting_curve,
        "description": (
            "Overfitting: a model fits the training data too closely, "
            "so training error keeps decreasing while test error rises "
            "with model complexity, producing a U-shaped test-error curve."
        ),
        "regex_aliases": [r"\boverfit(?:ting|ed|s)?\b"],
    },
    {
        "key": "bias_variance",
        "generator": G.bias_variance,
        "description": (
            "Bias-variance decomposition: total expected error decomposes "
            "into squared bias plus variance plus irreducible noise; bias "
            "falls and variance rises with model complexity."
        ),
        "regex_aliases": [
            r"\bbias[\s-]*variance\b",
            r"\bvariance[\s-]+decomposition\b",
        ],
    },
    {
        "key": "roc_curve",
        "generator": G.roc_curve,
        "description": (
            "ROC curve: true positive rate plotted against false positive "
            "rate as the classifier threshold sweeps; the area under the "
            "curve (AUC) summarises classifier quality."
        ),
        "regex_aliases": [
            r"\bROC\b",
            r"\breceiver operating characteristic\b",
        ],
    },
    {
        "key": "kfold_split",
        "generator": G.kfold_split,
        "description": (
            "k-fold cross-validation: the dataset is partitioned into k "
            "equal folds; each iteration trains on k-1 folds and tests "
            "on the held-out fold; results are averaged across iterations."
        ),
        "regex_aliases": [
            r"\bk[\s-]*fold\b",
            r"\bcross[\s-]*validation\b",
        ],
    },
    {
        "key": "gradient_descent",
        "generator": G.gradient_descent,
        "description": (
            "Gradient descent: iterative optimisation that updates "
            "parameters in the direction opposite to the loss gradient, "
            "tracing a path through the loss landscape toward a minimum."
        ),
        "regex_aliases": [
            r"\bgradient descent\b",
            r"\b(?:steepest|stochastic) descent\b",
        ],
    },
    {
        "key": "learning_curve",
        "generator": G.learning_curve,
        "description": (
            "Learning curve: training error and validation error plotted "
            "as functions of training set size; convergence and gap "
            "between them diagnose underfitting vs. data-starved regimes."
        ),
        "regex_aliases": [r"\blearning curve\b"],
    },
    {
        "key": "regularization_path",
        "generator": G.regularization_path,
        "description": (
            "Regularization path: trajectories of the coefficient values "
            "as the regularisation strength lambda is swept; lasso paths "
            "shrink coefficients to zero, ridge paths shrink smoothly."
        ),
        "regex_aliases": [
            r"\bregulariz(?:ation|ed) path\b",
            r"\b(?:lasso|ridge)\b.*\bpath\b",
        ],
    },
    {
        "key": "decision_boundary",
        "generator": G.decision_boundary,
        "description": (
            "Decision boundary: the surface in feature space that "
            "separates a classifier's predicted classes; for a linear "
            "classifier this is a hyperplane between two clusters."
        ),
        "regex_aliases": [
            r"\bdecision boundary\b",
            r"\blinear classifier\b",
        ],
    },
    {
        "key": "back_propagation",
        "generator": G.back_propagation,
        "description": (
            "Back-propagation: the algorithm that trains a neural network "
            "by running a forward pass to compute the loss and a backward "
            "pass that uses the chain rule to compute gradients of the "
            "loss with respect to every weight."
        ),
        "regex_aliases": [
            r"\bback[\s-]*propagation\b",
            r"\bbackprop(?:agation)?\b",
        ],
    },
    {
        "key": "activation_functions",
        "generator": G.activation_functions,
        "description": (
            "Activation function: the non-linear scalar function applied "
            "elementwise inside a neural network's hidden units, such as "
            "sigmoid, hyperbolic tangent, or rectified linear (ReLU)."
        ),
        "regex_aliases": [
            r"\bactivation\s+func",
            r"\b(?:sigmoid|tanh|ReLU|softmax)\b",
            r"\bnon[- ]?linearity\b",
        ],
    },
    {
        "key": "weight_matrix",
        "generator": G.weight_matrix,
        "description": (
            "Weight matrix in a neural network: the matrix W of learned "
            "weights connecting one layer's units to the next, applied "
            "via the affine transform z = W x + b before the activation."
        ),
        "regex_aliases": [
            r"\bweight\s+matri", r"\bweights?\s+of\s+the\s+(?:layer|network)",
        ],
    },
    {
        "key": "loss_function",
        "generator": G.loss_function,
        "description": (
            "Loss function R(theta): a scalar measure of model error as "
            "a function of its parameters; training minimises this loss "
            "by adjusting theta toward a minimum of the loss surface."
        ),
        "regex_aliases": [
            r"\bloss\s+function\b",
            r"\bcost\s+function\b",
            r"\bobjective\s+function\b",
            r"\berror\s+function\b",
        ],
    },
]


# ---------------------------------------------------------------------------
# Embedding cache — populated on first lookup
# ---------------------------------------------------------------------------

_emb_lock = threading.Lock()
_topic_embeddings: Optional[list[tuple[str, Callable, tuple[float, ...]]]] = None
_emb_unavailable = False


def _ensure_topic_embeddings() -> Optional[list[tuple[str, Callable, tuple[float, ...]]]]:
    """Pre-embed every topic description.  Returns None if the embedding
    server is unreachable, in which case callers fall back to regex.
    """
    global _topic_embeddings, _emb_unavailable
    with _emb_lock:
        if _topic_embeddings is not None or _emb_unavailable:
            return _topic_embeddings
        if _emb is None or not _emb.is_available():
            _emb_unavailable = True
            return None
        descriptions = [t["description"] for t in _TOPIC_CATALOG]
        try:
            vecs = _emb.embed_batch(descriptions)
        except Exception:
            _emb_unavailable = True
            return None
        rows: list[tuple[str, Callable, tuple[float, ...]]] = []
        for spec, v in zip(_TOPIC_CATALOG, vecs):
            if v:
                rows.append((spec["key"], spec["generator"], v))
        if not rows:
            _emb_unavailable = True
            return None
        _topic_embeddings = rows
        return _topic_embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_topics() -> list[str]:
    return [t["key"] for t in _TOPIC_CATALOG]


def find_visualization(
    *, title: str = "", question: str = "", body_preview: str = "",
    strict: bool = False,
) -> Optional[tuple[str, Callable]]:
    """Pick a curated visualisation for the given (title, question, body).

    Returns ``(topic_key, generator)`` or None.

    With ``strict=True``, only the regex unique-hit pass runs — the
    embedding fallback is skipped.  Used by per-clause callers where
    the input is a single sentence and embedding cosines spuriously
    match NN-adjacent prose to NN topics (e.g. an RBM sentence
    matching the ``activation_functions`` description because both
    sit in neural-network embedding space).
    """
    query = " ".join(s for s in (question, title) if s).strip()
    if not query:
        return None

    # ---- Tier-A: fast, unambiguous regex hit on the query --------------
    fast = _regex_unique_hit(query)
    if fast is not None:
        return fast
    if strict:
        return None

    # ---- Tier-B: embedding-based semantic match -----------------------
    rows = _ensure_topic_embeddings()
    if rows is not None and _emb is not None:
        qv = _emb.embed_text(query)
        if qv:
            scored = sorted(
                ((float(_emb.cosine(qv, v)), key, gen)
                 for key, gen, v in rows),
                key=lambda t: -t[0],
            )
            top_score, top_key, top_gen = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            if (top_score >= ACCEPT_THRESHOLD
                    and top_score - second_score >= MARGIN):
                return top_key, top_gen
            # NOTE: An older "body-preview rescue" averaged the question
            # vector with the passage body vector and accepted matches
            # under the same threshold.  In practice that path was
            # strictly more permissive (any passage mentioning ridge/
            # shrinkage/lambda would push regularization_path above the
            # bar) and produced the §3.5.2 Partial Least Squares →
            # regularization_path misfire the user reported.  Removed.
            return None

    # ---- Tier-C: regex fallback when embedder is down -----------------
    return _regex_loose_hit(query, body_preview)


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _compiled(spec: dict) -> list[re.Pattern]:
    cache = spec.setdefault("_compiled", [])
    if cache:
        return cache
    cache.extend(re.compile(p, re.I) for p in spec.get("regex_aliases", []))
    return cache


def _regex_unique_hit(text: str) -> Optional[tuple[str, Callable]]:
    """Return a topic *only* if exactly one catalog entry matches the query.

    This guards against ambiguous queries that mention several topics
    by name (in which case we'd rather defer to the embedding step).
    """
    matched: list[dict] = []
    for spec in _TOPIC_CATALOG:
        for p in _compiled(spec):
            if p.search(text):
                matched.append(spec)
                break
    if len(matched) == 1:
        return matched[0]["key"], matched[0]["generator"]
    return None


def _regex_loose_hit(
    query: str, body_preview: str,
) -> Optional[tuple[str, Callable]]:
    """Last-resort matcher when neither query embedding nor unique regex
    succeed.  Identical to the pre-embedding behaviour."""
    for spec in _TOPIC_CATALOG:
        for p in _compiled(spec):
            if p.search(query):
                return spec["key"], spec["generator"]
    body = (body_preview or "")[:1500].lower()
    if not body:
        return None
    for spec in _TOPIC_CATALOG:
        for p in _compiled(spec):
            if len(list(p.finditer(body))) >= 2:
                return spec["key"], spec["generator"]
    return None


# ---------------------------------------------------------------------------
# Back-compat alias retained for the trigger-time helper in
# serve.orchestrator (it iterates over patterns to find the spoken-word
# offset).  We expose the old shape here; values are computed lazily.
# ---------------------------------------------------------------------------

class _LegacyView:
    def __iter__(self):
        for spec in _TOPIC_CATALOG:
            yield (spec["key"], spec["generator"], _compiled(spec))


_TOPIC_REGISTRY = _LegacyView()
