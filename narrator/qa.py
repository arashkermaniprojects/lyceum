"""Closed-book Q&A — answer a user's question using ONLY the book corpus.

Two synthesis modes:

  * **Retrieval-only** (default, deterministic, no model dependency):
    BM25 over BookNode body_texts, take the top passages, splice them
    into a small NarrationPlan with a brief intro sentence.  No
    generation.  Honest about its limits — when no passage scores above
    the threshold, returns a plan that says *"I cannot answer that from
    this book."*

  * **Local Qwen synthesis** (opt-in via :func:`set_synth_backend`):
    The retrieved passages are passed to a locally-loaded Qwen model
    (``transformers`` + ``torch``).  Strict closed-book prompt: the
    model is told to use ONLY the supplied context, refuse otherwise.
    No external APIs.  Temperature 0 for reproducibility.

Both modes produce a :class:`NarrationPlan` that the orchestrator can
stream like any other narration.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from book.ir import Book, BookNode
from book import embeddings as _emb
from .planner import (
    NarrationPlan, NarrationClause,
    _bm25_scores, _tokenize, _split_sentences, _estimate_dur,
    _build_surface_regex, _tag_clause, _DEFAULT_CPS,
)


# ---------------------------------------------------------------------------
# Low-similarity routing thresholds
# ---------------------------------------------------------------------------
# When the question's best book match is weak, the orchestrator should
# *introduce* the topic from the local LLM rather than splice extracts
# from passages that don't really cover it.  Two thresholds:
#
#   * Dense path: cosine ≥ DENSE_THRESHOLD against any passage means we
#     trust the book.  Below that we treat the match as low-similarity.
#   * Sparse-only path: when no embeddings, top-BM25 ≥ BM25_THRESHOLD.
#
# Both override-able via env so we can sweep in eval without code edits.

DENSE_THRESHOLD = float(os.environ.get("QA_LOW_SIM_THRESHOLD", "0.50"))
BM25_THRESHOLD = float(os.environ.get("QA_LOW_SIM_BM25", "1.0"))


# ---------------------------------------------------------------------------
# Retrieval — hybrid BM25 + dense (Qwen3-Embedding) via reciprocal rank fusion
# ---------------------------------------------------------------------------

@dataclass
class RetrievedPassage:
    nid: str
    text: str
    score: float           # fused score (RRF when hybrid, BM25 otherwise)
    bm25_score: float = 0.0
    dense_score: float = 0.0


# RRF tunable: smaller = more weight on top hits.  60 is the canonical default.
_RRF_K = 60.0


def _rrf_fuse(
    bm25_ranks: dict[int, int],
    dense_ranks: dict[int, int],
) -> dict[int, float]:
    """Combine two ranking systems via reciprocal rank fusion.

    Score(doc) = Σ_{system} 1 / (k + rank_in_system).  Documents that
    appear in both rankings near the top get the highest fused score.
    """
    fused: dict[int, float] = {}
    all_idx = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    for idx in all_idx:
        s = 0.0
        if idx in bm25_ranks:
            s += 1.0 / (_RRF_K + bm25_ranks[idx])
        if idx in dense_ranks:
            s += 1.0 / (_RRF_K + dense_ranks[idx])
        fused[idx] = s
    return fused


_CITATION_SHORTCUT_RE = re.compile(
    r"(?:"
    r"\bsection\s+(?P<sec>\d+(?:\.\d+){0,2})"
    r"|\bchapter\s+(?P<ch>\d+)"
    r"|\btheorem\s+(?P<thm>\d+(?:\.\d+){0,2})"
    r"|\blemma\s+(?P<lem>\d+(?:\.\d+){0,2})"
    r"|\bdefinition\s+(?P<defn>\d+(?:\.\d+){0,2})"
    r"|\bequation\s+(?P<eq>\d+(?:\.\d+){0,2})"
    r"|\bfigure\s+(?P<fig>\d+(?:\.\d+){0,2})"
    r"|\btable\s+(?P<tbl>\d+(?:\.\d+){0,2})"
    r"|§\s*(?P<sec_alt>\d+(?:\.\d+){0,2})"
    r")(?!\d)",
    re.I,
)


def _citation_shortcut(book: Book, question: str) -> list[RetrievedPassage]:
    """When the question explicitly names a section / chapter / theorem
    by number, short-circuit retrieval and return that node + a few
    related ancestors directly.  No BM25 / dense scoring needed.

    Returns ``[]`` on miss so the caller can fall back to normal
    retrieval.  When multiple citations are mentioned, returns the
    first match (the user's primary anchor); the rest are caught by
    the orchestrator's reference-card emission downstream.
    """
    if not question or not book or not book.root:
        return []
    m = _CITATION_SHORTCUT_RE.search(question)
    if not m:
        return []
    # Pick whichever group matched.
    sec_num = m.group("sec") or m.group("sec_alt")
    ch_num = m.group("ch")
    target_node: Optional[BookNode] = None
    if sec_num:
        for n in book.root.walk():
            if n.kind in {"section", "subsection", "subsubsection"} \
                    and (n.number or "").strip() == sec_num:
                target_node = n
                break
    elif ch_num:
        for n in book.root.walk():
            if n.kind == "chapter" and (n.number or "").strip() == ch_num:
                target_node = n
                break
    else:
        # Theorem / Lemma / Definition / Equation / Figure / Table —
        # find an environment node by number, else any node with that
        # number.  Falls through to BM25 when nothing matches.
        for grp in ("thm", "lem", "defn", "eq", "fig", "tbl"):
            num = m.group(grp)
            if not num:
                continue
            for n in book.root.walk():
                if (n.number or "").strip() == num:
                    target_node = n
                    break
            if target_node is not None:
                break
    if target_node is None or not (target_node.body_text or "").strip():
        # Body-less environment nodes (theorem stubs etc.) — try the
        # parent nid derived from the slash-path so the shortcut
        # still anchors on something with content.
        if target_node is not None and "/" in target_node.nid:
            parent_nid = target_node.nid.rsplit("/", 1)[0]
            parent = book.find(parent_nid)
            if parent and (parent.body_text or "").strip():
                target_node = parent
            else:
                return []
        else:
            return []
    return [RetrievedPassage(
        nid=target_node.nid,
        text=target_node.body_text,
        score=10.0,    # synthetic high score — shortcut wins
    )]


def _retrieve(
    book: Book, question: str, *,
    top_k: int = 3,
    bm25_min_score: float = 0.3,
    use_embeddings: bool = True,
) -> list[RetrievedPassage]:
    """Hybrid BM25 + dense retrieval against every BookNode's body_text.

    Strategy:

      1. (NEW) Citation shortcut — when the question literally names a
         section / chapter / theorem / equation by number, return that
         node directly.
      2. (NEW) If an alias map is installed (``set_alias_map``), expand
         the query with discovered synonyms — so "what is the RBF
         kernel" also retrieves passages that say "radial basis
         function".
      3. Run BM25 over all candidate nodes (always, deterministic, ~ms).
      4. If embeddings are populated on the book AND the embedding server
         is reachable, also embed the query and rank by cosine.
      5. Fuse rankings with reciprocal rank fusion.
      6. Tie-break on (page_start, nid) for stable ordering.

    Falls back to pure BM25 silently when embeddings aren't available —
    every existing call site keeps working without any code change.
    """
    # Citation shortcut — short-circuit retrieval when the question
    # explicitly names a passage by number (e.g., "explain section
    # 5.8" / "what does Theorem 3.2 say").  Surface the node plus
    # whatever BM25 brings in for context.
    shortcut = _citation_shortcut(book, question)
    # Query expansion via the per-book alias map (lazily set by the
    # server at startup).  No-op when the map is missing or empty.
    if _alias_map:
        from book.aliases import expand_query as _expand
        question = _expand(question, _alias_map)
    candidates: list[BookNode] = [
        n for n in book.root.walk() if n.body_text and n.body_text.strip()
    ]
    if not candidates:
        return shortcut

    # ---- BM25 path (always) ----
    docs = [_tokenize(n.body_text) for n in candidates]
    q = _tokenize(question)
    q_content = [t for t in q if t not in ("what", "is", "are",
                                            "the", "a", "an", "of",
                                            "how", "why", "when",
                                            "explain", "define", "show")]
    bm25 = _bm25_scores(q, docs)
    # Title boost — passages whose title contains a content token from
    # the query receive a multiplicative bonus.  This rescues sections
    # like "Overfitting" or "Bias-Variance Tradeoff" that are sometimes
    # outranked by tangential discussions.
    if q_content:
        q_set = set(q_content)
        for i, n in enumerate(candidates):
            tt = _tokenize(n.title or "")
            tt_set = set(tt)
            overlap = len(tt_set & q_set)
            if overlap > 0:
                # Linear bonus per overlapping token, plus a strong
                # additional boost when the title is *predominantly*
                # the query terms (e.g. title "Overfitting" for query
                # "what is overfitting" → 1/1 = 100%).
                ratio = overlap / max(1, len(tt))
                bm25[i] = bm25[i] + 4.0 * overlap + 8.0 * ratio
    bm25_paired = sorted(
        ((sc, i) for i, sc in enumerate(bm25)),
        key=lambda sc_i: (-sc_i[0], candidates[sc_i[1]].page_start,
                           candidates[sc_i[1]].nid),
    )
    bm25_ranks = {idx: r for r, (_sc, idx) in enumerate(bm25_paired)
                  if bm25[idx] > 0}

    # ---- Dense path (when corpus has embeddings + server reachable) ----
    dense_paired: list[tuple[float, int]] = []
    dense_ranks: dict[int, int] = {}
    if use_embeddings:
        # Map candidates → vectors that survived ingestion.
        cand_vecs: list[tuple[float, ...]] = []
        for n in candidates:
            v = n.meta.get("embedding")
            cand_vecs.append(tuple(v) if v else ())
        if any(cand_vecs) and _emb.is_available():
            qv = _emb.embed_text(question)
            if qv:
                dense_paired = _emb.ranked_by_cosine(qv, cand_vecs)
                # Drop zero-vector docs (they have score 0 anyway).
                dense_ranks = {idx: r for r, (sc, idx) in enumerate(dense_paired)
                               if sc > 0}

    # ---- Fuse ----
    if dense_ranks:
        fused = _rrf_fuse(bm25_ranks, dense_ranks)
        ranked = sorted(
            fused.items(),
            key=lambda kv: (-kv[1], candidates[kv[0]].page_start,
                            candidates[kv[0]].nid),
        )
    else:
        # Pure BM25.  Use raw scores so the threshold behaves intuitively.
        ranked = [(idx, bm25[idx]) for _sc, idx in bm25_paired]

    out: list[RetrievedPassage] = []
    bm25_max = max(bm25, default=0.0)
    for idx, score in ranked[:top_k]:
        n = candidates[idx]
        bm = bm25[idx]
        if not dense_ranks and bm < bm25_min_score:
            break
        # Compute a dense score for the surfaced passages (for diagnostics).
        ds = 0.0
        if dense_ranks and idx in dense_ranks:
            for sc, di in dense_paired:
                if di == idx:
                    ds = sc
                    break
        out.append(RetrievedPassage(
            nid=n.nid, text=n.body_text, score=score,
            bm25_score=bm, dense_score=ds,
        ))
    # Splice in the citation shortcut at the front so the directly
    # referenced passage is always the first source the QA pipeline
    # sees — without losing the BM25 / dense context behind it.
    if shortcut:
        seen_nids = {p.nid for p in out}
        deduped_shortcut = [p for p in shortcut if p.nid not in seen_nids]
        if deduped_shortcut:
            out = deduped_shortcut + out[: max(0, top_k - len(deduped_shortcut))]
    return out


# ---------------------------------------------------------------------------
# Similarity gating
# ---------------------------------------------------------------------------

def is_low_similarity(
    passages: list[RetrievedPassage], *,
    dense_threshold: Optional[float] = None,
    bm25_threshold: Optional[float] = None,
) -> bool:
    """Return True when the best retrieved passage is too weak to ground
    a closed-book answer.

    Decision rule (cheap, deterministic):

      * No passages survived → low.
      * Some passage carries a non-zero ``dense_score`` → use cosine,
        compare to ``dense_threshold``.
      * Otherwise (BM25-only retrieval) → compare top ``bm25_score`` to
        ``bm25_threshold``.

    The thresholds default to the module-level ``DENSE_THRESHOLD`` and
    ``BM25_THRESHOLD`` (env-overridable) so callers don't usually need
    to pass them.
    """
    if not passages:
        return True
    dt = dense_threshold if dense_threshold is not None else DENSE_THRESHOLD
    bt = bm25_threshold if bm25_threshold is not None else BM25_THRESHOLD
    has_dense = any(p.dense_score > 0.0 for p in passages)
    if has_dense:
        top_dense = max(p.dense_score for p in passages)
        return top_dense < dt
    top_bm25 = max(p.bm25_score for p in passages)
    return top_bm25 < bt


# ---------------------------------------------------------------------------
# Narration sanitizer — strip/convert LaTeX so TTS never speaks markup
# ---------------------------------------------------------------------------

# Unicode Greek letters as they appear in OCR'd PDF text.  These are
# distinct from the LaTeX backslash-commands handled below.
_UNICODE_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
    "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta",
    "Ε": "epsilon", "Ζ": "zeta", "Η": "eta", "Θ": "theta",
    "Ι": "iota", "Κ": "kappa", "Λ": "lambda", "Μ": "mu",
    "Ν": "nu", "Ξ": "xi", "Π": "pi", "Ρ": "rho",
    "Σ": "sigma", "Τ": "tau", "Υ": "upsilon", "Φ": "phi",
    "Χ": "chi", "Ψ": "psi", "Ω": "omega",
}

# Unicode math operators that show up in OCR'd PDF text.
_UNICODE_OPS = {
    "∫": " the integral of ",
    "∑": " the sum of ",
    "∏": " the product of ",
    "√": " the square root of ",
    "∂": " partial ",
    "∇": " gradient of ",
    "∞": " infinity ",
    "∈": " in ",
    "∉": " not in ",
    "⊂": " subset of ",
    "⊆": " subset of ",
    "∪": " union ",
    "∩": " intersection ",
    "∀": " for all ",
    "∃": " there exists ",
    "→": " to ",
    "⇒": " implies ",
    "≤": " less than or equal to ",
    "≥": " greater than or equal to ",
    "≠": " not equal to ",
    "≈": " approximately ",
    "±": " plus or minus ",
    "·": " times ",
    "×": " times ",
    "÷": " divided by ",
    "⟨": " inner product of ",
    "⟩": " ",
    "ℝ": " the real numbers ",
    "ℕ": " the natural numbers ",
    "ℤ": " the integers ",
    "ℚ": " the rational numbers ",
    "ℂ": " the complex numbers ",
    "𝔼": " the expectation of ",
    "ℙ": " the probability of ",
    "−": " minus ",  # U+2212 (not the ASCII hyphen)
}

# Unicode subscript / superscript glyphs → spoken form.
_UNICODE_SUBSCRIPTS = {
    "₀": " sub 0", "₁": " sub 1", "₂": " sub 2", "₃": " sub 3",
    "₄": " sub 4", "₅": " sub 5", "₆": " sub 6", "₇": " sub 7",
    "₈": " sub 8", "₉": " sub 9",
    "ᵢ": " sub i", "ⱼ": " sub j", "ₖ": " sub k", "ₙ": " sub n",
    "ₘ": " sub m", "ₐ": " sub a", "ₑ": " sub e", "ₒ": " sub o",
    "ᵤ": " sub u", "ᵥ": " sub v", "ᵣ": " sub r", "ₛ": " sub s",
    "ₜ": " sub t",
}
_UNICODE_SUPERSCRIPTS = {
    "⁰": " to the 0", "¹": " to the 1", "²": " squared",
    "³": " cubed", "⁴": " to the 4", "⁵": " to the 5",
    "⁶": " to the 6", "⁷": " to the 7", "⁸": " to the 8",
    "⁹": " to the 9", "ⁿ": " to the n", "ⁱ": " to the i",
    "ᵀ": " transpose",
}


def _verbalize_function_calls(s: str) -> str:
    """Rewrite ``f(x)`` / ``L(y, f(x))`` as ``f of x`` / ``L of y, f of x``.

    OCR'd book passages contain math notation in function-call form
    (``L(y_i, f(x_i))``) which Kokoro otherwise reads as letter-by-
    letter spelling.  We only rewrite when the head before the paren
    is a short identifier (≤ 3 chars) to avoid mangling normal English
    parenthetical asides like ``regression (with regularization)``.
    """
    import re as _re
    # ``(?<![A-Za-z0-9])`` instead of ``\b`` so a leading Unicode
    # letter (Greek) doesn't suppress the boundary — ``λJ(f)`` should
    # rewrite ``J(f)`` to ``J of f`` even though ``\b`` between λ and
    # J is empty under Unicode-aware word semantics.
    pat = _re.compile(
        r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]{0,2})\s*\(([^()]*)\)"
    )
    # Apply iteratively to handle one level of nesting:
    # ``L(yi, f(xi))`` → ``L(yi, f of xi)`` → ``L of yi, f of xi``.
    for _ in range(4):
        new = pat.sub(lambda m: f"{m.group(1)} of {m.group(2)}", s)
        if new == s:
            break
        s = new
    return s


_LATEX_GREEK = {
    r"\\alpha": "alpha", r"\\beta": "beta", r"\\gamma": "gamma",
    r"\\delta": "delta", r"\\epsilon": "epsilon",
    r"\\varepsilon": "epsilon", r"\\zeta": "zeta", r"\\eta": "eta",
    r"\\theta": "theta", r"\\vartheta": "theta", r"\\iota": "iota",
    r"\\kappa": "kappa", r"\\lambda": "lambda", r"\\mu": "mu",
    r"\\nu": "nu", r"\\xi": "xi", r"\\pi": "pi", r"\\rho": "rho",
    r"\\sigma": "sigma", r"\\tau": "tau", r"\\upsilon": "upsilon",
    r"\\phi": "phi", r"\\varphi": "phi", r"\\chi": "chi",
    r"\\psi": "psi", r"\\omega": "omega",
    r"\\Gamma": "gamma", r"\\Delta": "delta", r"\\Theta": "theta",
    r"\\Lambda": "lambda", r"\\Xi": "xi", r"\\Pi": "pi",
    r"\\Sigma": "sigma", r"\\Phi": "phi", r"\\Psi": "psi",
    r"\\Omega": "omega",
}

_LATEX_OPS = {
    r"\\cdot": " times ",
    r"\\times": " times ",
    r"\\div": " divided by ",
    r"\\le(?:q)?\b": " less than or equal to ",
    r"\\ge(?:q)?\b": " greater than or equal to ",
    r"\\ne(?:q)?\b": " not equal to ",
    r"\\approx": " approximately ",
    r"\\to": " to ",
    r"\\rightarrow": " to ",
    r"\\Rightarrow": " implies ",
    r"\\in\b": " in ",
    r"\\notin\b": " not in ",
    r"\\subset\b": " subset of ",
    r"\\subseteq\b": " subset of ",
    r"\\forall": " for all ",
    r"\\exists": " there exists ",
    r"\\nabla": " gradient of ",
    r"\\partial": " partial ",
    r"\\infty": " infinity ",
    r"\\pm": " plus or minus ",
    r"\\mathbb\{E\}": " expectation of ",
    r"\\mathbb\{P\}": " probability of ",
    r"\\Pr\b": " probability of ",
    r"\\mathbf": "", r"\\mathrm": "", r"\\mathit": "", r"\\mathcal": "",
    r"\\mathbb": "",
    r"\\boldsymbol": "", r"\\text": "", r"\\operatorname": "",
    r"\\left": "", r"\\right": "",
}


def _balanced_brace_arg(s: str, start: int) -> tuple[str, int]:
    """Read ``{...}`` starting at *start* (which must be ``{``).

    Returns ``(inner, end_index_after_closing_brace)`` or ``("", start)``
    if no balanced argument exists.
    """
    if start >= len(s) or s[start] != "{":
        return "", start
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
    return "", start


def _verbalize_bigops(s: str) -> str:
    """Rewrite ``\\sum``/``\\prod``/``\\int`` with their bound forms.

    ``\\sum_{i=1}^{N} x_i`` → ``the sum from i = 1 to N of x sub i``.
    ``\\int_a^b f(x) dx`` → ``the integral from a to b of f(x) dx``.
    Bare ``\\sum`` (no bounds) → ``the sum of``.
    """
    import re as _re
    out = []
    i = 0
    name_map = {
        r"\sum": "sum",
        r"\prod": "product",
        r"\int": "integral",
        r"\oint": "contour integral",
        r"\bigcup": "union",
        r"\bigcap": "intersection",
    }
    while i < len(s):
        m = _re.match(
            r"\\(?:sum|prod|int|oint|bigcup|bigcap)(?![A-Za-z])", s[i:]
        )
        if not m:
            out.append(s[i])
            i += 1
            continue
        name = name_map[m.group(0)]
        i += len(m.group(0))
        # Optional limits: _{...} or _x, then ^{...} or ^x.
        lower = ""
        upper = ""
        if i < len(s) and s[i] == "_":
            i += 1
            if i < len(s) and s[i] == "{":
                lower, i = _balanced_brace_arg(s, i)
            else:
                # Single token (letter/digit/Greek-command).
                tm = _re.match(r"\\?[A-Za-z]+|\d+", s[i:])
                if tm:
                    lower = tm.group(0)
                    i += len(tm.group(0))
        if i < len(s) and s[i] == "^":
            i += 1
            if i < len(s) and s[i] == "{":
                upper, i = _balanced_brace_arg(s, i)
            else:
                tm = _re.match(r"\\?[A-Za-z]+|\d+", s[i:])
                if tm:
                    upper = tm.group(0)
                    i += len(tm.group(0))
        if lower and upper:
            out.append(f" the {name} from {lower} to {upper} of ")
        elif lower:
            out.append(f" the {name} over {lower} of ")
        else:
            out.append(f" the {name} of ")
    return "".join(out)


def _verbalize_frac(s: str) -> str:
    """Rewrite ``\\frac{a}{b}`` as ``a over b``."""
    out = []
    i = 0
    while i < len(s):
        if s.startswith(r"\frac", i):
            i += len(r"\frac")
            # Skip whitespace.
            while i < len(s) and s[i].isspace():
                i += 1
            num, j = _balanced_brace_arg(s, i)
            if j == i:
                out.append(" over ")  # malformed; let downstream clean up
                continue
            i = j
            while i < len(s) and s[i].isspace():
                i += 1
            den, k = _balanced_brace_arg(s, i)
            if k == i:
                out.append(f" {num} over ")
                continue
            i = k
            out.append(f" {num} over {den} ")
        elif s.startswith(r"\dfrac", i) or s.startswith(r"\tfrac", i):
            # Same handling, just a different-named command.
            i += 6
            while i < len(s) and s[i].isspace():
                i += 1
            num, j = _balanced_brace_arg(s, i)
            i = max(j, i)
            while i < len(s) and s[i].isspace():
                i += 1
            den, k = _balanced_brace_arg(s, i)
            i = max(k, i)
            out.append(f" {num} over {den} ")
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _verbalize_sqrt(s: str) -> str:
    """Rewrite ``\\sqrt{x}`` as ``the square root of x``."""
    out = []
    i = 0
    while i < len(s):
        if s.startswith(r"\sqrt", i):
            i += len(r"\sqrt")
            while i < len(s) and s[i].isspace():
                i += 1
            arg, j = _balanced_brace_arg(s, i)
            if j != i:
                i = j
                out.append(f" the square root of {arg} ")
            else:
                out.append(" square root ")
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _verbalize_decoration(s: str) -> str:
    """Rewrite ``\\hat{x}`` as ``x hat`` (also \\bar, \\tilde, \\dot)."""
    import re as _re
    rules = [
        (r"\\hat",   "hat"),
        (r"\\widehat", "hat"),
        (r"\\bar",   "bar"),
        (r"\\overline", "bar"),
        (r"\\tilde", "tilde"),
        (r"\\widetilde", "tilde"),
        (r"\\dot",   "dot"),
        (r"\\ddot",  "double dot"),
        (r"\\vec",   "vector"),
    ]
    for cmd, word in rules:
        # \cmd{x}  →  "x word"
        while True:
            m = _re.search(cmd + r"\s*\{", s)
            if not m:
                break
            arg, end = _balanced_brace_arg(s, m.end() - 1)
            if end == m.end() - 1:
                break
            s = s[:m.start()] + f" {arg} {word} " + s[end:]
        # \cmd x  (single token)  →  "x word"
        s = _re.sub(cmd + r"\s+([A-Za-z0-9])", lambda mm: f" {mm.group(1)} {word} ", s)
    return s


def _verbalize_norm(s: str) -> str:
    """Rewrite ``\\| x \\|`` and ``\\|x\\|^2`` as norm / squared-norm."""
    import re as _re
    # Squared norm: \|x\|^2  or  \|x\|^{2}
    s = _re.sub(r"\\\|\s*([^\\|]+?)\s*\\\|\s*\^\s*\{?2\}?",
                lambda m: f" the squared norm of {m.group(1).strip()} ", s)
    s = _re.sub(r"\\\|\s*([^\\|]+?)\s*\\\|",
                lambda m: f" the norm of {m.group(1).strip()} ", s)
    # \langle x, y \rangle → "inner product of x and y"
    s = _re.sub(r"\\langle\s*([^,\\]+?)\s*,\s*([^\\]+?)\s*\\rangle",
                lambda m: f" the inner product of {m.group(1).strip()} and {m.group(2).strip()} ", s)
    return s


def _verbalize_powers(s: str) -> str:
    """Rewrite ``x^2`` as ``x squared``, ``x^3`` as ``x cubed``,
    ``x^n`` and ``x^{n+1}`` as ``x to the n``/``x to the n + 1``."""
    import re as _re
    # x^{...}  (braced)
    def _rep_braced(m):
        base = m.group(1)
        exp = m.group(2).strip()
        if exp == "2":
            return f"{base} squared"
        if exp == "3":
            return f"{base} cubed"
        if exp == "-1":
            return f"{base} inverse"
        if exp == "T":
            return f"{base} transpose"
        return f"{base} to the {exp}"
    s = _re.sub(r"([A-Za-z0-9\)])\^\{([^{}]+)\}", _rep_braced, s)
    # x^c  (single character)
    def _rep_single(m):
        base = m.group(1)
        exp = m.group(2)
        if exp == "2":
            return f"{base} squared"
        if exp == "3":
            return f"{base} cubed"
        if exp == "T":
            return f"{base} transpose"
        return f"{base} to the {exp}"
    s = _re.sub(r"([A-Za-z0-9\)])\^([A-Za-z0-9])", _rep_single, s)
    return s


def _sanitize_for_narration(text: str) -> str:
    """Make *text* safe for direct TTS playback.

    Strips LaTeX delimiters and rewrites mathematical notation as
    spoken English so the user hears *words*, not symbols.  Sums and
    integrals become ``the sum from i = 1 to N of …``, ``\\frac{a}{b}``
    becomes ``a over b``, ``\\sqrt{x}`` becomes ``the square root of
    x``, ``\\hat{y}`` becomes ``y hat``, ``x^2`` becomes ``x squared``
    — the spoken stream matches what the chalkboard renders visually.
    """
    import re as _re
    if not text:
        return ""
    s = text
    # Strip display- and inline-math delimiters first so the rewrites
    # below see raw LaTeX content.
    s = _re.sub(r"\\\[|\\\]|\\\(|\\\)", " ", s)
    s = _re.sub(r"\$\$([^$]+)\$\$", r" \1 ", s)
    s = _re.sub(r"\$([^$]+)\$", r" \1 ", s)
    # Structural rewrites — order matters: bigops/frac/sqrt consume
    # their argument braces, so they run before the generic
    # "drop unknown command" pass that would strip them.
    s = _verbalize_bigops(s)
    s = _verbalize_frac(s)
    s = _verbalize_sqrt(s)
    s = _verbalize_decoration(s)
    s = _verbalize_norm(s)
    # Unicode math symbols (∫ ∑ λ ∈ …) — OCR'd book passages use these
    # instead of LaTeX backslash commands.  Translate before
    # function-call rewriting so the surface text is already English.
    for ch, repl in _UNICODE_OPS.items():
        s = s.replace(ch, repl)
    for ch, repl in _UNICODE_SUBSCRIPTS.items():
        s = s.replace(ch, repl)
    for ch, repl in _UNICODE_SUPERSCRIPTS.items():
        s = s.replace(ch, repl)
    # Function-call notation BEFORE Greek replacement so ``f(x)`` /
    # ``L(yi, f(xi))`` get rewritten.
    s = _verbalize_function_calls(s)
    # Unicode Greek letters last so any ``ψ(x)`` that survived (e.g.
    # because it was nested too deep for the function-call rewriter)
    # at least reads as "psi" rather than the unicode glyph itself.
    for ch, repl in _UNICODE_GREEK.items():
        s = s.replace(ch, " " + repl + " ")
    # Second function-call pass — Greek-rewritten heads are now ASCII
    # words (``psi (x)``).  Apply iteratively for one more round of
    # nesting that the first pass might have missed because of
    # Unicode boundary semantics.
    pat2 = _re.compile(
        r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]{0,8})\s*\(([^()]*)\)"
    )
    for _ in range(4):
        new = pat2.sub(lambda m: f"{m.group(1)} of {m.group(2)}", s)
        if new == s:
            break
        s = new
    # Greek + operator commands.  Use a negative lookahead instead of \b
    # so trailing _0 / _i / digits don't suppress the match.
    for pat, repl in _LATEX_GREEK.items():
        s = _re.sub(pat + r"(?![A-Za-z])", repl, s)
    for pat, repl in _LATEX_OPS.items():
        s = _re.sub(pat, repl, s)
    # Powers — run after Greek so ``\\alpha^2`` reads as ``alpha squared``.
    s = _verbalize_powers(s)
    # Drop unknown remaining commands like \xyz{ ... } — keep the inner
    # arg, drop the command name and braces.
    s = _re.sub(r"\\[A-Za-z]+\s*\{([^{}]*)\}", r"\1", s)
    s = _re.sub(r"\\[A-Za-z]+", "", s)
    # Spacing commands (``\,``, ``\;``, ``\!``, ``\:``, ``\ ``) — turn
    # into a plain space so they don't read as "backslash comma".
    s = _re.sub(r"\\[,;:! ]", " ", s)
    # Markdown emphasis / code fences / backticks.
    s = _re.sub(r"\*\*|__|`+", "", s)
    # Strip leftover braces.
    s = s.replace("{", "").replace("}", "")
    # Subscripts: x_i → "x sub i", but plain numerals stay attached
    # ("x_1" → "x 1" reads worse than "x sub 1" — keep "sub").
    s = _re.sub(r"_\{([^{}]+)\}", r" sub \1", s)
    s = _re.sub(r"_([A-Za-z0-9])", r" sub \1", s)
    # Fallback for any leftover ^c that didn't have a base letter
    # (e.g. starts the line) — read as "to the c".
    s = _re.sub(r"\^\{([^{}]+)\}", r" to the \1", s)
    s = _re.sub(r"\^([A-Za-z0-9])", r" to the \1", s)
    # Collapse whitespace.
    s = _re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Synthesis backends
# ---------------------------------------------------------------------------

# A backend takes (question, passages, book) and returns a list of strings,
# each becoming one narrated clause.  `None` means the backend cannot
# answer; caller should fall back to retrieval-only.
SynthBackend = Callable[[str, list[RetrievedPassage], Book], Optional[list[str]]]


_FIGURE_CAPTION_RE = __import__("re").compile(
    r"^\s*(?:figure|fig\.?|table|equation)\s+\d+(?:\.\d+){0,2}\b",
    __import__("re").I,
)
_HEADER_LINE_RE = __import__("re").compile(
    r"^\s*\d+(?:\.\d+){0,3}\s+[A-Z]"
)


def _is_caption_or_header(s: str) -> bool:
    """Is this sentence really a figure caption or section header?

    Captions read aloud during the intro of an answer give a useless
    first impression ("FIGURE 17.6 A restricted Boltzmann machine in
    which there are no connections..."), so we drop them from the
    extract and let real exposition lead.
    """
    if not s:
        return True
    s = s.strip()
    if _FIGURE_CAPTION_RE.match(s):
        return True
    if _HEADER_LINE_RE.match(s):
        return True
    # Mostly-uppercase one-line "FIGURE N.M." run-ons.
    upper_letters = sum(1 for c in s if c.isupper())
    letters = sum(1 for c in s if c.isalpha())
    if letters > 0 and upper_letters / letters > 0.6 and len(s) < 80:
        return True
    return False


# How many sentences per retrieved passage to include in the narrated
# answer.  Bumped from 2 → 5 so passages with several short paragraphs
# (e.g. ESLII §17.4.4 RBM) yield real exposition rather than just the
# figure caption + the next sentence.
# Sentences taken from the top retrieved passage.  Set to 0 to
# narrate the *whole* passage — that's the user-friendly default:
# they asked a question and expect a thorough explanation, not a
# 5-sentence snippet.
_SENTS_TOP_PASSAGE = int(
    __import__("os").environ.get("QA_SENTS_TOP_PASSAGE", "0")
)
# Sentences for non-top passages — kept short so the top passage's
# coverage stays prominent while supporting passages add breadth.
_SENTS_PER_PASSAGE = int(
    __import__("os").environ.get("QA_SENTS_PER_PASSAGE", "5")
)


def _retrieval_only_backend(
    question: str, passages: list[RetrievedPassage], book: Book,
) -> list[str]:
    """Default synthesis: assemble an answer from passage extracts.

    Strategy:
      1. Lead sentence summarising which passages were matched.
      2. For the **top** passage, walk every non-caption sentence in
         reading order — the user asked about this section and
         expects to hear the whole exposition (configurable via
         ``QA_SENTS_TOP_PASSAGE``; 0 = unlimited, default).
      3. For supporting passages, anchor on the BM25-best non-caption
         sentence and emit a forward window of
         ``_SENTS_PER_PASSAGE`` (default 5) sentences for breadth.

    Pure retrieval — no generation.
    """
    if not passages:
        return [
            "I cannot find anything about that in this book.",
            "Try rephrasing the question or asking about a different topic.",
        ]
    out: list[str] = [
        f"The book addresses this in {len(passages)} passages from "
        f"{', '.join(p.nid.split('/')[-1] for p in passages)}.",
    ]
    q_tokens = _tokenize(question)
    for rank, p in enumerate(passages):
        sents = _split_sentences(p.text)
        if not sents:
            continue
        if rank == 0:
            # Top passage — read in order, filter only captions/headers.
            cap = _SENTS_TOP_PASSAGE if _SENTS_TOP_PASSAGE > 0 else len(sents)
            emitted = 0
            for s in sents:
                if _is_caption_or_header(s):
                    continue
                out.append(s.strip())
                emitted += 1
                if emitted >= cap:
                    break
            continue
        # Supporting passages — anchor + window for breadth.
        scored: list[tuple[float, int, str]] = []
        sent_docs = [_tokenize(s) for s in sents]
        sent_scores = _bm25_scores(q_tokens, sent_docs)
        for i, s in enumerate(sents):
            if _is_caption_or_header(s):
                continue
            scored.append((sent_scores[i], i, s))
        if not scored:
            continue
        scored.sort(key=lambda t: -t[0])
        anchor_i = scored[0][1]
        emitted = 0
        i = anchor_i
        while i < len(sents) and emitted < _SENTS_PER_PASSAGE:
            if not _is_caption_or_header(sents[i]):
                out.append(sents[i].strip())
                emitted += 1
            i += 1
    return out


# Module-level synth backend; default is retrieval-only.
_synth_backend: SynthBackend = _retrieval_only_backend
# Optional streaming-aware backend that yields sentences as they
# arrive.  Used by ``answer_streaming`` to start TTS / chalkboard
# emission before the LLM has finished generating.  None disables
# streaming and the caller falls back to ``answer``.
_streaming_backend: Optional[Callable] = None
# Per-book alias map (``surface → canonical``) used to expand a
# user query with discovered synonyms before retrieval — so a
# question phrased with one term hits passages phrased with the
# other.  Server installs this at startup.
_alias_map: dict[str, str] = {}
# Per-book concept dependency graph (``concept → [(prereq, count), …]``)
# used by the ``dependencies`` planner to surface idea-level
# prerequisites alongside the citation-graph's passage prerequisites.
# May be empty for books whose prose doesn't match the extraction
# patterns; planner gracefully falls back to citation-graph only.
_concept_graph: dict = {}

# Module-level intro backend used when the question's similarity to the
# book is low.  When None, we fall back to the same closed-book path —
# no behavioural change vs. before this feature was added.
_intro_backend: Optional[SynthBackend] = None


def set_streaming_backend(backend: Optional[Callable]) -> None:
    """Install a streaming-aware backend.  Pass ``None`` to disable
    streaming and force ``answer_streaming`` to fall back to
    ``answer`` (non-streaming)."""
    global _streaming_backend
    _streaming_backend = backend


def get_streaming_backend() -> Optional[Callable]:
    return _streaming_backend


def set_alias_map(amap: Optional[dict[str, str]]) -> None:
    """Install the per-book ``surface → canonical`` alias map used by
    ``_retrieve`` for query expansion.  Pass ``None`` (or an empty
    dict) to disable expansion."""
    global _alias_map
    _alias_map = dict(amap or {})


def get_alias_map() -> dict[str, str]:
    return dict(_alias_map)


def set_concept_graph(graph: Optional[dict]) -> None:
    """Install the per-book concept dependency graph.  ``None`` clears."""
    global _concept_graph
    _concept_graph = dict(graph or {})


def get_concept_graph() -> dict:
    return dict(_concept_graph)


def set_synth_backend(backend: SynthBackend) -> None:
    """Replace the active synthesis backend (e.g. swap in local Qwen)."""
    global _synth_backend
    _synth_backend = backend


def get_synth_backend() -> SynthBackend:
    return _synth_backend


def set_intro_backend(backend: Optional[SynthBackend]) -> None:
    """Install the low-similarity intro backend.

    Pass ``None`` to disable the LLM intro path and revert to
    retrieval-only behaviour.
    """
    global _intro_backend
    _intro_backend = backend


def get_intro_backend() -> Optional[SynthBackend]:
    return _intro_backend


# ---------------------------------------------------------------------------
# Local-Qwen backend (opt-in; requires transformers + torch + a model)
# ---------------------------------------------------------------------------

_QWEN_PROMPT = """You are a closed-book teacher. Answer the question using ONLY the given context.
If the context does not contain enough information, say "The book does not cover this directly."
Keep your answer to 2-4 sentences.

Context:
{context}

Question: {question}

Answer:"""


_TUTOR_PROMPT_SYSTEM = (
    "You are a private mathematics tutor walking a student through "
    '"The Elements of Statistical Learning" by Hastie, Tibshirani, '
    "and Friedman.  Your reply will be spoken aloud by TTS *and* "
    "rendered on a chalkboard.  Real-time visualization picks up math "
    "expressions you write, the equation citations you mention, and "
    "the variable definitions you give.\n"
    "\n"
    "Be CONCISE.  The default is a short, focused answer:\n"
    "  - default length: 1–2 sentences plus one display formula on a "
    "separate line if relevant;\n"
    "  - only if they explicitly ask for depth ('explain in detail', "
    "'derive', 'walk me through', 'step by step') answer in 4–8 "
    "sentences.\n"
    "Do NOT pad short answers; do NOT lecture; the student will ask "
    "follow-ups if they need more.\n"
    "\n"
    "GROUND every claim in the supplied book passages.  When a passage "
    "names an Equation, Theorem, Lemma, Section, or Figure, cite it by "
    "number ('Equation 5.42', 'Theorem 3.2', 'Section 5.8') so the "
    "chalkboard can pull up the matching card.\n"
    "\n"
    "OUTPUT RULES (your reply is rendered literally):\n"
    "  - Plain prose paragraphs.  No markdown, no bullet lists, no "
    "headers, no code fences.\n"
    "  - Math goes in LaTeX delimiters: \\( … \\) for inline math "
    "embedded in a sentence, \\[ … \\] on its own line for display "
    "equations.  KaTeX renders these on the chalkboard.\n"
    "  - VERBALIZE every formula in plain English alongside the LaTeX, "
    "so a student listening through TTS hears words and not symbols.  "
    "Always speak the formula in words FIRST, then put the LaTeX on a "
    "display line.  Read \\sum_{i=1}^{N} x_i as 'the sum from i equals 1 "
    "to N of x sub i', not 'sigma i equals 1 N x i'.  Read \\int as "
    "'the integral of', \\frac{a}{b} as 'a over b' or 'a divided by b', "
    "x^2 as 'x squared', \\sqrt{x} as 'the square root of x', "
    "\\partial f / \\partial x as 'the partial derivative of f with "
    "respect to x', \\nabla as 'gradient of', \\hat{y} as 'y hat', "
    "\\bar{x} as 'x bar', \\mathbb{E} as 'the expectation of', "
    "\\Pr or \\mathbb{P} as 'the probability of', \\| x \\|^2 as 'the "
    "squared norm of x', \\in as 'in', \\to as 'goes to'.  Greek "
    "letters: read \\alpha as 'alpha', \\beta as 'beta', \\theta as "
    "'theta', \\lambda as 'lambda', \\sigma as 'sigma', \\mu as 'mu', "
    "etc. — never name them by symbol shape.\n"
    "  - When you introduce a symbol, say 'where x is the input "
    "vector' so the definition lands on the same box as the formula.\n"
    "  - Avoid numbering your own sentences (1., 2., …).  Speak naturally.\n"
    "  - When the passages don't cover the question, say so honestly "
    "in one sentence rather than inventing.\n"
)


def _format_history_for_prompt(
    history: Optional[list], *, max_turns: int = 4,
) -> str:
    """Compress a list of DialogueTurn-like records into a short
    'previously' summary for the LLM context.  Skips control turns;
    keeps the most recent ``max_turns`` content turns.
    """
    if not history:
        return ""
    summary: list[str] = []
    for t in history[-max_turns * 2:]:  # over-take, then filter
        intent = getattr(t, "intent", "") or ""
        if intent in {"control", "recap", "follow_up"}:
            continue
        text = (getattr(t, "user_text", "") or "").strip()
        topic = (getattr(t, "focus_topic", "") or "").strip()
        if not text and not topic:
            continue
        summary.append(text or topic)
    summary = summary[-max_turns:]
    if not summary:
        return ""
    bullets = "\n".join(f"  - {s}" for s in summary)
    return (
        "\nThe student has already asked, in this session:\n"
        + bullets +
        "\n(Don't re-introduce concepts you already covered.)\n"
    )


def make_vllm_tutor_backend(
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ",
    max_tokens: int = 700,
    timeout: float = 30.0,
) -> Optional[SynthBackend]:
    """A SynthBackend that runs Qwen as a chat-mode tutor.

    Differences from :func:`make_vllm_backend`:
      * lets the model decide answer length from the user's phrasing
      * allows LaTeX math delimiters (rendered by the chalkboard)
      * folds in the session's dialogue history when the caller
        threads it via ``qa.answer(..., history=...)``
      * generous max_tokens budget so deep follow-ups aren't cut

    Backend signature accepts the optional 4th positional ``history``
    so callers that don't have a history pass nothing and the call
    still works.  Returns ``None`` when vLLM is unreachable.
    """
    import json as _json
    import urllib.request
    import urllib.error

    # Probe /v1/models like the strict backend does.
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": "Bearer local-vllm"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
            served = {m.get("id") for m in payload.get("data", [])}
    except Exception as e:
        print(f"[narrator.qa] tutor: vLLM not reachable at {base_url}: {e}")
        return None

    if model not in served:
        if served:
            model = next(iter(served))
            print(f"[narrator.qa] tutor: using {model}")
        else:
            return None

    chat_url = base_url.rstrip("/") + "/chat/completions"

    def _backend(
        question: str, passages: list[RetrievedPassage], book: Book,
        history: Optional[list] = None,
    ) -> Optional[list[str]]:
        ctx = "\n\n".join(
            f"[from {p.nid}, page {getattr(p, 'page_start', '')}]\n"
            f"{p.text[:1600]}"
            for p in passages
        ) if passages else "(no relevant passages found)"
        history_blob = _format_history_for_prompt(history)
        user_msg = (
            f"Book passages:\n{ctx}\n"
            f"{history_blob}"
            f"\nThe student now asks: {question}"
        )
        body = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": _TUTOR_PROMPT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens,
            # Slight temperature so the tutor doesn't sound robotic
            # when length adapts; still low enough to be reproducible
            # within a session.
            "temperature": 0.2,
            "top_p": 0.95,
            "seed": 42,
        }).encode("utf-8")
        req = urllib.request.Request(
            chat_url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local-vllm",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            print(f"[narrator.qa] tutor: vLLM call failed: {e}")
            return None
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "").strip()
        if not text:
            return None
        # Strip trailing/leading "Here is..." preambles the model
        # sometimes adds despite the system prompt.
        text = re.sub(
            r"^(?:here(?:'s|\s+is)\s+[^.\n]*[.:]\s*)",
            "", text, flags=re.I,
        ).strip()
        sentences = _split_sentences(text)
        return sentences or [text]

    _backend.__name__ = "vllm_tutor_backend"
    _backend.supports_history = True
    return _backend


# ---------------------------------------------------------------------------
# Streaming tutor — yields each completed sentence as the LLM produces it.
# Cuts perceived latency: first audio plays in ~0.3 s instead of waiting
# 2-7 s for the full deep response.
# ---------------------------------------------------------------------------

# Sentence boundary recogniser — looks for ``.``/``!``/``?`` followed by
# whitespace + an upper-case letter or end of string.  Used to extract
# completed sentences from the streaming token buffer.  We avoid
# splitting inside ``\(...\)`` / ``\[...\]`` math by tracking depth.
_SENT_TAIL_RE = re.compile(
    r"([.!?])(?:[\"')\]]*)(\s+)(?=[A-Z\\(\[]|$)",
)


def _peel_sentence(buf: str) -> tuple[Optional[str], str]:
    """Try to peel one completed sentence off the front of *buf*.

    Returns ``(sentence, remaining)`` if a complete sentence is found,
    else ``(None, buf)``.  Skips sentence-enders that fall inside
    ``\\(...\\)`` or ``\\[...\\]`` math regions so a period inside a
    LaTeX formula doesn't break a sentence.
    """
    if not buf:
        return None, buf
    in_inline = 0
    in_display = 0
    i = 0
    n = len(buf)
    while i < n:
        # Track LaTeX math regions.
        if i + 1 < n and buf[i] == "\\" and buf[i + 1] == "(":
            in_inline += 1
            i += 2
            continue
        if i + 1 < n and buf[i] == "\\" and buf[i + 1] == ")":
            in_inline = max(0, in_inline - 1)
            i += 2
            continue
        if i + 1 < n and buf[i] == "\\" and buf[i + 1] == "[":
            in_display += 1
            i += 2
            continue
        if i + 1 < n and buf[i] == "\\" and buf[i + 1] == "]":
            in_display = max(0, in_display - 1)
            i += 2
            continue
        if (in_inline == 0 and in_display == 0
                and buf[i] in ".!?"):
            # Look ahead for closing quote/paren and required whitespace.
            j = i + 1
            while j < n and buf[j] in '"\')]':
                j += 1
            if j < n and buf[j] in " \t\n":
                # Need at least one space + upper-case-ish next char.
                k = j
                while k < n and buf[k] in " \t\n":
                    k += 1
                if k >= n:
                    # Trailing whitespace, no next char yet — wait.
                    return None, buf
                next_ch = buf[k]
                if (next_ch.isupper() or next_ch in "\\([")\
                        and (j - i) >= 1:
                    sentence = buf[: j].strip()
                    if sentence:
                        return sentence, buf[k:]
        i += 1
    return None, buf


# Soft phrase boundaries — used to peel a clause's worth of audio
# *before* the full sentence ends.  Cuts first-audio latency: TTS
# sees a 4-7 word chunk instead of waiting for 20-30 words.
# Triggered only after ``_PHRASE_MIN_WORDS`` whitespace tokens have
# accumulated since the last peel, so we don't emit "Bagging," alone.
_PHRASE_MIN_WORDS = 5


def _peel_phrase(buf: str) -> tuple[Optional[str], str]:
    """Try to peel a *phrase* (sub-sentence chunk) off the front of *buf*.

    Returns ``(phrase, remaining)`` if a comma/semicolon/em-dash boundary
    is found after at least ``_PHRASE_MIN_WORDS`` whitespace tokens —
    *outside* any LaTeX math region.  Returns ``(None, buf)`` otherwise.
    Used by the streaming backend so the TTS gets shorter chunks and
    audio plays sooner.
    """
    if not buf or " " not in buf:
        return None, buf
    in_inline = 0
    in_display = 0
    word_count = 0
    in_word = False
    n = len(buf)
    i = 0
    while i < n:
        ch = buf[i]
        # Track LaTeX math regions.
        if i + 1 < n and ch == "\\" and buf[i + 1] in "([":
            if buf[i + 1] == "(":
                in_inline += 1
            else:
                in_display += 1
            i += 2
            continue
        if i + 1 < n and ch == "\\" and buf[i + 1] in ")]":
            if buf[i + 1] == ")":
                in_inline = max(0, in_inline - 1)
            else:
                in_display = max(0, in_display - 1)
            i += 2
            continue
        # Track word boundaries (whitespace or strong punctuation
        # ends a word).
        if ch in " \t\n":
            if in_word:
                word_count += 1
                in_word = False
        elif ch in ",;.!?—":
            if in_word:
                word_count += 1
                in_word = False
        else:
            in_word = True
        # Look for a soft boundary, but only when we're outside math
        # AND the buffer has enough words to justify a phrase.
        if (in_inline == 0 and in_display == 0
                and word_count >= _PHRASE_MIN_WORDS
                and ch in ",;"):
            j = i + 1
            # Need a single whitespace then a non-whitespace continuation.
            if j < n and buf[j] in " \t\n":
                k = j
                while k < n and buf[k] in " \t\n":
                    k += 1
                if k >= n:
                    return None, buf
                # Don't peel inside numbers like "1,000" — the next
                # char after the space should be alphabetic or a math
                # opener (LaTeX command).
                next_ch = buf[k]
                if next_ch.isalpha() or next_ch in "\\([":
                    phrase = buf[: i + 1].strip()
                    if phrase:
                        return phrase, buf[k:]
        # Em-dash boundary (the unicode char or two ASCII hyphens
        # surrounded by spaces — both used in textbook prose).
        if (in_inline == 0 and in_display == 0
                and word_count >= _PHRASE_MIN_WORDS):
            if ch == "—" and i > 0 and buf[i - 1] == " ":
                # Match "<word> — <word>".
                if i + 1 < n and buf[i + 1] == " ":
                    phrase = buf[: i].rstrip()
                    if phrase:
                        return phrase, buf[i + 2:]
            if (i + 2 < n and ch == "-" and buf[i + 1] == "-"
                    and i > 0 and buf[i - 1] == " " and buf[i + 2] == " "):
                phrase = buf[: i].rstrip()
                if phrase:
                    return phrase, buf[i + 3:]
        i += 1
    return None, buf


def make_vllm_tutor_streaming_backend(
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ",
    max_tokens: int = 700,
    timeout: float = 60.0,
    peel_phrases: bool = True,
):
    """Streaming variant of the tutor backend.

    Returns a callable ``(question, passages, book, history) -> Iterator[str]``
    that yields each completed sentence as soon as the LLM finishes
    it, rather than waiting for the entire response.

    Detection: the returned callable has ``is_streaming = True`` so
    callers can pick the streaming path.
    """
    import json as _json
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": "Bearer local-vllm"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
            served = {m.get("id") for m in payload.get("data", [])}
    except Exception as e:
        print(f"[narrator.qa] streaming tutor: vLLM not reachable: {e}")
        return None

    if model not in served:
        if served:
            model = next(iter(served))
        else:
            return None

    chat_url = base_url.rstrip("/") + "/chat/completions"

    def _stream(question, passages, book, history=None):
        ctx = "\n\n".join(
            f"[from {p.nid}, page {getattr(p, 'page_start', '')}]\n"
            f"{p.text[:1600]}"
            for p in passages
        ) if passages else "(no relevant passages found)"
        history_blob = _format_history_for_prompt(history)
        user_msg = (
            f"Book passages:\n{ctx}\n"
            f"{history_blob}"
            f"\nThe student now asks: {question}"
        )
        body = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": _TUTOR_PROMPT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.95,
            "seed": 42,
            "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            chat_url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local-vllm",
            },
            method="POST",
        )
        buf = ""
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    line = line.decode("utf-8", errors="replace").rstrip("\n")
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        if data_str == "[DONE]":
                            break
                        continue
                    try:
                        chunk = _json.loads(data_str)
                    except Exception:
                        continue
                    delta = (
                        ((chunk.get("choices") or [{}])[0]
                         .get("delta") or {}).get("content", "")
                    )
                    if not delta:
                        continue
                    buf += delta
                    # Peel as many chunks as we can from the buffer.
                    # Prefer full sentences when they're available;
                    # otherwise fall back to phrase-level peeling so
                    # the TTS gets shorter input and first-audio
                    # latency stays low.
                    while True:
                        sentence, buf = _peel_sentence(buf)
                        if sentence is not None:
                            sentence = re.sub(
                                r"^(?:here(?:'s|\s+is)\s+[^.\n]*[.:]\s*)",
                                "", sentence, flags=re.I,
                            ).strip()
                            if sentence:
                                yield sentence
                            continue
                        if peel_phrases:
                            phrase, buf = _peel_phrase(buf)
                            if phrase is not None:
                                phrase = re.sub(
                                    r"^(?:here(?:'s|\s+is)\s+[^.\n]*[.:]\s*)",
                                    "", phrase, flags=re.I,
                                ).strip()
                                if phrase:
                                    yield phrase
                                continue
                        break
        except urllib.error.URLError as e:
            print(f"[narrator.qa] streaming tutor: vLLM stream failed: {e}")
            return
        # Flush any trailing tail that didn't end with sentence punctuation.
        tail = buf.strip()
        if tail:
            yield tail

    _stream.__name__ = "vllm_tutor_streaming_backend"
    _stream.is_streaming = True
    _stream.supports_history = True
    return _stream


def make_vllm_backend(
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ",
    max_tokens: int = 220,
    timeout: float = 30.0,
) -> Optional[SynthBackend]:
    """Construct a synth backend that calls a local vLLM server.

    The vLLM server speaks the OpenAI Chat Completions API at
    ``base_url``.  Start a vLLM instance hosting your chosen text-LLM
    on localhost (see https://docs.vllm.ai for the canonical
    invocation), or set the ``VLLM_BASE_URL`` environment variable to
    point at an existing endpoint.

    Returns None when the endpoint is unreachable, so callers can fall
    back to retrieval-only synthesis.

    Determinism: temperature=0 + top_p=1 + seed=42 → reproducible
    output for the same model + same input.  No internet, no Anthropic.
    """
    import json as _json
    import urllib.request
    import urllib.error

    # Probe /v1/models with a short timeout to confirm reachability.
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": "Bearer local-vllm"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
            served = {m.get("id") for m in payload.get("data", [])}
    except Exception as e:
        print(f"[narrator.qa] vLLM not reachable at {base_url}: {e}")
        return None

    if model not in served:
        # Fall back to whatever the server is actually serving.
        if served:
            model = next(iter(served))
            print(f"[narrator.qa] using available vLLM model: {model}")
        else:
            print(f"[narrator.qa] vLLM serves no models")
            return None

    chat_url = base_url.rstrip("/") + "/chat/completions"

    def _backend(
        question: str, passages: list[RetrievedPassage], book: Book,
    ) -> Optional[list[str]]:
        if not passages:
            return None
        ctx = "\n\n".join(
            f"[from {p.nid}, page-cluster]\n{p.text[:1500]}" for p in passages
        )
        body = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system",
                 "content": (
                     "You are a closed-book teacher for a math/stats "
                     "textbook narrated aloud by a TTS engine. Use ONLY "
                     "the supplied context. If the context does not "
                     "contain enough information, say so honestly. "
                     "Keep your answer to 2 to 4 sentences.\n"
                     "STRICT OUTPUT RULES (your reply is read literally):\n"
                     "- No LaTeX, no $...$ or \\( ... \\) or \\[ ... \\] markers.\n"
                     "- No markdown, no asterisks, no bullet lists.\n"
                     "- Spell out every Greek letter and operator in plain "
                     "English (write 'theta' not \\theta, 'sigma squared' not "
                     "\\sigma^2, 'sum from i equals 1 to n' not \\sum_{i=1}^n).\n"
                     "- Spell out function names with their arguments in "
                     "English (write 'the loss R of theta' not 'R(\\theta)').\n"
                     "- Plain prose only."
                 )},
                {"role": "user",
                 "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        }).encode("utf-8")
        req = urllib.request.Request(
            chat_url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local-vllm",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            print(f"[narrator.qa] vLLM call failed: {e}")
            return None
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "").strip()
        if not text:
            return None
        sentences = _split_sentences(text)
        return sentences or [text]

    return _backend


def make_vllm_intro_backend(
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ",
    max_tokens: int = 480,
    timeout: float = 30.0,
) -> Optional[SynthBackend]:
    """Open-book intro backend — used when the book has no good match.

    Differs from :func:`make_vllm_backend` in two ways:

      * The system prompt switches from closed-book ("use ONLY the
        context") to introductory ("introduce the topic for a student").
      * Retrieved passages, if any, are passed only as *soft* hints
        ("Optional context: …") — the model is free to ignore them.

    Returns None if the local vLLM endpoint is unreachable so callers
    can fall back to the retrieval-only template ("I cannot find …").
    Local-only by policy: never calls Anthropic / OpenAI.
    """
    import json as _json
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": "Bearer local-vllm"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
            served = {m.get("id") for m in payload.get("data", [])}
    except Exception as e:
        print(f"[narrator.qa] vLLM intro not reachable at {base_url}: {e}")
        return None

    if model not in served:
        if served:
            model = next(iter(served))
            print(f"[narrator.qa] intro using available vLLM model: {model}")
        else:
            return None

    chat_url = base_url.rstrip("/") + "/chat/completions"

    def _backend(
        question: str, passages: list[RetrievedPassage], book: Book,
    ) -> Optional[list[str]]:
        # Hint context — non-binding.  Keeps the answer in scope when
        # the book *does* contain something tangentially related, but
        # doesn't constrain the answer to those passages.
        hint = ""
        if passages:
            hint = "\n\nOptional context (may or may not be relevant):\n"
            hint += "\n\n".join(p.text[:600] for p in passages[:2])
        body = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system",
                 "content": (
                     "You are a math/stats teacher introducing a topic "
                     "to a student in a TTS-narrated lecture. The book "
                     "currently open does not cover this topic well, so "
                     "you answer from your general knowledge.\n"
                     "Goal: a SHORT 1–2 sentence introduction plus the "
                     "central formula on a display line.  Be concise — "
                     "the student will ask follow-ups if they need "
                     "more.  Every concept that has a formula MUST "
                     "include that formula.\n"
                     "\n"
                     "OUTPUT RULES (your reply is rendered literally):\n"
                     "- Plain prose paragraphs.  No markdown, no bullets, "
                     "no headers, no code fences.\n"
                     "- Math goes in LaTeX delimiters: \\( … \\) for "
                     "inline math inside a sentence, \\[ … \\] on its "
                     "own line for display equations.  KaTeX renders "
                     "these on the chalkboard.\n"
                     "- VERBALIZE every formula in plain English in the "
                     "same sentence, BEFORE you put the LaTeX on a "
                     "display line, so a TTS listener hears words and "
                     "not symbols.  Read \\sum_{i=1}^{N} x_i as 'the "
                     "sum from i equals 1 to N of x sub i', \\int as "
                     "'the integral of', \\frac{a}{b} as 'a over b', "
                     "x^2 as 'x squared', \\sqrt{x} as 'the square "
                     "root of x', \\nabla as 'gradient of', \\| x \\|^2 "
                     "as 'the squared norm of x'.  Read Greek letters "
                     "as their names (alpha, beta, theta, lambda, …) — "
                     "never name them by symbol shape.\n"
                     "- When you introduce a symbol, immediately say "
                     "'where x is the input vector' so the definition "
                     "lands in the same chalkboard card.\n"
                     "- Avoid sentence-numbering (1., 2., …); speak "
                     "naturally."
                 )},
                {"role": "user",
                 "content": f"Topic / question: {question}{hint}"},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        }).encode("utf-8")
        req = urllib.request.Request(
            chat_url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local-vllm",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            print(f"[narrator.qa] vLLM intro call failed: {e}")
            return None
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "").strip()
        if not text:
            return None
        sentences = _split_sentences(text)
        return sentences or [text]

    _backend.__name__ = "vllm_intro_backend"
    return _backend


def fetch_intro_formulas(
    question: str,
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ",
    max_formulas: int = 3,
    timeout: float = 15.0,
) -> list[str]:
    """Ask the local LLM for 1-3 canonical LaTeX formulas for *question*.

    Pure-symbolic side-channel — runs in addition to the prose intro so
    the spoken narration stays TTS-clean while the chalkboard still
    gets formula cards.  Returns a list of LaTeX strings (no
    delimiters); silently returns ``[]`` on any failure.

    Local-only by policy: never calls Anthropic / OpenAI.
    """
    import json as _json
    import urllib.request
    import urllib.error

    prompt = (
        "You output canonical mathematical formulas for a teaching "
        "whiteboard.  Given a topic, reply with a JSON object:\n"
        '{"formulas": ["latex1", "latex2", ...]}\n'
        f"Include 1 to {max_formulas} formulas, each one short, using LaTeX "
        "commands suitable for KaTeX.  No \\[ \\] delimiters — just the "
        "math source.  No prose, no markdown, only JSON."
    )
    body = _json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Topic: {question}"},
        ],
        "max_tokens": 240,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer local-vllm"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[narrator.qa] intro-formulas call failed: {e}")
        return []
    choice = (payload.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    if not content:
        return []
    # Strip code fences in case the model wrapped its reply.
    import re as _re
    if content.startswith("```"):
        content = _re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = _re.sub(r"\n?```\s*$", "", content)
    try:
        data = _json.loads(content)
    except Exception:
        return []
    raw = data.get("formulas") or []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw[:max_formulas]:
        s = str(item or "").strip()
        # Drop wrapping delimiters if the model included them.
        if s.startswith("$$") and s.endswith("$$"):
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1].strip()
        if s.startswith(r"\[") and s.endswith(r"\]"):
            s = s[2:-2].strip()
        if not s:
            continue
        norm = _re.sub(r"\s+", " ", s)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(s)
    return out


def make_qwen_backend(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    *,
    device: Optional[str] = None,
    max_new_tokens: int = 220,
) -> Optional[SynthBackend]:
    """In-process Qwen via transformers (heavyweight; prefer vLLM HTTP).

    Returns None when transformers/torch are not importable, or when the
    model cannot be loaded.  Provided as a fallback for environments
    without vLLM running.

    Determinism: greedy decoding (do_sample=False).
    """
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
    except Exception as e:
        print(f"[narrator.qa] cannot load {model_name}: {e}")
        return None

    def _backend(
        question: str, passages: list[RetrievedPassage], book: Book,
    ) -> Optional[list[str]]:
        if not passages:
            return None
        ctx = "\n\n".join(p.text[:1200] for p in passages)
        prompt = _QWEN_PROMPT.format(context=ctx, question=question)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # deterministic
                temperature=1.0,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        return _split_sentences(text) or [text]

    return _backend


def auto_qwen_backend(
    *, prefer: str = "vllm",
    base_url: str = "http://127.0.0.1:8000/v1",
    transformers_model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> Optional[SynthBackend]:
    """Try vLLM HTTP first, then in-process transformers, else None.

    The recommended path is vLLM — it loads the model once, serves multiple
    queries via HTTP, and matches the user's existing setup.
    """
    if prefer == "vllm":
        b = make_vllm_backend(base_url=base_url)
        if b:
            return b
    return make_qwen_backend(transformers_model)


# ---------------------------------------------------------------------------
# Public entry point — build a tangent NarrationPlan
# ---------------------------------------------------------------------------

def answer(
    book: Book, question: str,
    *,
    top_k: int = 3, cps: float = _DEFAULT_CPS,
    backend: Optional[SynthBackend] = None,
    intro_backend: Optional[SynthBackend] = None,
    dense_threshold: Optional[float] = None,
    bm25_threshold: Optional[float] = None,
    history: Optional[list] = None,
) -> NarrationPlan:
    """Build a NarrationPlan answering *question* against *book*.

    Parameters
    ----------
    backend:
        Override the active closed-book synth backend just for this
        call.  None → use the module's current backend (default
        retrieval-only).
    intro_backend:
        Open-book intro backend used when the retrieved passages are
        too weak to ground a closed-book answer.  None → fall back to
        the module's installed intro backend; if that is also None,
        the closed-book backend handles low-similarity questions like
        before (typically with the "I cannot find anything" template).
    dense_threshold, bm25_threshold:
        Override the cosine / BM25 cut-offs for the low-similarity
        decision.  None → use the module-level defaults
        (env-overridable via ``QA_LOW_SIM_THRESHOLD`` /
        ``QA_LOW_SIM_BM25``).
    """
    bk = backend or _synth_backend
    intro = intro_backend if intro_backend is not None else _intro_backend
    passages = _retrieve(book, question, top_k=top_k)
    low_sim = is_low_similarity(
        passages,
        dense_threshold=dense_threshold,
        bm25_threshold=bm25_threshold,
    )

    # Two LLM calls run at plan-build time, in parallel where possible:
    #   * the prose intro (only when low-similarity AND an intro
    #     backend is installed),
    #   * a semantic-graph **spec** describing the topic's structure,
    #     used by the chalkboard's own renderer to draw a diagram.
    # The system never asks the LLM for SVG markup directly — it only
    # asks for a structured description of the math entities; the
    # renderer that turns that into SVG lives in this repo.
    from concurrent.futures import ThreadPoolExecutor

    def _call_intro() -> Optional[list[str]]:
        if not (low_sim and intro is not None):
            return None
        try:
            return intro(question, passages, book)
        except Exception:
            return None

    def _call_spec():
        try:
            from viz.llm_spec import fetch_semantic_spec
            return fetch_semantic_spec(question)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_intro = ex.submit(_call_intro)
        f_spec = ex.submit(_call_spec)
        intro_sentences = f_intro.result()
        semantic_graph = f_spec.result()
    semantic_spec = (
        semantic_graph.to_dict() if semantic_graph is not None else None
    )

    used_intro = intro_sentences is not None
    sentences: Optional[list[str]] = intro_sentences
    if sentences is None:
        # Tutor-aware backends accept a 4th ``history`` argument so
        # follow-ups can avoid re-introducing concepts.  Strict legacy
        # backends only accept (question, passages, book) — call them
        # without history.  We probe a flag attribute set by the
        # tutor-backend factory.
        if getattr(bk, "supports_history", False):
            sentences = bk(question, passages, book, history)
        else:
            sentences = bk(question, passages, book)
    if sentences is None:
        # Backend declined; fall back to retrieval-only.
        sentences = _retrieval_only_backend(question, passages, book)
    # Belt-and-suspenders: sanitise every emitted sentence so TTS never
    # speaks LaTeX delimiters or markdown — even if the backend's system
    # prompt failed to suppress them.
    sentences = [
        _sanitize_for_narration(s) for s in sentences if s and s.strip()
    ]
    sentences = [s for s in sentences if s]

    surface_re, alias_to_cid = _build_surface_regex(book)

    default_nid = passages[0].nid if passages else "b"

    def _attribute(sentence: str) -> str:
        """Pick the nid of the passage that literally contains this sentence.

        Retrieval-only sentences are spliced verbatim from passage texts so
        substring matching is exact.  Synthesised sentences (vLLM / Qwen)
        rarely match, so they fall back to the top-scoring passage.
        """
        s_norm = sentence.strip()
        if not s_norm:
            return default_nid
        for p in passages:
            if s_norm in p.text:
                return p.nid
        return default_nid

    clauses: list[NarrationClause] = []
    for s in sentences:
        if not s.strip():
            continue
        clauses.append(NarrationClause(
            text=s.strip(),
            home_nid=_attribute(s),
            concepts=_tag_clause(s, surface_re, alias_to_cid),
            suggested_dur=_estimate_dur(s, cps),
        ))

    top_dense = max((p.dense_score for p in passages), default=0.0)
    top_bm25 = max((p.bm25_score for p in passages), default=0.0)

    return NarrationPlan(
        topic=f"Q: {question}",
        book_title=book.title,
        clauses=clauses,
        visited_nids=[p.nid for p in passages],
        meta={
            "mode": "tangent",
            "n_passages": len(passages),
            "question": question,
            "synth_backend": getattr(bk, "__name__", "lambda"),
            "intro_backend": getattr(intro, "__name__", "")
                              if (intro and used_intro) else "",
            "intro_source": "llm" if used_intro else "book",
            "low_similarity": low_sim,
            "top_dense_score": top_dense,
            "top_bm25_score": top_bm25,
            "passage_scores": [p.score for p in passages],
            "semantic_spec": semantic_spec,
        },
    )


# ---------------------------------------------------------------------------
# Streaming Q&A — yields clauses as the LLM produces sentences
# ---------------------------------------------------------------------------

def answer_streaming(
    book: Book, question: str,
    *,
    top_k: int = 3,
    history: Optional[list] = None,
) -> NarrationPlan:
    """Streaming counterpart of :func:`answer`.

    When a streaming backend is installed (via
    :func:`set_streaming_backend`), this returns a NarrationPlan whose
    ``clauses`` is a *generator* — clauses are yielded as the LLM
    finishes each sentence, so the orchestrator can begin TTS / visual
    emission while the LLM is still producing output.  Cuts perceived
    latency from "wait 3 s, then it speaks" to "first audio in <1 s".

    Falls back to :func:`answer` when no streaming backend is
    installed (silent degradation, unchanged behaviour for callers).
    """
    bk = _streaming_backend
    if bk is None or not getattr(bk, "is_streaming", False):
        return answer(book, question, top_k=top_k, history=history)

    passages = _retrieve(book, question, top_k=top_k)
    # If the book has no good match AND an intro backend is installed,
    # hand off to the non-streaming ``answer`` flow so the intro
    # backend (general-knowledge LLM with math-rich output) gets a
    # chance instead of forcing the streaming tutor to ground a claim
    # it can't find in the passages.  The intro response is short
    # (3–6 sentences), so the lost streaming buys only ~1–2 s in
    # exchange for an actual answer.  When no intro is installed,
    # stay on the streaming path — the tutor's "I can't find this in
    # the book" response is still better than nothing.
    if _intro_backend is not None and is_low_similarity(passages):
        return answer(book, question, top_k=top_k, history=history)

    default_nid = passages[0].nid if passages else "b"
    visited = [default_nid]

    # Fire the semantic-graph spec call so the orchestrator's Tier-2
    # canonical-card path has a graph to render.  Same contract as
    # ``answer()``: never asks the LLM for SVG markup, only structure.
    # Local-only — degrades to None silently when the endpoint is down.
    try:
        from viz.llm_spec import fetch_semantic_spec
        semantic_graph = fetch_semantic_spec(question)
    except Exception:
        semantic_graph = None
    semantic_spec = (
        semantic_graph.to_dict() if semantic_graph is not None else None
    )

    def _clause_gen():
        # Iterate the LLM's sentence stream and wrap each into a
        # NarrationClause on the fly.  Sanitisation happens here so
        # TTS receives speakable text while clause.text retains the
        # math the formula detector needs.
        try:
            for raw in bk(question, passages, book, history):
                if not raw or not raw.strip():
                    continue
                spoken = _sanitize_for_narration(raw)
                if not spoken:
                    continue
                yield NarrationClause(
                    text=spoken, home_nid=default_nid,
                    concepts=[],
                    suggested_dur=_estimate_dur(spoken),
                )
        except Exception as e:
            print(f"[narrator.qa] streaming clause-gen failed: {e}")
            return

    return NarrationPlan(
        topic=question or "<streaming-tutor>",
        book_title=book.title or "",
        clauses=_clause_gen(),
        visited_nids=visited,
        meta={
            "mode": "streaming_tutor",
            "question": question,
            "n_passages": len(passages),
            "synth_backend": getattr(bk, "__name__", "lambda"),
            "passage_scores": [p.score for p in passages],
            "semantic_spec": semantic_spec,
        },
        streaming=True,
    )


# ---------------------------------------------------------------------------
# Book-level overview — "what is this book about?"
# ---------------------------------------------------------------------------

def book_overview(
    book: Book, *, depth: str = "",
) -> NarrationPlan:
    """A high-level NarrationPlan that introduces the book itself.

    Mode:
      * ``depth == "short"`` → just title + author + a one-line tease.
      * ``depth == "deep"``  → title + a clause per top-level chapter
        with each chapter's title.  Closes with an invitation to
        drill in (``ask me about any chapter for depth``).
      * ``depth == ""`` (default) → middle ground: title + author +
        a clause listing the chapter titles in groups.
    """
    title = (book.title or "").strip() or "this book"
    author = (book.author or "").strip()
    chapters = [
        n for n in book.root.walk()
        if n.kind == "chapter" and (n.title or "").strip()
    ]

    clauses: list[NarrationClause] = []
    visited: list[str] = [book.root.nid] if book.root else []

    # Opening line.
    if author:
        opening = f"This book is {title}, by {author}."
    else:
        opening = f"This book is {title}."
    clauses.append(NarrationClause(
        text=opening, home_nid=book.root.nid,
        concepts=[], suggested_dur=2.4,
    ))

    if depth == "short":
        n_ch = len(chapters)
        if n_ch:
            clauses.append(NarrationClause(
                text=(f"It has {n_ch} chapters covering the field. "
                      f"Ask me about any chapter for depth."),
                home_nid=book.root.nid,
                concepts=[], suggested_dur=2.6,
            ))
        return NarrationPlan(
            topic="<book-overview>", book_title=title,
            clauses=clauses, visited_nids=visited,
            meta={"mode": "book_overview", "depth": depth,
                  "n_chapters": len(chapters)},
        )

    if depth == "deep":
        # One clause per chapter.
        for ch in chapters:
            num = (ch.number or "").strip()
            head = f"Chapter {num}." if num else "Chapter."
            clauses.append(NarrationClause(
                text=f"{head} {ch.title}.",
                home_nid=ch.nid,
                concepts=[], suggested_dur=2.0,
            ))
            visited.append(ch.nid)
        clauses.append(NarrationClause(
            text="Ask me about any chapter for a deeper dive.",
            home_nid=book.root.nid,
            concepts=[], suggested_dur=2.2,
        ))
        return NarrationPlan(
            topic="<book-overview>", book_title=title,
            clauses=clauses, visited_nids=visited,
            meta={"mode": "book_overview", "depth": depth,
                  "n_chapters": len(chapters)},
        )

    # Default depth: title + grouped chapter list.
    if chapters:
        # Group chapter titles in chunks of 4 so each clause stays
        # short enough for clean TTS pacing.
        chunk_size = 4
        for i in range(0, len(chapters), chunk_size):
            group = chapters[i:i + chunk_size]
            heads = []
            for ch in group:
                num = (ch.number or "").strip()
                t = (ch.title or "").strip()
                heads.append(f"{num}. {t}" if num else t)
            text = ("It opens with " if i == 0 else "It then covers ")
            text += "; ".join(heads) + "."
            clauses.append(NarrationClause(
                text=text, home_nid=book.root.nid,
                concepts=[], suggested_dur=3.0,
            ))
            for ch in group:
                visited.append(ch.nid)
        clauses.append(NarrationClause(
            text="Ask me about any chapter, section, or topic.",
            home_nid=book.root.nid,
            concepts=[], suggested_dur=2.4,
        ))

    return NarrationPlan(
        topic="<book-overview>", book_title=title,
        clauses=clauses, visited_nids=visited,
        meta={"mode": "book_overview", "depth": depth,
              "n_chapters": len(chapters)},
    )


# ---------------------------------------------------------------------------
# Recap — "what have we covered so far?"
# ---------------------------------------------------------------------------

def recap(
    book: Book, *,
    history=None,
    knowledge=None,
) -> NarrationPlan:
    """A short NarrationPlan that summarises what's been covered.

    Pulls from two sources:
      * ``history`` — list of ``DialogueTurn`` objects (or compatible
        duck-typed records with ``user_text`` / ``intent`` /
        ``focus_topic``).  Surfaces the topical questions the user
        has asked.
      * ``knowledge`` — a ``SessionKnowledge`` (or compatible
        duck-typed object).  Surfaces concrete primitives the
        chalkboard has shown: canonical topics, equation refs.

    No LLM, no retrieval — pure deterministic summary.
    """
    title = (book.title or "this book").strip()
    clauses: list[NarrationClause] = []
    visited: list[str] = [book.root.nid] if book.root else []
    home_nid = book.root.nid if book.root else ""

    # Topical asks from history (skip control / recap turns themselves).
    asked_topics: list[str] = []
    if history:
        seen: set[str] = set()
        for t in history:
            intent = getattr(t, "intent", "")
            if intent in {"control", "recap", "follow_up"}:
                continue
            topic = (getattr(t, "focus_topic", "")
                     or getattr(t, "user_text", "")).strip()
            if not topic:
                continue
            key = topic.lower()
            if key in seen:
                continue
            seen.add(key)
            asked_topics.append(topic)

    canonical_topics: list[str] = []
    cited_equations: list[str] = []
    if knowledge is not None:
        canonical_topics = sorted(
            getattr(knowledge, "seen_canonical_topics", set())
        )
        # ``seen_refs`` keys look like ``"Equation::5.42"`` — extract
        # the human label for the recap.
        for r in sorted(getattr(knowledge, "seen_refs", set())):
            if "::" not in r:
                continue
            kind, label = r.split("::", 1)
            cited_equations.append(f"{kind} {label}")

    # Opening.
    if not asked_topics and not canonical_topics and not cited_equations:
        clauses.append(NarrationClause(
            text=f"We haven't covered anything yet in this session of {title}.",
            home_nid=home_nid, concepts=[], suggested_dur=2.6,
        ))
    else:
        clauses.append(NarrationClause(
            text="Here is what we've covered so far.",
            home_nid=home_nid, concepts=[], suggested_dur=1.6,
        ))

    if asked_topics:
        # Group up to 5 topics per clause for a steady reading pace.
        for i in range(0, len(asked_topics), 5):
            chunk = asked_topics[i:i + 5]
            text = ("You asked about " if i == 0 else "And about ")
            text += "; ".join(chunk) + "."
            clauses.append(NarrationClause(
                text=text, home_nid=home_nid,
                concepts=[], suggested_dur=2.6,
            ))

    if canonical_topics:
        for i in range(0, len(canonical_topics), 5):
            chunk = canonical_topics[i:i + 5]
            text = ("We have diagrams for " if i == 0 else "And for ")
            text += "; ".join(chunk) + "."
            clauses.append(NarrationClause(
                text=text, home_nid=home_nid,
                concepts=[], suggested_dur=2.6,
            ))

    if cited_equations:
        # Cap to 6 to keep recap clauses short.
        eqs = cited_equations[:6]
        text = "We've cited " + ", ".join(eqs)
        if len(cited_equations) > 6:
            text += f", and {len(cited_equations) - 6} other references"
        text += "."
        clauses.append(NarrationClause(
            text=text, home_nid=home_nid,
            concepts=[], suggested_dur=2.6,
        ))

    # Invitation to continue.
    if asked_topics or canonical_topics or cited_equations:
        clauses.append(NarrationClause(
            text="What would you like to dive into next?",
            home_nid=home_nid, concepts=[], suggested_dur=1.8,
        ))

    return NarrationPlan(
        topic="<recap>", book_title=title,
        clauses=clauses, visited_nids=visited,
        meta={"mode": "recap",
              "n_topics": len(asked_topics),
              "n_canonical": len(canonical_topics),
              "n_equations": len(cited_equations)},
    )


# ---------------------------------------------------------------------------
# Cross-reference exploration — "what does this section reference?",
# "where is this cited?".  Walks the citation graph for a given nid.
# ---------------------------------------------------------------------------

def xref_explore(
    book: Book, focus_nid: str, *,
    max_outgoing: int = 6,
    max_incoming: int = 6,
) -> NarrationPlan:
    """Narrate the citation neighborhood around *focus_nid*.

    Two halves:
      * **Outgoing** — what this section cites (Figure / Section /
        Chapter / Table / Exercise references made *from* descendants
        of ``focus_nid``).  Surfaces the dependencies the user may
        want to follow.
      * **Incoming** — what cites this section.  Surfaces the
        passages that depend on this material.

    Both halves are deduped + grouped by label, then capped at
    ``max_outgoing`` / ``max_incoming`` so the recap stays short.
    """
    from book import xref as xref_mod

    title = (book.title or "this book").strip()
    home_nid = focus_nid or (book.root.nid if book.root else "")
    focus = book.find(focus_nid) if focus_nid else None
    focus_label = ""
    if focus is not None:
        num = (focus.number or "").strip()
        ttl = (focus.title or "").strip()
        kind = (focus.kind or "").strip()
        if num and ttl:
            focus_label = f"{kind.capitalize()} {num} ({ttl})"
        elif num:
            focus_label = f"{kind.capitalize()} {num}"
        elif ttl:
            focus_label = ttl

    clauses: list[NarrationClause] = []
    visited: list[str] = [home_nid] if home_nid else []

    if not focus_nid or focus is None:
        clauses.append(NarrationClause(
            text="I don't have a focus to explore — ask me about a "
                 "specific section first.",
            home_nid=home_nid, concepts=[], suggested_dur=2.4,
        ))
        return NarrationPlan(
            topic="<xref-explore>", book_title=title,
            clauses=clauses, visited_nids=visited,
            meta={"mode": "xref_explore",
                  "n_outgoing": 0, "n_incoming": 0},
        )

    out_refs = xref_mod.outgoing(book, focus_nid)
    in_refs = xref_mod.incoming(book, focus_nid)
    out_top = xref_mod.group_by_label(out_refs)[:max_outgoing]
    # Roll incoming up by *source* so the user sees which sections
    # outside the focus subtree cite it (not which descendant gets
    # cited internally).  Keep one entry per source chapter / section.
    in_top = xref_mod.group_by_source(in_refs)[:max_incoming]

    # Opening — anchor on what we're exploring.
    if focus_label:
        opening = f"Let's explore the citation neighborhood of {focus_label}."
    else:
        opening = "Let's explore the citation neighborhood of this section."
    clauses.append(NarrationClause(
        text=opening, home_nid=home_nid,
        concepts=[], suggested_dur=2.4,
    ))

    # Outgoing — what this section references.
    if out_top:
        labels = [lbl for lbl, _ in out_top]
        clauses.append(NarrationClause(
            text=("This section references " + "; ".join(labels) + "."),
            home_nid=home_nid, concepts=[], suggested_dur=3.0,
        ))
    else:
        clauses.append(NarrationClause(
            text=("This section makes no outgoing citations to other "
                  "labelled material."),
            home_nid=home_nid, concepts=[], suggested_dur=2.4,
        ))

    # Incoming — what cites this section.
    if in_top:
        names = []
        for nid, count, sample in in_top:
            target_node = book.find(nid)
            target_title = (target_node.title or "").strip() if target_node else ""
            target_num = (target_node.number or "").strip() if target_node else ""
            if target_num and target_title:
                tag = f"§{target_num} ({target_title})"
            elif target_title:
                tag = target_title
            elif target_num:
                tag = f"§{target_num}"
            else:
                tag = sample
            names.append(f"{tag} ({count}×)" if count > 1 else tag)
        clauses.append(NarrationClause(
            text=("It is cited by " + "; ".join(names) + "."),
            home_nid=home_nid, concepts=[], suggested_dur=3.2,
        ))
    else:
        clauses.append(NarrationClause(
            text="No other section cites this one.",
            home_nid=home_nid, concepts=[], suggested_dur=1.8,
        ))

    # Invitation.
    clauses.append(NarrationClause(
        text="Ask me about any of these to follow the trail.",
        home_nid=home_nid, concepts=[], suggested_dur=2.0,
    ))

    return NarrationPlan(
        topic="<xref-explore>", book_title=title,
        clauses=clauses, visited_nids=visited,
        meta={"mode": "xref_explore",
              "focus_nid": focus_nid,
              "n_outgoing": len(out_refs),
              "n_incoming": len(in_refs),
              "outgoing_labels": [lbl for lbl, _ in out_top],
              "incoming_sources": [nid for nid, _, _ in in_top]},
    )


# ---------------------------------------------------------------------------
# Dependencies — "what do I need to know first?", "prerequisites?"
# ---------------------------------------------------------------------------

def dependencies(
    book: Book, focus_nid: str, *, max_items: int = 6,
) -> NarrationPlan:
    """Walk the citation graph backwards: surface what *focus_nid*
    depends on (its outgoing references) as a list of prerequisites.

    Heuristic: a section's outgoing cross-refs are usually the
    chapters / sections / theorems it builds on.  Group them by
    target node, drop self-subtree refs, and announce the most-cited
    targets as prerequisites — those are the things the user should
    review before diving in.
    """
    from book import xref as xref_mod

    title = (book.title or "this book").strip()
    home_nid = focus_nid or (book.root.nid if book.root else "")
    focus = book.find(focus_nid) if focus_nid else None
    focus_label = ""
    if focus is not None:
        num = (focus.number or "").strip()
        ttl = (focus.title or "").strip()
        kind = (focus.kind or "").strip()
        if num and ttl:
            focus_label = f"{kind.capitalize()} {num} ({ttl})"
        elif num:
            focus_label = f"{kind.capitalize()} {num}"
        elif ttl:
            focus_label = ttl

    clauses: list[NarrationClause] = []
    visited: list[str] = [home_nid] if home_nid else []

    if not focus_nid or focus is None:
        clauses.append(NarrationClause(
            text=("To list prerequisites I need to know what we are "
                  "studying — ask me about a chapter or section first."),
            home_nid=home_nid, concepts=[], suggested_dur=2.6,
        ))
        return NarrationPlan(
            topic="<dependencies>", book_title=title,
            clauses=clauses, visited_nids=visited,
            meta={"mode": "dependencies", "n_prereqs": 0},
        )

    out_refs = xref_mod.outgoing(book, focus_nid)
    # Drop targets that share the same chapter as the focus — the
    # user is asking for *background* (i.e., things that come from
    # outside this chapter), not for the chapter's own internal
    # cross-references.  Keep the list ranked by reference count.
    focus_chapter = xref_mod._chapter_of(focus_nid)
    if focus_chapter:
        out_refs = [
            cr for cr in out_refs
            if xref_mod._chapter_of(cr.to_nid) != focus_chapter
        ]
    by_target = xref_mod.group_by_target(out_refs)[:max_items]

    # Opening anchored on the focus.
    if focus_label:
        clauses.append(NarrationClause(
            text=f"Here's the background you'll want for {focus_label}.",
            home_nid=home_nid, concepts=[], suggested_dur=2.4,
        ))
    else:
        clauses.append(NarrationClause(
            text="Here's the background you'll want for this section.",
            home_nid=home_nid, concepts=[], suggested_dur=2.2,
        ))

    # Concept-graph prerequisites — computed before the citation
    # branches so they can ride alongside both the present and the
    # absent-citation paths.
    concept_prereqs: list[str] = []
    if _concept_graph and focus is not None:
        try:
            from book.concept_graph import prerequisites as _cg_pre
            focus_title = (focus.title or "").strip()
            preqs = _cg_pre(_concept_graph, focus_title) if focus_title else []
            concept_prereqs = [p for p, _ in preqs]
        except Exception:
            pass

    if not by_target:
        if concept_prereqs:
            cap = 4
            names = concept_prereqs[:cap]
            clauses.append(NarrationClause(
                text=("It makes no outgoing citations, but at the "
                      "concept level it builds on " + "; ".join(names)
                      + "."),
                home_nid=home_nid, concepts=[], suggested_dur=3.0,
            ))
        else:
            clauses.append(NarrationClause(
                text=("This section makes no outgoing citations, so it "
                      "stands on its own."),
                home_nid=home_nid, concepts=[], suggested_dur=2.4,
            ))
        return NarrationPlan(
            topic="<dependencies>", book_title=title,
            clauses=clauses, visited_nids=visited,
            meta={"mode": "dependencies", "focus_nid": focus_nid,
                  "n_prereqs": 0,
                  "concept_prereqs": concept_prereqs},
        )

    # Walk top targets, narrating each as a "you may want to review …"
    # clause.  Visit them in the plan so the chalkboard's reading-list
    # surfaces them.
    intro_done = False
    prereq_nids: list[str] = []
    for tnid, count, sample in by_target:
        target_node = book.find(tnid)
        if target_node is None:
            target_label = sample
        else:
            tnum = (target_node.number or "").strip()
            ttitle = (target_node.title or "").strip()
            if tnum and ttitle:
                target_label = f"§{tnum} ({ttitle})"
            elif ttitle:
                target_label = ttitle
            elif tnum:
                target_label = f"§{tnum}"
            else:
                target_label = sample
        prereq_nids.append(tnid)
        visited.append(tnid)
        if not intro_done:
            text = (f"Most prominently, you'll want {target_label}")
            intro_done = True
        else:
            text = f"And {target_label}"
        if count > 2:
            text += f", which is referenced {count} times"
        text += "."
        clauses.append(NarrationClause(
            text=text, home_nid=tnid,
            concepts=[], suggested_dur=2.4,
        ))

    # Concept-graph prerequisites — added as a separate clause so the
    # user hears both the citation-level dependencies (what passages
    # this section cites) and the concept-level dependencies (what
    # ideas this section is built on, when those are in the graph).
    if concept_prereqs:
        cap = 4
        names = concept_prereqs[:cap]
        text = ("At the concept level, this builds on "
                + "; ".join(names) + ".")
        clauses.append(NarrationClause(
            text=text, home_nid=home_nid,
            concepts=[], suggested_dur=2.6,
        ))

    clauses.append(NarrationClause(
        text="Ask me to dive into any of those, or say 'see also' "
             "for the rest of the citation neighborhood.",
        home_nid=home_nid, concepts=[], suggested_dur=2.6,
    ))

    return NarrationPlan(
        topic="<dependencies>", book_title=title,
        clauses=clauses, visited_nids=visited,
        meta={"mode": "dependencies",
              "focus_nid": focus_nid,
              "n_prereqs": len(by_target),
              "prereq_nids": prereq_nids,
              "concept_prereqs": concept_prereqs},
    )
