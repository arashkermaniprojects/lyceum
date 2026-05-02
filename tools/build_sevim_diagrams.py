"""Pre-build per-section SeVim concept diagrams for a chapter-map.

For every node in the chapter-map sidecar, we ask the local Qwen to
extract 3-7 entity-relation triples from the node's narration text
(formula explanation + story paragraph) using SeVim's relation
ontology.  Then we construct a ``SceneGraph`` directly from those
triples and run SeVim's S3 (visual map) → S4 (layout) → S5 (render)
stages to produce the SVG.  Result is keyed by ``nid`` and saved
next to the chapter-map at

    <book_stem>.sevim_diagrams.<root>.json

LLM extraction beats SeVim's rule-based S2 on math-heavy prose because
the rule-based parser splits formula explanations like "the sum from
m equals 1 to capital M captures the combination of basis functions"
into nonsense fragments.  Qwen reads the same sentence and produces
triples like (weighted sum, contains, basis function), (basis
function, attribute_of, coefficient beta_m), (M, measures, number
of terms) — which then render as a clean three-box diagram.

Usage:

    python -m tools.build_sevim_diagrams \\
        books/ESLII.chapter_map.b_ch5.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from typing import Optional


LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"

# SeVim's relation ontology — closed set the LLM must obey.
_ALLOWED_RELATIONS = [
    "contains", "part_of", "causes", "sequence",
    "attribute_of", "similar_to", "opposes", "instance_of",
    "used_for", "requires", "reduces_to", "measures",
    "equals", "approximately_equal", "maps_to",
    "element_of", "subset_of", "labels", "points_to", "connects",
    "grouped_with", "aligned_with", "implies",
]


_EXTRACT_SYSTEM = """You extract concept diagrams from textbook
narration.  You receive ONE paragraph of plain English (a section's
story plus the parameter-by-parameter walk-through of its central
equation) and you return a small concept graph: 3-7 nodes naming the
mathematical objects the paragraph talks about, and 3-8 edges naming
how they relate.

OUTPUT FORMAT — JSON exactly like this:

  {
    "nodes": [
      {"id": "weighted_sum", "label": "weighted sum f(X)"},
      {"id": "basis_function", "label": "basis function h_m of X"},
      {"id": "coefficient", "label": "coefficient beta_m"},
      {"id": "M", "label": "number of basis functions M"}
    ],
    "edges": [
      {"from": "weighted_sum", "to": "basis_function", "relation": "contains"},
      {"from": "basis_function", "to": "coefficient", "relation": "attribute_of"},
      {"from": "M", "to": "weighted_sum", "relation": "measures"}
    ]
  }

HARD RULES:

1. NODES are concrete mathematical objects from the paragraph: a
   function, a coefficient, a parameter, a data point, a sum, a
   matrix, a region, a kernel, a basis function, etc.  Each label
   is in plain English (no LaTeX, no glyphs); spell math out
   ("coefficient beta sub m", "function f of X", "lambda").

2. EDGES connect two nodes via ONE of these relations EXACTLY:
   contains, part_of, causes, sequence, attribute_of, similar_to,
   opposes, instance_of, used_for, requires, reduces_to,
   measures, equals, approximately_equal, maps_to, element_of,
   subset_of, labels, points_to, connects, grouped_with,
   aligned_with, implies.  No other relation strings.

3. NODE IDs are short snake_case identifiers used as the JSON
   key.  Stable across re-runs would be ideal but isn't required.

4. PICK CONCEPTS THAT FORM A GRAPH.  Don't list 7 disconnected
   nodes.  Every node should be reachable from at least one
   other node by an edge.

5. NO COMMENTARY.  The whole response is the JSON object — first
   character "{", last character "}".  No prose, no markdown
   fence."""


def _call_llm_json(system: str, user: str, *,
                   max_tokens: int = 800,
                   temperature: float = 0.2,
                   retries: int = 1) -> Optional[dict]:
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode()
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                LLM_URL, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = json.loads(resp.read())
            content = (raw["choices"][0]["message"]
                       .get("content") or "").strip()
            if content:
                return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    print(f"  [llm] failed: {last_err}", file=sys.stderr)
    return None


_ID_SANITISE = re.compile(r"[^a-z0-9]+")


def _normalise_id(s: str) -> str:
    s = _ID_SANITISE.sub("_", (s or "").lower()).strip("_")
    return s or "x"


# --- Greek-letter and Unicode-subscript typesetting ---------------------
# SeVim's S5 renders <text> as flat strings, so we typeset labels at
# the source before they reach the renderer.  Replace named Greek
# letters with Unicode glyphs, and "x_m" / "beta_i" patterns with the
# proper Unicode subscript ("xₘ", "βᵢ") where one exists.

_GREEK_LOWER = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
}
_GREEK_UPPER = {k.capitalize(): v for k, v in {
    "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ",
    "Epsilon": "Ε", "Zeta": "Ζ", "Eta": "Η", "Theta": "Θ",
    "Iota": "Ι", "Kappa": "Κ", "Lambda": "Λ", "Mu": "Μ",
    "Nu": "Ν", "Xi": "Ξ", "Omicron": "Ο", "Pi": "Π",
    "Rho": "Ρ", "Sigma": "Σ", "Tau": "Τ", "Upsilon": "Υ",
    "Phi": "Φ", "Chi": "Χ", "Psi": "Ψ", "Omega": "Ω",
}.items()}

# Unicode subscripts only exist for a subset of letters and digits.
# When a letter has no subscript counterpart we keep the original
# token (e.g. "h_m(X)" → "hₘ(X)" works, "h_b(X)" → "h_b(X)" stays
# as is, since Unicode has no subscript "b").
_SUB_DIGITS = {str(d): chr(0x2080 + d) for d in range(10)}
_SUB_LETTERS = {
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
    "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ",
    "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
    "+": "₊", "-": "₋", "=": "₌",
}


def _to_subscript(token: str) -> Optional[str]:
    """Translate a short index token (e.g. "m", "i", "12", "i+1")
    into its Unicode-subscript equivalent.  Returns ``None`` when
    any character has no subscript form so the caller can fall
    back to the literal underscore."""
    if not token or len(token) > 4:
        return None
    out = []
    for ch in token:
        if ch in _SUB_DIGITS:
            out.append(_SUB_DIGITS[ch])
        elif ch.lower() in _SUB_LETTERS:
            out.append(_SUB_LETTERS[ch.lower()])
        elif ch == ",":
            out.append("︐")
        else:
            return None
    return "".join(out)


# Use letter-only look-around: ``\b`` treats ``_`` as a word char, so
# ``\bbeta\b`` doesn't match ``beta_m`` even though we want it to.
_GREEK_RE = re.compile(
    r"(?<![A-Za-z])("
    + "|".join(sorted(list(_GREEK_LOWER.keys())
                       + list(_GREEK_UPPER.keys()),
                       key=len, reverse=True))
    + r")(?![A-Za-z])"
)
_SUB_RE = re.compile(r"([A-Za-zα-ωΑ-Ω])_([A-Za-z0-9+\-=,]{1,3})\b")


_LABEL_MAX_CHARS = 22
# Filler words the LLM tends to prepend; dropping the *first* one
# usually makes the label fit a 100-150 px primitive container
# without losing the math identity ("matrix H of basis functions"
# → "H of basis functions" still reads like math).
_LABEL_PREFIX_FILLERS = (
    "estimated ", "original ", "transformed ", "intermediate ",
    "vector ", "matrix ", "function ", "parameter ",
    "coefficient ", "constant ", "operator ", "scalar ",
    "term ", "value ",
)


def _compact_label(label: str, max_chars: int = _LABEL_MAX_CHARS) -> str:
    """Shorten an over-long label so it fits SeVim's primitive
    containers (matrix brackets / diamonds skip ``_wrap_label`` and
    overflow at ~22 chars).  Strategy: drop a single filler prefix
    if that's enough; otherwise truncate at the nearest word
    boundary; ellipsise as last resort."""
    s = (label or "").strip()
    if len(s) <= max_chars:
        return s
    low = s.lower()
    for filler in _LABEL_PREFIX_FILLERS:
        if low.startswith(filler):
            trimmed = s[len(filler):].strip()
            if trimmed and len(trimmed) <= max_chars:
                return trimmed
            s = trimmed or s
            low = s.lower()
            break
    if len(s) <= max_chars:
        return s
    words = s.split()
    accum: list[str] = []
    n = 0
    for w in words:
        add = len(w) + (1 if accum else 0)
        if n + add > max_chars:
            break
        accum.append(w)
        n += add
    if accum:
        return " ".join(accum)
    return s[: max_chars - 1].rstrip() + "…"


def _typeset(label: str) -> str:
    """Pre-render Greek + subscripts in a label so SeVim's flat
    <text> output looks like real math."""
    if not label:
        return label
    # Greek words first ("beta_m" → "β_m" so the subscript pass below
    # sees a single Greek glyph followed by "_m").
    def _greek_sub(m):
        w = m.group(1)
        return (_GREEK_LOWER.get(w.lower())
                 or _GREEK_UPPER.get(w.capitalize())
                 or w)
    s = _GREEK_RE.sub(_greek_sub, label)
    # Subscripts ("xₘ", "βᵢ", "h₁₂"); fall back to literal "_" when
    # any char in the index has no subscript form.
    def _sub_sub(m):
        head = m.group(1)
        idx = m.group(2)
        sub = _to_subscript(idx)
        return f"{head}{sub}" if sub is not None else m.group(0)
    s = _SUB_RE.sub(_sub_sub, s)
    return s


def _build_scene_graph(triples: dict, *, src_label: str):
    """Turn the LLM's JSON triples into a SeVim ``SceneGraph`` ready
    for S3.  Drops edges whose endpoints don't resolve, and edges
    whose relation is outside the closed ontology.  Returns ``None``
    when the result has fewer than 2 nodes or zero edges."""
    from sevim.ir import SceneGraph, SceneNode, SceneEdge, SpanRef

    raw_nodes = triples.get("nodes") or []
    raw_edges = triples.get("edges") or []
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        return None
    span = SpanRef(start=0, end=0, utterance_id=src_label)
    nodes: list = []
    by_id: dict[str, SceneNode] = {}
    # Secondary index: normalised LABEL → full id, so an edge that
    # references a node by its label ("weighted sum f(X)") instead
    # of its id ("weighted_sum") still resolves.  Same trick with
    # the id field if it was passed in non-snake-case.
    by_label: dict[str, str] = {}
    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        raw_id = str(n.get("id") or n.get("label") or "")
        nid = _normalise_id(raw_id)
        if not nid:
            continue
        full_id = f"n_{nid}"
        if full_id in by_id:
            continue
        raw_label = (str(n.get("label") or n.get("id") or "").strip()
                      or full_id[2:].replace("_", " "))
        # Compact first (drops fillers, truncates at word-boundary),
        # then typeset (Greek + Unicode subscripts).  Order matters:
        # compacting is length-aware and won't see the right number
        # of characters once subscripts collapse.
        label = _typeset(_compact_label(raw_label))
        # Force rectangle as the visual primitive.  SeVim's S3
        # ``_label_to_primitive`` classifies labels containing "matrix",
        # "parameter", "weight", "kernel" etc. as math primitives
        # (bracket / diamond) which bypass ``_wrap_label`` and
        # overflow at any non-trivial length.  ``meta["kind"]`` is
        # the explicit-override path that S3 honours before any of
        # its label-classification rules run, so this guarantees
        # wrap-friendly boxes regardless of vocabulary.
        nodes.append(SceneNode(
            id=full_id, label=label, node_type="entity",
            embedding=(), salience=0.5, src_spans=[span],
            meta={"kind": "rect"},
        ))
        by_id[full_id] = nodes[-1]
        by_label[_normalise_id(label)] = full_id
        by_label[nid] = full_id

    def _resolve(token: str) -> str:
        """Return the full id for whatever the LLM wrote."""
        s = _normalise_id(token)
        if not s:
            return ""
        candidate = f"n_{s}"
        if candidate in by_id:
            return candidate
        if s in by_label:
            return by_label[s]
        # Try a longest-prefix label match — handles "weighted sum f(X)"
        # vs "weighted sum" mismatches.
        for k, v in by_label.items():
            if k and (k.startswith(s) or s.startswith(k)) and len(k) > 3:
                return v
        return ""

    edges: list = []
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        rel = str(e.get("relation") or "").strip().lower()
        # Map common LLM word-choices to ontology entries; anything
        # unrecognised falls back to ``connects`` rather than being
        # dropped, so we never end up with zero edges on a graph
        # that had real structure.
        if rel not in _ALLOWED_RELATIONS:
            rel = _RELATION_ALIAS.get(rel, "connects")
        a = _resolve(str(e.get("from") or ""))
        b = _resolve(str(e.get("to") or ""))
        if not a or not b or a == b:
            continue
        eid = f"e_{rel}_{a}_{b}"
        if any(x.id == eid for x in edges):
            continue
        edges.append(SceneEdge(
            id=eid, from_id=a, to_id=b, relation=rel, src_spans=[span],
        ))
    if len(nodes) < 2 or not edges:
        return None
    return SceneGraph(nodes=nodes, edges=edges, revision=0)


# Common LLM relation choices that don't exist in the SeVim ontology
# but map cleanly to one that does.  Anything not in this map (and
# not in _ALLOWED_RELATIONS) falls back to ``connects``.
_RELATION_ALIAS = {
    "is": "instance_of",
    "is_a": "instance_of",
    "isa": "instance_of",
    "has": "contains",
    "has_a": "contains",
    "represents": "instance_of",
    "describes": "labels",
    "models": "approximately_equal",
    "approximates": "approximately_equal",
    "depends_on": "requires",
    "uses": "used_for",
    "used_by": "used_for",
    "computes": "measures",
    "captures": "measures",
    "controls": "attribute_of",
    "weights": "attribute_of",
    "transforms_into": "maps_to",
    "transforms": "maps_to",
    "leads_to": "causes",
    "produces": "causes",
    "precedes": "sequence",
    "follows": "sequence",
    "applies_to": "used_for",
    "subsumes": "contains",
    "is_part_of": "part_of",
    "consists_of": "contains",
    "made_of": "contains",
}


def _render_scene_graph(graph) -> str:
    """Run S3 → S4 → S5 on a hand-built SceneGraph and return SVG."""
    from sevim.s3_map import map_visual
    from sevim.s4_layout import layout
    from sevim.s5_render import render
    vg = map_visual(graph)
    pg = layout(vg)
    return render(pg)


def _gather(node: dict) -> list[dict]:
    out = [node]
    for c in node.get("children", []) or []:
        out.extend(_gather(c))
    return out


def _text_for(node: dict) -> str:
    """Pick the richest narration text for one node.  We concatenate
    ``story_paragraph`` and ``formula_explanation`` so the SeVim
    extractor sees both the section's narrative *and* the
    parameter-by-parameter walkthrough — that yields a fuller
    relation graph than either alone."""
    parts = []
    sp = (node.get("story_paragraph") or "").strip()
    if sp:
        parts.append(sp)
    fe = (node.get("formula_explanation") or "").strip()
    if fe:
        parts.append(fe)
    if not parts:
        gist = (node.get("gist") or "").strip()
        if gist:
            parts.append(gist)
    return "  ".join(parts)


def build(sidecar_path: str, *, force: bool = False,
          min_nodes: int = 1) -> int:
    if not os.path.isfile(sidecar_path):
        print(f"[sevim-diagrams] no sidecar at {sidecar_path}",
              file=sys.stderr)
        return 1
    payload = json.load(open(sidecar_path))
    root = payload.get("root") or {}
    if not root:
        print("[sevim-diagrams] empty root", file=sys.stderr)
        return 1
    out_path = sidecar_path.replace(".chapter_map.",
                                     ".sevim_diagrams.")

    existing: dict[str, str] = {}
    if os.path.isfile(out_path) and not force:
        try:
            existing = json.load(open(out_path))
        except Exception:
            existing = {}

    nodes = _gather(root)
    print(f"[sevim-diagrams] {len(nodes)} node(s) to consider",
          flush=True)
    diagrams: dict[str, str] = dict(existing)
    written = 0
    skipped_existing = 0
    skipped_thin = 0

    for i, n in enumerate(nodes, 1):
        nid = (n.get("nid") or "").strip()
        if not nid:
            continue
        if nid in existing and not force:
            skipped_existing += 1
            continue
        text = _text_for(n)
        if not text:
            continue
        # Build a tight context blob — title, equation label/latex,
        # and the narration text — so Qwen has the math to anchor
        # its triples on, not just the prose.
        title = (n.get("title") or "").strip()
        cf_label = (n.get("canonical_formula_label") or "").strip()
        cf_latex = (n.get("canonical_formula_latex") or "").strip()
        user_prompt_parts = []
        if title:
            user_prompt_parts.append(f"SECTION: {title}")
        if cf_label and cf_latex:
            user_prompt_parts.append(
                f"EQUATION ({cf_label}): {cf_latex}"
            )
        user_prompt_parts.append(f"NARRATION: {text}")
        user_prompt_parts.append(
            "Extract 3-7 nodes (mathematical objects from this "
            "section) and 3-8 edges (relations from the closed "
            "ontology) as JSON, exactly per the system prompt."
        )
        user = "\n\n".join(user_prompt_parts)
        triples = _call_llm_json(_EXTRACT_SYSTEM, user)
        if not isinstance(triples, dict):
            print(f"  [{i}/{len(nodes)}] {nid} :: LLM returned no JSON",
                  flush=True)
            skipped_thin += 1
            continue
        graph = _build_scene_graph(triples, src_label=nid)
        if graph is None:
            print(f"  [{i}/{len(nodes)}] {nid} :: thin "
                  f"(nodes={len(triples.get('nodes') or [])}, "
                  f"edges={len(triples.get('edges') or [])}); skipping",
                  flush=True)
            skipped_thin += 1
            continue
        try:
            svg = _render_scene_graph(graph)
        except Exception as e:
            print(f"  [{i}/{len(nodes)}] {nid} :: render error: {e}",
                  flush=True)
            skipped_thin += 1
            continue
        if not svg or "<svg" not in svg:
            print(f"  [{i}/{len(nodes)}] {nid} :: empty SVG; skipping",
                  flush=True)
            skipped_thin += 1
            continue
        diagrams[nid] = svg
        written += 1
        print(f"  [{i}/{len(nodes)}] {nid} :: nodes={len(graph.nodes)} "
              f"edges={len(graph.edges)}  svg={len(svg)} chars",
              flush=True)

    if not written:
        print(f"[sevim-diagrams] nothing new to write "
              f"(existing={skipped_existing}, thin={skipped_thin})",
              flush=True)
        # Still write the file if it doesn't exist yet so the
        # endpoint has something to read (even if empty).
        if not os.path.isfile(out_path):
            with open(out_path, "w") as f:
                json.dump({}, f)
        return 0

    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(diagrams, f, ensure_ascii=False)
    os.replace(tmp, out_path)
    print(f"[sevim-diagrams] wrote {written} new diagram(s) "
          f"({skipped_existing} cached, {skipped_thin} thin) "
          f"to {out_path}", flush=True)
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sidecar", help="path to chapter_map.<root>.json")
    ap.add_argument("--force", action="store_true",
                    help="rebuild every node even when cached")
    ap.add_argument("--min-nodes", type=int, default=1,
                    help="skip nodes whose pipeline yields fewer "
                         "than this many concept nodes (default 1)")
    args = ap.parse_args()
    return build(args.sidecar, force=args.force,
                 min_nodes=args.min_nodes)


if __name__ == "__main__":
    sys.exit(main())
