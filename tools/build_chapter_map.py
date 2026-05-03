"""Build a hierarchical *chapter map* — the artefact behind the
zoom-in narration mode.

Per BookNode in a chapter's subtree we record:

    gist                  one sentence summarising the node
    canonical_formula_id  the single formula in the node's subtree that
                          serves as its anchor (max-incoming-references
                          in the math graph; deterministic tie-break)
    role_in_parent        one sentence: "how this node serves its
                          parent's gist" — used by the narrator's tour
                          pass so each child is framed by the whole.

Reuses the offline ``concepts.json`` sidecar for ``L0_gist`` whenever it
exists (the concept-layer builder is the source of truth for per-node
gists).  Calls the local Qwen2.5-14B endpoint only for the
``role_in_parent`` sentences, which the concepts builder never produced.

Output: ``books/<stem>.chapter_map.<root_nid>.json``.

Usage::

    python -m tools.build_chapter_map books/ESLII.json --root b/ch5

Idempotent: re-running merges new ``role_in_parent`` strings into an
existing on-disk map without overwriting hand-edited entries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional


# Local-only — same vLLM endpoint the concept-layer builder uses.
LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"


@dataclass
class MapNode:
    """One node in the chapter map.  ``children`` is recursive."""
    nid: str
    title: str
    kind: str
    number: str = ""
    depth: int = 0
    gist: str = ""
    canonical_formula_id: str = ""
    canonical_formula_latex: str = ""
    canonical_formula_label: str = ""
    role_in_parent: str = ""
    # The narrated paragraph for this node.  Generated with rolling
    # context across the chapter so consecutive paragraphs flow into
    # each other rather than reading as a list of bullet summaries.
    # 3-5 sentences of plain-English prose; math notation is named by
    # its citation label and explained in words first.  The runtime
    # planner splits this into sentence-clauses so each clause is
    # anchored at the node's home_nid for the treemap highlight.
    story_paragraph: str = ""
    children: list["MapNode"] = field(default_factory=list)


_SYSTEM_PROMPT = """You are an expert mathematics teacher writing one-sentence
explanations for a learner who already heard the parent section's punch-line.
Your job is to say IN ONE SENTENCE how the child section serves the parent's
big idea — what role it plays in the parent's story.

Hard rules:
  * One sentence, plain English, no LaTeX.  Maximum 25 words.
  * Reference the parent's idea explicitly, not the child's title.
  * Begin with a verb that names the relationship: "extends", "specialises",
    "applies", "proves", "motivates", "computes", "filters", "derives",
    "compares", "introduces", "anchors", "demonstrates", "regularises", …
  * Do not just rephrase the child's title.  The reader knows the title.

Return JSON: {"role_in_parent": "..."}"""


# A single big call generates the whole-chapter essay, so the
# narration reads as one continuous story instead of bullet summaries.
_STORY_SYSTEM_PROMPT = """You are an essayist explaining one chapter of a
mathematics textbook to a curious adult who has not done college math in
years.  Output ONE continuous narrative.  Hard rules — every one of these
matters:

1. NARRATIVE FLOW.  Read your output as one essay, not a table of
   contents.  FORBIDDEN OPENERS, no exceptions: "Section X.Y", "Subsection
   X.Y.Z", "In this section", "In this subsection", "Section five point
   X starts with", "We turn to", "Now we look at", "This section",
   "Moving on to", "The chapter introduces".  Every paragraph must open
   with a SUBSTANTIVE transition — a connective phrase ("But fitting a
   straight line through wiggly data leaves real patterns uncovered, …"),
   a callback to the previous paragraph's claim ("That balance shows up
   again, …"), or a question ("How smooth is too smooth?").  Pretend
   the reader doesn't know there are sections at all — they are
   listening to one continuous story.

2. PLAIN ENGLISH FIRST.  Every concept is introduced with a one-line
   intuition or concrete metaphor BEFORE any symbol or formula is named.
   Imagine the reader is curious but rusty — they remember what a graph
   is, not what an inner product is.  Never use a technical term without
   one short translating phrase next to it.

3. MATH IS A CONSTANT, INLINE PRESENCE — EVERY CITED EQUATION MUST
   APPEAR BY ITS LABEL.  This is the single hardest rule and the one
   most authors fail at.  The cited-equation list I give you is
   exhaustive.  Every label in it MUST be NAMED IN THE PROSE by its
   spelled-out citation ("equation five point nine", "equation five
   point forty-two") together with one short clause that says in
   plain English what the equation does.  Concrete pattern that works:
   "<intuition>; this is what equation five point nine says: <plain-
   English meaning>."  Multiple equations in one section get multiple
   such clauses, chained: "equation five point eleven … equation five
   point twelve …"  If the reader skips every formula, the prose
   alone still tells the whole story; if the reader keeps the formulas
   but skips the prose, they have raw symbols with no meaning.  Both
   readings have to work.  FAILURE STATE: an equation in the input
   list that does not appear by its spelled-out label in your prose
   is a hard failure of the task.

4. FIGURES MUST ALSO BE NAMED BY LABEL.  When the chapter has cited
   figures, every figure label in the figures list I give you MUST
   appear in the section's prose by its spelled-out citation
   ("figure five point one", "figure eleven point two") together with
   one short clause saying what the figure shows.  Concrete pattern:
   "<intuition>; figure eleven point two shows <one-line description
   of what's drawn>."  Place the figure-mention clause inside the
   prose of the section that owns the figure (the figure list tells
   you which nid each figure belongs to).  FAILURE STATE: a figure in
   the input list that does not appear by its spelled-out label in
   the prose of its owning section is a hard failure of the task.

5. NO RAW LATEX.  Variables get spoken names ("the smoothing parameter
   lambda" not "λ", "the sum from i equals 1 to N" not "Σᵢ").  Citation
   labels are spelled out ("equation five point nine", not "(5.9)";
   "figure five point three", not "Fig. 5.3").

6. HONOR THE STRUCTURE.  Output a JSON map
   ``{"paragraphs": {nid: text, ...}, "covered_equations": [labels...],
   "covered_figures": [labels...]}`` where every input section's nid
   appears as a key, and the two ``covered_*`` lists name every
   equation / figure label you actually mentioned in the prose (not
   just the ones I gave you).  Each paragraph is 3-6 sentences;
   longer when the section owns many cited equations or figures."""


def _call_llm(system: str, user: str, *,
              max_tokens: int = 200,
              temperature: float = 0.4,
              retries: int = 2) -> Optional[dict]:
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read())
            content = (raw["choices"][0]["message"]
                       .get("content") or "").strip()
            if not content:
                return None
            return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    print(f"  [llm] failed after {retries + 1} tries: {last_err}",
          file=sys.stderr)
    return None


_CITE_RE = __import__("re").compile(r"\((\d+\.\d+)\)")


def _pick_canonical_formula(node, math_graph, *, by_label=None):
    """Pick THE one formula that anchors this node.

    Three stages — each stage falls back to the next:

    A. **Body-text attribution.**  The math graph collapses many cited
       formulas to one or two home_nids per chapter, so per-section
       home_nid attribution is sparse.  Scan the section's
       ``body_text`` for ``(N.M)`` cite markers (the form the book
       itself uses).  Try each cite in order; if any of them resolves
       in the graph, return that formula.

    B. **Numeric-floor fallback.**  When body cites point at equations
       the math-graph extractor missed (its coverage is partial), pick
       the LARGEST graph-resolvable label that is still ``≤`` the
       smallest cite this section mentions.  That yields a formula
       that comes from earlier in the same chapter and is therefore
       the closest plausible canonical for the section.

    C. **Subtree home_nid attribution** (legacy).  Last resort: pick
       the most-incoming-references formula whose home_nid equals or
       descends from this node's nid.

    ``by_label`` is an optional pre-built ``"5.42" -> Formula`` index
    so callers running over a chapter can build it once and pass it in.
    """
    if math_graph is None:
        return None
    if by_label is None:
        by_label = _build_label_index(math_graph)

    # Stage A: cited equations from the body, in order.
    body = (getattr(node, "body_text", "") or "")
    cites_in_body: list[str] = []
    if body:
        cites_in_body = [m.group(1) for m in _CITE_RE.finditer(body)]
        for cite in cites_in_body:
            f = by_label.get(cite)
            if f and (f.latex or "").strip():
                return f

    # Stage B: the largest label in the graph that is ≤ the smallest
    # cite mentioned in the body.  Same chapter, earliest plausible
    # ancestor formula.
    if cites_in_body:
        def _key(s: str):
            try:
                return tuple(int(x) for x in s.split("."))
            except Exception:
                return (10**9,)
        smallest = min(cites_in_body, key=_key)
        smallest_t = _key(smallest)
        best = None
        best_t = None
        for lab, f in by_label.items():
            t = _key(lab)
            # same chapter prefix
            if not lab.startswith(smallest.split(".")[0] + "."):
                continue
            if t > smallest_t:
                continue
            if (f.latex or "").strip() and (best is None or t > best_t):
                best, best_t = f, t
        if best is not None:
            return best

    # Stage C: subtree home_nid attribution.
    nid_prefix = node.nid + "/"
    candidates = []
    for f in math_graph.formulas.values():
        if not f.home_nid:
            continue
        if f.home_nid != node.nid and not f.home_nid.startswith(nid_prefix):
            continue
        candidates.append(f)
    if not candidates:
        return None
    def _score(f):
        n = 0
        for e in math_graph.in_edges(f.id):
            if e.type in ("references", "derived_from"):
                n += 1
        return n
    candidates.sort(key=lambda f: (-_score(f),
                                    -len(f.latex or ""),
                                    f.id))
    return candidates[0]


def _build_label_index(math_graph) -> dict:
    """``"5.42" -> Formula`` for every cited formula in the graph."""
    by_label: dict = {}
    for f in math_graph.formulas.values():
        for lab in (f.cite_labels or []):
            norm = (lab.replace("Equation ", "")
                    .replace("(", "").replace(")", "").strip())
            if norm and norm not in by_label and (f.latex or "").strip():
                by_label[norm] = f
    return by_label


def _walk(node, *, depth, max_depth,
          parent_gist: str,
          concepts: dict,
          math_graph,
          existing_roles: dict,
          parent_formula: Optional[tuple] = None,
          label_index: Optional[dict] = None) -> Optional[MapNode]:
    """Build a MapNode for *node* recursively.  Returns None for nodes
    that should be skipped (front matter, exercise leaves, etc.).

    ``parent_formula`` is ``(id, latex, label)`` from the parent so we
    can inherit when this node's own body has no cite the graph
    resolves; ensures every cell in a chapter map has a formula to
    show.
    """
    SKIP = {"bibliography", "index", "glossary",
            "exercise", "exercises", "bibliographic_notes"}
    if node.kind in SKIP:
        return None
    if depth > max_depth:
        return None

    entry = concepts.get("by_home_nid", {}).get(node.nid, {})
    gist = (entry.get("L0_gist") or "").strip()

    # Canonical formula via the math graph.  Inherit from the parent
    # when no own attribution is found, so every cell in the chapter
    # map carries one — math should be a constant presence.
    f = _pick_canonical_formula(node, math_graph, by_label=label_index)
    if f is not None:
        canonical_id = f.id
        canonical_latex = (f.latex or "")
        canonical_label = (f.cite_labels[0] if f.cite_labels else "")
    elif parent_formula is not None:
        canonical_id, canonical_latex, canonical_label = parent_formula
    else:
        canonical_id = ""
        canonical_latex = ""
        canonical_label = ""

    # role_in_parent — re-use cached if present, else LLM call.
    role = existing_roles.get(node.nid, "").strip()
    if not role and depth > 0 and parent_gist and gist:
        title = (node.title or "").strip()
        kind = node.kind.replace("_", " ")
        user = (
            f"Parent gist:\n{parent_gist}\n\n"
            f"Child {kind} title: {title}\n"
            f"Child gist: {gist}\n\n"
            "Write one sentence in JSON describing how this child serves "
            "the parent's gist."
        )
        out = _call_llm(_SYSTEM_PROMPT, user)
        if out and isinstance(out.get("role_in_parent"), str):
            role = out["role_in_parent"].strip()
            print(f"    role[{node.nid}] = {role[:80]}", flush=True)

    children: list[MapNode] = []
    own_formula = ((canonical_id, canonical_latex, canonical_label)
                   if canonical_id else parent_formula)
    for c in node.children:
        child_map = _walk(
            c,
            depth=depth + 1,
            max_depth=max_depth,
            parent_gist=gist or parent_gist,
            concepts=concepts,
            math_graph=math_graph,
            existing_roles=existing_roles,
            parent_formula=own_formula,
            label_index=label_index,
        )
        if child_map is not None:
            children.append(child_map)

    return MapNode(
        nid=node.nid,
        title=(node.title or "").strip(),
        kind=node.kind,
        number=(node.number or "").strip(),
        depth=depth,
        gist=gist,
        canonical_formula_id=canonical_id,
        canonical_formula_latex=canonical_latex,
        canonical_formula_label=canonical_label,
        role_in_parent=role,
        children=children,
    )


def _collect_chapter_figures(chapter_root_nid: str,
                              figures_index: dict) -> list[dict]:
    """Every cited figure under the chapter, as
    ``[{"label": "Figure 11.2", "home_nid": "b/ch11/s11_3",
        "caption": "Schematic ..."}, ...]`` deduped on label and
    ordered by numeric label suffix.  The story builder walks this
    list and the LLM is required to name each label in the prose of
    its owning section."""
    out: list[dict] = []
    seen: set[str] = set()
    if not figures_index or not chapter_root_nid:
        return out
    by_nid = figures_index.get("by_nid", {}) or {}
    prefix = chapter_root_nid + "/"
    for home, entries in by_nid.items():
        if home != chapter_root_nid and not home.startswith(prefix):
            continue
        for entry in (entries or []):
            label = (entry.get("label") or "").strip()
            if not label or label in seen:
                continue
            seen.add(label)
            cap = (entry.get("caption") or "").strip().replace("\n", " ")
            cap = re.sub(r"\s+", " ", cap)[:160]
            out.append({
                "label": label,
                "home_nid": home,
                "caption": cap,
            })
    def _sort_key(e):
        m = re.search(r"(\d+)(?:\.(\d+))?", e["label"])
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2) or 0))
    out.sort(key=_sort_key)
    return out


def _collect_chapter_equations(chapter_root_nid: str,
                                math_graph) -> list[dict]:
    """Every cited equation under the chapter, as
    ``[{"label": "Equation 5.9", "latex": "..."}, ...]`` deduped on
    label and ordered by numeric label suffix.  The narrative builder
    walks this list and the ``covered_equations`` field of the LLM
    response is checked against it so we know which equations the
    essay actually named."""
    out: list[dict] = []
    seen: set[str] = set()
    if math_graph is None or not chapter_root_nid:
        return out
    prefix = chapter_root_nid + "/"
    for f in math_graph.formulas.values():
        if not f.home_nid:
            continue
        if (f.home_nid != chapter_root_nid
                and not f.home_nid.startswith(prefix)):
            continue
        for lab in (f.cite_labels or []):
            lab = lab.strip()
            if not lab or lab in seen:
                continue
            seen.add(lab)
            out.append({"label": lab, "latex": (f.latex or "").strip()})

    def _sort_key(d: dict):
        # "Equation 5.42" → (5, 42); "5.42" → (5, 42); "Eq A.1" → (∞, …)
        s = d["label"].replace("Equation ", "").strip()
        parts = s.split(".")
        try:
            return tuple(int(p) for p in parts)
        except Exception:
            return (10**9,)
    out.sort(key=_sort_key)
    return out


def _build_chapter_story(tree, *, math_graph,
                          figures_index: dict,
                          existing_paragraphs: dict) -> tuple[dict, list[str]]:
    """Single big LLM call that produces the whole-chapter narrative.

    Returns ``(paragraphs_by_nid, missing_equation_labels)``.  Reuses
    cached paragraphs from ``existing_paragraphs`` if the LLM call is
    skipped or fails; the result is monotone — never wipes a previously-
    generated paragraph just because this run couldn't produce one.
    """
    # Build the section list (in DFS order, depth-bounded by the tree).
    sections: list[dict] = []
    def _walk(node):
        sections.append({
            "nid": node.nid,
            "title": node.title,
            "kind": node.kind,
            "number": node.number,
            "depth": node.depth,
            "gist": node.gist,
        })
        for c in node.children:
            _walk(c)
    _walk(tree)

    eqs = _collect_chapter_equations(tree.nid, math_graph)
    figs = _collect_chapter_figures(tree.nid, figures_index)

    # Skip the call entirely when every section already has a cached
    # paragraph — keeps re-builds cheap.
    all_cached = (
        existing_paragraphs and
        all(s["nid"] in existing_paragraphs for s in sections)
    )
    if all_cached:
        print(f"[chapter-map] using cached story_paragraph for "
              f"{len(sections)} sections; skipping the chapter-story call",
              flush=True)
        return existing_paragraphs, []

    def _build_user_prompt(sections_subset, *, total_sections):
        section_lines = []
        for s in sections_subset:
            gist = s["gist"] or "(no gist)"
            kind = s["kind"].replace("_", " ")
            num = s["number"] or ""
            section_lines.append(
                f"  {s['nid']}  ({kind} {num})  {s['title']}: {gist}"
            )
        eq_lines = []
        for e in eqs:
            latex_short = (e["latex"][:120] + "…") if len(e["latex"]) > 120 \
                          else e["latex"]
            eq_lines.append(f"  {e['label']}: {latex_short}")
        fig_lines = []
        for f in figs:
            cap = f["caption"] or "(no caption)"
            fig_lines.append(
                f"  {f['label']}  (owns: {f['home_nid']}): {cap}"
            )
        fig_block = ""
        if fig_lines:
            fig_block = (
                "\n\nCITED FIGURES (every one whose owning nid is in "
                "this batch must be named in the prose of its owning "
                "section, with one short clause saying what the figure "
                "shows):\n" + "\n".join(fig_lines)
            )
        partial_note = ""
        if len(sections_subset) < total_sections:
            partial_note = (
                f" — this is a partial batch of {len(sections_subset)} of "
                f"{total_sections} chapter sections; produce paragraphs "
                f"only for the nids listed below"
            )
        return (
            f"CHAPTER: {tree.title}\n\n"
            f"PUNCH-LINE TO HONOR THROUGHOUT:\n  {tree.gist}\n\n"
            f"SECTIONS (in narrative order{partial_note}):\n"
            + "\n".join(section_lines) + "\n\n"
            f"CITED EQUATIONS (every one whose owning nid is in this "
            f"batch must be named in the prose, with its meaning "
            f"explained in plain English right there):\n"
            + "\n".join(eq_lines) + fig_block + "\n\n"
            f"Write the essay as JSON: "
            f'{{"paragraphs": {{"<nid>": "..."}}, '
            f'"covered_equations": ["Equation N.M", ...], '
            f'"covered_figures": ["Figure N.M", ...]}}.'
        )

    def _call_for_subset(sections_subset, *, max_tokens):
        prompt = _build_user_prompt(
            sections_subset, total_sections=len(sections),
        )
        return _call_llm(_STORY_SYSTEM_PROMPT, prompt,
                         max_tokens=max_tokens, temperature=0.5)

    def _call_chunked(sections_subset, *, max_tokens, depth=0):
        """Try the call; on failure (None / no paragraphs / 400) split
        the section list in half and recurse.  Returns the merged
        ``out`` dict or ``None`` if every chunk failed."""
        if not sections_subset:
            return None
        out = _call_for_subset(sections_subset, max_tokens=max_tokens)
        good = (
            isinstance(out, dict)
            and isinstance(out.get("paragraphs"), dict)
            and out.get("paragraphs")
        )
        if good:
            return out
        if len(sections_subset) <= 1 or depth >= 4:
            print(f"  [chapter-story] chunk of {len(sections_subset)} "
                  f"sections failed at depth={depth}; giving up on chunk",
                  flush=True)
            return None
        mid = len(sections_subset) // 2
        print(f"  [chapter-story] chunk of {len(sections_subset)} sections "
              f"failed; retrying as {mid}+{len(sections_subset)-mid}",
              flush=True)
        a = _call_chunked(sections_subset[:mid],
                          max_tokens=max(800, max_tokens // 2),
                          depth=depth + 1)
        b = _call_chunked(sections_subset[mid:],
                          max_tokens=max(800, max_tokens // 2),
                          depth=depth + 1)
        merged_paragraphs: dict = {}
        merged_eq: list = []
        merged_fig: list = []
        for sub in (a, b):
            if not isinstance(sub, dict):
                continue
            p = sub.get("paragraphs") or {}
            if isinstance(p, dict):
                merged_paragraphs.update(p)
            merged_eq.extend(sub.get("covered_equations") or [])
            merged_fig.extend(sub.get("covered_figures") or [])
        if not merged_paragraphs:
            return None
        return {
            "paragraphs": merged_paragraphs,
            "covered_equations": merged_eq,
            "covered_figures": merged_fig,
        }

    # The vLLM context for Qwen2.5-14B-AWQ is 8192.  The system prompt
    # is ~600 tokens, the user prompt with all cited equations + section
    # gists is ~2500 tokens; we leave the rest for the response.  Long
    # chapters (Ch.14 / Ch.18 in ESLII) overflow even that budget on
    # the single-shot call — ``_call_chunked`` halves the section list
    # on failure and recurses.  But empirically the LLM also TRUNCATES
    # its output mid-chapter even when no error fires (it gives back a
    # well-formed JSON with paragraphs for the first 6–8 sections and
    # silently drops the rest).  So we PROACTIVELY split any chapter
    # of more than ~12 sections into batches up front and merge.
    BATCH_MAX = 12
    if len(sections) > BATCH_MAX:
        merged_paragraphs: dict = {}
        merged_eq: list = []
        merged_fig: list = []
        n_batches = (len(sections) + BATCH_MAX - 1) // BATCH_MAX
        size = (len(sections) + n_batches - 1) // n_batches
        print(f"[chapter-map] {len(sections)} sections > {BATCH_MAX} — "
              f"splitting into {n_batches} batches of ≈{size}",
              flush=True)
        for i in range(0, len(sections), size):
            batch = sections[i:i + size]
            sub = _call_chunked(batch, max_tokens=3200)
            if isinstance(sub, dict):
                p = sub.get("paragraphs") or {}
                if isinstance(p, dict):
                    merged_paragraphs.update(p)
                merged_eq.extend(sub.get("covered_equations") or [])
                merged_fig.extend(sub.get("covered_figures") or [])
        out = {
            "paragraphs": merged_paragraphs,
            "covered_equations": merged_eq,
            "covered_figures": merged_fig,
        } if merged_paragraphs else None
    else:
        out = _call_chunked(sections, max_tokens=4200)

    # Fill-missing retry: the LLM tends to drop "boring" sections
    # (intros, bibliographic notes, exercises) even when explicitly
    # asked to cover all nids.  Find any section that still lacks a
    # paragraph (in either ``out`` OR ``existing_paragraphs``) and
    # ask for them in a smaller, focused call.
    if isinstance(out, dict):
        already_covered = set(
            (out.get("paragraphs") or {}).keys()
        ) | set((existing_paragraphs or {}).keys())
        missing_sections = [s for s in sections
                            if s["nid"] not in already_covered]
        if missing_sections:
            print(f"[chapter-map] retrying {len(missing_sections)} "
                  f"sections the LLM skipped on the first pass",
                  flush=True)
            # Up to 2 retry rounds, each batch sized to BATCH_MAX.
            for _round in range(2):
                if not missing_sections:
                    break
                still_missing: list = []
                for i in range(0, len(missing_sections), BATCH_MAX):
                    batch = missing_sections[i:i + BATCH_MAX]
                    sub = _call_chunked(batch, max_tokens=2400)
                    if not isinstance(sub, dict):
                        still_missing.extend(batch)
                        continue
                    p = sub.get("paragraphs") or {}
                    if isinstance(p, dict):
                        out["paragraphs"].update(p)
                        out["covered_equations"] = (
                            (out.get("covered_equations") or [])
                            + (sub.get("covered_equations") or [])
                        )
                        out["covered_figures"] = (
                            (out.get("covered_figures") or [])
                            + (sub.get("covered_figures") or [])
                        )
                    for s in batch:
                        if s["nid"] not in p:
                            still_missing.append(s)
                missing_sections = still_missing
    if not isinstance(out, dict):
        print("  [chapter-story] all chunks failed; "
              "keeping any existing paragraphs", flush=True)
        return existing_paragraphs, [e["label"] for e in eqs]
    paragraphs = out.get("paragraphs") or {}
    if not isinstance(paragraphs, dict) or not paragraphs:
        print("  [chapter-story] LLM JSON had no paragraphs; "
              "keeping cached", flush=True)
        return existing_paragraphs, [e["label"] for e in eqs]

    # Strip the section-self-reference boilerplate the LLM keeps
    # putting in despite explicit instructions.  Patterns like
    #   "as introduced in section five point one"
    #   ", covered in section five point two,"
    #   ", discussed in subsection five point five point one"
    # all collapse to nothing.  After stripping we still have to
    # fix any "double comma", "comma at start" and " ." artefacts.
    import re as _re
    _STRIP_PATTERNS = [
        _re.compile(
            r",?\s*(?:as\s+)?(?:introduced|covered|discussed|detailed|"
            r"described|explained|explored|presented|outlined|examined|"
            r"investigated|delved\s+into|seen)\s+in\s+(?:section|subsection)"
            r"\s+\w+(?:\s+point\s+\w+)*",
            _re.IGNORECASE,
        ),
        _re.compile(
            r"^\s*(?:Section|Subsection|Chapter)\s+\w+(?:\s+point\s+\w+)*\s+"
            r"(?:introduces|describes|covers|discusses|presents|"
            r"explains|delves\s+into|details|outlines|examines)\s+",
            _re.IGNORECASE,
        ),
        _re.compile(
            r"\bIn\s+(?:section|subsection)\s+\w+(?:\s+point\s+\w+)*\s*,?\s*",
            _re.IGNORECASE,
        ),
    ]
    def _clean(p: str) -> str:
        for pat in _STRIP_PATTERNS:
            p = pat.sub("", p)
        # fix doubled punctuation + leading punctuation
        p = _re.sub(r"\s*,\s*,", ",", p)
        p = _re.sub(r"\s*\.\s*\.", ".", p)
        p = _re.sub(r"\s+([.,;:])", r"\1", p)
        p = _re.sub(r"^[\s,;:.\-]+", "", p)
        p = _re.sub(r"\s{2,}", " ", p).strip()
        # capitalise first letter if we lopped off the opener.
        if p and p[0].islower():
            p = p[0].upper() + p[1:]
        return p

    # Fill missing nids from existing cache so we never go backwards.
    merged = dict(existing_paragraphs or {})
    for nid, txt in paragraphs.items():
        if isinstance(txt, str) and txt.strip():
            merged[nid] = _clean(txt.strip())

    covered_eq = set((out.get("covered_equations") or []))
    expected_eq = {e["label"] for e in eqs}
    missing_eq = sorted(expected_eq - covered_eq)
    if missing_eq:
        print(f"  [chapter-story] LLM did not reference {len(missing_eq)} "
              f"of {len(expected_eq)} cited equations: "
              f"{', '.join(missing_eq[:8])}"
              f"{'…' if len(missing_eq) > 8 else ''}",
              flush=True)

    covered_fig = set((out.get("covered_figures") or []))
    expected_fig = {f["label"] for f in figs}
    missing_fig = sorted(expected_fig - covered_fig)
    if missing_fig:
        print(f"  [chapter-story] LLM did not reference {len(missing_fig)} "
              f"of {len(expected_fig)} cited figures: "
              f"{', '.join(missing_fig[:8])}"
              f"{'…' if len(missing_fig) > 8 else ''}",
              flush=True)
    return merged, missing_eq


def _flatten_existing_paragraphs(loaded: dict) -> dict:
    out: dict[str, str] = {}
    def _visit(n: dict):
        if not isinstance(n, dict):
            return
        nid = n.get("nid")
        sp = n.get("story_paragraph") or ""
        if nid and sp:
            out[nid] = sp
        for c in n.get("children", []) or []:
            _visit(c)
    if loaded and "root" in loaded:
        _visit(loaded["root"])
    return out


def _attach_paragraphs(tree, paragraphs: dict) -> None:
    tree.story_paragraph = paragraphs.get(tree.nid, "") or tree.story_paragraph
    for c in tree.children:
        _attach_paragraphs(c, paragraphs)


def _flatten_existing_roles(loaded: dict) -> dict:
    """Pull every role_in_parent already present in *loaded* into a
    flat dict keyed by nid so we don't re-call the LLM on rebuild."""
    roles: dict[str, str] = {}
    def _visit(n: dict):
        if not isinstance(n, dict):
            return
        nid = n.get("nid")
        role = n.get("role_in_parent") or ""
        if nid and role:
            roles[nid] = role
        for c in n.get("children", []) or []:
            _visit(c)
    if loaded and "root" in loaded:
        _visit(loaded["root"])
    return roles


def build(book_path: str, root_nid: str, out_path: str, *,
          max_depth: int = 3,
          concepts_path: Optional[str] = None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from book.corpus import load_corpus
    from sevim.math_graph import MathGraph, graph_path_for_book

    book = load_corpus(book_path)
    root = book.find(root_nid)
    if root is None:
        print(f"[chapter-map] unknown root_nid: {root_nid}", file=sys.stderr)
        return 1

    if concepts_path is None:
        stem, _ = os.path.splitext(book_path)
        concepts_path = stem + ".concepts.json"
    concepts: dict = {}
    if os.path.isfile(concepts_path):
        concepts = json.load(open(concepts_path))
        n = len(concepts.get("by_home_nid", {}))
        print(f"[chapter-map] loaded concepts.json: {n} entries", flush=True)
    # Figures sidecar (optional but enables figure-mention enforcement
    # in the chapter-wide narrative).
    figures_index: dict = {}
    fig_stem, _ = os.path.splitext(book_path)
    fig_path = fig_stem + ".figures.json"
    if os.path.isfile(fig_path):
        try:
            figures_index = json.load(open(fig_path))
            n_nids = len(figures_index.get("by_nid", {}) or {})
            print(f"[chapter-map] loaded figures.json: "
                  f"{n_nids} owning sections", flush=True)
        except Exception as e:
            print(f"[chapter-map] figures.json load failed: {e}",
                  flush=True)
            figures_index = {}
    else:
        print(f"[chapter-map] no figures.json at {fig_path} — "
              "story_paragraphs will not enforce figure mentions",
              flush=True)
    if not concepts:
        print(f"[chapter-map] no concepts.json at {concepts_path} — "
              "gists will be empty (fix: build_concept_layer first)",
              flush=True)

    gpath = graph_path_for_book(book_path)
    math_graph = (MathGraph.load(gpath, book_id=book.title or book_path)
                  if os.path.isfile(gpath) else None)

    loaded = {}
    if os.path.isfile(out_path):
        try:
            loaded = json.load(open(out_path))
        except Exception:
            loaded = {}
    existing_roles = _flatten_existing_roles(loaded)
    existing_paragraphs = _flatten_existing_paragraphs(loaded)
    if existing_roles:
        print(f"[chapter-map] re-using {len(existing_roles)} cached "
              f"role_in_parent strings", flush=True)
    if existing_paragraphs:
        print(f"[chapter-map] re-using {len(existing_paragraphs)} cached "
              f"story_paragraph entries", flush=True)

    print(f"[chapter-map] walking {root_nid} (max_depth={max_depth})…",
          flush=True)
    label_index = (_build_label_index(math_graph)
                   if math_graph is not None else {})
    tree = _walk(
        root,
        depth=0,
        max_depth=max_depth,
        parent_gist="",
        concepts=concepts,
        math_graph=math_graph,
        existing_roles=existing_roles,
        parent_formula=None,
        label_index=label_index,
    )
    if tree is None:
        print(f"[chapter-map] root was skipped — nothing to write",
              file=sys.stderr)
        return 1

    # Single big LLM call for the whole-chapter essay.  Reuses the
    # cached paragraphs when every section already has one; otherwise
    # asks Qwen for the full narrative in one shot, with the list of
    # cited equations included so each one is named in the prose.
    print(f"[chapter-map] generating whole-chapter narrative…", flush=True)
    paragraphs, missing = _build_chapter_story(
        tree,
        math_graph=math_graph,
        figures_index=figures_index,
        existing_paragraphs=existing_paragraphs,
    )
    _attach_paragraphs(tree, paragraphs)
    n_with_paragraph = sum(
        1 for v in paragraphs.values() if (v or "").strip()
    )
    print(f"[chapter-map] story_paragraph populated for "
          f"{n_with_paragraph} nodes", flush=True)

    payload = {
        "schema_version": 2,
        "book": book.title or "",
        "book_path": book_path,
        "root_nid": root_nid,
        "model": LLM_MODEL,
        "max_depth": max_depth,
        "missing_equations": missing,
        "root": _to_dict(tree),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
    n_nodes = _count(tree)
    print(f"[chapter-map] wrote {out_path} — {n_nodes} nodes", flush=True)
    return 0


def _to_dict(tree: MapNode) -> dict:
    return {
        "nid": tree.nid,
        "title": tree.title,
        "kind": tree.kind,
        "number": tree.number,
        "depth": tree.depth,
        "gist": tree.gist,
        "canonical_formula_id": tree.canonical_formula_id,
        "canonical_formula_latex": tree.canonical_formula_latex,
        "canonical_formula_label": tree.canonical_formula_label,
        "role_in_parent": tree.role_in_parent,
        "story_paragraph": tree.story_paragraph,
        "children": [_to_dict(c) for c in tree.children],
    }


def _count(tree: MapNode) -> int:
    return 1 + sum(_count(c) for c in tree.children)


def chapter_map_path_for_root(book_path: str, root_nid: str) -> str:
    """Where the sidecar lives.  ``b/ch5`` → ``…chapter_map.b_ch5.json``."""
    stem, _ = os.path.splitext(book_path)
    safe = root_nid.replace("/", "_")
    return f"{stem}.chapter_map.{safe}.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("book_path")
    ap.add_argument("--root", required=True,
                    help="Root nid (e.g. b/ch5)")
    ap.add_argument("--out", default=None,
                    help="Output path (default: derived from book + root)")
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()
    out = args.out or chapter_map_path_for_root(args.book_path, args.root)
    return build(args.book_path, args.root, out, max_depth=args.max_depth)


if __name__ == "__main__":
    raise SystemExit(main())
