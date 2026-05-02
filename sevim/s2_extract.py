"""S2 Extract — semantic triple extraction from clause tokens.

Architecture: two cascaded extraction paths that share the same 12-relation
ontology, node-deduplication logic, and provenance tracking.

Path A — dependency-parse (preferred)
    Uses spaCy `en_core_web_sm` to parse each clause.  A verb-lemma dispatch
    table (VERB_RELATION_MAP, seeded from corpus clustering) maps the root verb
    to one of the 12 relations.  Handles arbitrary surface variation, including
    conjunct verbs, copular constructions, possessives, "X of Y" nominals, and
    relative-clause input/output patterns.  If the parse yields no triple, the
    regex cascade fires as a backstop for that clause.

Path B — regex cascade (fallback)
    Hand-authored compiled patterns covering the most common surface forms of
    each relation (e.g. "X causes Y", "X is a part of Y", "X requires Y").
    Used when spaCy is absent (encoder-off mode) or as a regression target.

Both paths converge on _ensure_node / _add_edge, which implement:
    - Stable node IDs derived from normalised labels ("n_gradient_descent")
    - Cosine-merge deduplication: near-duplicate labels are merged into an
      existing node when their embedding similarity exceeds MERGE_TAU (0.85)
      AND the lemma-proxy gate confirms they share a common word base-form
    - Provenance: every node and edge carries the SpanRef of the source clause

Public entry point: extract(tokens, graph=None) → SceneGraph
"""
from __future__ import annotations

import math
import re
from threading import Lock

from .ir import SceneEdge, SceneGraph, SceneNode, SpanRef, SpanToken
from .math_lex import (
    classify_math_label,
    find_latex_spans,
    latex_to_unicode,
    parse_matrix_literal,
)

MERGE_TAU = 0.85     # cosine threshold: candidates above this may be merged
HIGH_TAU  = 0.95     # cosine threshold: candidates above this merge without lemma check
MIN_LEMMA_LEN = 3    # minimum label length for the prefix-based lemma-proxy gate

_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_TRAIL_CONJ = re.compile(r"\s+(?:and|or|but|nor)\s*$", re.IGNORECASE)
_TRAIL_PUNCT = ".,;:!?"


# ---------- shared normalisation / node helpers -----------------------------

# Words that look plural but are not (or whose singular changes meaning).
_NOT_PLURAL = frozenset({
    'series', 'species', 'analysis', 'basis', 'axis', 'thesis', 'crisis',
    'hypothesis', 'diagnosis', 'emphasis', 'synthesis', 'corpus', 'calculus',
    'nexus', 'status', 'bonus', 'census', 'campus', 'focus', 'radius',
    'process', 'success', 'physics', 'mathematics', 'statistics', 'electronics',
    'dynamics', 'semantics', 'economics', 'ethics', 'logistics', 'robotics',
    'news', 'loss', 'class', 'mass', 'pass', 'access', 'address', 'express',
    'ress', 'stress', 'progress', 'process', 'excess', 'bias', 'alias',
    'canvas', 'atlas', 'gas', 'chaos', 'atlas',
    # Mass nouns used in technical writing — do NOT singularize
    'data', 'information', 'knowledge', 'feedback', 'software', 'hardware',
    'evidence', 'research', 'output', 'input',
})

_IRREGULAR_PLURAL: dict[str, str] = {
    'criteria': 'criterion', 'phenomena': 'phenomenon',
    'matrices': 'matrix', 'indices': 'index', 'vertices': 'vertex',
    'analyses': 'analysis', 'hypotheses': 'hypothesis', 'theses': 'thesis',
    'crises': 'crisis', 'bases': 'basis', 'axes': 'axis',
    # "data" is used as a mass noun in technical writing; do NOT singularize it.
}


def _singularize(word: str) -> str:
    """Return the singular form of an English word (best-effort)."""
    w = word.lower()
    if w in _NOT_PLURAL:
        return word
    if w in _IRREGULAR_PLURAL:
        return _IRREGULAR_PLURAL[w]
    # -ies → -y  (policies → policy, strategies → strategy)
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    # -sses / -xes / -zes / -ches / -shes → remove 'es'
    if (w.endswith('sses') or w.endswith('xes') or w.endswith('zes')
            or w.endswith('ches') or w.endswith('shes')):
        return w[:-2]
    # -s (but not -ss, and at least 4 chars)
    if w.endswith('s') and not w.endswith('ss') and len(w) > 3:
        return w[:-1]
    return word


def _lemmatize_label(label: str) -> str:
    """Singularize every content noun in a multi-word label.

    Works on the whole label so that both "control policies" and
    "control policy" normalise to "control policy".
    """
    words = label.split()
    result = []
    for i, word in enumerate(words):
        # Skip prepositions and determiners that aren't nouns
        lw = word.lower()
        if lw in ('of', 'for', 'in', 'on', 'at', 'by', 'to', 'with', 'from',
                  'and', 'or', 'the', 'a', 'an'):
            result.append(lw)
        else:
            result.append(_singularize(lw))
    return ' '.join(result)


def _normalize(label: str) -> str:
    label = label.strip().rstrip(_TRAIL_PUNCT)
    label = _ARTICLE.sub("", label)
    label = _lemmatize_label(label.strip())
    label = _TRAIL_CONJ.sub("", label)  # strip trailing "and", "or" etc.
    return label.strip()


def _is_stop(label: str) -> bool:
    return label.strip().lower() in _STOP_LABELS


def _nid(label: str) -> str:
    return "n_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _eid(frm: str, to: str, rel: str) -> str:
    return f"e_{rel}_{frm}_{to}"


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return 0.0 if na == 0 or nb == 0 else dot / math.sqrt(na * nb)


def _lemma_compatible(a: str, b: str) -> bool:
    la, lb = a.strip().lower(), b.strip().lower()
    if not la or not lb:
        return False
    if la == lb:
        return True
    shorter = la if len(la) <= len(lb) else lb
    longer = lb if shorter is la else la
    if len(shorter) < MIN_LEMMA_LEN:
        return False
    return longer.startswith(shorter)


def _ensure_node(
    g: SceneGraph,
    label: str,
    span: SpanRef,
    embedding: tuple[float, ...] = (),
) -> str:
    nid = _nid(label)
    for n in g.nodes:
        if n.id == nid:
            if span not in n.src_spans:
                n.src_spans.append(span)
            return nid
    if embedding:
        best_id: str | None = None
        best_sim = -1.0
        for n in g.nodes:
            if n.node_type != "entity" or not n.embedding:
                continue
            sim = _cosine(embedding, n.embedding)
            if sim < MERGE_TAU:
                continue
            if sim < HIGH_TAU and not _lemma_compatible(label, n.label):
                continue
            if sim > best_sim:
                best_id = n.id
                best_sim = sim
        if best_id is not None:
            for n in g.nodes:
                if n.id == best_id:
                    if span not in n.src_spans:
                        n.src_spans.append(span)
                    return best_id
    g.nodes.append(SceneNode(
        id=nid, label=label, node_type="entity",
        embedding=embedding, salience=0.5, src_spans=[span],
    ))
    return nid


def _add_edge(g: SceneGraph, frm: str, to: str, rel: str, span: SpanRef) -> None:
    eid = _eid(frm, to, rel)
    for e in g.edges:
        if e.id == eid:
            if span not in e.src_spans:
                e.src_spans.append(span)
            return
    g.edges.append(SceneEdge(
        id=eid, from_id=frm, to_id=to, relation=rel,  # type: ignore[arg-type]
        src_spans=[span],
    ))


# ---------- dep-parse extractor ---------------------------------------------

# Seeded from discovery/findings.md (corpus-clustered verb lemmas).
VERB_RELATION_MAP: dict[str, str] = {
    # causes
    "cause": "causes", "lead": "causes", "result": "causes", "trigger": "causes",
    "produce": "causes", "generate": "causes", "induce": "causes",
    "yield": "causes", "enable": "causes", "allow": "causes",
    "improve": "causes", "increase": "causes", "decrease": "causes",
    "maximise": "causes", "maximize": "causes",
    # used_for
    "use": "used_for", "utilize": "used_for", "employ": "used_for",
    "serve": "used_for", "apply": "used_for", "train": "used_for",
    "learn": "used_for", "estimate": "used_for", "approximate": "used_for",
    "extract": "used_for", "process": "used_for", "represent": "used_for",
    "select": "used_for", "sample": "used_for", "optimise": "used_for",
    "optimize": "used_for", "update": "used_for", "solve": "used_for",
    "evaluate": "used_for",
    # requires
    "require": "requires", "need": "requires", "depend": "requires",
    "necessitate": "requires", "assume": "requires",
    # reduces_to
    "reduce": "reduces_to", "simplify": "reduces_to", "minimise": "reduces_to",
    "minimize": "reduces_to", "collapse": "reduces_to", "approximate": "reduces_to",
    # measures
    "equal": "measures", "measure": "measures", "quantify": "measures",
    "compute": "measures", "define": "measures", "calculate": "measures",
    "estimate": "measures",
    # contains
    "contain": "contains", "enclose": "contains", "include": "contains",
    "encompass": "contains", "comprise": "contains", "consist": "contains",
    # measures (additional)
    "determine": "measures", "describe": "measures", "characterize": "measures",
    "specify": "measures", "indicate": "measures",
    # I/O verbs (input feeds X / X produces output)
    "receive": "requires", "accept": "requires",
    "output": "causes", "emit": "causes",
    # attribute_of
    "have": "attribute_of", "possess": "attribute_of",
    # sequence
    "follow": "sequence", "precede": "sequence",
    # British spelling variants
    "utilise": "used_for", "recognise": "measures",
    # similar_to
    "resemble": "similar_to", "mirror": "similar_to",
    "compare": "similar_to",
    # opposes
    "oppose": "opposes", "contradict": "opposes",
}

# Labels that are never valid standalone nodes — conjunctions, pronouns,
# articles. When the extractor would otherwise create a node from one of
# these tokens (typically from a fragmentary clause like "and requires
# memory"), the triple is dropped instead.
_STOP_LABELS = frozenset({
    # Conjunctions (usually from clause-splitting fragments we no longer
    # produce, but kept as defence-in-depth).
    "and", "but", "or", "then", "so", "because",
    # Pronouns — these appear when questions are fed through the extractor
    # ("what does it require?") and would otherwise become orphan nodes.
    "it", "this", "that", "they", "them", "these", "those",
    "he", "she", "we", "us", "you", "i",
    # Question pronouns (likewise from questions).
    "what", "who", "whom", "which", "where", "when", "why", "how",
    # Articles are already stripped by _normalize; do not list them here,
    # or single-character labels like "A" / "An" in test cases would be
    # dropped after case-folding.
})

# Copular predicates: "X is a <PRED> of Y" → relation is determined by PRED.
COP_PREDICATE_MAP: dict[str, str] = {
    "part": "part_of",
    "example": "instance_of",
    "instance": "instance_of",
    "kind": "instance_of",
    "type": "instance_of",
    "sort": "instance_of",
    "form": "instance_of",
    "similar": "similar_to",
    "like": "similar_to",
    "opposite": "opposes",
    "contrary": "opposes",
    # — math copular predicates —
    "element": "element_of",
    "member": "element_of",
    "subset": "subset_of",
    "superset": "contains",
    "perpendicular": "perpendicular",
    "parallel": "parallel",
    "tangent": "tangent_to",
    "congruent": "congruent",
    "isomorphic": "isomorphic_to",
    "homeomorphic": "isomorphic_to",
    "diffeomorphic": "isomorphic_to",
    "equivalent": "equals",
    "equal": "equals",
    "disjoint": "disjoint",
    # University: logical / categorical
    "adjoint": "adjoint_to",
    "left-adjoint": "adjoint_to",
    "right-adjoint": "adjoint_to",
    "natural": "natural_transformation",
}

# Math verbs that map to relations (added to VERB_RELATION_MAP via .update).
_MATH_VERBS: dict[str, str] = {
    # Geometric incidence
    "lie": "lies_on",        # "P lies on L"
    "pass": "lies_on",       # "L passes through P"  (pass through pobj)
    "intersect": "connects",
    "cross": "connects",
    "meet": "connects",
    # Functional / mappings
    "map": "maps_to",
    "send": "maps_to",
    "project": "maps_to",
    # Set / membership (verbal forms)
    "belong": "element_of",
    # Algebraic
    "satisfy": "equals",
    "evaluate": "equals",
    "simplify": "equals",
    # University: logical entailment / category theory
    "imply": "implies",
    "entail": "implies",
    "commute": "commutes",
    # Animation verbs — map to the new `transforms` relation, which the
    # renderer turns into an SMIL <animateTransform> element on the target.
    "rotate":    "transforms",
    "translate": "transforms",
    "scale":     "transforms",
    # "transform" itself is now an animation cue, not a maps_to alias.
    "transform": "transforms",
}
VERB_RELATION_MAP.update(_MATH_VERBS)


# ---------- animation parsing helpers ---------------------------------------
#
# Animation is opt-in: we only emit `meta["animate"]` when the source clause
# carries an explicit motion verb (rotate/translate/scale/transform) plus the
# numeric parameter "by N degrees" / "by N°" / "by N px" / "by Nx".
# Anything we cannot parse confidently is left as a plain `transforms` edge
# without an `animate` payload, so the renderer simply renders the static
# scene plus the connector — never an undefined animation.

_ANIM_DEG_RE = re.compile(
    r"by\s+(?P<n>\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)\b",
    re.IGNORECASE,
)
_ANIM_PX_RE = re.compile(
    r"by\s+(?P<n>\d+(?:\.\d+)?)\s*(?:px|pixels?)\b",
    re.IGNORECASE,
)
_ANIM_FACTOR_RE = re.compile(
    r"by\s+(?:a\s+factor\s+of\s+)?(?P<n>\d+(?:\.\d+)?)\s*(?:x|times)?\b",
    re.IGNORECASE,
)


def _parse_animation_meta(text: str, verb_lemma: str) -> dict | None:
    """Parse a clause like 'rotates X by 90 degrees' and return an `animate` meta.

    Returns None if no numeric magnitude can be extracted — in that case the
    caller still records the `transforms` edge but does not attach motion meta.
    """
    if not verb_lemma:
        return None
    verb = verb_lemma.lower()
    if verb in ("rotate", "transform"):
        m = _ANIM_DEG_RE.search(text)
        if not m:
            return None
        n = float(m.group("n"))
        return {
            "kind": "rotate", "from": 0.0, "to": n,
            "dur": "2s", "repeat": "indefinite", "around": "center",
        }
    if verb == "translate":
        m = _ANIM_PX_RE.search(text)
        if not m:
            return None
        n = float(m.group("n"))
        return {
            "kind": "translate", "from": 0.0, "to": n,
            "dur": "2s", "repeat": "indefinite",
        }
    if verb == "scale":
        m = _ANIM_FACTOR_RE.search(text)
        if not m:
            return None
        n = float(m.group("n"))
        return {
            "kind": "scale", "from": 1.0, "to": n,
            "dur": "2s", "repeat": "indefinite",
        }
    return None


def _attach_animate(g: SceneGraph, target_id: str, animate_meta: dict) -> None:
    """Attach `animate_meta` to the target node's meta dict in place."""
    if not animate_meta:
        return
    for n in g.nodes:
        if n.id == target_id:
            n.meta["animate"] = animate_meta
            return


_nlp = None
_nlp_lock = Lock()
_nlp_tried = False


def _get_nlp():
    global _nlp, _nlp_tried
    with _nlp_lock:
        if _nlp_tried:
            return _nlp
        _nlp_tried = True
        try:
            import spacy
            # Lemmatizer is load-bearing: the verb map keys on lemmas.
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            _nlp = None
        return _nlp


def dep_parse_available() -> bool:
    return _get_nlp() is not None


def _subtree_text(token) -> str:
    # Exclude determiners, punctuation, and the token's own determiner children
    # from the rendered label so "the body" → "body".
    parts = [t.text for t in token.subtree if t.pos_ not in ("PUNCT", "DET")]
    return " ".join(parts).strip()


def _head_label(token) -> str:
    """Label for a noun excluding poss-pronoun children, prep-of branches,
    and all subordinate clauses (relcl, acl, advcl) to keep labels short.

    "its internal state"  → "internal state"
    "internal state of the emulator" → "internal state"
    "behaviour which can be problematic" → "behaviour"
    """
    skip: set[int] = set()
    for child in token.children:
        if child.dep_ == "poss" and child.pos_ == "PRON":
            skip.update(t.i for t in child.subtree)
        elif child.dep_ == "prep" and child.text.lower() == "of":
            skip.update(t.i for t in child.subtree)
        elif child.dep_ in ("relcl", "acl", "advcl", "ccomp", "xcomp"):
            skip.update(t.i for t in child.subtree)
    parts = [
        t.text for t in token.subtree
        if t.i not in skip and t.pos_ not in ("PUNCT", "DET")
    ]
    return " ".join(parts).strip()


def _find_subject(verb):
    for c in verb.children:
        if c.dep_ in ("nsubj", "nsubjpass"):
            return _head_label(c)
    return None


def _find_object(verb):
    for c in verb.children:
        if c.dep_ in ("dobj", "attr", "acomp"):
            return _head_label(c)
    for c in verb.children:
        if c.dep_ == "prep":
            for gc in c.children:
                if gc.dep_ == "pobj":
                    return _head_label(gc)
    return None


def _handle_copular(root):
    """Return (relation, object) for 'X is <attr> (prep <pobj>)' patterns.
    Requires the parent caller to supply the subject via nsubj."""
    attr = next(
        (c for c in root.children if c.dep_ in ("attr", "acomp")),
        None,
    )
    if attr is None:
        return None, None
    # Agent-of-passive ('is used as/for/in') — handled by the passive branch.
    predicate = attr.lemma_.lower() if attr.lemma_ else attr.text.lower()
    if predicate in COP_PREDICATE_MAP:
        rel = COP_PREDICATE_MAP[predicate]
        prep = next((c for c in attr.children if c.dep_ == "prep"), None)
        if prep is not None:
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj is not None:
                return rel, _subtree_text(pobj)
        # "X is similar to Y" — sometimes parser makes "to" a direct child of attr.
        for c in attr.children:
            if c.dep_ == "prep":
                for gc in c.children:
                    if gc.dep_ == "pobj":
                        return rel, _subtree_text(gc)
        return None, None
    # Generic 'X is a Y' / 'X is an N' — instance_of when the predicate has
    # a determiner (so it is used nominally), even if spaCy tags it ADJ.
    if attr.pos_ in ("NOUN", "PROPN", "ADJ"):
        has_det = any(c.dep_ == "det" for c in attr.children)
        if has_det or attr.pos_ in ("NOUN", "PROPN"):
            return "instance_of", _subtree_text(attr)
    return None, None


_POSS_PRONOUNS = frozenset({"its", "their", "his", "her"})


def _resolve_possessor(
    sent, poss_token, g: SceneGraph, preexisting: set[str]
) -> str | None:
    """Return the graph nid of the most likely referent of a possessive pronoun.

    Priority order:
    1. Nearest preceding noun in the SAME sentence that already exists in the
       graph (search right-to-left, stop at the first match).
    2. Among *pre-existing* graph entities that appear as a ``contains`` source,
       pick the one added most recently (rightmost in g.nodes).
    3. The first pre-existing entity in g.nodes (most-prominent heuristic).
    """
    # 1. Same-sentence search.
    for t in reversed(list(sent)):
        if t.i >= poss_token.i:
            continue
        if t.pos_ not in ("NOUN", "PROPN"):
            continue
        if t.dep_ not in ("nsubj", "nsubjpass", "pobj", "dobj", "attr", "ROOT"):
            continue
        label = _normalize(_head_label(t))
        if not label or _is_stop(label):
            continue
        candidate = _nid(label)
        for n in g.nodes:
            if n.id == candidate:
                return candidate
    if not g.nodes:
        return None
    # 2. Prefer pre-existing entities that are already containers.
    existing_containers = {
        e.from_id for e in g.edges
        if e.relation == "contains" and e.from_id in preexisting
    }
    if existing_containers:
        for n in reversed(g.nodes):
            if n.id in existing_containers:
                return n.id
    # 3. First pre-existing entity.
    for n in g.nodes:
        if n.id in preexisting:
            return n.id
    return None


def _extract_possessives(
    sent, g: SceneGraph, span: SpanRef, emb, preexisting: set[str]
) -> None:
    """Emit part_of edges for possessive constructions.

    "its internal state" → part_of(internal_state, <resolved referent>)
    "the emulator's parameters" → part_of(parameters, emulator)
    """
    for token in sent:
        if token.dep_ != "poss":
            continue
        poss_text = token.text.lower()
        head = token.head  # the noun being possessed

        owned_text = _normalize(_head_label(head))
        if not owned_text or _is_stop(owned_text):
            continue

        if poss_text in _POSS_PRONOUNS:
            possessor_nid = _resolve_possessor(sent, token, g, preexisting)
            if possessor_nid is None:
                continue
            owned_id = _ensure_node(g, owned_text, span, emb(owned_text))
            _add_edge(g, owned_id, possessor_nid, "part_of", span)
        else:
            # Noun possessive: "emulator's X" — token is the possessor noun.
            possessor_text = _normalize(token.text)
            if not possessor_text or _is_stop(possessor_text):
                continue
            possessor_id = _ensure_node(g, possessor_text, span, emb(possessor_text))
            owned_id = _ensure_node(g, owned_text, span, emb(owned_text))
            _add_edge(g, owned_id, possessor_id, "part_of", span)


def _extract_noun_of(sent, g: SceneGraph, span: SpanRef, emb) -> None:
    """Emit part_of edges for 'X of Y' nominal patterns.

    "internal state of the emulator" → part_of(internal_state, emulator)

    Only fires when the preposition is 'of' and both X and Y are already
    non-stop nouns.  This avoids false positives on verbal 'of' phrases.
    """
    for token in sent:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue
        for child in token.children:
            if child.dep_ != "prep" or child.text.lower() != "of":
                continue
            for gc in child.children:
                if gc.dep_ != "pobj":
                    continue
                owned_text = _normalize(_head_label(token))
                possessor_text = _normalize(_subtree_text(gc))
                if (not owned_text or _is_stop(owned_text)
                        or not possessor_text or _is_stop(possessor_text)
                        or owned_text == possessor_text):
                    continue
                owned_id = _ensure_node(g, owned_text, span, emb(owned_text))
                possessor_id = _ensure_node(g, possessor_text, span,
                                            emb(possessor_text))
                _add_edge(g, owned_id, possessor_id, "part_of", span)


# Relational nouns that signal input/output roles in technical text.
_IO_INPUT_NOUNS  = frozenset({'input', 'inputs'})
_IO_OUTPUT_NOUNS = frozenset({'output', 'outputs', 'result', 'results', 'return', 'returns'})
_IO_NOUNS        = _IO_INPUT_NOUNS | _IO_OUTPUT_NOUNS


def _collect_conjuncts(token) -> list:
    """Return token plus all conj-linked siblings, recursively."""
    result = [token]
    for child in token.children:
        if child.dep_ == 'conj':
            result.extend(_collect_conjuncts(child))
    return result


_ENUM_WORDS = frozenset({'including', 'include', 'such'})


def _head_label_no_conj(token) -> str:
    """Like _head_label but also excludes conj-linked siblings from the subtree.

    Used for list items so "convolutional networks, multilayer perceptrons"
    doesn't create a node whose label contains all the other list members.
    """
    skip: set[int] = set()
    for child in token.children:
        if child.dep_ == 'poss' and child.pos_ == 'PRON':
            skip.update(t.i for t in child.subtree)
        elif child.dep_ == 'prep' and child.text.lower() == 'of':
            skip.update(t.i for t in child.subtree)
        elif child.dep_ == 'conj':
            skip.update(t.i for t in child.subtree)
    parts = [
        t.text for t in token.subtree
        if t.i not in skip and t.pos_ not in ('PUNCT', 'DET')
    ]
    return ' '.join(parts).strip()


def _extract_including_list(sent, g: SceneGraph, span: SpanRef, emb) -> None:
    """Extract enumeration lists: 'architectures, including X, Y and Z'.

    Emits  X instance_of <parent>  for each item in the list.
    Also handles "methods such as X, Y, Z" (pobj of "as" prep child of "such").
    """
    for token in sent:
        if token.lower_ not in _ENUM_WORDS:
            continue

        # Locate the pobj head of the list
        list_head = None
        if token.lower_ in ('including', 'include'):
            for ch in token.children:
                if ch.dep_ == 'pobj':
                    list_head = ch
                    break
        elif token.lower_ == 'such':
            for ch in token.children:
                if ch.lower_ == 'as' and ch.dep_ in ('prep', 'amod'):
                    for gc in ch.children:
                        if gc.dep_ == 'pobj':
                            list_head = gc
                            break
                    if list_head:
                        break

        if list_head is None:
            continue

        # Walk up to find the nearest NOUN parent (what is being enumerated).
        # Build the parent label EXCLUDING the "including..." subtree so it
        # doesn't end up as "neural network architecture including X Y Z".
        parent = token.head
        hops = 0
        while parent.pos_ not in ('NOUN', 'PROPN') and hops < 5:
            if parent.dep_ == 'ROOT':
                break
            parent = parent.head
            hops += 1
        if parent.pos_ not in ('NOUN', 'PROPN'):
            continue

        # Build parent label excluding the enum token's whole subtree
        enum_subtree_idx = {t.i for t in token.subtree}
        skip: set[int] = set()
        for child in parent.children:
            if child.dep_ == 'poss' and child.pos_ == 'PRON':
                skip.update(t.i for t in child.subtree)
            elif child.dep_ == 'prep' and child.text.lower() == 'of':
                skip.update(t.i for t in child.subtree)
        skip.update(enum_subtree_idx)
        parent_parts = [
            t.text for t in parent.subtree
            if t.i not in skip and t.pos_ not in ('PUNCT', 'DET')
        ]
        parent_label = _normalize(' '.join(parent_parts))
        if not parent_label or _is_stop(parent_label):
            continue
        parent_id = _ensure_node(g, parent_label, span, emb(parent_label))

        for item in _collect_conjuncts(list_head):
            item_label = _normalize(_head_label_no_conj(item))
            if not item_label or _is_stop(item_label) or item_label == parent_label:
                continue
            item_id = _ensure_node(g, item_label, span, emb(item_label))
            _add_edge(g, item_id, parent_id, 'instance_of', span)


def _extract_relcl_io(sent, g: SceneGraph, span: SpanRef, emb) -> None:
    """Extract input/output dependencies from relative clauses.

    Handles patterns like:
      "X whose input  is Y"  → X requires Y  (Y feeds into X)
      "X whose output is Y"  → X causes   Y  (X produces Y)
      "X takes Y as input"   → X requires Y
      "X produces/outputs Y" → X causes   Y  (covered by VERB_RELATION_MAP too)

    In spaCy, "network whose input is raw pixels" parses as:
      input  (nsubj of relcl-verb "is") ← whose (poss)
      relcl-verb "is" (dep_="relcl", head = antecedent "network")
      pixels (attr of "is")
    """
    for token in sent:
        lm = token.lemma_.lower()
        if lm not in _IO_NOUNS:
            continue

        # Must have "whose" as a possessive child → it's a relative clause IO noun
        has_whose = any(c.lower_ == 'whose' and c.dep_ == 'poss' for c in token.children)
        if not has_whose:
            continue

        # token should be nsubj of the relcl verb
        io_verb = token.head
        if io_verb.dep_ != 'relcl' or io_verb.lemma_ != 'be':
            continue

        # Antecedent = the noun the relative clause modifies
        antecedent = io_verb.head
        ant_label = _normalize(_head_label(antecedent))
        if not ant_label or _is_stop(ant_label):
            continue

        # Complement = what input/output IS (attr/acomp child of the copula)
        comp_token = None
        for child in io_verb.children:
            if child.dep_ in ('attr', 'acomp') and child != token:
                comp_token = child
                break
        if comp_token is None:
            continue

        # Build label; strip participial modifiers ("estimating future rewards")
        comp_parts = []
        for t in comp_token.subtree:
            if t.dep_ in ('relcl', 'advcl', 'acl') and t != comp_token:
                break
            if t.pos_ not in ('PUNCT', 'DET'):
                comp_parts.append(t.text)
        comp_label = _normalize(' '.join(comp_parts))
        if not comp_label or _is_stop(comp_label):
            continue

        ant_id  = _ensure_node(g, ant_label,  span, emb(ant_label))
        comp_id = _ensure_node(g, comp_label, span, emb(comp_label))

        if lm in _IO_INPUT_NOUNS:
            # comp → ant  (raw pixels feed INTO the network)
            _add_edge(g, comp_id, ant_id, 'requires', span)
        else:
            # ant → comp  (network produces value function)
            _add_edge(g, ant_id, comp_id, 'causes', span)


def _extract_from_sent(sent, g: SceneGraph, span: SpanRef, emb):
    # Snapshot which nodes exist before this sentence adds new ones.
    # Used by possessive resolution to avoid picking a freshly-minted node
    # as the referent of "its".
    preexisting: set[str] = {n.id for n in g.nodes}

    root = sent.root
    verbs = [root]
    for c in root.children:
        if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX"):
            verbs.append(c)
        # Also pick up adverbial/complement clauses: "X changes as Y learns Z"
        # spaCy sometimes mislabels a verb as NOUN at root level, but its
        # advcl children are correctly tagged VERB with full subject+object.
        elif c.dep_ in ("advcl", "ccomp", "xcomp") and c.pos_ in ("VERB", "AUX"):
            verbs.append(c)

    for verb in verbs:
        if verb.pos_ not in ("VERB", "AUX"):
            continue

        subj = _find_subject(verb)
        # Conjunct verbs often inherit the subject from the parent.
        if not subj and verb is not root:
            subj = _find_subject(root)
        if not subj:
            continue

        if verb.lemma_ == "be":
            rel, obj = _handle_copular(verb)
        else:
            rel = VERB_RELATION_MAP.get(verb.lemma_)
            obj = _find_object(verb) if rel else None

        if not rel or not obj:
            continue

        s_label = _normalize(subj)
        o_label = _normalize(obj)
        if not s_label or not o_label:
            continue
        if _is_stop(s_label) or _is_stop(o_label):
            continue

        s_id = _ensure_node(g, s_label, span, emb(s_label))
        o_id = _ensure_node(g, o_label, span, emb(o_label))
        if rel == "attribute_of":
            _add_edge(g, o_id, s_id, rel, span)
        else:
            _add_edge(g, s_id, o_id, rel, span)
        # Animation: when the verb is a motion verb, attach a SMIL animate
        # spec to the *target* (the thing being moved).
        if rel == "transforms":
            anim = _parse_animation_meta(sent.text, verb.lemma_)
            if anim is not None:
                _attach_animate(g, o_id, anim)

    # Structural extraction — runs regardless of whether a verbal triple fired.
    _extract_possessives(sent, g, span, emb, preexisting)
    _extract_noun_of(sent, g, span, emb)
    _extract_relcl_io(sent, g, span, emb)
    _extract_including_list(sent, g, span, emb)


def _try_animation_regex(text: str, g: SceneGraph, span: SpanRef, emb) -> bool:
    """Try animation-specific regexes (rotate / translate / scale).

    These run before the generic rule cascade because the surface form
    "rotates X by N degrees" would otherwise be claimed by `_M_SYM_*` or
    a generic copular pattern.  When matched, they create a `transforms`
    edge and attach the SMIL animate meta to the target node.

    Returns True iff a triple fired.
    """
    cases: list[tuple[re.Pattern[str], str, str]] = [
        (_M_ROTATE_AGENT,    "rotate",    "agent"),
        (_M_ROTATE_SELF,     "rotate",    "self"),
        (_M_TRANSLATE_AGENT, "translate", "agent"),
        (_M_TRANSLATE_SELF,  "translate", "self"),
        (_M_SCALE_AGENT,     "scale",     "agent"),
        (_M_SCALE_SELF,      "scale",     "self"),
    ]
    for rx, kind, mode in cases:
        m = rx.match(text)
        if not m:
            continue
        n = float(m.group("n"))
        o_label = _normalize(m.group("o"))
        if not o_label or _is_stop(o_label):
            continue
        o_id = _ensure_node(g, o_label, span, emb(o_label))
        if mode == "agent":
            s_label = _normalize(m.group("s"))
            if s_label and not _is_stop(s_label):
                s_id = _ensure_node(g, s_label, span, emb(s_label))
                _add_edge(g, s_id, o_id, "transforms", span)
        # Build animate meta from the parsed magnitude.
        if kind == "rotate":
            anim = {"kind": "rotate", "from": 0.0, "to": n,
                    "dur": "2s", "repeat": "indefinite", "around": "center"}
        elif kind == "translate":
            anim = {"kind": "translate", "from": 0.0, "to": n,
                    "dur": "2s", "repeat": "indefinite"}
        else:  # scale
            anim = {"kind": "scale", "from": 1.0, "to": n,
                    "dur": "2s", "repeat": "indefinite"}
        _attach_animate(g, o_id, anim)
        return True
    return False


def _regex_try_one(text: str, g: SceneGraph, span: SpanRef, emb) -> bool:
    """Run the regex cascade on a single clause. Returns True if a triple fired."""
    # Animation patterns get first refusal — they have a stronger lexical
    # signature ("rotates ... by N degrees") and we don't want generic copular
    # rules to claim them.
    if _try_animation_regex(text, g, span, emb):
        return True
    for rx, rel in _RULES:
        m = rx.match(text)
        if not m:
            continue
        s_label = _normalize(m.group("s"))
        o_label = _normalize(m.group("o"))
        if not s_label or not o_label:
            continue
        if _is_stop(s_label) or _is_stop(o_label):
            continue
        s_id = _ensure_node(g, s_label, span, emb(s_label))
        o_id = _ensure_node(g, o_label, span, emb(o_label))
        if rel == "attribute_of":
            _add_edge(g, o_id, s_id, rel, span)
        else:
            _add_edge(g, s_id, o_id, rel, span)
        return True
    return False


def _extract_math_literals(
    text: str, g: SceneGraph, span: SpanRef, emb,
) -> str:
    """Detect inline LaTeX and matrix literals; add nodes; return cleaned text.

    For each found LaTeX or matrix literal, creates a SceneNode with the
    appropriate primitive metadata (`equation_block` / `matrix_bracket`) and
    splices a short placeholder into the returned text so downstream
    extraction does not re-parse the math source as English.

    Node IDs are derived from the literal body so identical formulas dedupe
    across the document.
    """
    out_text = text
    # Inline LaTeX delimited by $…$, $$…$$, \(…\), \[…\].
    spans = find_latex_spans(text)
    # Splice from the right so earlier offsets stay valid.
    for start, end, body in reversed(spans):
        if not body.strip():
            continue
        mat = parse_matrix_literal(body)
        if mat is not None:
            _add_matrix_node(g, body, mat, span, emb)
            out_text = out_text[:start] + " matrix " + out_text[end:]
        else:
            _add_equation_node(g, body, span, emb)
            out_text = out_text[:start] + " equation " + out_text[end:]

    # Bare (non-LaTeX) matrix literal — at most one per clause for v1.
    bare_mat = parse_matrix_literal(out_text)
    if bare_mat is not None:
        _add_matrix_node(g, out_text, bare_mat, span, emb)

    return out_text


def _stable_short_id(prefix: str, body: str) -> str:
    """Deterministic short ID derived from *body* for math-literal nodes."""
    h = abs(hash(body)) % (10 ** 8)
    return f"n_{prefix}_{h:08d}"


def _add_equation_node(
    g: SceneGraph, latex: str, span: SpanRef, emb,
) -> str:
    """Create an `equation_block` node; return its ID."""
    unicode_form = latex_to_unicode(latex)
    label = unicode_form or latex.strip()
    if len(label) > 64:
        label = label[:61] + "…"
    nid = _stable_short_id("eq", latex.strip())
    for n in g.nodes:
        if n.id == nid:
            if span not in n.src_spans:
                n.src_spans.append(span)
            return nid
    g.nodes.append(SceneNode(
        id=nid, label=label, node_type="entity",
        embedding=emb(label), salience=0.5, src_spans=[span],
        meta={"latex": latex.strip(), "unicode": unicode_form,
              "kind": "equation_block"},
    ))
    return nid


def _add_matrix_node(
    g: SceneGraph, src: str, mat: dict, span: SpanRef, emb,
) -> str:
    """Create a `matrix_bracket` node; return its ID."""
    nid = _stable_short_id("mat", src.strip())
    for n in g.nodes:
        if n.id == nid:
            if span not in n.src_spans:
                n.src_spans.append(span)
            return nid
    label = f"matrix {mat['nrows']}×{mat['ncols']}"
    g.nodes.append(SceneNode(
        id=nid, label=label, node_type="entity",
        embedding=emb(label), salience=0.5, src_spans=[span],
        meta={**mat, "kind": "matrix_bracket"},
    ))
    return nid


def _extract_dep(tokens: list[SpanToken], g: SceneGraph) -> SceneGraph:
    from .embed import encode

    nlp = _get_nlp()
    assert nlp is not None

    label_emb: dict[str, tuple[float, ...]] = {}

    def emb(label: str) -> tuple[float, ...]:
        if label not in label_emb:
            label_emb[label] = encode(label)
        return label_emb[label]

    for tok in tokens:
        text = tok.text.rstrip(_TRAIL_PUNCT).strip()
        if not text:
            continue
        span = SpanRef(tok.start, tok.end, tok.utterance_id)
        # Math literals first: produces equation_block / matrix_bracket nodes
        # and returns the text with literals replaced by short placeholders.
        text = _extract_math_literals(text, g, span, emb)
        before_edges = len(g.edges)
        doc = nlp(text)
        for sent in doc.sents:
            _extract_from_sent(sent, g, span, emb)
        # If the parse yielded nothing, try the regex cascade as a backstop.
        if len(g.edges) == before_edges:
            _regex_try_one(text, g, span, emb)

    g.revision += 1
    return g


# ---------- regex fallback (original cascade) -------------------------------

_PART_OF = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+(?:a\s+)?part\s+of|belongs\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_SIMILAR = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+similar\s+to|is\s+like|resembles)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_OPPOSES = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:opposes?|contradicts?|is\s+(?:the\s+)?opposite\s+of|is\s+contrary\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_INSTANCE_EXAMPLE = re.compile(
    r"^\s*(?P<s>.+?)\s+is\s+an?\s+example\s+of\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_CONTAINS = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:contains?|encloses?|encompasses?)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_USED_FOR = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+used\s+(?:as|for|in)|used\s+(?:as|for|in)|uses?)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_REQUIRES = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:requires?|needs?|depends?\s+on)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_REDUCES_TO = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:reduces?\s+to|simplifies?\s+to|collapses?\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_MEASURES = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:equals?|is\s+divisible\s+by|is\s+divided\s+by|divided\s+by)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_CAUSE = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:causes?|leads?\s+to|results?\s+in|triggers?)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_SEQ = re.compile(
    r"^\s*(?P<s>.+?)\s+then\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_ATTR = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:has|have)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_INSTANCE_ISA = re.compile(
    r"^\s*(?P<s>.+?)\s+is\s+an?\s+(?P<o>\w+(?:\s+\w+)?)\s*$",
    re.IGNORECASE,
)

# Input/output sentence patterns (regex fallback)
_INPUT_OF = re.compile(
    r"^\s*input\s+(?:to|of)\s+(?P<o>.+?)\s+is\s+(?P<s>.+?)\s*$",
    re.IGNORECASE,
)
_OUTPUT_OF = re.compile(
    r"^\s*output\s+(?:of|from)\s+(?P<s>.+?)\s+is\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_TAKES_AS_INPUT = re.compile(
    r"^\s*(?P<s>.+?)\s+takes?\s+(?P<o>.+?)\s+as\s+inputs?\s*$",
    re.IGNORECASE,
)

# ---------- math-specific regex patterns ----------
# Order: these run BEFORE the concept-diagram rules so "X is parallel to Y"
# is recognised as `parallel`, not as a generic copular `instance_of`.

_M_LIES_ON = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:lies?\s+on|sits?\s+on|"
    r"is\s+(?:located\s+)?on|passes?\s+through)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_PERP = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+perpendicular\s+to|"
    r"perpendicular\s+to|"
    r"is\s+orthogonal\s+to|orthogonal\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_PARALLEL = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+parallel\s+to|parallel\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_TANGENT = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+tangent\s+to|tangent\s+to|touches)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_CONGRUENT = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+congruent\s+to|congruent\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_ISOMORPHIC = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+isomorphic\s+to|isomorphic\s+to|"
    r"is\s+homeomorphic\s+to|homeomorphic\s+to|"
    r"is\s+diffeomorphic\s+to|diffeomorphic\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_APPROX = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+approximately\s+(?:equal\s+to)?|"
    r"approximately\s+equals?|≈)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_EQUALS = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:equals?|is\s+equal\s+to|=)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_ELEMENT_OF = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+(?:a|an)\s+(?:element|member)\s+of|"
    r"is\s+in|belongs\s+to|∈)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_SUBSET_OF = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+a\s+(?:proper\s+)?subset\s+of|"
    r"is\s+contained\s+in|⊆|⊂|"
    r"is\s+included\s+in)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_DISJOINT = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+disjoint\s+from|"
    r"are\s+disjoint(?:\s+from)?|do(?:es)?\s+not\s+intersect)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_MAPS_TO = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:maps?\s+to|sends?\s+\S+\s+to|→|↦|"
    r"is\s+a\s+(?:function|map|mapping|morphism|homomorphism)\s+from\s+\S+\s+to|"
    r"is\s+(?:a\s+)?(?:function|map)\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_BETWEEN = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+between|lies\s+between)\s+"
    r"(?P<o>.+?)\s+(?:and|,)\s+\S+.*$",
    re.IGNORECASE,
)

# Unicode-symbol relation patterns (for narration containing typeset symbols).
_M_SYM_PERP   = re.compile(r"^\s*(?P<s>.+?)\s*⊥\s*(?P<o>.+?)\s*$")
_M_SYM_PAR    = re.compile(r"^\s*(?P<s>.+?)\s*∥\s*(?P<o>.+?)\s*$")
_M_SYM_IN     = re.compile(r"^\s*(?P<s>.+?)\s*∈\s*(?P<o>.+?)\s*$")
_M_SYM_NOT_IN = re.compile(r"^\s*(?P<s>.+?)\s*∉\s*(?P<o>.+?)\s*$")
_M_SYM_SUB    = re.compile(r"^\s*(?P<s>.+?)\s*(?:⊆|⊂)\s*(?P<o>.+?)\s*$")
_M_SYM_SUP    = re.compile(r"^\s*(?P<s>.+?)\s*(?:⊇|⊃)\s*(?P<o>.+?)\s*$")
_M_SYM_CONG   = re.compile(r"^\s*(?P<s>.+?)\s*≅\s*(?P<o>.+?)\s*$")
_M_SYM_APPROX = re.compile(r"^\s*(?P<s>.+?)\s*≈\s*(?P<o>.+?)\s*$")
_M_SYM_NEQ    = re.compile(r"^\s*(?P<s>.+?)\s*≠\s*(?P<o>.+?)\s*$")
_M_SYM_MAPSTO = re.compile(r"^\s*(?P<s>.+?)\s*(?:↦|→)\s*(?P<o>.+?)\s*$")

# — university: logical / category-theory relation patterns —

_M_IMPLIES = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:implies?|entails?|⇒|=>|"
    r"therefore|hence|thus)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_ADJOINT = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:is\s+)?(?:left\s+adjoint\s+to|"
    r"right\s+adjoint\s+to|adjoint\s+to|⊣)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_NAT_TRANS = re.compile(
    # ⇒ is reserved for `implies` (logic) — for nat. transformations the user
    # must explicitly say "natural transformation" to disambiguate.
    r"^\s*(?P<s>.+?)\s+(?:is\s+)?(?:a\s+)?(?:natural\s+transformation\s+(?:from\s+\S+\s+)?to|"
    r"transforms\s+naturally\s+to)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_COMMUTES = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:commutes?\s+with|equals\s+(?:the\s+composition\s+of\s+)?"
    r"|the\s+square\s+(?:of\s+)?(?P<sq>.+?)\s+commutes)\s+(?P<o>.+?)\s*$",
    re.IGNORECASE,
)
_M_SYM_IMPLIES   = re.compile(r"^\s*(?P<s>.+?)\s*(?:⇒|⟹|=>)\s*(?P<o>.+?)\s*$")
_M_SYM_NAT_TRANS = re.compile(r"^\s*(?P<s>.+?)\s*⇒\s*(?P<o>.+?)\s*$")  # context-dependent
_M_SYM_ADJOINT   = re.compile(r"^\s*(?P<s>.+?)\s*⊣\s*(?P<o>.+?)\s*$")

# — animation: motion verbs with explicit numeric magnitude —
#
# Three accepted surface forms.  In all of them the *target* (object being
# moved) is captured as group "o"; the *agent* (rotation matrix, etc.) is
# group "s" and may be a synthetic placeholder for the subject-less variant.
#
#   "X rotates by 90 degrees"                       → s=X, o=X, agent=None
#   "rotates X by 90 degrees"                        → s=X, o=X, agent=None
#   "the rotation matrix R rotates X by 90 degrees"  → s=R, o=X
_M_ROTATE_AGENT = re.compile(
    r"^\s*(?P<s>.+?)\s+(?:rotates?|spins?|turns?)\s+(?P<o>.+?)\s+"
    r"by\s+(?P<n>\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)\b.*$",
    re.IGNORECASE,
)
_M_ROTATE_SELF = re.compile(
    r"^\s*(?P<o>.+?)\s+(?:rotates?|spins?|turns?)\s+"
    r"by\s+(?P<n>\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)\b.*$",
    re.IGNORECASE,
)
_M_TRANSLATE_AGENT = re.compile(
    r"^\s*(?P<s>.+?)\s+translates?\s+(?P<o>.+?)\s+"
    r"by\s+(?P<n>\d+(?:\.\d+)?)\s*(?:px|pixels?)\b.*$",
    re.IGNORECASE,
)
_M_TRANSLATE_SELF = re.compile(
    r"^\s*(?P<o>.+?)\s+translates?\s+"
    r"by\s+(?P<n>\d+(?:\.\d+)?)\s*(?:px|pixels?)\b.*$",
    re.IGNORECASE,
)
_M_SCALE_AGENT = re.compile(
    r"^\s*(?P<s>.+?)\s+scales?\s+(?P<o>.+?)\s+"
    r"by\s+(?:a\s+factor\s+of\s+)?(?P<n>\d+(?:\.\d+)?)\s*(?:x|times)?\b.*$",
    re.IGNORECASE,
)
_M_SCALE_SELF = re.compile(
    r"^\s*(?P<o>.+?)\s+scales?\s+"
    r"by\s+(?:a\s+factor\s+of\s+)?(?P<n>\d+(?:\.\d+)?)\s*(?:x|times)?\b.*$",
    re.IGNORECASE,
)

_RULES: list[tuple[re.Pattern[str], str]] = [
    # — university (must run before generic math) —
    (_M_ADJOINT,    "adjoint_to"),
    (_M_NAT_TRANS,  "natural_transformation"),
    (_M_COMMUTES,   "commutes"),
    (_M_IMPLIES,    "implies"),
    (_M_SYM_IMPLIES, "implies"),
    (_M_SYM_ADJOINT, "adjoint_to"),
    # — math (specific phrasings; must run first) —
    (_M_LIES_ON,    "lies_on"),
    (_M_PERP,       "perpendicular"),
    (_M_PARALLEL,   "parallel"),
    (_M_TANGENT,    "tangent_to"),
    (_M_CONGRUENT,  "congruent"),
    (_M_ISOMORPHIC, "isomorphic_to"),
    (_M_APPROX,     "approximately_equal"),
    (_M_ELEMENT_OF, "element_of"),
    (_M_SUBSET_OF,  "subset_of"),
    (_M_DISJOINT,   "disjoint"),
    (_M_MAPS_TO,    "maps_to"),
    (_M_BETWEEN,    "between"),
    (_M_EQUALS,     "equals"),
    # — Unicode-symbol relation forms —
    (_M_SYM_PERP,   "perpendicular"),
    (_M_SYM_PAR,    "parallel"),
    (_M_SYM_IN,     "element_of"),
    (_M_SYM_NOT_IN, "disjoint"),
    (_M_SYM_SUB,    "subset_of"),
    (_M_SYM_SUP,    "contains"),
    (_M_SYM_CONG,   "congruent"),
    (_M_SYM_APPROX, "approximately_equal"),
    (_M_SYM_NEQ,    "opposes"),
    (_M_SYM_MAPSTO, "maps_to"),
    # — original concept-diagram rules —
    (_INPUT_OF, "requires"),
    (_OUTPUT_OF, "causes"),
    (_TAKES_AS_INPUT, "requires"),
    (_PART_OF, "part_of"),
    (_SIMILAR, "similar_to"),
    (_OPPOSES, "opposes"),
    (_INSTANCE_EXAMPLE, "instance_of"),
    (_REDUCES_TO, "reduces_to"),
    (_MEASURES, "measures"),
    (_USED_FOR, "used_for"),
    (_REQUIRES, "requires"),
    (_CONTAINS, "contains"),
    (_CAUSE, "causes"),
    (_SEQ, "sequence"),
    (_ATTR, "attribute_of"),
    (_INSTANCE_ISA, "instance_of"),
]


def _extract_regex(tokens: list[SpanToken], g: SceneGraph) -> SceneGraph:
    from .embed import encode

    label_emb: dict[str, tuple[float, ...]] = {}

    def emb(label: str) -> tuple[float, ...]:
        if label not in label_emb:
            label_emb[label] = encode(label)
        return label_emb[label]

    for tok in tokens:
        text = tok.text.rstrip(_TRAIL_PUNCT).strip()
        if not text:
            continue
        span = SpanRef(tok.start, tok.end, tok.utterance_id)
        # Math literals first; placeholders go through the regex cascade.
        text = _extract_math_literals(text, g, span, emb)
        # Animation patterns claim the clause first if present.
        if _try_animation_regex(text, g, span, emb):
            continue
        for rx, rel in _RULES:
            m = rx.match(text)
            if not m:
                continue
            s_label = _normalize(m.group("s"))
            o_label = _normalize(m.group("o"))
            if not s_label or not o_label:
                continue
            if _is_stop(s_label) or _is_stop(o_label):
                continue
            s_id = _ensure_node(g, s_label, span, emb(s_label))
            o_id = _ensure_node(g, o_label, span, emb(o_label))
            if rel == "attribute_of":
                _add_edge(g, o_id, s_id, rel, span)
            else:
                _add_edge(g, s_id, o_id, rel, span)
            break
    g.revision += 1
    return g


# ---------- public entry point ----------------------------------------------

def extract(tokens: list[SpanToken], graph: SceneGraph | None = None) -> SceneGraph:
    """Extract semantic triples from *tokens* and add them to *graph*.

    Dispatches to the dependency-parse extractor (Path A) when spaCy is
    available, otherwise falls back to the regex cascade (Path B).  Both
    paths use the same 12-relation ontology and cosine-merge deduplication.

    Pass an existing SceneGraph to extend it incrementally (streaming mode).
    Pass None to start a fresh graph.
    """
    g = graph or SceneGraph()
    if dep_parse_available():
        return _extract_dep(tokens, g)
    return _extract_regex(tokens, g)
