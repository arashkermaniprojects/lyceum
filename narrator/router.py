"""Question router — classify a user utterance into a teaching intent.

Rules-first classifier so the high-traffic intents (overview,
chapter X, follow-up, control) never round-trip through the LLM.
Falls back to ``topic_qa`` when nothing else matches — that path
runs the existing BM25 + dense retrieval.

Intents
-------
  * ``book_overview``    — "what is this book about", "summarise the book"
  * ``chapter_overview`` — "explain chapter 2", "what is chapter 5 about"
  * ``section_overview`` — "explain section 5.8", "what is §5.8.1"
  * ``follow_up``        — "tell me more", "go deeper", "explain that"
  * ``control``          — "pause", "stop", "next", "go on", "skip"
  * ``topic_qa``         — everything else (default)

Each classification returns a ``RoutedIntent`` with the resolved
target nid (when applicable) so downstream planners can act
without re-parsing the utterance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from book.ir import Book


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@dataclass
class RoutedIntent:
    intent: str                       # one of the constants below
    raw: str = ""                     # original utterance
    target_nid: str = ""              # for chapter / section overviews
    topic: str = ""                   # for topic_qa / follow_up
    control_action: str = ""          # for control intent
    depth: str = ""                   # "short" | "deep" | "" (default)
    notes: dict = field(default_factory=dict)


INTENT_BOOK_OVERVIEW = "book_overview"
INTENT_CHAPTER_OVERVIEW = "chapter_overview"
INTENT_SECTION_OVERVIEW = "section_overview"
INTENT_FOLLOW_UP = "follow_up"
INTENT_CONTROL = "control"
INTENT_TOPIC_QA = "topic_qa"
INTENT_RECAP = "recap"
INTENT_RESHOW = "reshow"
INTENT_XREF_EXPLORE = "xref_explore"
INTENT_DEPENDENCIES = "dependencies"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "what is this book about", "summarise the book", "tell me about this book"
_BOOK_OVERVIEW_RE = re.compile(
    r"\b(?:"
    r"(?:what(?:'?s| is)|tell me).{0,15}\bthis\s+book\b"
    r"|(?:summari[sz]e|overview\s+of)\s+(?:the\s+)?book"
    r"|book\s+(?:overview|summary)"
    r"|what\s+is\s+the\s+book\s+about"
    r")\b",
    re.I,
)

# "explain chapter 2", "what is chapter 5 about", "chapter 3 please"
_CHAPTER_RE = re.compile(
    r"\bchapter\s+(\d+)\b",
    re.I,
)

# "section 5.8", "§5.8", "5.8.1"
_SECTION_RE = re.compile(
    r"\bsection\s+(\d+(?:\.\d+){0,2})\b"
    r"|\b§\s*(\d+(?:\.\d+){0,2})\b"
    r"|(?<![\w.])(\d+\.\d+(?:\.\d+)?)(?![\w.])",
    re.I,
)

# "tell me more", "go deeper", "explain that", "more about that", "more"
_FOLLOW_UP_RE = re.compile(
    r"\b(?:"
    r"tell me more|more about (?:that|this|it)|"
    r"go deeper|deeper|elaborat(?:e|ing)|"
    r"explain (?:that|this|it|more)|"
    r"continue|carry on|keep going|go on|"
    r"(?:say|tell)\s+more"
    r")\b",
    re.I,
)

# Pure follow-up indicators we accept on their own (1-2 word turns).
_FOLLOW_UP_SOLO = frozenset({
    "more", "deeper", "continue", "elaborate",
    "go on", "carry on", "keep going",
})

# Control intents.  Map utterance -> action.
_CONTROL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpause\b", re.I), "pause"),
    (re.compile(r"\b(?:resume|un\s*pause|continue\s+reading)\b", re.I),
     "resume"),
    (re.compile(r"\b(?:stop|cancel|that's enough|enough)\b", re.I), "stop"),
    (re.compile(r"\bskip(?:\s+ahead)?\b", re.I), "skip"),
    (re.compile(r"\b(?:louder|speak\s+up)\b", re.I), "louder"),
    (re.compile(r"\b(?:quieter|softer)\b", re.I), "quieter"),
    (re.compile(r"\b(?:slow(?:er)?\s+down|slow(?:er)?)\b", re.I), "slow"),
    (re.compile(r"\b(?:speed\s+up|faster)\b", re.I), "fast"),
    # Memory controls — flush the SessionKnowledge so the next tangent
    # treats every concept as fresh.  Triggered by "forget", "reset",
    # "start fresh", "clear what we covered".
    (re.compile(
        r"\b(?:forget(?:\s+(?:that|what\s+we\s+(?:covered|said)|everything))?"
        r"|start\s+fresh|reset(?:\s+(?:memory|context))?|"
        r"clear\s+(?:what\s+we\s+covered|memory|context))\b",
        re.I,
    ), "forget"),
]


# "what have we covered", "what did you cover", "summarize what we discussed"
_RECAP_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:have\s+we|did\s+(?:you|we))\s+"
    r"(?:cover(?:ed)?|discuss(?:ed)?|gone\s+over)"
    r"|recap(?:\s+(?:what|this|so\s+far))?"
    r"|(?:summari[sz]e|summary\s+of)\s+(?:what\s+we|our|this\s+session|so\s+far)"
    r"|where\s+are\s+we\s+(?:so\s+far|now)"
    r")\b",
    re.I,
)


# "see also", "what else references this", "related sections",
# "where else is this used"
_XREF_EXPLORE_RE = re.compile(
    r"\b(?:"
    r"see\s+also"
    r"|what\s+else\s+(?:references?\s+this|cites?\s+this|uses?\s+this)"
    r"|related\s+(?:section|chapter|content|material)s?"
    r"|where\s+(?:else\s+)?is\s+this\s+(?:used|cited|referenced)"
    r"|citation(?:s)?\s+(?:graph|neighbor(?:hood)?|context)"
    r"|cross[\s-]?references?"
    r")\b",
    re.I,
)

# "what do I need to know first", "prerequisites", "what does X depend on",
# "background for X", "before X what", "what comes before X"
_DEPENDENCIES_RE = re.compile(
    r"\b(?:"
    r"prerequisite(?:s)?"
    r"|what\s+(?:do\s+i|should\s+i)\s+(?:need\s+to\s+)?know\s+(?:first|before)"
    r"|what\s+(?:does\s+this|do\s+i)\s+need\s+to\s+understand"
    r"|background\s+(?:for|on)\s+\S+"
    r"|what\s+(?:does\s+this\s+depend\s+on"
    r"|depends?\s+on)"
    r"|before\s+(?:i|we)\s+(?:learn|study|read)"
    r"|what\s+comes\s+before"
    r")\b",
    re.I,
)


# "show me X again", "explain X again", "again", "repeat that", "once more"
_RESHOW_RE = re.compile(
    r"\b(?:"
    r"(?:show\s+me|explain|tell\s+me\s+about)\s+.{1,80}?\s+again\b"
    r"|repeat(?:\s+that|\s+it)?"
    r"|(?:once|one)\s+more(?:\s+time)?"
    r"|say\s+that\s+again"
    r")\b",
    re.I,
)
# Bare "again" gets reshow only when it's the entire utterance — too
# many follow-ups end with "again" otherwise.
_RESHOW_BARE_RE = re.compile(
    r"^\s*again\s*[!?.]?\s*$", re.I,
)

# Depth modifiers — orthogonal to intent.
_DEPTH_SHORT_RE = re.compile(
    r"\b(?:short(?:ly)?|brief(?:ly)?|in\s+a\s+(?:sentence|line)|tl;?dr)\b",
    re.I,
)
_DEPTH_DEEP_RE = re.compile(
    r"\b(?:deep(?:ly|er)?|in\s+detail|thorough(?:ly)?|"
    r"step\s+by\s+step|long(?:er)?)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def route(
    utterance: str,
    *,
    book: Optional[Book] = None,
    last_focus_nid: str = "",
    last_focus_topic: str = "",
) -> RoutedIntent:
    """Classify *utterance* into a :class:`RoutedIntent`.

    ``last_focus_nid`` / ``last_focus_topic`` come from the session's
    dialogue history: when the user says "tell me more", the resolver
    treats this as a follow-up to the last topic.
    """
    raw = (utterance or "").strip()
    if not raw:
        return RoutedIntent(intent=INTENT_TOPIC_QA, raw=raw)

    depth = ""
    if _DEPTH_SHORT_RE.search(raw):
        depth = "short"
    elif _DEPTH_DEEP_RE.search(raw):
        depth = "deep"

    # 1. Control — short-circuit, never an LLM call.
    short = raw.lower().strip(" .?!")
    for pat, action in _CONTROL_PATTERNS:
        if pat.search(raw):
            return RoutedIntent(
                intent=INTENT_CONTROL, raw=raw,
                control_action=action, depth=depth,
            )

    # 1b. Recap — "what have we covered" / "summarize what we discussed".
    if _RECAP_RE.search(raw):
        return RoutedIntent(intent=INTENT_RECAP, raw=raw, depth=depth)

    # 1d. Xref explore — "see also" / "what else references this".
    # Anchored on the last focus, just like follow-up.
    if _XREF_EXPLORE_RE.search(raw):
        return RoutedIntent(
            intent=INTENT_XREF_EXPLORE, raw=raw,
            target_nid=last_focus_nid,
            topic=last_focus_topic,
            depth=depth,
        )

    # 1e. Dependencies — "prerequisites", "what do I need to know
    # first".  Walks the citation graph for the focus and surfaces
    # what it depends on.
    if _DEPENDENCIES_RE.search(raw):
        return RoutedIntent(
            intent=INTENT_DEPENDENCIES, raw=raw,
            target_nid=last_focus_nid,
            topic=last_focus_topic,
            depth=depth,
        )

    # 1c. Re-show — "show me X again" / "repeat that" / bare "again".
    if _RESHOW_RE.search(raw) or _RESHOW_BARE_RE.match(raw):
        # The "topic" is the carrier phrase ("X" in "show me X again");
        # extract it for downstream re-anchoring.  When the request is
        # bare ("again", "repeat that"), fall back to last_focus_topic.
        m = re.search(
            r"\b(?:show\s+me|explain|tell\s+me\s+about)\s+(.+?)\s+again\b",
            raw, re.I,
        )
        topic = m.group(1).strip() if m else (last_focus_topic or "")
        return RoutedIntent(
            intent=INTENT_RESHOW, raw=raw,
            topic=topic,
            target_nid=last_focus_nid,
            depth=depth,
        )

    # 2. Book overview.
    if _BOOK_OVERVIEW_RE.search(raw):
        return RoutedIntent(
            intent=INTENT_BOOK_OVERVIEW, raw=raw, depth=depth,
        )

    # 3. Chapter overview — match "chapter N" first, optionally with
    # number-only follow-up "explain section 5.8 of chapter 5".
    ch_match = _CHAPTER_RE.search(raw)
    sec_match = _SECTION_RE.search(raw)
    if ch_match and (not sec_match or "section" not in raw.lower()):
        ch_num = ch_match.group(1)
        target = _resolve_chapter_nid(book, ch_num) if book else ""
        return RoutedIntent(
            intent=INTENT_CHAPTER_OVERVIEW, raw=raw,
            target_nid=target, topic=f"chapter {ch_num}",
            depth=depth,
            notes={"chapter_number": ch_num},
        )

    # 4. Section overview.
    if sec_match:
        sec_num = (sec_match.group(1) or sec_match.group(2)
                   or sec_match.group(3) or "").strip()
        target = _resolve_section_nid(book, sec_num) if book else ""
        if target:
            return RoutedIntent(
                intent=INTENT_SECTION_OVERVIEW, raw=raw,
                target_nid=target, topic=f"section {sec_num}",
                depth=depth,
                notes={"section_number": sec_num},
            )

    # 5. Follow-up — match the regex OR the solo set.
    if _FOLLOW_UP_RE.search(raw) or short in _FOLLOW_UP_SOLO:
        return RoutedIntent(
            intent=INTENT_FOLLOW_UP, raw=raw,
            target_nid=last_focus_nid,
            topic=last_focus_topic,
            depth=depth,
        )

    # 6. Default — topic-driven QA.
    return RoutedIntent(
        intent=INTENT_TOPIC_QA, raw=raw,
        topic=raw, depth=depth,
    )


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

def _resolve_chapter_nid(book: Book, ch_num: str) -> str:
    """Find the BookNode for ``Chapter <ch_num>``."""
    if book is None or not ch_num:
        return ""
    target = ch_num.strip()
    for n in book.root.walk():
        if n.kind != "chapter":
            continue
        if (n.number or "").strip() == target:
            return n.nid
    # Fall-back: the conventional ``b/chN`` slug.
    nid = f"b/ch{target}"
    if book.find(nid) is not None:
        return nid
    return ""


def _resolve_section_nid(book: Book, sec_num: str) -> str:
    """Find the BookNode whose ``number`` matches *sec_num* and is a
    section / subsection / subsubsection."""
    if book is None or not sec_num:
        return ""
    target = sec_num.strip()
    section_kinds = {"section", "subsection", "subsubsection"}
    # Prefer an exact-number match in section-like kinds.
    for n in book.root.walk():
        if n.kind in section_kinds and (n.number or "").strip() == target:
            return n.nid
    # Otherwise any node with that number.
    for n in book.root.walk():
        if (n.number or "").strip() == target:
            return n.nid
    return ""
