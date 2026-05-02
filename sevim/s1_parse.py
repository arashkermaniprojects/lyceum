"""S1 Parse — clause-unit tokeniser.

Splits raw input text into clause-level SpanTokens, preserving character
offsets so that every token can be traced back to its source span (I3).

Splitting strategy
------------------
We split only on sentence-final punctuation (., ; : ,) followed by
whitespace, and on a small set of discourse connectives (but / because / so).

We deliberately do NOT split on "and" even though it connects clauses,
because the S2 dependency-parse extractor handles conjunct verbs correctly
(they inherit the subject of the root verb).  Splitting at "and" produces
subject-less fragments that the regex fallback in S2 then mis-extracts.

Node-level embeddings are NOT attached at this stage.  They are computed
in S2 Extract once subject and object label strings are known, so that the
two arguments of a single triple receive distinct vectors (attaching a single
clause embedding to both would collapse them under the cosine merge gate).
"""
from __future__ import annotations

import re

from .ir import SpanToken

# Matches the position between two clause units — either punctuation+space
# or a discourse connective preceded by whitespace.
_CLAUSE_SPLIT = re.compile(
    r"(?<=[.,;:])\s+|\s+(?=(?:but|because|so)\s)",
    re.IGNORECASE,
)


def parse_text(text: str, utterance_id: str = "u0") -> list[SpanToken]:
    """Split *text* into clause-level SpanTokens.

    Parameters
    ----------
    text:
        Raw natural-language input (one or more sentences).
    utterance_id:
        Identifier attached to every token; increment per turn in multi-turn
        streaming mode so provenance can be traced back to specific utterances.

    Returns
    -------
    list[SpanToken]
        One token per clause, with character offsets into *text*.
        Empty input returns an empty list.
    """
    if not text:
        return []

    tokens: list[SpanToken] = []
    idx = 0  # current scan position in text

    for m in _CLAUSE_SPLIT.finditer(text):
        end = m.start()
        chunk = text[idx:end].strip()
        if chunk:
            start = text.find(chunk, idx, end)
            tokens.append(SpanToken(chunk, start, start + len(chunk), utterance_id))
        idx = m.end()

    # Append whatever remains after the last split point.
    tail = text[idx:].strip()
    if tail:
        start = text.find(tail, idx)
        tokens.append(SpanToken(tail, start, start + len(tail), utterance_id))

    return tokens
