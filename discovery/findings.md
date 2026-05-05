# Discovery run — full-corpus results

**Source:** 109 files from a private internal teaching corpus (90 LaTeX write-ups + 19 VTT lecture transcripts).  The corpus contains personally-identifiable student records and is not redistributed; the aggregate counts below are anonymous and contain no per-student information.

**Pipeline:** sentences → Qwen2.5-14B-Instruct-AWQ (vllm on :8000) → JSON triples → Qwen2.5-7B mean-pool embeddings → greedy single-link clustering at τ = 0.92.

**Scale:** 2244 sentences → 681 triples (≈30% yield) → 199 unique relation phrases → 129 clusters after embedding merge. Full run completed in 304 s (≈ 7.4 sentences/s).

## Top data-driven clusters

| Rank | Count | Representative phrase | Maps to v2 relation |
|---|---|---|---|
| 1 | 79 | used as / used in / used for | **used_for** ✓ |
| 2 | 77 | do not provide | grading-report noise, filter |
| 3 | 76 | requires | **requires** ✓ |
| 4 | 75 | require (plural) | **requires** ✓ (morphology — merge) |
| 5 | 62 | reduced | **reduces_to** ✓ |
| 6 | 34 | is | copula (instance_of / attribute_of fallthrough) |
| 7 | 21 | is like / is composite / is greater than | heterogeneous "is X" cluster |
| 11 | 10 | equals | **measures** ✓ |
| 13 | 8 | uses | **used_for** ✓ (morphology — merge) |
| 14 | 7 | follows | **sequence** ✓ |
| 15 | 7 | is an example of / is a mix of / is based on | **instance_of** ✓ |
| 18 | 5 | sorts / extracts / compresses | algorithmic operation verbs |
| 19 | 5 | splits / sends / solves | decomposition verbs |

## Validation of v2 ontology

The four relations added after the 1000-sentence pilot (`used_for`, `requires`, `reduces_to`, `measures`) are heavily represented in the full corpus:

- **`used_for`**: 79 + 8 = 87 hits — now the most frequent semantic relation in the corpus
- **`requires`**: 76 + 75 = 151 hits (after morphology merge) — the single largest signal
- **`reduces_to`**: 62 hits — strongly validated
- **`measures`**: 10 + 4 = 14 hits — smallest of the four but real

Combined, these four v2 relations would have covered ≈45% of all triples in the corpus. The original v1 ontology would have routed nearly all of those to the `else` branch.

## No new relation classes beyond v2

Longer-tail candidates inspected:
- `costs` (10) — overlaps with `measures`; could be a sub-pattern (cost annotation) rather than a new relation.
- `sorts / extracts / compresses` (5) — domain-specific operation verbs; composing with `used_for` covers them (*"algorithm uses sorting"*).
- `splits / sends / solves` (5) — decomposition verbs; `reduces_to` + sequence composition covers.

No cluster with > 5 hits demands a new visual pattern, so the ontology remains at 12 relations — still inside the 15–20 visual-distinguishability ceiling.

## Known noise

- `do not provide` (77) --- all from grading-report sections in the source corpus. A regex block-list on section headers (``Grading``, ``Submission``, ``CLO``) would remove this noise cleanly and likely raise the useful-triple yield from 30% to ≈ 40%.
- `will do` (11) — planning verbs from student narratives; not a semantic relation.

## Morphology note

`requires` and `require` (plural/conjugation) remain separate clusters under cosine τ = 0.92 because Qwen-7B base embeddings of single inflected verbs don't collapse tightly. A trivial lemmatiser pre-pass (spaCy is already loaded for S2 — reusing its lemmatiser is a one-line change) would merge these automatically. Until then, the S2 dep-parse path already uses the verb's spaCy lemma, so this morphology gap does not affect runtime extraction, only the discovery-side cluster counts.

## Next steps

1. Block grading-report sections in the extractor and rerun (one-liner), then refresh this file with a noise-free distribution.
2. Pipe the spaCy lemmatiser into `discovery/cluster_relations.py` so morphology-only splits collapse automatically.
3. The v2 ontology is corpus-validated — wire the `VERB_RELATION_MAP` additions made so far into an explicit JSON artefact that the article can cite and that `s2_extract.py` imports from, rather than being hand-copied.
