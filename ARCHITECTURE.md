# SeVim — Code Architecture Reference

> **Note:** This document replaces the prior design-spec draft. It is generated
> directly from reading the source code in `sevim/` and describes what the
> code *actually does*, not aspirational plans.

**Version:** 0.1.0  
**Package root:** `sevim/`  
**Entry point:** `sevim/cli.py → main()` or `sevim/pipeline.py → run_pipeline()`/pipeline.py → run_pipeline()`

---

## Table of Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Component diagram](#2-component-diagram)
3. [Module-by-module reference](#3-module-by-module-reference)
4. [Data flow with exact types](#4-data-flow-with-exact-types)
5. [Neural vs. symbolic classification](#5-neural-vs-symbolic-classification)
6. [Neural components in detail](#6-neural-components-in-detail)
7. [The W matrix in s3_map.py](#7-the-w-matrix-in-s3_mappy)
8. [Environment variables](#8-environment-variables)
9. [Multi-turn / streaming mode](#9-multi-turn--streaming-mode)

---

## 1. Pipeline Overview

```
text: str
  │
  ▼  S1  s1_parse.py
list[SpanToken]
  │
  ▼  S2  s2_extract.py
SceneGraph  (nodes + edges)
  │
  ▼  S2b  s2b_improve.py   [optional, disabled by default]
SceneGraph  (possibly merged / relabelled)
  │
  ▼  S3  s3_map.py
VisualGraph  (shapes + connectors + containers)
  │
  ▼  S4  s4_layout.py
PlacedGraph  (pixel coordinates, scaled to canvas)
  │
  ▼  S4.5  overlap.py   [geometry check]
PlacedGraph  (unchanged; raises or logs if overlaps found)
  │
  ▼  S5  s5_render.py
str  (SVG markup)
```

The single public function `run_pipeline(text, utterance_id, graph)` in
`pipeline.py` calls these stages in order and collects `TraceEvent` records
for each stage. `graph=<prev>` enables incremental / multi-turn updates.

---

## 2. Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         SeVim pipeline                          │
│                                                                  │
│  ┌──────────┐     ┌─────────────────────────────────────────┐   │
│  │  S1      │     │  S2  s2_extract.py                      │   │
│  │ SYMBOLIC │────▶│  SYMBOLIC (dep-parse + regex cascade)   │   │
│  │ s1_parse │     │  + embed.py calls for cosine-merge dedup│   │
│  └──────────┘     └──────────────────┬──────────────────────┘   │
│                                      │                           │
│                     embed.py (NEURAL)│ called per node label     │
│                    ┌─────────────────▼──────────────────────┐   │
│                    │  Qwen/Qwen2.5-7B  (FROZEN ENCODER)     │   │
│                    │  mean-pool last_hidden_state            │   │
│                    │  output: tuple[float, ...]  (D-dim)     │   │
│                    └────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  S2b  s2b_improve.py  (NEURAL, optional, default OFF)   │    │
│  │  claude-haiku-4-5  — text generation, JSON output       │    │
│  │  Input:  graph JSON + source text                       │    │
│  │  Output: revised graph JSON (nodes + edges)             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                 │
│  │  S3      │     │  S4      │     │  S5      │                 │
│  │ SYMBOLIC │────▶│ SYMBOLIC │────▶│ SYMBOLIC │                 │
│  │ s3_map   │     │ s4_layout│     │ s5_render│                 │
│  │ + φ(W)   │     │ Sugiyama │     │  SVG gen │                 │
│  └──────────┘     └──────────┘     └──────────┘                 │
│                                                                  │
│  ┌──────────────────────────┐                                    │
│  │  overlap.py  SYMBOLIC    │  (geometry checker, S4.5)          │
│  └──────────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────┘

Legend:
  NEURAL   — involves a trained model (Qwen or Claude)
  SYMBOLIC — pure Python / regex / rule tables / deterministic math
  φ(W)     — linear projection with a FROZEN RANDOM weight matrix (see §7)
```

---

## 3. Module-by-module Reference

### `__init__.py`
Package initialiser. Exports nothing public; sets `__version__ = "0.1.0"`.

---

### `ir.py` — Intermediate Representation types
Pure data-class definitions. No logic, no I/O.

| Class | Role |
|---|---|
| `SpanRef` | Frozen pointer into source text: `(start, end, utterance_id)` |
| `SpanToken` | A clause-sized chunk from S1: `(text, start, end, utterance_id, embedding)` |
| `SceneNode` | A graph node: `(id, label, node_type, embedding, salience, src_spans)` |
| `SceneEdge` | A directed relation: `(id, from_id, to_id, relation, src_spans)` |
| `SceneGraph` | Container for `list[SceneNode]` + `list[SceneEdge]` + `revision: int` |
| `VisualShape` | Visual specification of a node: primitive, size, font, fill index |
| `VisualConn` | Visual specification of an edge: relation pattern string |
| `VisualGraph` | `shapes + connectors + containers` (container hierarchy from `part_of`/`contains`) |
| `PlacedShape` | `VisualShape` with absolute `(x, y)` coordinates |
| `PlacedConn` | `VisualConn` with a `list[tuple[float,float]]` polyline (2 points: border-to-border) |
| `PlacedGraph` | `placed_shapes + conns + canvas_w + canvas_h` |
| `TraceEvent` | `(stage, message, refs)` — one per pipeline stage |
| `PipelineResult` | Final output: `(svg, graph, placed, trace)` |

**Type constants (Literals):**
- `NodeType`: `"entity" | "process" | "attribute" | "value" | "group"`
- `RelationType`: 12 values — `contains, part_of, causes, sequence, attribute_of, similar_to, opposes, instance_of, used_for, requires, reduces_to, measures`
- `Primitive`: `"rect" | "ellipse" | "line" | "arrow" | "text" | "group" | "diamond" | "hexagon" | "parallelogram"`

---

### `cli.py` — Command-line interface
**Public function:** `main(argv=None) -> int`

Reads text from a file or stdin, calls `run_pipeline`, writes three output files:
- `<prefix>.svg` — the SVG string
- `<prefix>.ir.json` — serialised `SceneGraph` (nodes + edges)
- `<prefix>.trace.json` — `list[TraceEvent]`

Usage: `sevim <input_file_or_-> [--out <prefix>]`

---

### `pipeline.py` — Orchestrator
**Public function:** `run_pipeline(text, utterance_id="u0", graph=None) -> PipelineResult`

Calls S1 → S2 → S2b → S3 → S4 → overlap check → S5 in sequence.

- `graph=None` starts a fresh `SceneGraph`.
- `graph=<prev>` extends an existing graph (multi-turn / streaming mode); S2
  merges new nodes into the existing graph via `_ensure_node`.
- Between S4 and S5, runs `detect_overlaps` (or `assert_no_overlaps` when
  `SEVIM_STRICT_OVERLAPS=1`).
- Appends one `TraceEvent` per stage, plus one for the overlap check (S4.5).

---

### `embed.py` — Frozen neural encoder
See §6.1 for the full neural breakdown.

**Public functions:**

| Function | Signature | Returns |
|---|---|---|
| `encode` | `(text: str) -> tuple[float, ...]` | Mean-pooled last-hidden-state vector, or `()` on failure/disabled |
| `is_available` | `() -> bool` | Whether the model loaded successfully |
| `embedding_dim` | `() -> int` | `model.config.hidden_size` of the loaded model (0 if not loaded) |

Module-level state: `_model`, `_tokenizer`, `_device` — all `None` until first
call; protected by `threading.Lock`.

---

### `s1_parse.py` — Text tokeniser (S1)
**Public function:** `parse_text(text, utterance_id="u0") -> list[SpanToken]`

Splits input text into clause-level chunks using a single compiled regex:

```python
_CLAUSE_SPLIT = re.compile(
    r"(?<=[.,;:])\s+|\s+(?=(?:but|because|so)\s)",
    re.IGNORECASE,
)
```

Splits on sentence-final punctuation and the conjunctions `but / because / so`.
Does **not** split on `and` — the dep-parse extractor handles conjunct verbs
correctly without that split. Each chunk becomes a `SpanToken` with character
offsets into the original string.

`SpanToken.embedding` is always `()` here. Node-level embeddings are deferred
to S2 where the extracted subject/object labels are available (attaching a
single clause embedding to both subject and object would collapse them under
cosine merge).

**Classification: fully symbolic.** One regex, no model calls.

---

### `s2_extract.py` — Semantic triple extraction (S2)
**Public function:** `extract(tokens: list[SpanToken], graph: SceneGraph | None = None) -> SceneGraph`

Dispatches to one of two internal extractors depending on whether spaCy is
importable and `en_core_web_sm` loads successfully:

#### Path A — `_extract_dep` (preferred, requires `spacy` + `en_core_web_sm`)

1. Loads `en_core_web_sm` via `spacy.load(..., disable=["ner"])` — lazy, cached,
   not retried after first failure.
2. For each `SpanToken`, calls `nlp(text)` to produce a spaCy `Doc`.
3. For each sentence in the doc, calls `_extract_from_sent` which:
   - Walks the root verb and any conjunct/adverbial-clause verbs.
   - Maps verb **lemmas** to relation types via `VERB_RELATION_MAP` (60-entry
     hardcoded dict, labelled as "seeded from corpus-clustered verb lemmas").
   - Handles copular patterns (`"X is a Y"`) via `_handle_copular` +
     `COP_PREDICATE_MAP` (9-entry dict).
   - Also fires four structural sub-extractors regardless of whether a main
     verb triple was found: `_extract_possessives`, `_extract_noun_of`,
     `_extract_relcl_io`, `_extract_including_list`.
   - If the dep-parse yields no new edges for a clause, falls back to the regex
     cascade (`_regex_try_one`) on that clause.
4. Every unique subject/object label string is encoded via `embed.encode(label)`
   and stored in the node's `embedding` field (memoised per-call in
   `label_emb: dict[str, tuple]`).

#### Path B — `_extract_regex` (fallback, no spaCy needed)

Runs a 16-rule regex cascade (`_RULES`) on each `SpanToken`. Each rule is a
`(compiled_regex, relation_string)` pair. First match wins per clause. Also
calls `embed.encode()` for node embeddings.

#### Node deduplication — `_ensure_node(g, label, span, embedding)`

Before creating a new `SceneNode`, checks in order:
1. **Exact ID match:** `n_<normalised_label>` already in graph → reuse.
2. **Cosine merge** (only when embedding is non-empty):
   - Finds all existing `entity` nodes with non-empty embeddings.
   - If `cosine(new, existing) >= MERGE_TAU (0.85)`:
     - If `0.85 <= sim < HIGH_TAU (0.95)`: also requires `_lemma_compatible`
       (one label is a prefix of the other, min length 3).
     - If `sim >= 0.95`: merges unconditionally.
   - Merges into the highest-similarity match.

Label normalisation (`_normalize`): strip leading articles, strip trailing
punctuation, singularise each content word via `_singularize` (handles
`-ies → -y`, `-sses/-xes/-zes/-ches/-shes → -s` removal, irregular plurals),
strip trailing conjunctions.

**Classification:** Mostly symbolic. spaCy's tagger/parser is a statistical
model (packaged as a fixed artefact), but the triple-extraction logic on top
of it is entirely rule-based. Qwen embeddings are used only for the
cosine-merge dedup decision.

---

### `s2b_improve.py` — LLM graph inspector (S2b, optional)
See §6.2 for the full neural breakdown.

**Public function:** `improve(g: SceneGraph, source_text: str) -> SceneGraph`

Default: no-op (returns `g` unchanged). Activated only when both
`SEVIM_IMPROVE=1` and `ANTHROPIC_API_KEY` are set.

---

### `s3_map.py` — Semantic-to-visual mapping (S3)
**Public function:** `map_visual(graph: SceneGraph) -> VisualGraph`

Four sub-tasks, all run unconditionally:

#### 3a — Shape primitive selection (`_label_to_primitive`)

Symbolic keyword rules checked in order (first match wins):

| Pattern | Primitive | Example labels matched |
|---|---|---|
| `node_type == "attribute"` | `"ellipse"` | any attribute node |
| neural network vocabulary regex | `"hexagon"` | `neural network, transformer, lstm, q-network` |
| numeric parameter vocabulary regex | `"diamond"` | `weight, gradient, loss, reward, bias` |
| layer/transform vocabulary regex | `"parallelogram"` | `layer, encoder, attention, softmax, embedding` |
| default | `"rect"` | everything else |

#### 3b — Visual sizing (`_phi`)
See §7 for full explanation. Produces `(width, height, font_size, stroke_width, fill_index)`.

#### 3c — Connector pattern lookup

Hard-coded 12-entry dict `_RELATION_PATTERN` mapping `RelationType` → a
descriptive string (e.g., `"causes" → "arrow-directed"`). The string itself
is carried through to S5 where `_render_conn` dispatches on the `.relation`
field directly, not on this pattern string. The pattern string is metadata only.

#### 3d — Container hierarchy

Builds `containers: list[tuple[str, list[str]]]` from the graph edges:
- `part_of(A, B)` → A is a child of B
- `contains(A, B)` → B is a child of A

Uses `dict.setdefault` so only the first edge wins for each child (no cycles
from a single child having two parents). Containers list is sorted by parent ID
for stable output.

---

### `s4_layout.py` — Deterministic layout (S4)
**Public function:** `layout(vg: VisualGraph) -> PlacedGraph`

All symbolic. No model calls.

**Layout algorithms (all deterministic, tie-broken by nid ascending):**

| Algorithm | Trigger | Description |
|---|---|---|
| `_strip` | dominant relation is `sequence` | Horizontal row, items sorted by nid |
| `_stack` | container has ≤ 1 child | Single vertical column |
| `_grid` | no dominant relation / mixed | Square grid, `ceil(sqrt(n))` columns, uniform cell size |
| `_sugiyama` | most semantic relations | Longest-path topological layering + 3-pass barycenter crossing reduction (alternating forward/backward sweeps) |

`_dominant_rule` picks the algorithm by counting relations between siblings in
a container group and selecting the relation with the most edges (priority:
recognised relation > count > nid).

After placing all containers recursively, the entire layout is uniformly scaled
down if it overflows the canvas:
```python
sf = min(CANVAS_W / max_right, CANVAS_H / max_bottom, 1.0)
```

Connector endpoints are clipped to the rectangle boundary via `_clip_to_rect`
(parametric line-vs-axis-aligned-rect intersection). Redundant connectors
(those that mirror a part_of/contains parent-child containment) are suppressed.

**Canvas defaults:** 700 × 440 px. Override with `SEVIM_CANVAS_W` / `SEVIM_CANVAS_H`.

---

### `s5_render.py` — SVG serialiser (S5)
**Public function:** `render(pg: PlacedGraph) -> str`

Purely symbolic. Converts `PlacedGraph` to a complete SVG string.

- **Shapes:** `_render_shape` emits the correct SVG element per `Primitive`.
  Label wrapping (`_wrap_label`) tries three decreasing font sizes
  (`fs, fs*0.85, fs*0.70`) to fit the label into ≤ 3 lines. Long labels that
  still exceed 3 lines are capped at 3 with the third line truncated.
- **Z-order:** containers (larger, background) rendered before non-containers.
  Within each group, sorted by descending area so smaller shapes appear on top.
- **Connectors:** `_render_conn` dispatches per relation to specialised
  renderers:
  - `reduces_to` → `_funnel` (filled triangle)
  - `similar_to` → `_parallel` (two lines + `≈` label)
  - `measures` → `_measures_annot` (dotted line + `=` label)
  - `opposes` → `_opposes` (bar-end at both ends)
  - `requires` → `_requires` (hollow triangle marker)
  - all others → cubic Bézier with optional dash array and arrow marker
- **Palette:** 5 colours indexed by `fill_index % 5`:
  `["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]`
- **SVG defs block:** defines three reusable markers: `arrow` (filled
  triangle), `hollow-tri` (hollow triangle), `bar` (thick rectangle).

---

### `overlap.py` — Geometry checker (S4.5)
**Public functions:**

| Function | Signature | Returns |
|---|---|---|
| `detect_overlaps` | `(pg: PlacedGraph) -> list[dict]` | Stable-sorted list of finding dicts |
| `assert_no_overlaps` | `(pg: PlacedGraph) -> None` | Raises `OverlapError` if any findings |

Two check types:
1. **`shape_overlap`** — axis-aligned bounding-box intersection with 0.5 px
   tolerance. Skips pairs where one shape is a container and the other's rect
   is fully contained inside it (legitimate nesting).
2. **`label_overflow`** — estimated text width (`longest_line_chars × 0.55 × font_size`)
   exceeds available box width (`shape.width - 8`). Skips container shapes.

Called by `pipeline.py` after S4. Entirely symbolic / geometric. No model calls.

`OverlapError` (raised only under `SEVIM_STRICT_OVERLAPS=1`) is a `RuntimeError`
subclass that carries `findings: list[dict]`.

---

## 4. Data Flow with Exact Types

```
run_pipeline(text: str, utterance_id: str, graph: SceneGraph | None)
│
├── S1: parse_text(text, utterance_id)
│       in:  str
│       out: list[SpanToken]
│               SpanToken.embedding is always () here
│
├── S2: extract(tokens: list[SpanToken], graph: SceneGraph | None)
│       in:  list[SpanToken], optional prior SceneGraph
│       out: SceneGraph
│               SceneNode.embedding: tuple[float, ...]  ← from embed.encode()
│               SceneNode.salience:  float              ← hardcoded 0.5 for all nodes
│               SceneEdge.relation:  RelationType       ← 12-value Literal
│
│       embed.encode(label: str) → tuple[float, ...]
│           Calls Qwen2.5-7B, returns D-dimensional vector (D = hidden_size)
│           Returns () if model unavailable or SEVIM_DISABLE_EMBED=1
│
├── S2b: improve(g: SceneGraph, source_text: str) → SceneGraph
│       in:  SceneGraph, str
│       out: SceneGraph  (mutated in place, same object returned)
│       HTTP POST to https://api.anthropic.com/v1/messages
│           request:  {model, max_tokens, system, messages: [{role:"user", content:"..."}]}
│           response: {content: [{text: "<JSON string or prose+JSON>"}]}
│           extracted: {nodes: [{id,label}], edges: [{from,to,relation}]}
│       On any error: returns g unchanged
│
├── S3: map_visual(graph: SceneGraph) → VisualGraph
│       in:  SceneGraph
│       out: VisualGraph
│               VisualShape.primitive:    Primitive  ← from _label_to_primitive()
│               VisualShape.width:        float      ← from _phi(), range [100, 200]
│               VisualShape.height:       float      ← from _phi(), range [74, 90]
│               VisualShape.font_size:    float      ← constant 16.0 (from _phi())
│               VisualShape.stroke_width: float      ← constant 1.2 (from _phi())
│               VisualShape.fill_index:   int        ← from _phi(), range [0, 4]
│               VisualConn.pattern:       str        ← from _RELATION_PATTERN dict
│
├── S4: layout(vg: VisualGraph) → PlacedGraph
│       in:  VisualGraph
│       out: PlacedGraph
│               PlacedShape.x, .y:        float  (absolute canvas coords, post-scale)
│               PlacedConn.points:        list[tuple[float,float]]  (2 points)
│               PlacedGraph.canvas_w/h:   float  (from env vars, default 700 × 440)
│
├── S4.5: detect_overlaps(pg) → list[dict]
│       in:  PlacedGraph
│       out: list[dict]  (each has 'kind', affected nids, geometry details)
│            raises OverlapError if SEVIM_STRICT_OVERLAPS=1 and list is non-empty
│
└── S5: render(pg: PlacedGraph) → str
        in:  PlacedGraph
        out: str  (complete SVG document as UTF-8 string)
```

---

## 5. Neural vs. Symbolic Classification

| Component | Classification | Honest notes |
|---|---|---|
| `s1_parse` | **Symbolic** | Single regex |
| `s2_extract` — dep-parse path | **Symbolic** | spaCy `en_core_web_sm` is a statistical model (packaged artefact), but the triple-extraction logic on top of its parse tree is entirely rule-based |
| `s2_extract` — regex path | **Symbolic** | Pure regex cascade, 16 rules |
| `embed.encode` | **Neural (frozen encoder)** | Qwen2.5-7B, `eval()` mode, no gradients, mean-pool only — not generation |
| `s2b_improve` | **Neural (LLM generation, optional)** | Claude claude-haiku-4-5 via API, text generation, non-deterministic, disabled by default |
| `s3_map` — shape selection | **Symbolic** | 3 keyword regexes + node_type rule |
| `s3_map` — `_phi` sizing | **Pseudo-neural / deterministic** | Fixed random matrix W; see §7 for honest assessment |
| `s3_map` — connector patterns | **Symbolic** | Dict lookup |
| `s4_layout` | **Symbolic** | Sugiyama + grid/strip/stack, all deterministic |
| `s5_render` | **Symbolic** | String formatting |
| `overlap` | **Symbolic** | Geometry arithmetic |

---

## 6. Neural Components in Detail

### 6.1 `embed.py` — Qwen2.5-7B as frozen encoder

**Model:** `Qwen/Qwen2.5-7B` (base, not Instruct). Override with `SEVIM_EMBED_MODEL`.

**Loading (lazy, once, thread-safe):**
```python
torch.set_grad_enabled(False)          # global, permanent
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME,
            torch_dtype=torch.float16,   # float32 on CPU
            low_cpu_mem_usage=True)
_model = _model.to(device).eval()
```

**What `encode(text)` does — step by step:**

1. Tokenise `text` with `AutoTokenizer`, `truncation=True, max_length=128`.
2. Run `model(**inputs, output_hidden_states=False, use_cache=False)`.
3. Take `out.last_hidden_state` of shape `(1, T, D)`.
4. Attention-mask-weighted mean over the token dimension:
   ```
   pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
   ```
5. Cast to `float32`, move to CPU, return as `tuple[float, ...]` of length `D`.

**This is not text generation.** Qwen produces no output tokens, no logits,
and no sampling. It is used exclusively as a frozen feature extractor that
maps a short string to a single dense vector.

**Where the output is used:**
- `s2_extract._ensure_node`: cosine similarity against existing node embeddings
  to decide whether to merge two nodes (threshold 0.85).
- `s3_map._phi`: as input to the W-matrix linear projection that determines
  visual width, height, and colour index for each shape.

**Graceful degradation:** If `transformers`/`torch` are absent, `SEVIM_DISABLE_EMBED=1`
is set, or the model fails to load for any reason, `encode()` returns `()`.
All downstream code has explicit `if not embedding:` fallback paths:
- `_ensure_node` skips the cosine-merge step entirely (ID-match only).
- `_phi` falls back to `(100 + 60*salience, 50 + 20*salience, 14.0, 1.5, 0)`
  (currently all nodes get the same size because `salience` is hardcoded 0.5).

**Precision:** `float16` on CUDA, `float32` on CPU. All outputs converted to
`float32` before return regardless of device.

**Determinism:** Fully deterministic on a fixed machine + model revision.
`SEVIM_STRICT_DET=1` additionally calls `torch.set_num_threads(1)` for
cross-run reproducibility.

---

### 6.2 `s2b_improve.py` — Claude claude-haiku-4-5 as graph editor

**Model:** `claude-haiku-4-5-20251001` (hard-coded in source).

**Activation guard:** the first two lines of `improve()` are:
```python
if os.environ.get("SEVIM_IMPROVE") != "1":
    return g
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    return g
```
Both conditions must pass. If either fails, the graph is returned unchanged
immediately — no side effects, no error.

**What it does:**

1. Serialises the current `SceneGraph` to a JSON payload:
   ```json
   { "nodes": [{"id": "n_...", "label": "..."}],
     "edges": [{"from": "n_...", "to": "n_...", "relation": "..."}] }
   ```
2. Builds a prompt: system prompt (graph-QC instructions) + user message
   containing the original source text and the graph JSON.
3. POSTs to `https://api.anthropic.com/v1/messages`, `max_tokens=1024`,
   timeout=20 s.
4. Extracts the first `{...}` block from the response text (the LLM sometimes
   adds prose or backtick fences around the JSON).
5. Parses `{nodes: [...], edges: [...]}` from the extracted JSON.
6. Calls `_apply_improvements(g, improved)` which mutates `g.nodes` and
   `g.edges` in place: rebuilds nodes from the LLM's list, reusing `embedding`,
   `salience`, and `src_spans` from the original node where an ID or label
   match can be found; builds edges from the LLM's edge list, rejecting any
   with a relation not in `_VALID_RELATIONS`.

**This is text generation.** Claude reasons about the graph and returns a
corrected version. The output is non-deterministic (default temperature
sampling). Any failure — HTTP error, timeout, malformed JSON, no JSON in
response — causes the function to print a warning and return `g` unchanged.
S2b is never a fatal error.

---

## 7. The W matrix in `s3_map.py`

### Code

```python
_PHI_SEED = 42
_PHI_ROWS = 5   # one row per output: w, h, font_size, stroke_width, fill_index

def _init_W(dim: int) -> None:
    rng = random.Random(_PHI_SEED)          # Python stdlib RNG, seeded 42
    scale = 1.0 / math.sqrt(dim + 1)        # Xavier-style scale
    _W = tuple(
        tuple(rng.gauss(0.0, 1.0) * scale for _ in range(dim + 1))
        for _ in range(_PHI_ROWS)
    )
```

### Initialisation

On the first call to `_phi()` with a non-empty embedding, `_init_W(d)` is
called with `d = len(embedding)`. It creates a `5 × (d+1)` matrix by drawing
from `Normal(0, 1/sqrt(d+1))` using Python's `random.Random` seeded with the
constant `42`. The matrix is built at runtime from a fixed seed. **There is
no file to load, no checkpoint, no saved parameters.**

### Is W trained?

**No.** There is no training loop, no loss function, no gradient descent,
no dataset, no parameter file. The matrix is constructed fresh from a fixed
seed on every process start.

### What `_phi` actually computes

```python
aug = list(embedding) + [salience]          # length d+1
raw = [dot(row, aug) for row in _W]         # 5 raw scalars

w            = 150.0 + 50.0 * tanh(raw[0]) # node width  ∈ [100, 200] px
h            = 82.0  +  8.0 * tanh(raw[1]) # node height ∈  [74,  90] px
font_size    = 16.0                         # raw[2] is IGNORED — constant
stroke_width = 1.2                          # raw[3] is IGNORED — constant
fill_index   = abs(int(raw[4] * 1000)) % 5 # colour index ∈ {0,1,2,3,4}
```

### Honest assessment

`_phi` is a random projection masquerading as a learned "numeric mapping."

- Rows 2 and 3 of W are wasted — their outputs are discarded in favour of
  hardcoded constants.
- Rows 0 and 1 modulate node size by ±50 px in width and ±8 px in height
  relative to a fixed baseline. These variations are driven by the dot product
  of a random matrix with the Qwen embedding, which means different concept
  labels will get slightly different sizes — but the assignment is arbitrary,
  not semantically meaningful.
- Row 4 assigns one of 5 palette colours. Again arbitrary — the colour a node
  gets is a function of its Qwen embedding dotted with a random row.
- The `tanh` keeps values in range but adds no semantic structure.
- `salience` is hardcoded to 0.5 for all nodes, so the last column of W
  contributes a constant offset to every node equally.

In the current implementation, `_phi` is effectively: "project the Qwen
embedding through a fixed noise matrix, apply tanh, rescale." The variation
it produces is content-dependent but not content-meaningful. Size and colour
are effectively random per-concept assignments that are stable across runs
(given fixed embedding model + fixed W seed).

---

## 8. Environment Variables

| Variable | Module | Default | Effect |
|---|---|---|---|
| `SEVIM_EMBED_MODEL` | `embed.py` | `"Qwen/Qwen2.5-7B"` | Which HuggingFace model to load for encoding |
| `SEVIM_EMBED_MAX_TOKENS` | `embed.py` | `128` | Maximum token sequence length for the encoder |
| `SEVIM_DISABLE_EMBED` | `embed.py` | unset | Set to `"1"` to skip model loading entirely; all `encode()` calls return `()` |
| `SEVIM_STRICT_DET` | `embed.py` | unset | Set to `"1"` to call `torch.set_num_threads(1)` for cross-run reproducibility on CPU |
| `SEVIM_IMPROVE` | `s2b_improve.py` | unset | Set to `"1"` to activate the S2b LLM graph-improvement pass |
| `ANTHROPIC_API_KEY` | `s2b_improve.py` | unset | Anthropic API key for claude-haiku-4-5 (required when `SEVIM_IMPROVE=1`) |
| `SEVIM_STRICT_OVERLAPS` | `pipeline.py` | unset | Set to `"1"` to raise `OverlapError` on any geometry finding; default is log-only |
| `SEVIM_CANVAS_W` | `s4_layout.py` | `700` | Output SVG canvas width in pixels |
| `SEVIM_CANVAS_H` | `s4_layout.py` | `440` | Output SVG canvas height in pixels |

---

## 9. Multi-turn / Streaming Mode

`run_pipeline` accepts an optional `graph: SceneGraph` argument. When supplied,
S2's `_ensure_node` / `_add_edge` operate on the existing graph rather than a
fresh one:

- New triples whose subject/object normalise to the same ID as an existing
  node are merged silently (provenance `src_spans` is appended).
- New nodes with cosine similarity ≥ 0.85 to an existing entity node
  (subject to `_lemma_compatible` check at the 0.85–0.95 borderline) are
  merged into that node.
- All other new nodes and edges are appended.

The returned `PipelineResult.graph` is the merged graph. Pass it back on the
next call to extend the scene. `SceneGraph.revision` increments on each S2
pass (each call to `_extract_dep` or `_extract_regex`).

S3–S5 re-run on the full merged graph every call; there is no incremental
diff at those stages in the current implementation.

