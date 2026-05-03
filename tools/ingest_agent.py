"""Local-LLM-driven ingest agent for Lyceum.

Given a book corpus JSON (``books/<stem>.json``), this agent drives the
full sidecar build until every observed quality metric clears its bar
OR the budget runs out.

Loop, in plain words:

    while budget remaining:
        observe()                      # what sidecars exist + how good they are
        if every check passes: break   # we're done
        decision = think(state)        # ask local Qwen what to do next
        outcome  = act(decision)       # run a build_*.py / reingest_*.py
        log(decision, outcome)

The ``think`` step is the only place an LLM is called (Qwen2.5-14B-AWQ
on ``127.0.0.1:8000`` — same endpoint the rest of Lyceum uses).  The
``act`` step shells out to the existing ``tools/build_*.py`` and
``tools/reingest_*.py`` modules — no logic is reimplemented here.

Local-only: the only outbound HTTP this script makes is to localhost
(Qwen vLLM at :8000 for thinking, and the actual builders may also
hit :8000 / :8003 / :8004).  No Anthropic, no OpenAI.

Usage:
    .venv/bin/python3 -m tools.ingest_agent books/ESLII.json
    .venv/bin/python3 -m tools.ingest_agent books/ESLII.json \\
        --max-actions 60 --max-thinks 200

The agent writes a structured trace to ``books/<stem>.ingest_log.json``
so the operator can audit every decision after the fact.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
PROJECT = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT / ".venv" / "bin" / "python3")

# Quality thresholds the agent tries to clear.  These are intentionally
# *minimum* bars: above these we declare a sidecar "good enough" and
# move on.  Below these the agent re-runs or asks for a fix.
QUALITY = {
    "min_figures":              5,    # books with fewer than this are suspicious
    "min_formulas":             20,
    "min_concepts":             5,
    "max_truncated_formulas":   3,    # formulas ending in ``= -`` etc.
    "min_chapter_map_coverage": 0.75, # populated_sections / total
    "min_eq_mention_coverage":  0.0,  # we don't fail on this — just observe
    "min_fig_mention_coverage": 0.0,  # likewise
}


# ---------------------------------------------------------------------------
# Observation — read sidecars, compute quality metrics
# ---------------------------------------------------------------------------

@dataclass
class BookState:
    """Snapshot of what's on disk for a book + quality metrics."""
    book_json:                 str
    stem:                      str
    pdf_path:                  Optional[str] = None
    has_corpus:                bool = False
    has_figures_sidecar:       bool = False
    has_math_graph:            bool = False
    has_concepts:              bool = False
    has_formulas_layer:        bool = False
    has_equations_sidecar:     bool = False
    chapters:                  list[str] = field(default_factory=list)
    chapters_with_map:         list[str] = field(default_factory=list)
    chapter_map_coverage:      dict[str, float] = field(default_factory=dict)
    n_figures:                 int = 0
    n_formulas:                int = 0
    n_concepts:                int = 0
    n_truncated_formulas:      int = 0
    sample_truncated:          list[str] = field(default_factory=list)
    last_runs:                 list[dict] = field(default_factory=list)

    def to_summary(self) -> dict:
        """Compact JSON the LLM sees as input."""
        return {
            "book":               os.path.basename(self.stem),
            "has_corpus":         self.has_corpus,
            "has_figures.json":   self.has_figures_sidecar,
            "has_math_graph":     self.has_math_graph,
            "has_concepts":       self.has_concepts,
            "has_formulas_layer": self.has_formulas_layer,
            "has_equations_OCR":  self.has_equations_sidecar,
            "n_figures":          self.n_figures,
            "n_formulas":         self.n_formulas,
            "n_concepts":         self.n_concepts,
            "n_truncated":        self.n_truncated_formulas,
            "sample_truncated":   self.sample_truncated[:3],
            "n_chapters":         len(self.chapters),
            "chapters_with_map":  len(self.chapters_with_map),
            "chapters_missing_map": [
                c for c in self.chapters
                if c not in self.chapters_with_map
            ][:6],
            "low_coverage_chapters": [
                {"nid": c, "pct": round(self.chapter_map_coverage[c], 2)}
                for c in self.chapters_with_map
                if self.chapter_map_coverage.get(c, 1.0)
                   < QUALITY["min_chapter_map_coverage"]
            ][:6],
        }


_TRUNC_RE = re.compile(r"=\s*-\s*$|^\s*X\s+\w+\s*=\s*\d+\s*$")


def _read_json(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _walk_chapters(corpus: dict) -> list[str]:
    """Same heuristic as serve.ingest_pipeline._is_real_chapter, abridged."""
    out: list[str] = []
    front_matter = {
        "cover", "title page", "title", "copyright", "dedication",
        "contents", "table of contents", "index", "indices",
        "preface", "preface to the first edition",
        "preface to the second edition", "preface to the third edition",
        "front matter", "back matter", "bibliography", "references",
        "selected bibliography", "acknowledgments", "acknowledgements",
        "errata", "colophon", "name index", "subject index",
        "list of figures", "list of tables", "notation", "abbreviations",
        "glossary", "statement",
    }
    def walk(n: dict):
        kind = (n.get("kind") or "").lower()
        title = (n.get("title") or "").strip().lower()
        if kind == "chapter" and title not in front_matter:
            out.append(n.get("nid", ""))
            return
        for c in n.get("children", []) or []:
            walk(c)
    walk(corpus.get("root", corpus) or {})
    return [n for n in out if n]


def observe(book_json: str) -> BookState:
    """Read every relevant sidecar, compute a BookState snapshot."""
    stem = book_json[: -len(".json")] if book_json.endswith(".json") \
           else book_json
    st = BookState(book_json=book_json, stem=stem)
    pdf_candidates = [stem + ".pdf"]
    for p in pdf_candidates:
        if os.path.isfile(p):
            st.pdf_path = p
            break

    corpus = _read_json(book_json)
    if isinstance(corpus, dict):
        st.has_corpus = True
        st.chapters = _walk_chapters(corpus)

    figures_path = stem + ".figures.json"
    figures = _read_json(figures_path)
    if isinstance(figures, dict):
        by_nid = figures.get("by_nid") or {}
        if by_nid:
            st.has_figures_sidecar = True
            st.n_figures = sum(len(v or []) for v in by_nid.values())

    mg_path = stem + ".math_graph.json"
    mg = _read_json(mg_path)
    if isinstance(mg, dict):
        formulas = mg.get("formulas") or {}
        if formulas:
            st.has_math_graph = True
            st.n_formulas = len(formulas)
            for fid, f in formulas.items():
                if not isinstance(f, dict):
                    continue
                latex = (f.get("latex") or "").strip()
                if not latex or _TRUNC_RE.search(latex):
                    st.n_truncated_formulas += 1
                    if len(st.sample_truncated) < 4:
                        cite = (f.get("cite_labels") or [None])[0]
                        st.sample_truncated.append(
                            f"{cite or fid}: {latex[:80]!r}"
                        )

    cc_path = stem + ".concepts.json"
    cc = _read_json(cc_path)
    if isinstance(cc, dict):
        by_home = cc.get("by_home_nid") or {}
        if by_home:
            st.has_concepts = True
            st.n_concepts = len(by_home)

    fl_path = stem + ".formulas.json"
    fl = _read_json(fl_path)
    if isinstance(fl, dict) and fl.get("by_formula_id"):
        st.has_formulas_layer = True

    eq_path = stem + "_equations.json"
    eq = _read_json(eq_path)
    if isinstance(eq, dict) and eq.get("equations"):
        st.has_equations_sidecar = True

    for ch in st.chapters:
        flat = ch.replace("/", "_")
        cm_path = f"{stem}.chapter_map.{flat}.json"
        cm = _read_json(cm_path)
        if not isinstance(cm, dict):
            continue
        st.chapters_with_map.append(ch)
        total = pop = 0
        def walk(n: dict):
            nonlocal total, pop
            total += 1
            if (n.get("story_paragraph") or "").strip():
                pop += 1
            for c in n.get("children", []) or []:
                walk(c)
        walk(cm.get("root", cm) or {})
        st.chapter_map_coverage[ch] = pop / max(total, 1)

    return st


def is_complete(st: BookState) -> tuple[bool, list[str]]:
    """Return (done, list_of_open_complaints).  Done iff no complaints."""
    why: list[str] = []
    if not st.has_corpus:
        why.append("missing corpus JSON")
    if not st.has_figures_sidecar or st.n_figures < QUALITY["min_figures"]:
        why.append(f"figures sidecar missing or has only "
                   f"{st.n_figures} figures")
    if not st.has_math_graph or st.n_formulas < QUALITY["min_formulas"]:
        why.append(f"math_graph missing or has only "
                   f"{st.n_formulas} formulas")
    if st.n_truncated_formulas > QUALITY["max_truncated_formulas"]:
        why.append(f"{st.n_truncated_formulas} truncated formulas in "
                   f"math_graph")
    if not st.has_concepts or st.n_concepts < QUALITY["min_concepts"]:
        why.append(f"concepts missing or has only {st.n_concepts} entries")
    if not st.has_formulas_layer:
        why.append("formulas_layer (formulas.json) missing")
    missing_maps = [c for c in st.chapters
                    if c not in st.chapters_with_map]
    if missing_maps:
        why.append(f"chapter_map missing for {len(missing_maps)} chapter(s)")
    low_cov = [c for c in st.chapters_with_map
               if st.chapter_map_coverage.get(c, 1.0)
                  < QUALITY["min_chapter_map_coverage"]]
    if low_cov:
        why.append(f"low chapter_map coverage for {len(low_cov)} chapter(s): "
                   f"{', '.join(c for c in low_cov[:3])}")
    return (not why, why)


# ---------------------------------------------------------------------------
# Tool registry — subprocess wrappers around the existing build_*.py.
# Each callable returns ``(ok, stdout_tail)`` so the agent can log it.
# ---------------------------------------------------------------------------

def _run(cmd: list[str], *, timeout: float = 1800.0) -> tuple[bool, str]:
    print(f"  [agent] $ {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT),
            capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    tail = (proc.stdout or "").splitlines()[-12:] + \
           (proc.stderr or "").splitlines()[-4:]
    return (proc.returncode == 0), "\n".join(tail)


def tool_extract_book_figures(book_json: str, *,
                                force: bool = False) -> tuple[bool, str]:
    cmd = [PYTHON, "-m", "tools.extract_book_figures", book_json]
    if force:
        cmd.append("--force")
    return _run(cmd, timeout=900)


def tool_build_math_graph(book_json: str, *,
                            fresh: bool = False) -> tuple[bool, str]:
    if fresh:
        stem = book_json[: -len(".json")]
        for p in (stem + ".math_graph.json", stem + ".formulas.json"):
            try: os.remove(p)
            except FileNotFoundError: pass
    return _run([PYTHON, "-m", "tools.build_math_graph", book_json],
                timeout=900)


def tool_build_concept_layer(book_json: str) -> tuple[bool, str]:
    return _run([PYTHON, "-m", "tools.build_concept_layer", book_json],
                timeout=3600)


def tool_build_formula_layer(book_json: str) -> tuple[bool, str]:
    return _run([PYTHON, "-m", "tools.build_formula_layer", book_json],
                timeout=3600)


def tool_build_chapter_map(book_json: str, *,
                            root: str,
                            fresh: bool = False) -> tuple[bool, str]:
    stem = book_json[: -len(".json")]
    flat = root.replace("/", "_")
    cm_path = f"{stem}.chapter_map.{flat}.json"
    if fresh:
        try: os.remove(cm_path)
        except FileNotFoundError: pass
    return _run(
        [PYTHON, "-m", "tools.build_chapter_map", book_json, "--root", root],
        timeout=900,
    )


def tool_reingest_equations(book_json: str) -> tuple[bool, str]:
    return _run([PYTHON, "-m", "tools.reingest_equations", book_json],
                timeout=1800)


TOOLS = {
    "extract_book_figures":  tool_extract_book_figures,
    "build_math_graph":      tool_build_math_graph,
    "build_concept_layer":   tool_build_concept_layer,
    "build_formula_layer":   tool_build_formula_layer,
    "build_chapter_map":     tool_build_chapter_map,
    "reingest_equations":    tool_reingest_equations,
}

# Names the LLM can pick from, with one-line summaries for the prompt.
TOOL_DESCRIPTIONS = """
extract_book_figures(force=False)
    Caption-region cropper.  Produces <stem>.figures.json + per-fig PNGs.
    Run when figures sidecar missing or n_figures suspiciously low.

build_math_graph(fresh=False)
    PyMuPDF text → math semantic graph.  Produces <stem>.math_graph.json.
    Pass fresh=True to wipe the prior graph + formulas first — needed
    after detector fixes so stale truncated entries don't linger.

build_concept_layer()
    Per-section L0…L4 concept narratives.  Produces <stem>.concepts.json.
    Required upstream of build_chapter_map (chapter_map uses concepts'
    L0 gist).

build_formula_layer()
    Per-formula F0/F1/F2 explanations.  Produces <stem>.formulas.json.
    Required for runtime narration (formula card audio).  Slow — one
    LLM call per formula.

build_chapter_map(root="b/chN", fresh=False)
    Chapter-wide narrative + canonical formula picker per node.
    Produces <stem>.chapter_map.<root>.json.  Pass fresh=True to wipe
    cached story_paragraphs.  Already chunks long chapters internally.

reingest_equations()
    VLM-OCR pass via Qwen2.5-VL on :8004.  Produces <stem>_equations.json.
    Optional but improves equation latex when PyMuPDF text is messy.

DONE
    No more useful actions — every check passes (or further actions are
    out of the agent's scope).
"""


# ---------------------------------------------------------------------------
# Thinking — local Qwen picks the next action
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are an INGEST AGENT for a math-textbook tutoring
system called Lyceum.  Your goal is to drive a fresh book corpus to a
state where every quality check passes.  You decide ONE next action per
turn.  The actual tools are subprocesses you cannot see; you only see
the tool name + parameters.

Available tools and when to use them:
{TOOL_DESCRIPTIONS}

Strict ordering (the system enforces this in spirit but please respect
it in your choice too):
  1. extract_book_figures must run before any chapter_map (chapter_map
     wants figures.json so the chapter-wide LLM can name figures).
  2. build_math_graph must run before build_chapter_map (chapter_map
     picks each node's canonical formula from the math graph).
  3. build_concept_layer must run before build_chapter_map (gists fill
     in chapter_map cells).
  4. build_formula_layer is optional for chapter_map but required for
     full runtime narration.
  5. reingest_equations is optional but improves equation OCR — run
     after build_math_graph if many formulas look truncated.
  6. build_chapter_map runs ONCE per chapter; pass each chapter nid in
     turn until every chapter has a map at the required coverage.

Output format — strict JSON, no prose around it:
  {{"action": "<tool_name>", "args": {{...}}, "rationale": "<one sentence>"}}

Reply with {{"action": "DONE", "args": {{}}, "rationale": "..."}} when
no useful next action exists.

Hard rules:
- Choose ONLY tools listed above plus DONE.
- Never repeat the same (action, args) pair twice in a row — if a
  tool just ran and didn't fix the state, pick a DIFFERENT tool or
  pass fresh=True.
- Prefer the cheapest action that addresses the largest open complaint.
"""


def _llm_call(prompt: str, *, max_tokens: int = 220) -> Optional[dict]:
    payload = json.dumps({
        "model":          LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":     max_tokens,
        "temperature":    0.2,
        "response_format": {"type": "json_object"},
    }).encode()
    try:
        req = urllib.request.Request(
            LLM_URL, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read())
        content = (raw["choices"][0]["message"].get("content") or "").strip()
        return json.loads(content) if content else None
    except Exception as e:
        print(f"  [agent] LLM call failed: {e}", flush=True)
        return None


def think(state: BookState, complaints: list[str],
          history: list[dict]) -> Optional[dict]:
    state_json = json.dumps(state.to_summary(), indent=2)
    history_summary = "\n".join(
        f"  step {h['step']}: {h['action']}({h['args']}) -> "
        f"{'OK' if h['ok'] else 'FAIL'}"
        for h in history[-5:]
    )
    prompt = (
        f"CURRENT STATE\n{state_json}\n\n"
        f"OPEN COMPLAINTS (resolve these):\n  - "
        + "\n  - ".join(complaints) + "\n\n"
        f"RECENT ACTIONS (most recent last):\n{history_summary or '  (none)'}\n\n"
        f"What is the highest-priority next action?  Reply JSON only."
    )
    decision = _llm_call(prompt, max_tokens=240)
    if not isinstance(decision, dict) or "action" not in decision:
        return None
    return decision


def act(book_json: str, decision: dict) -> tuple[bool, str]:
    name = decision.get("action") or ""
    args = decision.get("args") or {}
    if name == "DONE":
        return True, "agent declared DONE"
    fn = TOOLS.get(name)
    if fn is None:
        return False, f"unknown action: {name!r}"
    safe_args = {k: v for k, v in args.items()
                 if k in {"force", "fresh", "root"}}
    return fn(book_json, **safe_args)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(book_json: str, *,
        max_actions: int = 60,
        max_thinks:  int = 200,
        verbose:     bool = True) -> dict:
    log_path = (book_json[: -len(".json")]
                if book_json.endswith(".json") else book_json) \
               + ".ingest_log.json"
    history: list[dict] = []
    n_actions = n_thinks = 0
    started = time.time()

    while True:
        st = observe(book_json)
        done, complaints = is_complete(st)
        if verbose:
            print(f"\n[agent] step={n_actions} thinks={n_thinks} "
                  f"complaints={len(complaints)}", flush=True)
            for c in complaints:
                print(f"  - {c}", flush=True)
        if done:
            print("\n[agent] every check passes — DONE", flush=True)
            break
        if n_actions >= max_actions:
            print(f"\n[agent] hit max_actions={max_actions} — stopping",
                  flush=True)
            break
        if n_thinks >= max_thinks:
            print(f"\n[agent] hit max_thinks={max_thinks} — stopping",
                  flush=True)
            break

        decision = think(st, complaints, history)
        n_thinks += 1
        if decision is None:
            print("[agent] LLM didn't produce a usable decision; "
                  "stopping", flush=True)
            break

        # Reject loops: if last decision == this decision, force a
        # different action (prevents infinite re-runs).
        if (history and history[-1]["action"] == decision["action"]
                and history[-1]["args"] == (decision.get("args") or {})
                and not history[-1]["ok"]):
            print(f"[agent] LLM repeated a failed action "
                  f"({decision['action']}); forcing DONE", flush=True)
            break

        ok, tail = act(book_json, decision)
        n_actions += 1
        history.append({
            "step":      n_actions,
            "action":    decision.get("action"),
            "args":      decision.get("args", {}),
            "rationale": (decision.get("rationale") or "")[:200],
            "ok":        ok,
            "tail":      tail[-800:],
            "wall_t":    round(time.time() - started, 1),
        })
        if decision.get("action") == "DONE":
            break

    final_state = observe(book_json).to_summary()
    report = {
        "book":            book_json,
        "actions":         n_actions,
        "thinks":          n_thinks,
        "wall_seconds":    round(time.time() - started, 1),
        "final_state":     final_state,
        "final_complaints": is_complete(observe(book_json))[1],
        "trace":           history,
    }
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[agent] trace written → {log_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("book_json", help="path to books/<stem>.json")
    p.add_argument("--max-actions", type=int, default=60,
                   help="hard cap on subprocess invocations (default 60)")
    p.add_argument("--max-thinks",  type=int, default=200,
                   help="hard cap on LLM-think calls (default 200)")
    args = p.parse_args(argv)
    if not os.path.isfile(args.book_json):
        print(f"book corpus not found: {args.book_json}", file=sys.stderr)
        return 1
    rep = run(args.book_json,
              max_actions=args.max_actions,
              max_thinks=args.max_thinks)
    return 0 if not rep["final_complaints"] else 2


if __name__ == "__main__":
    sys.exit(main())
