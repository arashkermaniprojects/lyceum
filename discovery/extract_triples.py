"""Walk a corpus directory, extract (subject, relation, object)
triples via the local Qwen2.5-14B-Instruct-AWQ vLLM server.

Output: discovery/triples.jsonl — one triple per line with provenance
{"file": str, "sentence": str, "s": str, "r": str, "o": str}.

Usage:
    python3 extract_triples.py --corpus-root /path/to/corpus \\
        --max-sentences 500 --max-files 20

The corpus root can also be set via the ``LYCEUM_TRIPLE_CORPUS``
environment variable.  This script was originally exercised on a
private teaching corpus; the corpus is not redistributed and is not
required for any reported result in the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

CORPUS_ROOT = Path(os.environ.get("LYCEUM_TRIPLE_CORPUS", ""))
OUT_PATH = Path(__file__).parent / "triples.jsonl"

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"

SYSTEM = (
    "You extract (subject, relation, object) triples from academic text. "
    "Output ONLY a JSON array. No commentary, no markdown fences."
)
USER_TMPL = (
    "Extract semantic triples from this sentence. "
    "Keep relations short (1–4 words, a verb phrase or canonical relation like "
    "\"causes\", \"is part of\", \"is an example of\"). "
    "Return [] if no clear relation exists. "
    "Sentence: {sentence}"
)


# ---------------------- text extraction ----------------------

_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})*")
_LATEX_ENV_BEGIN = re.compile(r"\\begin\{[^}]+\}")
_LATEX_ENV_END = re.compile(r"\\end\{[^}]+\}")
_LATEX_MATH = re.compile(r"\$[^$]*\$|\$\$[^$]*\$\$", re.DOTALL)
_LATEX_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_VTT_TS = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->.*$", re.MULTILINE)
_VTT_CUE_ID = re.compile(r"^\d+\s*$", re.MULTILINE)
_MULTI_WS = re.compile(r"\s+")


def strip_latex(text: str) -> str:
    text = _LATEX_COMMENT.sub("", text)
    text = _LATEX_MATH.sub(" ", text)
    text = _LATEX_ENV_BEGIN.sub(" ", text)
    text = _LATEX_ENV_END.sub(" ", text)
    text = _LATEX_CMD.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = _MULTI_WS.sub(" ", text)
    return text.strip()


def strip_vtt(text: str) -> str:
    lines = text.splitlines()
    keep: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("WEBVTT") or s.startswith("NOTE"):
            continue
        if _VTT_TS.match(s) or _VTT_CUE_ID.match(s):
            continue
        keep.append(s)
    return _MULTI_WS.sub(" ", " ".join(keep)).strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def sentences(text: str, min_words: int = 6, max_words: int = 40) -> list[str]:
    out = []
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        wc = len(s.split())
        if min_words <= wc <= max_words:
            out.append(s)
    return out


# ---------------------- corpus walking ----------------------

def iter_source_files(max_files: int | None = None):
    files: list[Path] = []
    for p in CORPUS_ROOT.rglob("*_codex.tex"):
        if "_codex_independent" in p.name:
            continue
        files.append(p)
    for p in CORPUS_ROOT.rglob("*.vtt"):
        files.append(p)
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    return files


def load_text(path: Path) -> str:
    raw = path.read_text(errors="replace")
    if path.suffix == ".tex":
        return strip_latex(raw)
    if path.suffix == ".vtt":
        return strip_vtt(raw)
    return raw


# ---------------------- vllm client ----------------------

def _post(payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        VLLM_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def extract_triples_one(sentence: str) -> list[dict]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(sentence=sentence)},
        ],
        "temperature": 0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }
    # Some vllm builds don't support array-only response_format.
    # Fall back if the server rejects it.
    try:
        r = _post(payload)
    except Exception:
        payload.pop("response_format", None)
        r = _post(payload)
    content = r["choices"][0]["message"]["content"].strip()
    # Strip potential markdown fences.
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    # Accept either a bare array or {"triples": [...]}.
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        # {"triples": [...]} / {"data": [...]} etc.
        inner = None
        for key in ("triples", "data", "result", "items", "relations"):
            if key in parsed and isinstance(parsed[key], list):
                inner = parsed[key]
                break
        if inner is not None:
            parsed = inner
        else:
            # Model returned a single-triple object — wrap it.
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        s = str(item.get("s") or item.get("subject") or "").strip()
        r = str(item.get("r") or item.get("relation") or "").strip()
        o = str(item.get("o") or item.get("object") or "").strip()
        if s and r and o:
            out.append({"s": s, "r": r, "o": o})
    return out


# ---------------------- orchestration ----------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT,
                    help="root directory to walk for source files; "
                         "may also be set via the LYCEUM_TRIPLE_CORPUS "
                         "environment variable")
    ap.add_argument("--max-sentences", type=int, default=500)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ns = ap.parse_args()

    if not ns.corpus_root or not Path(ns.corpus_root).is_dir():
        ap.error("--corpus-root is required (or set $LYCEUM_TRIPLE_CORPUS); "
                 "no default corpus is shipped with the repository.")

    global CORPUS_ROOT
    CORPUS_ROOT = Path(ns.corpus_root)
    files = iter_source_files(ns.max_files)
    print(f"sources: {len(files)} files", flush=True)

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    processed = 0
    t0 = time.perf_counter()

    with open(ns.out, "w") as fout:
        for f in files:
            if processed >= ns.max_sentences:
                break
            try:
                text = load_text(f)
            except Exception as exc:
                print(f"skip {f.name}: {exc}", flush=True)
                continue
            for sent in sentences(text):
                if processed >= ns.max_sentences:
                    break
                processed += 1
                try:
                    triples = extract_triples_one(sent)
                except Exception as exc:
                    print(f"vllm error on sentence {processed}: {exc}", flush=True)
                    continue
                for t in triples:
                    fout.write(json.dumps({
                        "file": f.relative_to(CORPUS_ROOT).as_posix(),
                        "sentence": sent,
                        **t,
                    }) + "\n")
                    written += 1
                if processed % 25 == 0:
                    el = time.perf_counter() - t0
                    print(f"  {processed} sentences, {written} triples, {el:.0f}s ({processed/el:.2f} sent/s)", flush=True)

    el = time.perf_counter() - t0
    print(f"DONE: {processed} sentences → {written} triples in {el:.0f}s, written to {ns.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
