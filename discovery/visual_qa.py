"""Render one minimal scene per relation, rasterise to PNG via headless
Chrome, and ask Qwen2.5-VL whether the visual encoding is clear.

Writes discovery/visual_qa.md with the verdicts.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from svggpt.pipeline import run_pipeline  # noqa: E402

OUT = Path(__file__).parent / "visual_qa.md"
VL_URL = "http://127.0.0.1:8001/v1/chat/completions"
VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

CASES: list[tuple[str, str]] = [
    ("causes",       "A causes B."),
    ("sequence",     "Wake up then drink coffee."),
    ("part_of",      "The arm is part of the body."),
    ("contains",     "The box contains a toy."),
    ("attribute_of", "The body has a heart."),
    ("similar_to",   "A dolphin is similar to a whale."),
    ("opposes",      "Heat opposes cold."),
    ("instance_of",  "A dolphin is an example of a mammal."),
    ("used_for",     "The algorithm uses a binary heap."),
    ("requires",     "Merge sort requires additional memory."),
    ("reduces_to",   "The problem reduces to sorting."),
    ("measures",     "The perimeter equals four times the side."),
]

_PROMPT_USER = (
    "This diagram is intended to depict the semantic relation "
    "\"{relation}\" between the two concepts drawn. Answer two things "
    "briefly:\n"
    "(A) Is the visual encoding between the two concepts (line style, "
    "arrow, marker, position) distinctive and does it intuitively "
    "suggest \"{relation}\" rather than some other relation?\n"
    "(B) Are there any overlapping shapes, text that overlaps another "
    "text or is obscured by a shape, or labels that are cut off?\n"
    "Give one sentence for (A) and one for (B). Then end on exactly "
    "one of:\n"
    "  VERDICT: YES     (A is clear and B reports no overlaps)\n"
    "  VERDICT: NO-RELATION   (A fails)\n"
    "  VERDICT: NO-OVERLAP    (B reports an overlap/occlusion)\n"
    "  VERDICT: NO-BOTH       (both problems)"
)


def svg_to_png(svg: str, out_path: Path, width: int = 1200, height: int = 800) -> None:
    with NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(svg)
        svg_path = f.name
    with TemporaryDirectory() as td:
        subprocess.run(
            [
                "google-chrome",
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                f"--user-data-dir={td}",
                f"--screenshot={out_path}",
                f"--window-size={width},{height}",
                f"file://{svg_path}",
            ],
            check=True,
            capture_output=True,
        )
    Path(svg_path).unlink(missing_ok=True)


def _vl_describe(png_path: Path, relation: str) -> str:
    image_b64 = base64.b64encode(png_path.read_bytes()).decode()
    payload = {
        "model": VL_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise visual design reviewer."},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text",
                 "text": _PROMPT_USER.format(relation=relation)},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 220,
    }
    req = urllib.request.Request(
        VL_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"].strip()


def _wait_for_vl() -> None:
    print("waiting for Qwen-VL on :8001 ...", flush=True)
    while True:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8001/v1/models", timeout=3) as r:
                if r.status == 200:
                    print("Qwen-VL ready.", flush=True)
                    return
        except Exception:
            time.sleep(5)


def main() -> int:
    _wait_for_vl()
    rows = []
    for rel, sent in CASES:
        print(f"\n=== {rel} ===\n  sentence: {sent}", flush=True)
        try:
            result = run_pipeline(sent)
        except Exception as exc:
            rows.append((rel, sent, f"pipeline failed: {exc}"))
            continue
        png = OUT.parent / f"qa_{rel}.png"
        svg_to_png(result.svg, png)
        try:
            verdict = _vl_describe(png, rel)
        except Exception as exc:
            verdict = f"VL error: {exc}"
        print(verdict, flush=True)
        rows.append((rel, sent, verdict))

    def _icon(v: str) -> str:
        if "VERDICT: YES" in v:
            return "✅"
        if "NO-BOTH" in v:
            return "❌❌"
        if "NO-RELATION" in v:
            return "❌rel"
        if "NO-OVERLAP" in v:
            return "❌ovl"
        return "❔"

    with open(OUT, "w") as f:
        f.write("# Visual QA — Qwen2.5-VL verdicts per relation\n\n")
        f.write("Each row: relation, source sentence, Qwen-VL review "
                "covering (A) relation clarity and (B) overlap / occlusion.\n\n")
        for rel, sent, verdict in rows:
            f.write(f"## {_icon(verdict)} `{rel}`\n\n")
            f.write(f"*Source:* {sent}\n\n")
            f.write(f"*Qwen-VL:* {verdict}\n\n")
            f.write(f"![{rel}](qa_{rel}.png)\n\n")
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
