"""Command-line interface for SeVim.

Usage:
    sevim <input_file>           # read text from file
    sevim -                      # read text from stdin
    sevim <input_file> --out out/diagram

Outputs three files (default prefix "out"):
    out.svg          canonical SVG diagram
    out.ir.json      scene graph in JSON (nodes + edges + revision counter)
    out.trace.json   per-stage diagnostic trace
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sevim",
        description="Convert natural-language text to deterministic SVG diagrams.",
    )
    ap.add_argument(
        "input",
        help="Path to a plain-text input file, or '-' to read from stdin.",
    )
    ap.add_argument(
        "--out",
        default="out",
        help="Output path prefix (default: 'out').  "
             "Suffixes .svg / .ir.json / .trace.json are appended automatically.",
    )
    ns = ap.parse_args(argv)

    # Read input text.
    text = sys.stdin.read() if ns.input == "-" else Path(ns.input).read_text()

    result = run_pipeline(text)

    # Write outputs — create parent directories if they don't exist yet.
    out = Path(ns.out)
    if out.parent and str(out.parent) not in (".", ""):
        out.parent.mkdir(parents=True, exist_ok=True)

    out.with_suffix(".svg").write_text(result.svg)

    out.with_suffix(".ir.json").write_text(json.dumps({
        "nodes": [asdict(n) for n in result.graph.nodes],
        "edges": [asdict(e) for e in result.graph.edges],
        "revision": result.graph.revision,
    }, indent=2))

    out.with_suffix(".trace.json").write_text(json.dumps([
        {"stage": t.stage, "message": t.message, "refs": t.refs}
        for t in result.trace
    ], indent=2))

    print(f"SVG:   {out.with_suffix('.svg')}")
    print(f"IR:    {out.with_suffix('.ir.json')}")
    print(f"Trace: {out.with_suffix('.trace.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
