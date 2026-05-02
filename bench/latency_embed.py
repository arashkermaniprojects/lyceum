"""Warm-path latency benchmark with the local Qwen encoder enabled.

Assumes SEVIM_EMBED_MODEL is loadable via transformers on this machine.
Run with the venv that has torch+transformers installed.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sevim.embed import encode, is_available  # noqa: E402
from sevim.pipeline import run_pipeline  # noqa: E402


WARMUP_INPUTS = [
    "Gravity causes objects to fall.",
    "The arm is part of the body.",
]


def _percentile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(q * (len(s) - 1)))
    return s[i] * 1000.0


def main() -> int:
    if not is_available():
        print("embed.is_available() = False — cannot run embedding bench.", file=sys.stderr)
        return 1

    print("warming encoder ...", flush=True)
    t0 = time.perf_counter()
    for w in WARMUP_INPUTS:
        encode(w)
    print(f"warmup done ({time.perf_counter() - t0:.1f}s)\n", flush=True)

    sizes = [10, 50, 100]
    runs = 20
    print(f"{'|V|':>4} {'stage':>6} {'p50_ms':>8} {'p95_ms':>8} {'p99_ms':>8} {'mean_ms':>8}")
    for n in sizes:
        text = " ".join(f"node{i} causes node{i+1}." for i in range(0, n - 1, 2))
        totals: list[float] = []
        encs: list[float] = []
        for _ in range(runs):
            t_enc0 = time.perf_counter()
            _ = encode(text[:256])  # one warm encode call, standalone measurement
            encs.append(time.perf_counter() - t_enc0)

            t0 = time.perf_counter()
            run_pipeline(text)
            totals.append(time.perf_counter() - t0)

        print(
            f"{n:>4} {'encode':>6} {_percentile(encs, 0.5):>8.2f} "
            f"{_percentile(encs, 0.95):>8.2f} {_percentile(encs, 0.99):>8.2f} "
            f"{statistics.mean(encs) * 1000:>8.2f}"
        )
        print(
            f"{n:>4} {'total':>6} {_percentile(totals, 0.5):>8.2f} "
            f"{_percentile(totals, 0.95):>8.2f} {_percentile(totals, 0.99):>8.2f} "
            f"{statistics.mean(totals) * 1000:>8.2f}"
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
