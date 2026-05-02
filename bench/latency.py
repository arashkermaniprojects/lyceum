"""Per-stage latency benchmark against the I2 budget (≤50 ms p95 incremental).

Synthetic inputs at 10/50/100/500 nodes. Reports p50/p95/p99 per stage.
Uses stdlib only — no numpy required.
"""
from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sevim.s1_parse import parse_text
from sevim.s2_extract import extract
from sevim.s3_map import map_visual
from sevim.s4_layout import layout
from sevim.s5_render import render


def _synthetic_text(n_nodes: int) -> str:
    """Build text that produces ~n_nodes distinct entities via causes chains."""
    pairs = []
    for i in range(0, n_nodes - 1, 2):
        pairs.append(f"node{i} causes node{i + 1}.")
    return " ".join(pairs)


@dataclass
class StageStats:
    name: str
    samples: list[float]

    def p(self, q: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        i = min(len(s) - 1, int(q * (len(s) - 1)))
        return s[i] * 1000.0  # → ms

    def mean_ms(self) -> float:
        return statistics.mean(self.samples) * 1000.0 if self.samples else 0.0


def bench_size(n_nodes: int, runs: int = 50) -> dict[str, StageStats]:
    text = _synthetic_text(n_nodes)
    stats = {name: StageStats(name, []) for name in ("S1", "S2", "S3", "S4", "S5", "total")}

    for _ in range(runs):
        t0 = time.perf_counter()
        tokens = parse_text(text)
        t1 = time.perf_counter()
        graph = extract(tokens)
        t2 = time.perf_counter()
        vg = map_visual(graph)
        t3 = time.perf_counter()
        pg = layout(vg)
        t4 = time.perf_counter()
        _svg = render(pg)
        t5 = time.perf_counter()

        stats["S1"].samples.append(t1 - t0)
        stats["S2"].samples.append(t2 - t1)
        stats["S3"].samples.append(t3 - t2)
        stats["S4"].samples.append(t4 - t3)
        stats["S5"].samples.append(t5 - t4)
        stats["total"].samples.append(t5 - t0)

    return stats


def print_report(sizes: list[int], runs: int = 50) -> None:
    print(f"{'n_nodes':>8} {'stage':>6} {'p50_ms':>8} {'p95_ms':>8} {'p99_ms':>8} {'mean_ms':>8}")
    for n in sizes:
        stats = bench_size(n, runs=runs)
        for stage in ("S1", "S2", "S3", "S4", "S5", "total"):
            s = stats[stage]
            print(f"{n:>8} {stage:>6} {s.p(0.5):>8.2f} {s.p(0.95):>8.2f} {s.p(0.99):>8.2f} {s.mean_ms():>8.2f}")
        print()


if __name__ == "__main__":
    print_report(sizes=[10, 50, 100, 500], runs=50)
