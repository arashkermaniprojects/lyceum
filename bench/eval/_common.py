"""Shared helpers for the eval harness.

Keep this file dependency-light (stdlib only) so each driver runs
without any local service.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "bench" / "eval"
RESULTS = EVAL_ROOT / "results"
TABLES = EVAL_ROOT / "tables"
RUBRICS = EVAL_ROOT / "rubrics"
LEDGER = EVAL_ROOT / "cost_ledger.json"


# Make `import narrator`, `import book`, `import serve`, `import viz` work
# regardless of cwd when a driver is invoked directly.
sys.path.insert(0, str(REPO_ROOT))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger_record(call: dict[str, Any]) -> None:
    """Append an API-call record to ``cost_ledger.json`` with cap enforcement."""
    led = read_json(LEDGER)
    spent = float(led.get("spent_usd", 0.0)) + float(call.get("usd", 0.0))
    cap = float(led.get("cap_usd", 0.0))
    if spent > cap:
        raise RuntimeError(
            f"cost ledger cap exceeded: spent={spent:.4f} usd "
            f"would exceed cap={cap:.4f} usd"
        )
    led["spent_usd"] = round(spent, 6)
    led.setdefault("calls", []).append({"ts": now_iso(), **call})
    write_json(LEDGER, led)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}\\%"


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("_", r"\_")
         .replace("#", r"\#")
         .replace("$", r"\$")
    )


def have_anthropic_key() -> bool:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    return bool(key) and key.startswith("sk-ant-")
