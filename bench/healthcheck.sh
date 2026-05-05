#!/usr/bin/env bash
# Periodic health check for SeVim. Writes one status line per run to the log.
# No Claude involved — pure local shell + Python.
#
# Install as cron (every 15 min):
#   crontab -e
#   */15 * * * * /path/to/repo/bench/healthcheck.sh
#
# The Python interpreter is taken from $LYCEUM_PY if set, otherwise the
# repo-local virtual environment is used.  Override either via cron's
# environment block.

set -eu
cd "$(dirname "$0")/.."

VENV_PY="${LYCEUM_PY:-$(pwd)/.venv/bin/python3}"
LOG="${LYCEUM_HEALTH_LOG:-/tmp/lyceum_health.log}"
TS="$(date -Iseconds)"

status_line() {
  printf '%s %s\n' "$TS" "$1" | tee -a "$LOG"
}

# 1. Test suite (encoder-off for speed).
tests_pass=$(SEVIM_DISABLE_EMBED=1 "$VENV_PY" -c "
import sys; sys.path.insert(0, '.')
import tests.test_determinism, tests.test_layout, tests.test_embeddings, tests.test_relations, tests.test_routing, tests.test_property_determinism, tests.test_cosine_merge, tests.test_dep_parse, tests.test_render_per_relation
mods=[tests.test_determinism, tests.test_layout, tests.test_embeddings, tests.test_relations, tests.test_routing, tests.test_property_determinism, tests.test_cosine_merge, tests.test_dep_parse, tests.test_render_per_relation]
fails=0; total=0
for m in mods:
    for name in sorted(dir(m)):
        if name.startswith('test_'):
            total+=1
            try: getattr(m,name)()
            except Exception: fails+=1
print(f'{total-fails}/{total}')
" 2>/dev/null)

# 2. One-clause latency (encoder-off).
p95=$(SEVIM_DISABLE_EMBED=1 "$VENV_PY" -c "
import sys, time; sys.path.insert(0, '.')
from sevim.pipeline import run_pipeline
for _ in range(3): run_pipeline('A causes B.')
s=[]
for _ in range(20):
    t=time.perf_counter(); run_pipeline('A causes B.'); s.append((time.perf_counter()-t)*1000)
s.sort(); print(f'{s[18]:.2f}')
" 2>/dev/null)

# 3. vllm :8000 health.
if curl -sf -o /dev/null -m 3 http://127.0.0.1:8000/v1/models; then
  vllm8000="up"
else
  vllm8000="DOWN"
fi

# 4. Report.
status_line "tests=$tests_pass  clause_p95_ms=$p95  vllm_8000=$vllm8000"

# 5. Non-zero exit if anything is broken — so cron MAILTO / notifier can act.
[[ "$tests_pass" == 45/45 ]] || exit 1
[[ "$vllm8000" == "up" ]] || exit 2
