#!/usr/bin/env bash
# Control-experiment matrix (after the ontology fix, D39). Each job: "<config> <seed> <model>".
# All runs are analyzed with the same observer model (sonnet) so observer differences don't masquerade as effects.
# Usage: scripts/run_controls.sh <parallel> <job-file>
set -u
cd "$(dirname "$0")/.."
PAR=${1:-4}
JOBS=${2:-scripts/controls.jobs}
grep -v '^#' "$JOBS" | grep -v '^$' | xargs -P "$PAR" -L 1 sh -c '
  cfg=$0; seed=$1; model=$2; out=runs/c2_${cfg}_${model}_s${seed}
  if [ ! -f $out/manifest.json ] || ! grep -q "\"status\": \"finished\"" $out/manifest.json 2>/dev/null; then
    .venv/bin/python -m backend.cli run --config configs/$cfg.yaml --set seed=$seed llm.model=$model llm.max_workers=4 --out $out > $out.log 2>&1
  fi
  .venv/bin/python -m backend.cli analyze $out --model sonnet > $out.analysis.log 2>&1
  echo "$out done"'
