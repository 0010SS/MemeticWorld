#!/usr/bin/env bash
# Run every experimental mode with the same seed and population, then analyze.
# Usage: scripts/run_experiments.sh [seed] [parallel]
set -u
cd "$(dirname "$0")/.."
SEED=${1:-42}
PAR=${2:-3}
CONDS="baseline social_reward perfect_memory high_noise event_rich"
printf "%s\n" $CONDS | xargs -P "$PAR" -I{} sh -c \
  ".venv/bin/python -m backend.cli run --config configs/{}.yaml --set seed=$SEED llm.max_workers=4 --out runs/{}_s$SEED --analyze > runs/{}_s$SEED.log 2>&1; echo '{} done'"
