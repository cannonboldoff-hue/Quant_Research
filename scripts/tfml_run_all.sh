#!/usr/bin/env bash
# Reproduce every experiment reported in the paper (sequential; each run is parallel inside).
#   bash scripts/tfml_run_all.sh [n_jobs]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
J=${1:-6}
for cfg in benchmark robust_holdout crypto_intraday_4h crypto_intraday_1h robust_single_stocks robust_rolling; do
  echo "### $(date) start $cfg"
  "$PY" scripts/tfml_run.py --config configs/tfml/$cfg.yaml --n-jobs "$J"
  echo "### $(date) end $cfg (exit $?)"
done
