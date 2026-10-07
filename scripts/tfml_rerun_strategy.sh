#!/usr/bin/env bash
# Re-run one strategy in every run (experiments + placebo), e.g. after an indicator fix.
#   bash scripts/tfml_rerun_strategy.sh mcginley_14 [n_jobs]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
S=$1; J=${2:-6}
for cfg in benchmark robust_holdout crypto_intraday_4h crypto_intraday_1h robust_single_stocks robust_rolling; do
  echo "### $(date) rerun $S in $cfg"
  "$PY" scripts/tfml_run.py --config configs/tfml/$cfg.yaml --n-jobs "$J" --strategies "$S"
  "$PY" scripts/tfml_placebo.py --config configs/tfml/$cfg.yaml --seeds 20 --n-jobs "$J" --strategies "$S"
done
echo "### $(date) rerun $S done"
