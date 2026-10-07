#!/usr/bin/env bash
# Chain A: remaining placebos, then McGinley rerun (must follow placebos of the same runs).
set -u; cd "$(dirname "$0")/.."; PY=${PY:-.venv/Scripts/python.exe}; J=${1:-6}
for cfg in robust_single_stocks robust_rolling; do
  echo "### $(date) placebo $cfg"; "$PY" scripts/tfml_placebo.py --config configs/tfml/$cfg.yaml --seeds 20 --n-jobs "$J"
done
bash scripts/tfml_rerun_strategy.sh mcginley_14 "$J"
echo "### chain A done"
