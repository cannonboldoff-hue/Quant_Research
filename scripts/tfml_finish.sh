#!/usr/bin/env bash
# Final stretch after an interruption: remaining placebos -> McGinley rerun (post-fix)
# -> indicator-family ablation -> data validation -> report. Single sequential chain.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
J=${1:-9}
for cfg in robust_single_stocks robust_rolling; do
  echo "### $(date) placebo $cfg"
  "$PY" scripts/tfml_placebo.py --config configs/tfml/$cfg.yaml --seeds 20 --n-jobs "$J"
done
bash scripts/tfml_rerun_strategy.sh mcginley_14 "$J"
bash scripts/tfml_ablation_all.sh "$J"
echo "### $(date) report"
"$PY" scripts/tfml_validate_data.py
"$PY" scripts/tfml_report.py
echo "### finish done"
