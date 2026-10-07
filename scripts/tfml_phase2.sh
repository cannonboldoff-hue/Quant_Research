#!/usr/bin/env bash
# Second phase (after scripts/tfml_run_all.sh): leakage null test, placebo controls,
# indicator-family ablation, then data validation, report, paper and notebook.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
J=${1:-6}
echo "### $(date) null test"
"$PY" scripts/tfml_null_test.py --n-instruments 60 --n-jobs "$J"
for cfg in benchmark robust_holdout crypto_intraday_4h crypto_intraday_1h robust_single_stocks robust_rolling; do
  echo "### $(date) placebo $cfg"
  "$PY" scripts/tfml_placebo.py --config configs/tfml/$cfg.yaml --seeds 20 --n-jobs "$J"
done
bash scripts/tfml_ablation_all.sh "$J"
echo "### $(date) report"
"$PY" scripts/tfml_validate_data.py
"$PY" scripts/tfml_report.py
echo "### $(date) phase2 done"
