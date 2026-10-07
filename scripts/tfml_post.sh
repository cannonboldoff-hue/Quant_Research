#!/usr/bin/env bash
# Post-processing after scripts/tfml_run_all.sh: placebo controls, report, paper, notebook.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
J=${1:-6}
for cfg in benchmark robust_holdout crypto_intraday_4h crypto_intraday_1h robust_single_stocks robust_rolling; do
  "$PY" scripts/tfml_placebo.py --config configs/tfml/$cfg.yaml --seeds 20 --n-jobs "$J"
done
"$PY" scripts/tfml_validate_data.py
"$PY" scripts/tfml_report.py
"$PY" paper/build_paper.py
"$PY" scripts/tfml_make_notebook.py
