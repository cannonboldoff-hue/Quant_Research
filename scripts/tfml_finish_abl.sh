#!/usr/bin/env bash
# Corrected ablation (single-family blocks on 15 strategies) then validation + report.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-.venv/Scripts/python.exe}
bash scripts/tfml_ablation_all.sh "${1:-11}"
echo "### $(date) report"; "$PY" scripts/tfml_validate_data.py; "$PY" scripts/tfml_report.py
echo "### finish done"
