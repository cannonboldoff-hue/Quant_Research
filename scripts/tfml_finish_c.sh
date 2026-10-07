#!/usr/bin/env bash
# Waits for chains A and B, then validation + report.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-.venv/Scripts/python.exe}
until grep -q "chain A done" results/tfml/finish_a.log 2>/dev/null && grep -q "chain B done" results/tfml/finish_b.log 2>/dev/null; do sleep 60; done
echo "### $(date) report"; "$PY" scripts/tfml_validate_data.py; "$PY" scripts/tfml_report.py
echo "### finish done"
