#!/usr/bin/env bash
# Indicator-family ablation runs (see scripts/tfml_ablation.py).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
J=${1:-6}
for b in all momentum volatility trend oscillator_channel other; do
  echo "### $(date) ablation $b"
  "$PY" scripts/tfml_ablation.py --block "$b" --n-jobs "$J"
done
