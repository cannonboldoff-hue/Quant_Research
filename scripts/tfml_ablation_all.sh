#!/usr/bin/env bash
# Indicator-family ablation (see scripts/tfml_ablation.py). The "all"-features reference runs
# on every strategy (done once); the single-family blocks run on 15 representative strategies.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
J=${1:-5}
if [ "$("$PY" scripts/tfml_missing.py ablation_all | wc -w)" != "0" ]; then
  echo "### $(date) ablation all"; "$PY" scripts/tfml_ablation.py --block all --n-jobs "$J"
fi
for b in context_only momentum volatility trend oscillator_channel other; do
  echo "### $(date) ablation $b"
  "$PY" scripts/tfml_ablation.py --block "$b" --n-jobs "$J" --subset
done
echo "### ablation done"
