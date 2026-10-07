#!/usr/bin/env bash
# Resume the experiment sequence after an interruption: for every config, run only the
# strategies missing from the registry; then phase 2. Fewer workers to limit memory use.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
J=${1:-4}
# if an experiment process is still running (e.g. it survived the interruption), let it finish
while [ "$(powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tfml_running.ps1 | tr -d '\r')" != "0" ]; do sleep 60; done
for cfg in benchmark robust_holdout crypto_intraday_4h crypto_intraday_1h robust_single_stocks robust_rolling; do
  rid=$(grep '^run_id:' configs/tfml/$cfg.yaml | awk '{print $2}')
  missing=$("$PY" scripts/tfml_missing.py "$rid" | tr -d '\r')
  if [ -z "$missing" ]; then echo "### complete: $cfg"; continue; fi
  echo "### $(date) start $cfg ($(echo $missing | wc -w) strategies)"
  "$PY" scripts/tfml_run.py --config configs/tfml/$cfg.yaml --n-jobs "$J" --strategies $missing
  echo "### $(date) end $cfg (exit $?)"
done
bash scripts/tfml_phase2.sh "$J"
