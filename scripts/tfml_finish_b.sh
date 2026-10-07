#!/usr/bin/env bash
# Chain B: indicator-family ablation (independent of chain A).
set -u; cd "$(dirname "$0")/.."
bash scripts/tfml_ablation_all.sh "${1:-5}"
echo "### chain B done"
