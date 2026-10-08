#!/usr/bin/env bash
# Cap the ablation at 4 blocks (all, momentum, volatility, trend): when chain B starts the
# 5th block, stop it and mark chain B done so the final report can proceed.
cd "$(dirname "$0")/.."
until grep -q "ablation oscillator_channel" results/tfml/finish_b.log; do sleep 30; done
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tfml_kill_ablation.ps1
sleep 5
echo "### $(date) ablation capped at 4 blocks (oscillator_channel and other not run)" >> results/tfml/finish_b.log
echo "### chain B done" >> results/tfml/finish_b.log
