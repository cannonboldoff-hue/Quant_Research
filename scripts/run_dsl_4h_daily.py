"""Run just the new kama_dsl/lhp_dsl 4h+daily registry entries into the v2
campaign store, then reprint the full per-strategy summary (old + new rows).

Scoped to these 4 strategy_ids only -- run_campaign's markets= filter would
also re-run the existing (already-scored) crypto/forex 1m entries.

Run: python scripts/run_dsl_4h_daily.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from qresearch.campaign import ProvenanceStore, StrategyRegistry, run_campaign
from qresearch.stats import deflated_sharpe
from qresearch.utils.logging_config import get_logger

log = get_logger("qresearch.dsl_4h_daily")
CAMPAIGN_ID = "campaign_20260713_v2"
STRATEGY_IDS = ["forex_lhp_dsl_4h", "forex_lhp_dsl_daily", "crypto_kama_dsl_4h", "crypto_kama_dsl_daily"]


def main() -> None:
    registry = StrategyRegistry()
    subset = [(sid, registry.get(sid)) for sid in STRATEGY_IDS]

    t0 = time.perf_counter()
    result = run_campaign(subset, campaign_id=CAMPAIGN_ID, markets=["forex", "crypto"], n_jobs=1)
    elapsed = time.perf_counter() - t0
    log.info("dsl 4h/daily run finished in %.1fs", elapsed)

    if result.empty:
        log.error("no tasks ran -- check data/processed and configs/strategies.yaml")
        sys.exit(1)
    total_errors = sum(len(e) for e in result["errors"])
    if total_errors:
        log.warning("%d task-level errors -- sample:", total_errors)
        for e in result["errors"]:
            for msg in e[:3]:
                log.warning("  %s", msg)

    store = ProvenanceStore()
    trades = store.read(campaign_id=CAMPAIGN_ID)

    summary_rows = []
    for strategy_id, g in trades.groupby("strategy_id"):
        r = g["ret"].to_numpy(dtype=np.float64)
        n_trials = g.groupby("params").ngroups
        dsr = deflated_sharpe(r, n_trials=n_trials)
        summary_rows.append({
            "strategy_id": strategy_id, "n_trades": len(g), "n_tickers": g["ticker"].nunique(),
            "n_param_combos": n_trials, "win_rate": float((r > 0).mean()),
            "mean_ret": float(r.mean()), "deflated_sharpe": dsr,
        })
    summary = pd.DataFrame(summary_rows).sort_values("deflated_sharpe", ascending=False)

    out_path = ROOT / "data" / "campaign_summaries" / f"{CAMPAIGN_ID}_summary.csv"
    summary.to_csv(out_path, index=False)
    log.info("per-strategy summary -> %s", out_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
