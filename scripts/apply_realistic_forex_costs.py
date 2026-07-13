"""Recompute forex net returns with realistic per-pair spread+swap costs
(backtest/costs.py) instead of the flat cross-asset 7bps fee baked into the
campaign run, and compare the two -- the direct test of whether "drop forex"
was right or an artifact of overcharging it.

No backtest re-run needed: trade rows already carry entry_price/exit_price/
entry_time/exit_time, so gross return is recoverable as ``ret + flat_fee``.

Run: python scripts/apply_realistic_forex_costs.py [campaign_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.backtest.costs import realistic_net_ret  # noqa: E402
from qresearch.campaign import ProvenanceStore  # noqa: E402
from qresearch.config.settings import get_settings  # noqa: E402
from qresearch.stats import deflated_sharpe  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.realistic_forex_costs")


def main() -> None:
    campaign_id = sys.argv[1] if len(sys.argv) > 1 else "campaign_20260713_v2"
    store = ProvenanceStore()
    trades = store.read(campaign_id=campaign_id)
    trades = trades[trades["market"] == "forex"].copy()
    if trades.empty:
        log.error("no forex trades found for campaign_id=%s", campaign_id)
        sys.exit(1)

    flat_fee = (get_settings().fee_bps + get_settings().slippage_bps) / 1e4
    trades["gross"] = trades["ret"] + flat_fee
    trades["ret_realistic"] = realistic_net_ret(trades)

    rows = []
    for (strategy_id, params), g in trades.groupby(["strategy_id", "params"]):
        r_flat = g["ret"].to_numpy(dtype=np.float64)
        r_real = g["ret_realistic"].to_numpy(dtype=np.float64)
        n_trials = trades[trades["strategy_id"] == strategy_id].groupby("params").ngroups
        rows.append({
            "strategy_id": strategy_id, "params": params, "n_trades": len(g),
            "n_tickers": g["ticker"].nunique(),
            "flat_mean_ret": r_flat.mean(), "flat_win_rate": (r_flat > 0).mean(),
            "realistic_mean_ret": r_real.mean(), "realistic_win_rate": (r_real > 0).mean(),
            "realistic_dsr": deflated_sharpe(r_real, n_trials=n_trials),
        })
    detail = pd.DataFrame(rows)

    # per-strategy rollup: best realistic combo, not the blended average
    # (same "look for the best combo, not the mean" logic as the v2/walk-forward gate).
    best = detail.sort_values("realistic_mean_ret", ascending=False).groupby("strategy_id").head(3)
    print("=== per-strategy: top 3 combos by realistic net mean_ret ===")
    print(best.to_string(index=False))

    strategy_summary = detail.groupby("strategy_id").agg(
        n_combos=("params", "count"),
        flat_mean_ret_blended=("flat_mean_ret", "mean"),
        realistic_mean_ret_blended=("realistic_mean_ret", "mean"),
        realistic_mean_ret_best=("realistic_mean_ret", "max"),
    ).reset_index()
    print("\n=== flat-fee vs realistic-cost, blended and best combo ===")
    print(strategy_summary.to_string(index=False))

    out_path = ROOT / "data" / "campaign_summaries" / f"{campaign_id}_forex_realistic_costs.csv"
    detail.to_csv(out_path, index=False)
    log.info("detail -> %s", out_path)


if __name__ == "__main__":
    main()
