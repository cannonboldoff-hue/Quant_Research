"""Walk-forward gate for the v2 alpha-search finalists (Lever 3).

v2 campaign showed 4 (market, timeframe) JMA configs with a positive net
mean_ret for their BEST individual (fast, slow, sl_mult, tp_mult) combo:
commodities/indices at daily and 4h. That's still in-sample -- this script
re-optimizes per chronological fold and evaluates on the next (anchored
walk-forward, reusing optimize.walk_forward.split_data_by_periods), so a
"lead" is only real if it survives out-of-sample.

walk_forward_optimize() isn't reused directly: it only sweeps the signal's
param_grid and hardcodes sl/tp at 1.5/3.0, so it can't validate the sl/tp
grid finding from Lever 2. This sweeps both, same grids as the campaign
runner (STOP_GRID) and registry (fast/slow), via the fast numba engine.

The fold-by-fold selection/OOS-eval logic itself lives in
``qresearch.optimize.finalist_selection`` (extracted so
``scripts/build_portfolio.py`` can reuse the exact same walk-forward-selected
trades to build the portfolio-level Sharpe/CAGR, instead of re-deriving a
second, possibly-diverging selection).

Run: python scripts/walk_forward_finalists.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.optimize.finalist_selection import run_finalist_trades  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.walk_forward_finalists")
PROCESSED = ROOT / "data" / "processed"

FINALISTS = [
    ("commodities_jma_atr_daily", "commodities", "daily"),
    ("indices_jma_atr_daily", "indices", "daily"),
    ("commodities_jma_atr_4h", "commodities", "4h"),
    ("indices_jma_atr_4h", "indices", "4h"),
    # v2.1 (2026-07-13): crypto_jma_atr_daily's blended mean_ret (+156bps) is
    # the strongest of any strategy/timeframe so far; crypto_jma_atr_4h's best
    # combo (+61bps, n=1653) also clears cost with room. Both added here.
    ("crypto_jma_atr_daily", "crypto", "daily"),
    ("crypto_jma_atr_4h", "crypto", "4h"),
    # v2.1 (2026-07-13): realistic per-pair spread/swap (backtest/costs.py)
    # flipped these from net-negative-under-flat-fee to net-positive -- gated
    # here using the SAME realistic cost model for both fold selection and
    # OOS evaluation (not just post-hoc), so the walk-forward result reflects
    # what a live account would actually pay, not the flat 7bps placeholder.
    ("forex_jma_atr_4h", "forex", "4h"),
    ("forex_jma_atr_daily", "forex", "daily"),
    # v2.1: indian_equities_jma_atr_daily (+16.3bps blended, dsr=1.0) and
    # _4h (+8.8bps blended, dsr=1.0, unlike every other market's 4h so far).
    ("indian_equities_jma_atr_daily", "indian_equities", "daily"),
    ("indian_equities_jma_atr_4h", "indian_equities", "4h"),
]


def run_finalist(strategy_id: str, market: str, timeframe: str) -> pd.DataFrame:
    """Per ticker-fold OOS summary (n_trades/mean_ret/win_rate), regrouped
    from the shared per-trade walk-forward output."""
    trades = run_finalist_trades(strategy_id, market, timeframe, PROCESSED)
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["ticker", "fold"])
        .agg(fast=("fast", "first"), slow=("slow", "first"),
             sl_mult=("sl_mult", "first"), tp_mult=("tp_mult", "first"),
             n_trades=("ret", "size"), mean_ret=("ret", "mean"),
             win_rate=("ret", lambda r: float((r > 0).mean())))
        .reset_index()
    )


def main() -> None:
    summary_rows = []
    for strategy_id, market, timeframe in FINALISTS:
        log.info("walk-forward: %s (%s/%s)", strategy_id, market, timeframe)
        res = run_finalist(strategy_id, market, timeframe)
        if res.empty:
            log.warning("%s: no fold produced a qualifying OOS result", strategy_id)
            continue
        pos_frac = (res["mean_ret"] > 0).mean()
        summary_rows.append({
            "strategy_id": strategy_id, "n_ticker_folds": len(res),
            "oos_positive_frac": pos_frac, "oos_mean_ret": res["mean_ret"].mean(),
            "oos_mean_win_rate": res["win_rate"].mean(),
        })
        print(f"\n--- {strategy_id}: per ticker-fold OOS ---")
        print(res.to_string(index=False))

    summary = pd.DataFrame(summary_rows).sort_values("oos_positive_frac", ascending=False)
    print("\n=== walk-forward gate summary ===")
    print(summary.to_string(index=False))
    out_path = ROOT / "data" / "campaign_summaries" / "walk_forward_finalists_summary.csv"
    summary.to_csv(out_path, index=False)
    log.info("summary -> %s", out_path)


if __name__ == "__main__":
    main()
