"""Part C of the drawdown-reduction plan: trailing stop.

`risk/stops.py:37 trailing_stop` was defined but never wired into any
backtest path -- only the fixed ATR stop/target is used. This tests the new
`backtest_trades_trailing_fast` (ratcheting ATR trailing stop, no fixed
take-profit) against the fixed-stop baseline, on: (1) all-48 baseline,
(2) the annual rolling-rebalance curated set (Parts A/B's "best curated
book"), and (3) that same curated set WITH the Part B market-regime overlay
stacked on top, to see whether all three fixes compound.

Run: python scripts/campaign_trailing_stop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qresearch.backtest.engine import backtest_trades_trailing_fast  # noqa: E402
from qresearch.optimize.finalist_selection import run_finalist_trades  # noqa: E402
from qresearch.portfolio.combine import portfolio_metrics, trades_to_daily_returns  # noqa: E402
from qresearch.risk.leverage_sim import simulate  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402
import campaign_market_regime_overlay as cmro  # noqa: E402 -- reuse the composite market mask
import campaign_rolling_rebalance as crr  # noqa: E402 -- reuse the annual-rebalance curated selection

log = get_logger("qresearch.campaign_trailing_stop")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"
STRATEGY_ID, MARKET, TIMEFRAME = crr.STRATEGY_ID, crr.MARKET, crr.TIMEFRAME
TRAILING_STOP_GRID = {"sl_mult": [1.0, 1.5, 2.0], "trail_mult": [1.5, 2.0, 3.0]}


def _curated_from(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    start = trades["entry_time"].min() + pd.DateOffset(years=crr.WARMUP_YEARS)
    end = trades["entry_time"].max()
    curated, _, _ = crr._run_frequency(trades, "annual", crr.REBALANCE_OFFSETS["annual"], start, end)
    return curated


def main() -> None:
    log.info("building all-48 baseline (fixed stop)")
    baseline_fixed = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED)
    log.info("building all-48 baseline (trailing stop)")
    baseline_trailing = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED,
                                             backtest_fn=backtest_trades_trailing_fast,
                                             stop_grid=TRAILING_STOP_GRID)

    log.info("curating (annual rebalance) both fixed and trailing baselines")
    curated_fixed = _curated_from(baseline_fixed)
    curated_trailing = _curated_from(baseline_trailing)

    log.info("stacking market-regime overlay on top of curated+trailing (full 3-fix stack)")
    market_mask = cmro._build_market_mask()
    curated_tickers = set(curated_trailing["ticker"].unique())
    full_stack = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED, tickers=curated_tickers,
                                      market_mask=market_mask, backtest_fn=backtest_trades_trailing_fast,
                                      stop_grid=TRAILING_STOP_GRID)
    full_stack = _curated_from(full_stack)

    variants = [
        ("baseline_fixed_stop", baseline_fixed), ("baseline_trailing_stop", baseline_trailing),
        ("curated_fixed_stop", curated_fixed), ("curated_trailing_stop", curated_trailing),
        ("curated_market_gate_trailing_stop", full_stack),
    ]
    # Trailing-stop trades are fatter-tailed (no fixed target caps the ride)
    # -- the UNCAPPED sequential leverage_sim compounds thousands of trades
    # one at a time as if they were never concurrent, wildly overstating real
    # achievable growth. Part A's max_concurrent cap (reused, not re-derived)
    # gives the honest number for anything using the trailing engine.
    RISK_CAP = 10
    rows = []
    for label, trades in variants:
        if trades.empty:
            continue
        daily = trades_to_daily_returns(trades)
        pm = portfolio_metrics(daily)
        sim_uncapped = simulate(trades, leverage=100, sizing_mode="disciplined")
        sim_capped = simulate(trades, leverage=100, sizing_mode="disciplined", max_concurrent=RISK_CAP)
        stop_rate = (trades["reason"] == "stop").mean()
        rows.append({
            "variant": label, "n_tickers": trades["ticker"].nunique(), "n_trades": len(trades),
            "stop_rate": stop_rate, **pm,
            "uncapped_max_drawdown": sim_uncapped.max_drawdown, "uncapped_final_equity": sim_uncapped.final_equity,
            f"cap{RISK_CAP}_max_drawdown": sim_capped.max_drawdown, f"cap{RISK_CAP}_final_equity": sim_capped.final_equity,
        })

    summary = pd.DataFrame(rows).set_index("variant")
    print("\n=== fixed stop vs trailing stop (baseline, curated, full 3-fix stack) ===")
    print(summary.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "trailing_stop_comparison.csv")
    log.info("wrote %s", OUT / "trailing_stop_comparison.csv")


if __name__ == "__main__":
    main()
