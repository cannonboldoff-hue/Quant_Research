"""Part A of the drawdown-reduction plan: portfolio-level concurrent-risk cap.

The drawdown diagnostic found `leverage_sim.simulate()` has no concept of how
many positions are actually open at once -- it sorts trades by entry_time and
compounds them sequentially with no cap on aggregate exposure. The real trade
book has ~30 of 48 tickers open concurrently on average (up to 47), so fixed
1%-per-trade sizing means ~30% of equity at risk simultaneously with no limit.

This sweeps `max_concurrent` (see `leverage_sim.simulate`'s new param) on both
the all-48 baseline book and the best curated book found so far (annual
rolling rebalance, Sharpe 1.92 / real maxDD -47.4%), to see whether capping
concurrent positions reduces real drawdown without giving back too much
return.

Run: python scripts/campaign_risk_cap_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qresearch.optimize.finalist_selection import run_finalist_trades  # noqa: E402
from qresearch.portfolio.combine import portfolio_metrics, trades_to_daily_returns  # noqa: E402
from qresearch.risk.leverage_sim import _admission_mask, simulate  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402
import campaign_rolling_rebalance as crr  # noqa: E402 -- reuse the annual-rebalance curated book

log = get_logger("qresearch.campaign_risk_cap_sweep")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"
STRATEGY_ID, MARKET, TIMEFRAME = crr.STRATEGY_ID, crr.MARKET, crr.TIMEFRAME

MAX_CONCURRENT_SWEEP = [5, 10, 15, 20, None]  # None = uncapped (existing behavior)


def main() -> None:
    log.info("building all-48 baseline trade book")
    all_trades = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED)
    all_trades["entry_time"] = pd.to_datetime(all_trades["entry_time"])

    log.info("rebuilding annual rolling-rebalance curated book (best variant so far)")
    start = all_trades["entry_time"].min() + pd.DateOffset(years=crr.WARMUP_YEARS)
    end = all_trades["entry_time"].max()
    curated, baseline, _ = crr._run_frequency(all_trades, "annual", crr.REBALANCE_OFFSETS["annual"], start, end)

    rows = []
    for variant_name, trades in [("baseline_all48", baseline), ("curated_annual_rebalance", curated)]:
        for cap in MAX_CONCURRENT_SWEEP:
            sim = simulate(trades, leverage=100, sizing_mode="disciplined", max_concurrent=cap)
            rows.append({
                "variant": variant_name, "max_concurrent": cap if cap is not None else "uncapped",
                "n_trades_admitted": sim.n_trades, "n_skipped_capacity": sim.n_skipped_capacity,
                "position_fraction": sim.position_fraction, "final_equity": sim.final_equity,
                "max_drawdown": sim.max_drawdown, "ruined": sim.ruined,
                "ruin_probability_mc": sim.ruin_probability_mc,
            })

    summary = pd.DataFrame(rows).set_index(["variant", "max_concurrent"])
    print("\n=== concurrent-risk-cap sweep (disciplined sizing, 1% risk/trade, leverage=100x) ===")
    print(summary.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "risk_cap_sweep.csv")
    log.info("wrote %s", OUT / "risk_cap_sweep.csv")

    # Sharpe/CAGR of the leading candidate (curated + cap=10) on the exact
    # admitted-trade subset, not just leverage_sim's scalar maxDD/final_equity
    # -- confirms the drawdown cut isn't just a sizing-model artifact.
    best_cap = 10
    curated_sorted = curated.sort_values("entry_time")
    admitted = _admission_mask(curated_sorted["entry_time"], curated_sorted["exit_time"], best_cap)
    capped_trades = curated_sorted.loc[admitted]
    pm_uncapped = portfolio_metrics(trades_to_daily_returns(curated_sorted))
    pm_capped = portfolio_metrics(trades_to_daily_returns(capped_trades))
    print(f"\n=== curated_annual_rebalance: Sharpe/CAGR, uncapped vs max_concurrent={best_cap} ===")
    print(pd.DataFrame({"uncapped": pm_uncapped, f"cap_{best_cap}": pm_capped}).to_string())


if __name__ == "__main__":
    main()
