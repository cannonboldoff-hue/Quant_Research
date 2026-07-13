"""Definitive nested holdout for the FULL drawdown-reduction pipeline.

Every structural decision made in this plan so far -- use the trailing-stop
engine, skip the market-gate stack, freeze max_concurrent=10 -- was picked by
looking at results on the SAME 2016-2024 book being reported. That's exactly
the pattern that made the static one-shot sector pick look great and then
fail a real holdout (campaign_sector_holdout_validation.py: Sharpe 1.68 in-
sample vs 0.84 on data it never touched). The rolling sector rebalance fixed
that specific bias (selection always strictly precedes trading), but the
CHOICE of trailing-vs-fixed engine, market-gate on/off, and max_concurrent
value were never tested the same way.

This freezes every one of those choices using ONLY pre-2021-12-03 data (the
same 70/30 calendar split as the sector holdout test), then grades the frozen
recipe PURELY on 2021-2024 data none of those choices ever saw.

Run: python scripts/campaign_full_pipeline_holdout_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qresearch.backtest.engine import backtest_trades_fast, backtest_trades_trailing_fast  # noqa: E402
from qresearch.optimize.finalist_selection import run_finalist_trades  # noqa: E402
from qresearch.portfolio.combine import portfolio_metrics, trades_to_daily_returns  # noqa: E402
from qresearch.risk.leverage_sim import _admission_mask, simulate  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402
import campaign_market_regime_overlay as cmro  # noqa: E402
import campaign_rolling_rebalance as crr  # noqa: E402

log = get_logger("qresearch.campaign_full_pipeline_holdout_validation")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"
STRATEGY_ID, MARKET, TIMEFRAME = crr.STRATEGY_ID, crr.MARKET, crr.TIMEFRAME
TRAILING_STOP_GRID = {"sl_mult": [1.0, 1.5, 2.0], "trail_mult": [1.5, 2.0, 3.0]}
CAP_SWEEP = [5, 10, 15, 20, None]
SELECTION_FRAC = 0.7  # same convention as campaign_sector_holdout_validation.py


def _sharpe_capped(trades: pd.DataFrame, cap) -> float:
    """Naive (unsized) Sharpe on whichever trades survive a max_concurrent
    admission filter -- used to pick a cap using ONLY the metric the frozen
    decision is allowed to see (Sharpe on daily returns), independent of
    leverage_sim's separate sequential-compounding maxDD/equity numbers."""
    if trades.empty:
        return -999.0
    if cap is None:
        subset = trades
    else:
        sorted_t = trades.sort_values("entry_time")
        admitted = _admission_mask(sorted_t["entry_time"], sorted_t["exit_time"], cap)
        subset = sorted_t.loc[admitted]
    if subset.empty:
        return -999.0
    return portfolio_metrics(trades_to_daily_returns(subset))["sharpe"]


def _pick_best_cap(selection_trades: pd.DataFrame) -> tuple:
    scored = [(cap, _sharpe_capped(selection_trades, cap)) for cap in CAP_SWEEP]
    best_cap, best_sharpe = max(scored, key=lambda x: x[1])
    return best_cap, best_sharpe, scored


def main() -> None:
    log.info("building all-48 trailing-stop book (plain) and market-gated book")
    trailing_plain = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED,
                                          backtest_fn=backtest_trades_trailing_fast,
                                          stop_grid=TRAILING_STOP_GRID)
    market_mask = cmro._build_market_mask()
    trailing_gated = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED,
                                          backtest_fn=backtest_trades_trailing_fast,
                                          stop_grid=TRAILING_STOP_GRID, market_mask=market_mask)
    baseline_fixed = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED)

    for df in (trailing_plain, trailing_gated, baseline_fixed):
        df["entry_time"] = pd.to_datetime(df["entry_time"])
    start = trailing_plain["entry_time"].min() + pd.DateOffset(years=crr.WARMUP_YEARS)
    end = trailing_plain["entry_time"].max()

    log.info("rolling annual sector rebalance on both the plain and gated trailing books")
    curated_plain, _, _ = crr._run_frequency(trailing_plain, "annual", crr.REBALANCE_OFFSETS["annual"], start, end)
    curated_gated, _, _ = crr._run_frequency(trailing_gated, "annual", crr.REBALANCE_OFFSETS["annual"], start, end)

    split_date = trailing_plain["entry_time"].min() + SELECTION_FRAC * (
        trailing_plain["entry_time"].max() - trailing_plain["entry_time"].min())
    log.info("selection window < %s, holdout window >= %s", split_date.date(), split_date.date())

    def split(df):
        sel = df[df["entry_time"] < split_date]
        hold = df[df["entry_time"] >= split_date]
        return sel, hold

    plain_sel, plain_hold = split(curated_plain)
    gated_sel, gated_hold = split(curated_gated)

    # --- freeze gate on/off using ONLY selection-window Sharpe ---
    sel_sharpe_plain = portfolio_metrics(trades_to_daily_returns(plain_sel))["sharpe"] if not plain_sel.empty else -999
    sel_sharpe_gated = portfolio_metrics(trades_to_daily_returns(gated_sel))["sharpe"] if not gated_sel.empty else -999
    use_gate = sel_sharpe_gated > sel_sharpe_plain
    log.info("selection-window Sharpe: plain=%.3f gated=%.3f -> frozen gate choice: %s",
              sel_sharpe_plain, sel_sharpe_gated, use_gate)
    frozen_sel, frozen_hold = (gated_sel, gated_hold) if use_gate else (plain_sel, plain_hold)

    # --- freeze max_concurrent using ONLY selection-window Sharpe ---
    best_cap, best_cap_sharpe, cap_scores = _pick_best_cap(frozen_sel)
    log.info("selection-window cap sweep: %s -> frozen cap: %s (Sharpe %.3f)", cap_scores, best_cap, best_cap_sharpe)

    # --- apply the FROZEN recipe (gate choice + cap) to the HOLDOUT window ---
    holdout_sim = simulate(frozen_hold, leverage=100, sizing_mode="disciplined", max_concurrent=best_cap)
    holdout_pm_uncapped = portfolio_metrics(trades_to_daily_returns(frozen_hold))
    if best_cap is not None and not frozen_hold.empty:
        sorted_h = frozen_hold.sort_values("entry_time")
        admitted = _admission_mask(sorted_h["entry_time"], sorted_h["exit_time"], best_cap)
        holdout_pm_capped = portfolio_metrics(trades_to_daily_returns(sorted_h.loc[admitted]))
    else:
        holdout_pm_capped = holdout_pm_uncapped

    # --- benchmark: baseline (all-48, fixed stop) on the SAME holdout window ---
    _, baseline_hold = split(baseline_fixed)
    baseline_pm = portfolio_metrics(trades_to_daily_returns(baseline_hold)) if not baseline_hold.empty else {}

    print(f"\n=== FROZEN RECIPE (chosen using ONLY data before {split_date.date()}) ===")
    print(f"use_market_gate={use_gate}  max_concurrent={best_cap}")

    print(f"\n=== holdout window ({split_date.date()} -> {end.date()}) -- frozen recipe never saw this data ===")
    result = pd.DataFrame([
        {"variant": "baseline_all48_fixed_stop_holdout", "n_trades": len(baseline_hold), **baseline_pm},
        {"variant": "frozen_recipe_holdout_uncapped_sharpe", "n_trades": len(frozen_hold), **holdout_pm_uncapped,
         "sized_max_drawdown": simulate(frozen_hold, leverage=100, sizing_mode="disciplined").max_drawdown
         if not frozen_hold.empty else None},
        {"variant": f"frozen_recipe_holdout_cap{best_cap}_sharpe", "n_trades": holdout_sim.n_trades,
         **holdout_pm_capped, "sized_max_drawdown": holdout_sim.max_drawdown},
    ]).set_index("variant")
    print(result.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "full_pipeline_holdout_decision.csv")
    log.info("wrote %s", OUT / "full_pipeline_holdout_decision.csv")

    headline = result.loc[f"frozen_recipe_holdout_cap{best_cap}_sharpe", "sharpe"] > baseline_pm.get("sharpe", 0)
    print(f"\nheadline: frozen recipe beats baseline on TRUE holdout = {bool(headline)}")


if __name__ == "__main__":
    main()
