"""The real test: apply the FROZEN drawdown-reduction recipe to genuinely new
market data it has never touched, fetched live from Yahoo Finance
(scripts/fetch_yahoo_indian_equities.py) past data/processed's 2024-11-08
cutoff -- not another backtest split, actual forward-in-time data.

Frozen recipe (chosen using ONLY pre-2022-05-25 data in
campaign_full_pipeline_holdout_validation.py, and already confirmed to
survive the 2022-05-2024-11 holdout): trailing-stop engine, rolling annual
sector rebalance, max_concurrent=15, no market-gate. NONE of these choices
are re-tuned here -- that would defeat the point of a forward test.

Uses a GLOBAL cutoff (2024-11-08, the validated dataset's actual boundary for
47/48 tickers) to decide what counts as "genuinely new", not each ticker's
own last row -- one ticker (LT) has a pre-existing data gap back to 2021-12
in data/processed, unrelated to this fetch, and using a per-ticker cutoff
would wrongly count LT's backfilled 2021-2024 rows as forward-test data.
TATAMOTORS.NS 404'd (delisted post its 2024 demerger into separate CV/PV
listings) -- self-excludes since it has zero rows past the cutoff.

Run: python scripts/campaign_forward_validation.py
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
from qresearch.risk.leverage_sim import _admission_mask, simulate  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402
import campaign_rolling_rebalance as crr  # noqa: E402

log = get_logger("qresearch.campaign_forward_validation")
PROCESSED_FRESH = ROOT / "data" / "processed_fresh_validation"
OUT = ROOT / "data" / "campaign_summaries"
STRATEGY_ID, MARKET, TIMEFRAME = crr.STRATEGY_ID, crr.MARKET, crr.TIMEFRAME
TRAILING_STOP_GRID = {"sl_mult": [1.0, 1.5, 2.0], "trail_mult": [1.5, 2.0, 3.0]}
FROZEN_MAX_CONCURRENT = 15  # from campaign_full_pipeline_holdout_validation.py's frozen decision
GLOBAL_CUTOFF = pd.Timestamp("2024-11-08")  # data/processed's actual validated-data boundary


def main() -> None:
    log.info("running frozen recipe (trailing stop, no gate) on the EXTENDED (validated+fresh) data tree")
    trailing = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED_FRESH,
                                    backtest_fn=backtest_trades_trailing_fast, stop_grid=TRAILING_STOP_GRID)
    log.info("running baseline (fixed stop, all 48) on the same extended tree for comparison")
    baseline = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED_FRESH)

    trailing["entry_time"] = pd.to_datetime(trailing["entry_time"])
    baseline["entry_time"] = pd.to_datetime(baseline["entry_time"])

    log.info("rolling annual sector rebalance, continuing forward through the new data")
    start = trailing["entry_time"].min() + pd.DateOffset(years=crr.WARMUP_YEARS)
    end = trailing["entry_time"].max()
    curated, _, picks = crr._run_frequency(trailing, "annual", crr.REBALANCE_OFFSETS["annual"], start, end)

    forward_curated = curated[curated["entry_time"] > GLOBAL_CUTOFF]
    forward_baseline = baseline[baseline["entry_time"] > GLOBAL_CUTOFF]
    log.info("forward window (> %s): %d curated trades, %d baseline trades",
              GLOBAL_CUTOFF.date(), len(forward_curated), len(forward_baseline))

    rows = []
    for label, trades, cap in [("baseline_all48_fixed_stop_FORWARD", forward_baseline, None),
                                ("frozen_recipe_FORWARD_uncapped", forward_curated, None),
                                (f"frozen_recipe_FORWARD_cap{FROZEN_MAX_CONCURRENT}", forward_curated,
                                 FROZEN_MAX_CONCURRENT)]:
        if trades.empty:
            rows.append({"variant": label, "n_trades": 0})
            continue
        if cap is not None:
            sorted_t = trades.sort_values("entry_time")
            admitted = _admission_mask(sorted_t["entry_time"], sorted_t["exit_time"], cap)
            graded = sorted_t.loc[admitted]
            sim = simulate(trades, leverage=100, sizing_mode="disciplined", max_concurrent=cap)
        else:
            graded = trades
            sim = simulate(trades, leverage=100, sizing_mode="disciplined") if not trades.empty else None
        pm = portfolio_metrics(trades_to_daily_returns(graded)) if not graded.empty else {}
        rows.append({
            "variant": label, "n_trades": len(trades), "n_graded": len(graded),
            "n_tickers": trades["ticker"].nunique(), **pm,
            "sized_max_drawdown": sim.max_drawdown if sim else None,
            "sized_final_equity": sim.final_equity if sim else None,
        })

    result = pd.DataFrame(rows).set_index("variant")
    print(f"\n=== FORWARD test: data after {GLOBAL_CUTOFF.date()} that the frozen recipe never saw before now ===")
    print(result.to_string())

    print("\n=== which sectors did the rolling rebalance keep in the newest (2025-2026) periods? ===")
    picks_df = pd.DataFrame(picks)
    print(picks_df[picks_df["rebalance_date"] > GLOBAL_CUTOFF.date()].to_string(index=False)
          if not picks_df.empty else "no rebalance periods start after cutoff")

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "forward_validation_result.csv")
    log.info("wrote %s", OUT / "forward_validation_result.csv")


if __name__ == "__main__":
    main()
