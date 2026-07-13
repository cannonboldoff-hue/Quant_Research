"""Rolling sector rebalance for indian_equities_jma_atr_daily.

The static one-shot sector pick (campaign_sector_analysis.py) failed a nested
holdout test (campaign_sector_holdout_validation.py): sectors picked on
2015-2021 data underperformed baseline when graded on 2021-2024 data they
never saw, and the ranking flipped almost entirely across the two windows.
That doesn't rule out the underlying idea -- it rules out picking ONCE and
never updating. This tests the proper version: re-pick sectors periodically
using only past trades, trade that set forward, roll. Selection is then
PERMANENTLY out-of-sample, at every step, not just once.

Key simplification: a ticker's walk-forward OOS trades (from
run_finalist_trades) don't depend on which sector group it's in -- a ticker
trades the same whether "curated" or not. So the full 48-ticker OOS trade
book is computed ONCE, and every rebalance frequency is just pandas filtering
of that one book (no repeated backtesting):
  - sector score at rebalance date t: composite over trades with
    entry_time < t only (strictly past -- OOS by construction)
  - forward trades for the period [t, t_next): whichever tickers were kept,
    filtered to entry_time in that window
Selection and the trades it's graded on never share a bar -- there is no
lookahead to guard against here (unlike the static pick, which reused the
very sample it was graded on).

Run: python scripts/campaign_rolling_rebalance.py
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
from qresearch.risk.leverage_sim import simulate  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402
import campaign_sector_analysis as csa  # noqa: E402 -- reuse SECTOR_MAP/_sector_metrics/_composite

log = get_logger("qresearch.campaign_rolling_rebalance")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"
STRATEGY_ID, MARKET, TIMEFRAME = csa.STRATEGY_ID, csa.MARKET, csa.TIMEFRAME
TICKER_SECTOR = {t: sector for sector, tickers in csa.SECTOR_MAP.items() for t in tickers}

WARMUP_YEARS = 3  # first rebalance only once this much history exists to score sectors on
REBALANCE_OFFSETS = {
    "annual": pd.DateOffset(years=1),
    "semiannual": pd.DateOffset(months=6),
    "quarterly": pd.DateOffset(months=3),
}


def _rebalance_periods(start: pd.Timestamp, end: pd.Timestamp, offset: pd.DateOffset) -> list[tuple]:
    """[(period_start, period_end), ...] covering [start, end], each exactly
    one rebalance step wide (last one clipped to end)."""
    bounds = [start]
    while bounds[-1] < end:
        bounds.append(min(bounds[-1] + offset, end))
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _pick_sectors(hist_trades: pd.DataFrame) -> set[str]:
    """Composite-score every sector using only hist_trades (already filtered
    to entry_time < rebalance date), keep the top half -- exact same rule as
    campaign_sector_analysis.py's static pick, just re-run on a growing
    expanding window instead of once on the full history."""
    rows = []
    for sector, tickers in csa.SECTOR_MAP.items():
        sector_trades = hist_trades[hist_trades["ticker"].isin(tickers)]
        m = csa._sector_metrics(sector_trades)
        m["sector"] = sector
        rows.append(m)
    ranking = pd.DataFrame(rows).set_index("sector")
    ranking["composite"] = csa._composite(ranking)
    kept = ranking["composite"] >= ranking["composite"].median()
    return {t for sector, tickers in csa.SECTOR_MAP.items() for t in tickers if kept[sector]}


def _run_frequency(all_trades: pd.DataFrame, freq_name: str, offset: pd.DateOffset,
                    start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    periods = _rebalance_periods(start, end, offset)
    curated_frames, baseline_frames, picks = [], [], []
    for period_start, period_end in periods:
        hist = all_trades[all_trades["entry_time"] < period_start]
        curated_tickers = _pick_sectors(hist)
        window = all_trades[(all_trades["entry_time"] >= period_start) & (all_trades["entry_time"] < period_end)]
        baseline_frames.append(window)
        curated_frames.append(window[window["ticker"].isin(curated_tickers)])
        picks.append({
            "freq": freq_name, "rebalance_date": period_start.date(),
            "kept_sectors": ",".join(sorted({TICKER_SECTOR[t] for t in curated_tickers})),
            "n_curated_tickers": len(curated_tickers),
        })
    curated = pd.concat(curated_frames, ignore_index=True) if curated_frames else pd.DataFrame()
    baseline = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    return curated, baseline, picks


def main() -> None:
    log.info("computing full 48-ticker OOS trade book once (shared across every frequency)")
    all_trades = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED)
    all_trades["entry_time"] = pd.to_datetime(all_trades["entry_time"])

    start = all_trades["entry_time"].min() + pd.DateOffset(years=WARMUP_YEARS)
    end = all_trades["entry_time"].max()
    log.info("trade book spans %s -> %s; rebalancing begins %s (after %dy warmup)",
              all_trades["entry_time"].min().date(), end.date(), start.date(), WARMUP_YEARS)

    summary_rows, all_picks = [], []
    for freq_name, offset in REBALANCE_OFFSETS.items():
        log.info("rolling rebalance: %s", freq_name)
        curated, baseline, picks = _run_frequency(all_trades, freq_name, offset, start, end)
        all_picks.extend(picks)

        for label, trades in [("baseline", baseline), ("curated", curated)]:
            if trades.empty:
                continue
            daily = trades_to_daily_returns(trades)
            pm = portfolio_metrics(daily)
            sim = simulate(trades, leverage=100, sizing_mode="disciplined")
            summary_rows.append({
                "freq": freq_name, "variant": label, "n_trades": len(trades),
                "n_tickers": trades["ticker"].nunique(), **pm,
                "sized_max_drawdown": sim.max_drawdown, "sized_final_equity": sim.final_equity,
            })

    summary = pd.DataFrame(summary_rows).set_index(["freq", "variant"])
    for freq_name in REBALANCE_OFFSETS:
        if (freq_name, "curated") in summary.index and (freq_name, "baseline") in summary.index:
            summary.loc[(freq_name, "curated"), "beats_baseline"] = (
                summary.loc[(freq_name, "curated"), "sharpe"] > summary.loc[(freq_name, "baseline"), "sharpe"])

    picks_df = pd.DataFrame(all_picks)

    print("\n=== rolling rebalance summary (selection always strictly out-of-sample) ===")
    print(summary.to_string())
    print("\n=== sector-pick stability per rebalance (churn check) ===")
    print(picks_df.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "rolling_rebalance_summary.csv")
    picks_df.to_csv(OUT / "rolling_rebalance_picks.csv", index=False)
    log.info("wrote rolling_rebalance_summary.csv and rolling_rebalance_picks.csv")

    any_beats = summary.xs("curated", level="variant")["beats_baseline"].any() if "beats_baseline" in summary else False
    print(f"\nheadline: curated beats baseline at ANY frequency = {bool(any_beats)}")


if __name__ == "__main__":
    main()
