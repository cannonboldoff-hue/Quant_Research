"""Sector-curated universe test for indian_equities_jma_atr_daily.

The strategy trades all 48 NIFTY-50 daily stocks indiscriminately. A JMA
trend-crossover works better on trending sectors (metals, energy) than
choppy ones (FMCG, banks). This script walk-forward-optimizes the strategy
per sector, ranks sectors by a composite OOS score, keeps the top half,
rebuilds a curated universe from the kept sectors, and reruns to see
whether the curated book beats the all-48 baseline
(data/campaign_summaries/portfolio_metrics.csv: sharpe=1.059).

Reuses qresearch.optimize.finalist_selection.run_finalist_trades (same
walk-forward param optimization every other campaign script uses) via its
new ``tickers`` subset filter -- no new optimizer, no parquet copying.

Run: python scripts/campaign_sector_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.optimize.finalist_selection import run_finalist_trades  # noqa: E402
from qresearch.portfolio.combine import portfolio_metrics, trades_to_daily_returns  # noqa: E402
from qresearch.risk.leverage_sim import simulate  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.campaign_sector_analysis")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"
STRATEGY_ID, MARKET, TIMEFRAME = "indian_equities_jma_atr_daily", "indian_equities", "daily"

# NIFTY-50 sector classification for the 48 daily tickers on disk. No such
# mapping exists in the repo (data/processed/*.parquet is OHLCV only), so
# this is hand-classified from public NIFTY-50 sector membership.
SECTOR_MAP = {
    "financials": ["AXISBANK", "BAJAJFINSV", "BAJFINANCE", "ICICIBANK", "INDUSINDBK",
                   "KOTAKBANK", "SBIN", "SBILIFE", "SHRIRAMFIN"],
    "it": ["HCLTECH", "INFY", "TCS", "TECHM", "WIPRO"],
    "auto": ["BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "MARUTI", "MM", "TATAMOTORS"],
    "energy": ["BPCL", "COALINDIA", "NTPC", "ONGC", "POWERGRID", "RELIANCE"],
    "metals": ["HINDALCO", "JSWSTEEL", "TATASTEEL"],
    "fmcg": ["BRITANNIA", "HINDUNILVR", "ITC", "NESTLEIND", "TATACONSUM"],
    "consumer": ["ASIANPAINT", "TITAN", "TRENT"],
    "pharma": ["APOLLOHOSP", "CIPLA", "DRREDDY", "SUNPHARMA"],
    "materials": ["GRASIM", "ULTRACEMCO"],
    "telecom": ["BHARTIARTL"],  # single stock -- thin, flagged low-confidence below
    "capgoods": ["LT", "BEL"],
    "adani": ["ADANIENT", "ADANIPORTS"],
}
MIN_TICKERS_FOR_CONFIDENCE = 2

# Option 3a: tighter stops than the default STOP_GRID (sl_mult:[1,1.5,2],
# tp_mult:[1.5,2,3,4]) -- smaller sl_mult caps loss-per-trade tighter, trading
# off some return for lower drawdown.
TIGHT_STOP_GRID = {"sl_mult": [0.75, 1.0], "tp_mult": [2.0, 3.0]}

# Option 3b: risk_pct values to sweep on the best-Sharpe variant, alongside
# the default 0.01 (1% of equity risked per trade) every other row uses --
# leverage_sim.simulate's disciplined sizing scales position size (and hence
# drawdown) roughly linearly with risk_pct.
RISK_PCT_SWEEP = [0.005, 0.0075, 0.01]


def _sector_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0, "n_tickers": 0, "oos_mean_ret": 0.0,
                "oos_mean_win_rate": 0.0, "oos_positive_frac": 0.0, "sharpe": 0.0}
    per_fold = (
        trades.groupby(["ticker", "fold"])["ret"]
        .agg(mean_ret="mean", win_rate=lambda r: float((r > 0).mean()))
    )
    daily = trades_to_daily_returns(trades)
    return {
        "n_trades": len(trades),
        "n_tickers": trades["ticker"].nunique(),
        "oos_mean_ret": float(per_fold["mean_ret"].mean()),
        "oos_mean_win_rate": float(per_fold["win_rate"].mean()),
        "oos_positive_frac": float((per_fold["mean_ret"] > 0).mean()),
        "sharpe": portfolio_metrics(daily)["sharpe"],
    }


def _composite(df: pd.DataFrame) -> pd.Series:
    cols = ["oos_mean_ret", "oos_mean_win_rate", "oos_positive_frac"]
    norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min()).replace(0, 1)
    return norm.mean(axis=1)


def main() -> None:
    rows = []
    for sector, tickers in SECTOR_MAP.items():
        log.info("walk-forward: sector=%s (%d tickers)", sector, len(tickers))
        trades = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED, tickers=set(tickers))
        m = _sector_metrics(trades)
        m["sector"] = sector
        m["thin"] = len(tickers) < MIN_TICKERS_FOR_CONFIDENCE
        rows.append(m)

    ranking = pd.DataFrame(rows).set_index("sector")
    ranking["composite"] = _composite(ranking)
    ranking = ranking.sort_values("composite", ascending=False)
    # ponytail: median split is the keep threshold -- tune to a fixed composite
    # bar if you want fewer/more sectors kept.
    ranking["kept"] = ranking["composite"] >= ranking["composite"].median()

    print("\n=== sector ranking (composite = mean of normalized oos_mean_ret/win_rate/positive_frac) ===")
    print(ranking.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUT / "sector_ranking.csv")
    log.info("wrote %s", OUT / "sector_ranking.csv")

    curated_tickers = {t for sector, tickers in SECTOR_MAP.items()
                        for t in tickers if ranking.loc[sector, "kept"]}
    log.info("curated universe: %d tickers (kept sectors: %s)",
              len(curated_tickers), sorted(ranking.index[ranking["kept"]]))

    baseline_trades = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED)
    curated_trades = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED, tickers=curated_tickers)
    # ADX regime filter already tried on curated: Sharpe 1.68 -> 0.47, dropped.
    # Two remaining drawdown levers instead: MFI volume filter, tighter stops.
    log.info("curated + MFI volume filter: re-optimizing to see if gating weak money-flow cuts drawdown")
    curated_volume_trades = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED,
                                                 tickers=curated_tickers, volume_filter=True)
    log.info("curated + tight stops: re-optimizing with sl_mult capped at 0.75-1.0")
    curated_tight_stop_trades = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED,
                                                      tickers=curated_tickers, stop_grid=TIGHT_STOP_GRID)

    variants = [("baseline_all48", baseline_trades), ("curated", curated_trades),
                ("curated_volume_filtered", curated_volume_trades),
                ("curated_tight_stops", curated_tight_stop_trades)]

    decision_rows = []
    for label, trades in variants:
        daily = trades_to_daily_returns(trades)
        pm = portfolio_metrics(daily)
        decision_rows.append({
            "universe": label, "n_tickers": trades["ticker"].nunique() if not trades.empty else 0,
            "n_trades": len(trades), "oos_mean_ret": float(trades["ret"].mean()) if not trades.empty else 0.0,
            **pm,
        })
    decision = pd.DataFrame(decision_rows).set_index("universe")
    decision["curated_beats_baseline"] = decision["sharpe"] > decision.loc["baseline_all48", "sharpe"]

    print("\n=== baseline (all 48) vs curated (kept sectors) vs curated+regime-filtered ===")
    print(decision.to_string())
    decision.to_csv(OUT / "sector_curated_decision.csv")
    log.info("wrote %s", OUT / "sector_curated_decision.csv")

    # trade-stacked daily-return maxDD above ignores position sizing (every
    # trade gets full portfolio weight for its exit day) -- overlay the same
    # disciplined fixed-fractional sizing build_portfolio.py already validates
    # to see the REAL capital-at-risk drawdown, not an artifact of pooling.
    print("\n=== disciplined-sizing overlay (leverage=100x, risk_pct=1%/trade, sequential equity walk) ===")
    sizing_rows = []
    for label, trades in variants:
        if trades.empty:
            continue
        sim = simulate(trades, leverage=100, sizing_mode="disciplined")
        sizing_rows.append({
            "universe": label, "position_fraction": sim.position_fraction,
            "final_equity": sim.final_equity, "max_drawdown": sim.max_drawdown,
            "ruined": sim.ruined, "ruin_probability_mc": sim.ruin_probability_mc,
        })
    sizing = pd.DataFrame(sizing_rows).set_index("universe")
    print(sizing.to_string())

    # Option 3b: sweep risk_pct on the best-Sharpe variant only (sizing-only
    # lever -- doesn't touch the strategy, drawdown scales ~linearly with it).
    best_variant = decision["sharpe"].idxmax()
    best_trades = dict(variants)[best_variant]
    print(f"\n=== risk_pct sweep on best-Sharpe variant ({best_variant}) ===")
    sweep_rows = []
    for risk_pct in RISK_PCT_SWEEP:
        sim = simulate(best_trades, leverage=100, sizing_mode="disciplined", risk_pct=risk_pct)
        sweep_rows.append({
            "universe": best_variant, "risk_pct": risk_pct, "position_fraction": sim.position_fraction,
            "final_equity": sim.final_equity, "max_drawdown": sim.max_drawdown,
            "ruined": sim.ruined, "ruin_probability_mc": sim.ruin_probability_mc,
        })
    sweep = pd.DataFrame(sweep_rows).set_index("risk_pct")
    print(sweep.to_string())

    sizing.to_csv(OUT / "sector_curated_leverage_sim.csv")
    sweep.to_csv(OUT / "sector_curated_risk_pct_sweep.csv")
    log.info("wrote %s and %s", OUT / "sector_curated_leverage_sim.csv", OUT / "sector_curated_risk_pct_sweep.csv")


if __name__ == "__main__":
    main()
