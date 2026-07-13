"""Part B of the drawdown-reduction plan: broad-market regime overlay.

The per-ticker ADX regime filter (Follow-up section, options 2&3 earlier in
this plan) was already tried and made things WORSE (Sharpe 1.68 -> 0.47) --
it's too local/noisy, filtering out plenty of good individual trades on
tickers whose own ADX happened to dip even while the broader market was fine.

The drawdown diagnostic found the real pattern is different: every major
drawdown episode hit 46-48 of 48 tickers SIMULTANEOUSLY, during multi-month
broad-market chop (2018-19, 2021, 2023-24), not sharp crashes. That's a
market-wide phenomenon, not a per-ticker one. This tests a coarser, shared
gate instead: one equal-weight composite built from all 48 tickers, ADX'd
once, and used to block NEW entries (on every ticker) on days the broad
market isn't confirmed-trending -- open positions still run to their own
stop/target as normal, only new risk is gated.

Run: python scripts/campaign_market_regime_overlay.py
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
from qresearch.signals.regime import trend_regime  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402
import campaign_rolling_rebalance as crr  # noqa: E402 -- reuse the annual-rebalance curated book

log = get_logger("qresearch.campaign_market_regime_overlay")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"
STRATEGY_ID, MARKET, TIMEFRAME = crr.STRATEGY_ID, crr.MARKET, crr.TIMEFRAME
ADX_THRESHOLD = 25.0


def _build_market_mask() -> pd.Series:
    """Equal-weight composite OHLC from every ticker (each rebased to start
    at 1.0 so no single stock's price level dominates the average), ADX'd
    once -- one shared 'is the broad market trending' signal."""
    src_dir = PROCESSED / MARKET / TIMEFRAME
    highs, lows, closes = {}, {}, {}
    for path in sorted(src_dir.glob("*.parquet")):
        df = pd.read_parquet(path).set_index("Date")
        base = df["close"].iloc[0]
        highs[path.stem] = df["high"] / base
        lows[path.stem] = df["low"] / base
        closes[path.stem] = df["close"] / base
    composite = pd.DataFrame({
        "high": pd.DataFrame(highs).mean(axis=1),
        "low": pd.DataFrame(lows).mean(axis=1),
        "close": pd.DataFrame(closes).mean(axis=1),
    }).sort_index()
    mask = trend_regime(composite, adx_length=14, adx_threshold=ADX_THRESHOLD)
    log.info("composite market mask: %d/%d days trending (%.1f%%)",
              int(mask.sum()), len(mask), 100 * mask.mean())
    return mask


def main() -> None:
    market_mask = _build_market_mask()

    log.info("baseline (all 48), with vs without market regime overlay")
    baseline_plain = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED)
    baseline_gated = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, PROCESSED, market_mask=market_mask)

    # Apply the SAME rolling-rebalance sector selection to both books (plain
    # and market-gated) independently -- reusing curated_plain's ticker set
    # for the gated variant would silently swap "rolling per-period pick" for
    # "static union of every ticker ever picked, full unrestricted history",
    # an invalid comparison (it would also inflate trade count, not shrink it,
    # defeating the point of a gate that's supposed to suppress entries).
    log.info("rebuilding annual rolling-rebalance curation on both plain and gated books")
    baseline_plain["entry_time"] = pd.to_datetime(baseline_plain["entry_time"])
    baseline_gated["entry_time"] = pd.to_datetime(baseline_gated["entry_time"])
    start_plain = baseline_plain["entry_time"].min() + pd.DateOffset(years=crr.WARMUP_YEARS)
    end_plain = baseline_plain["entry_time"].max()
    curated_plain, _, _ = crr._run_frequency(baseline_plain, "annual", crr.REBALANCE_OFFSETS["annual"],
                                              start_plain, end_plain)
    start_gated = baseline_gated["entry_time"].min() + pd.DateOffset(years=crr.WARMUP_YEARS)
    end_gated = baseline_gated["entry_time"].max()
    curated_gated, _, _ = crr._run_frequency(baseline_gated, "annual", crr.REBALANCE_OFFSETS["annual"],
                                              start_gated, end_gated)

    variants = [
        ("baseline_plain", baseline_plain), ("baseline_market_gated", baseline_gated),
        ("curated_plain", curated_plain), ("curated_market_gated", curated_gated),
    ]
    rows = []
    for label, trades in variants:
        daily = trades_to_daily_returns(trades)
        pm = portfolio_metrics(daily)
        sim = simulate(trades, leverage=100, sizing_mode="disciplined") if not trades.empty else None
        rows.append({
            "variant": label, "n_tickers": trades["ticker"].nunique() if not trades.empty else 0,
            "n_trades": len(trades), **pm,
            "sized_max_drawdown": sim.max_drawdown if sim else None,
            "sized_final_equity": sim.final_equity if sim else None,
        })
    summary = pd.DataFrame(rows).set_index("variant")
    print("\n=== market regime overlay: plain vs gated (baseline and curated) ===")
    print(summary.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "market_regime_overlay.csv")
    log.info("wrote %s", OUT / "market_regime_overlay.csv")


if __name__ == "__main__":
    main()
