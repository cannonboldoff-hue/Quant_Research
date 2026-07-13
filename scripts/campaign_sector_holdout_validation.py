"""Nested holdout validation for the sector-curation edge found in
campaign_sector_analysis.py.

That script picked the top 6/12 sectors using the SAME out-of-sample
walk-forward trades it then re-graded the curated universe on -- sectors were
kept because they scored well on data the curated rerun's Sharpe (1.68) was
then measured against. That circularity can inflate the result (the same
selection-bias risk the repo's own DSR/PBO checks guard against elsewhere,
e.g. campaign_cross_sectional.py's fit/eval split).

This splits the full 2015-02-02..2024-11-08 history 70/30 by calendar date
(matching build_portfolio.py's FIT_FRAC convention): sector ranking + the
curated ticker list are chosen using ONLY the selection window (<split_date).
That list is then frozen and re-graded on the holdout window (>=split_date),
which the selection step never saw. If curated still beats baseline there,
the edge survives a real out-of-sample test of the selection step itself.

Run: python scripts/campaign_sector_holdout_validation.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qresearch.optimize.finalist_selection import run_finalist_trades  # noqa: E402
from qresearch.portfolio.combine import portfolio_metrics, trades_to_daily_returns  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402
import campaign_sector_analysis as csa  # noqa: E402 -- reuse SECTOR_MAP/_sector_metrics/_composite

log = get_logger("qresearch.campaign_sector_holdout_validation")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"
STRATEGY_ID, MARKET, TIMEFRAME = csa.STRATEGY_ID, csa.MARKET, csa.TIMEFRAME
SELECTION_FRAC = 0.7  # matches build_portfolio.py's FIT_FRAC convention
MIN_BARS = 100  # below this a 5-fold anchored walk-forward has too few bars/fold to mean anything


def _write_sliced_market(src_dir: Path, dst_root: Path, split_date: pd.Timestamp, before: bool) -> Path:
    """Copy every ticker parquet under src_dir, keeping only rows before/after
    split_date, into dst_root/MARKET/TIMEFRAME -- lets run_finalist_trades's
    existing glob-and-read logic run unmodified against a calendar sub-period,
    no core module changes needed."""
    dst_dir = dst_root / MARKET / TIMEFRAME
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        sliced = df[df["Date"] < split_date] if before else df[df["Date"] >= split_date]
        if len(sliced) < MIN_BARS:
            continue
        sliced.to_parquet(dst_dir / path.name, index=False)
    return dst_dir


def main() -> None:
    src_dir = PROCESSED / MARKET / TIMEFRAME
    all_dates = pd.concat(
        [pd.read_parquet(p, columns=["Date"]) for p in sorted(src_dir.glob("*.parquet"))]
    )["Date"]
    all_dates = pd.to_datetime(all_dates)
    start, end = all_dates.min(), all_dates.max()
    split_date = start + SELECTION_FRAC * (end - start)
    log.info("full history %s -> %s; selection <%s (%.1fy), holdout >=%s (%.1fy)",
              start.date(), end.date(), split_date.date(), (split_date - start).days / 365.25,
              split_date.date(), (end - split_date).days / 365.25)

    with tempfile.TemporaryDirectory(prefix="qr_holdout_") as tmp:
        tmp_root = Path(tmp)
        sel_processed = tmp_root / "selection"
        hold_processed = tmp_root / "holdout"
        _write_sliced_market(src_dir, sel_processed, split_date, before=True)
        _write_sliced_market(src_dir, hold_processed, split_date, before=False)

        # --- sector ranking + curated-universe selection, SELECTION WINDOW ONLY ---
        rows = []
        for sector, tickers in csa.SECTOR_MAP.items():
            log.info("selection-window walk-forward: sector=%s (%d tickers)", sector, len(tickers))
            trades = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, sel_processed, tickers=set(tickers))
            m = csa._sector_metrics(trades)
            m["sector"] = sector
            rows.append(m)
        ranking = pd.DataFrame(rows).set_index("sector")
        ranking["composite"] = csa._composite(ranking)
        ranking["kept"] = ranking["composite"] >= ranking["composite"].median()
        curated_tickers = {t for sector, tickers in csa.SECTOR_MAP.items()
                            for t in tickers if ranking.loc[sector, "kept"]}
        log.info("curated universe (selection-window only): %d tickers, sectors=%s",
                  len(curated_tickers), sorted(ranking.index[ranking["kept"]]))
        ranking.to_csv(OUT / "sector_holdout_selection_ranking.csv")

        # --- grade baseline vs curated on the HOLDOUT window selection never saw ---
        baseline_holdout = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, hold_processed)
        curated_holdout = run_finalist_trades(STRATEGY_ID, MARKET, TIMEFRAME, hold_processed,
                                               tickers=curated_tickers)

        decision_rows = []
        for label, trades in [("baseline_all48_holdout", baseline_holdout),
                               ("curated_holdout", curated_holdout)]:
            daily = trades_to_daily_returns(trades)
            pm = portfolio_metrics(daily)
            decision_rows.append({
                "universe": label, "n_tickers": trades["ticker"].nunique() if not trades.empty else 0,
                "n_trades": len(trades), "oos_mean_ret": float(trades["ret"].mean()) if not trades.empty else 0.0,
                **pm,
            })
        decision = pd.DataFrame(decision_rows).set_index("universe")
        decision["curated_beats_baseline"] = (decision.loc["curated_holdout", "sharpe"]
                                               > decision.loc["baseline_all48_holdout", "sharpe"])

        print("\n=== sector ranking (selection window only, never sees holdout) ===")
        print(ranking.to_string())
        print(f"\n=== holdout window ({split_date.date()} -> {end.date()}, "
              f"~{(end - split_date).days / 365.25:.1f}y) grading -- selection never saw this data ===")
        print(decision.to_string())
        decision.to_csv(OUT / "sector_holdout_decision.csv")
        log.info("wrote sector_holdout_selection_ranking.csv and sector_holdout_decision.csv")


if __name__ == "__main__":
    main()
