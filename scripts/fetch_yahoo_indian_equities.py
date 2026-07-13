"""Fetch genuinely NEW NIFTY-50 daily data (past data/processed's 2024-11-08
cutoff) from Yahoo Finance, and append it to each ticker's validated history.

This is the true forward test the drawdown-reduction plan's frozen recipe
needs: every prior validation (walk-forward folds, rolling sector rebalance,
the nested holdout) reused SOME slice of the same 2015-2024 historical book.
This data literally did not exist when any part of that pipeline was built --
today's date is far past the processed data's cutoff, so whatever the frozen
recipe does here is a real out-of-time test, not another backtest split.

Writes to a SEPARATE tree, data/processed_fresh_validation/indian_equities/daily/,
so the validated data/processed is never touched. Each output file is the
FULL OLD history + newly-fetched rows appended (old rows kept verbatim on any
overlapping date) -- so the walk-forward/indicator warmup context is intact,
and only the genuinely-new tail is what gets graded.

Run: python scripts/fetch_yahoo_indian_equities.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.data.loaders import load_yfinance  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.fetch_yahoo_indian_equities")
OLD_DIR = ROOT / "data" / "processed" / "indian_equities" / "daily"
OUT_DIR = ROOT / "data" / "processed_fresh_validation" / "indian_equities" / "daily"


def _yahoo_symbol(stem: str) -> str:
    # local ticker stem -> Yahoo NSE symbol; only M&M needs an override
    # (local stem "MM" has no ampersand, Yahoo's actual symbol does).
    if stem == "MM":
        return "M&M.NS"
    return f"{stem}.NS"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = []
    for old_path in sorted(OLD_DIR.glob("*.parquet")):
        stem = old_path.stem
        old = pd.read_parquet(old_path)
        old_max = old["Date"].max()
        symbol = _yahoo_symbol(stem)
        log.info("fetching %s (%s) from %s", symbol, stem, old_max.date())
        fresh = load_yfinance(symbol, start=(old_max - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), interval="1d")
        if fresh.empty:
            log.warning("%s: no fresh data returned, keeping old history only", symbol)
            merged = old
        else:
            fresh = fresh.drop(columns=["Ticker"]).assign(Ticker=stem)[old.columns]
            merged = (pd.concat([old, fresh], ignore_index=True)
                        .drop_duplicates(subset="Date", keep="first")
                        .sort_values("Date").reset_index(drop=True))
        merged.to_parquet(OUT_DIR / f"{stem}.parquet", index=False)
        n_new = (merged["Date"] > old_max).sum()
        coverage.append({"ticker": stem, "old_max_date": old_max.date(), "new_rows": n_new,
                          "new_max_date": merged["Date"].max().date()})

    cov = pd.DataFrame(coverage)
    print("\n=== fresh-data coverage per ticker ===")
    print(cov.to_string(index=False))
    print(f"\ntotal new bars: {cov['new_rows'].sum()}  "
          f"min new_max_date: {cov['new_max_date'].min()}  max: {cov['new_max_date'].max()}")
    log.info("wrote extended parquet tree -> %s", OUT_DIR)


if __name__ == "__main__":
    main()
