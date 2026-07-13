"""Fetch longer daily OHLC history for the 3-market subset (commodities,
indices, forex) from Yahoo Finance, to widen the walk-forward eval window --
existing data/processed forex only spans 2020-2025 (~1680 rows), the binding
constraint on scripts/build_portfolio_subset.py's held-out significance gate.

Writes to a SEPARATE tree, data/processed_yf/<market>/daily/, so the
validated data/processed is never touched.

Run: python scripts/fetch_yahoo_daily.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.data.loaders import load_yfinance, save_parquet_by_ticker  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.fetch_yahoo_daily")
OUT = ROOT / "data" / "processed_yf"
START = "2000-01-01"

# yahoo symbol -> (market, local ticker name matching the existing sleeve's convention)
SYMBOLS = {
    "forex": {
        # 7 majors + 21 crosses -- all C(8,2) pairs of EUR/GBP/USD/JPY/CHF/CAD/AUD/NZD,
        # the standard "major currencies" universe.
        "EURUSD=X": "EURUSD", "GBPUSD=X": "GBPUSD", "USDJPY=X": "USDJPY", "USDCHF=X": "USDCHF",
        "USDCAD=X": "USDCAD", "AUDUSD=X": "AUDUSD", "NZDUSD=X": "NZDUSD",
        "EURGBP=X": "EURGBP", "EURJPY=X": "EURJPY", "EURCHF=X": "EURCHF",
        "EURCAD=X": "EURCAD", "EURAUD=X": "EURAUD", "EURNZD=X": "EURNZD",
        "GBPJPY=X": "GBPJPY", "GBPCHF=X": "GBPCHF", "GBPCAD=X": "GBPCAD",
        "GBPAUD=X": "GBPAUD", "GBPNZD=X": "GBPNZD",
        "AUDJPY=X": "AUDJPY", "AUDCHF=X": "AUDCHF", "AUDCAD=X": "AUDCAD", "AUDNZD=X": "AUDNZD",
        "NZDJPY=X": "NZDJPY", "NZDCHF=X": "NZDCHF", "NZDCAD=X": "NZDCAD",
        "CADJPY=X": "CADJPY", "CADCHF=X": "CADCHF",
        "CHFJPY=X": "CHFJPY",
    },
    "commodities": {
        "GC=F": "XAUUSD", "SI=F": "XAGUSD", "CL=F": "WTI", "BZ=F": "BRENT", "NG=F": "NATGAS",
    },
    "indices": {
        "^GSPC": "USA500", "^IXIC": "USATECH", "^DJI": "USA30", "^GDAXI": "DEU", "^FTSE": "GBR",
    },
}


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])
    return df.reset_index(drop=True)


def main() -> None:
    try:
        import yfinance  # noqa: F401
    except ImportError:
        log.error("yfinance not installed -- run: pip install yfinance")
        sys.exit(1)

    coverage = []
    failed = []
    for market, symbols in SYMBOLS.items():
        rows = []
        for yahoo_symbol, local_ticker in symbols.items():
            log.info("fetching %s (%s/%s)", yahoo_symbol, market, local_ticker)
            df = load_yfinance(yahoo_symbol, start=START, interval="1d")
            df = _validate(df)
            if df.empty:
                log.warning("no data returned for %s -- skipping", yahoo_symbol)
                failed.append((market, yahoo_symbol))
                continue
            df["Ticker"] = local_ticker
            rows.append(df)
            coverage.append({
                "market": market, "ticker": local_ticker, "yahoo_symbol": yahoo_symbol,
                "rows": len(df), "min_date": df["Date"].min(), "max_date": df["Date"].max(),
            })
        if rows:
            combined = pd.concat(rows, ignore_index=True)
            save_parquet_by_ticker(combined, OUT / market / "daily")

    cov_df = pd.DataFrame(coverage)
    print("\n=== coverage ===")
    print(cov_df.to_string(index=False))
    if failed:
        print(f"\nFAILED (no data returned, not written): {failed}")
    log.info("wrote parquet trees under %s", OUT)


if __name__ == "__main__":
    main()
