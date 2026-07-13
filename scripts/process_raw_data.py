"""Manifest-driven ETL: data/raw/* -> data/processed/{market}/{timeframe}/{ticker}.parquet.

One parameterized reader per storage format (CSV / Parquet); per-source quirks
(column renames, tz offsets, date formats) are manifest parameters, not new
functions. Two sources aren't OHLCV (tick quotes, a pre-derived signal
dataset) and get bespoke passthrough handling.

Run: python scripts/process_raw_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.csv as pcsv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.config.constants import COL_DATE, COL_TICKER, OHLCV  # noqa: E402
from qresearch.data.loaders import save_parquet_by_ticker  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402
from qresearch.utils.timing import timeit  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
log = get_logger("qresearch.etl")


def _drop_index_col(df: pd.DataFrame) -> pd.DataFrame:
    junk = [c for c in df.columns if c == "" or str(c).startswith("Unnamed")]
    return df.drop(columns=junk) if junk else df


def normalize_csv(path: Path, date_col: str, ticker_col: str | None = None,
                   ticker_value: str | None = None, rename: dict | None = None,
                   drop_cols: list[str] | None = None, dayfirst: bool = False,
                   tz_strip_pattern: str | None = None, tz_offset_hours: float = 0.0,
                   use_pyarrow: bool = False) -> pd.DataFrame:
    df = pcsv.read_csv(path).to_pandas() if use_pyarrow else pd.read_csv(path)
    df = _drop_index_col(df)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if rename:
        df = df.rename(columns=rename)
    if tz_strip_pattern:
        df[date_col] = df[date_col].astype(str).str.replace(tz_strip_pattern, "", regex=True)
    df[COL_DATE] = pd.to_datetime(df[date_col], dayfirst=dayfirst)
    if tz_offset_hours:
        df[COL_DATE] = df[COL_DATE] - pd.Timedelta(hours=tz_offset_hours)
    df[COL_TICKER] = df[ticker_col] if ticker_col else ticker_value
    return df[[COL_DATE, COL_TICKER, *OHLCV]]


def normalize_parquet(path: Path, rename: dict | None = None,
                       drop_cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if rename:
        df = df.rename(columns=rename)
    return df[[COL_DATE, COL_TICKER, *OHLCV]]


def _validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[COL_DATE, "close"])
    # positive_ok catches all-zero (or negative) OHLC rows -- these trivially
    # pass the high>=max/low<=min checks below (0 >= 0), so a bad tick/print
    # with price==0 would otherwise slip through and later produce inf/nan
    # returns (division by entry_price==0) downstream in the backtest engine.
    positive_ok = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    hi_ok = df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9
    lo_ok = df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9
    df = df[positive_ok & hi_ok & lo_ok & (df["volume"] >= 0)]
    dropped = before - len(df)
    if dropped:
        log.warning("%s: dropped %d/%d rows failing OHLCV sanity checks", name, dropped, before)
    return df


def _finalize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.sort_values([COL_TICKER, COL_DATE])
    before = len(df)
    df = df.drop_duplicates(subset=[COL_TICKER, COL_DATE], keep="first")
    if before != len(df):
        log.info("%s: dropped %d exact-duplicate (ticker,date) rows", name, before - len(df))
    return _validate(df, name).reset_index(drop=True)


@timeit
def _process_source(spec: dict) -> None:
    name = spec["name"]
    if spec["kind"] == "csv":
        df = normalize_csv(spec["path"], **spec["reader_kwargs"])
    else:
        df = normalize_parquet(spec["path"], **spec["reader_kwargs"])
    df = _finalize(df, name)
    out_dir = PROCESSED / spec["market"] / spec["timeframe"]
    save_parquet_by_ticker(df, str(out_dir))
    log.info("%s -> %s (%d rows, %d tickers)", name, out_dir, len(df), df[COL_TICKER].nunique())


MANIFEST = [
    dict(
        name="crypto_t7", kind="parquet", path=RAW / "Binance_Crypto_T7_2020_25.parquet",
        market="crypto", timeframe="1m",
        reader_kwargs=dict(drop_cols=["__index_level_0__"]),
    ),
    dict(
        name="forex_t8", kind="parquet", path=RAW / "Forex_T8_2020_202505.parquet",
        market="forex", timeframe="1m",
        reader_kwargs=dict(rename={"Datetime": COL_DATE, "Open": "open", "High": "high",
                                    "Low": "low", "Close": "close", "Volume": "volume"}),
    ),
    dict(
        name="commodities_merged", kind="csv",
        path=RAW / "Comodoties_Data" / "Comodoties_Data_merged.csv",
        market="commodities", timeframe="1h",
        reader_kwargs=dict(date_col="Datetime", ticker_col="Ticker"),
    ),
    dict(
        name="natgas", kind="csv",
        path=RAW / "Comodoties_Data" / "GAS.CMDUSD_Candlestick_1_Hour_BID_01.01.2020-05.07.2025.csv",
        market="commodities", timeframe="1h",
        reader_kwargs=dict(date_col="Local time", ticker_value="NATGAS", dayfirst=True,
                            tz_strip_pattern=r"\s*GMT[+-]\d{4}$", tz_offset_hours=5.5,
                            rename={"Open": "open", "High": "high", "Low": "low",
                                    "Close": "close", "Volume": "volume"}),
    ),
    dict(
        name="wti", kind="csv",
        path=RAW / "Comodoties_Data" / "LIGHT.CMDUSD_Candlestick_1_Hour_BID_01.01.2015-05.07.2025.csv",
        market="commodities", timeframe="1h",
        reader_kwargs=dict(date_col="Local time", ticker_value="WTI", dayfirst=True,
                            tz_strip_pattern=r"\s*GMT[+-]\d{4}$", tz_offset_hours=5.5,
                            rename={"Open": "open", "High": "high", "Low": "low",
                                    "Close": "close", "Volume": "volume"}),
    ),
    dict(
        name="xauusd_1m", kind="csv", path=RAW / "XAU_1m_data.csv",
        market="commodities", timeframe="1m",
        reader_kwargs=dict(date_col="Date", ticker_value="XAUUSD", use_pyarrow=True,
                            rename={"Open": "open", "High": "high", "Low": "low",
                                    "Close": "close", "Volume": "volume"}),
    ),
    dict(
        name="indices_merged", kind="csv", path=RAW / "Indices_Data" / "Indices_Data_merged.csv",
        market="indices", timeframe="1h",
        reader_kwargs=dict(date_col="Datetime", ticker_col="Ticker"),
    ),
    dict(
        name="nifty50_5m", kind="csv", path=RAW / "NIFTY_50_5min_2015_24.csv",
        market="indian_equities", timeframe="5m",
        reader_kwargs=dict(date_col="Date", ticker_col="Ticker", use_pyarrow=True),
    ),
    dict(
        name="niftybees", kind="csv", path=RAW / "niftybees_aug.csv",
        market="indian_equities", timeframe="1m",
        reader_kwargs=dict(date_col="Date_Time", ticker_col="Ticker", dayfirst=True,
                            drop_cols=["Open Interest"],
                            rename={"Open": "open", "High": "high", "Low": "low",
                                    "Close": "close", "Volume": "volume"}),
    ),
    dict(
        name="niftyfut", kind="csv", path=RAW / "niftyfut_aug.csv",
        market="futures", timeframe="1m",
        reader_kwargs=dict(date_col="Date_Time", ticker_col="Ticker", dayfirst=False,
                            drop_cols=["Open Interest"],
                            rename={"Open": "open", "High": "high", "Low": "low",
                                    "Close": "close", "Volume": "volume"}),
    ),
]


def process_ticks() -> None:
    """Quarter/*.parquet: L1 tick/quote data (LTP/BuyPrice/SellPrice/LTQ/OpenInterest).
    Not OHLCV -- clean/dedupe/sort only, keep the native schema."""
    src_dir = RAW / "Quarter"
    out_dir = PROCESSED / "indian_equities" / "ticks"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.parquet"))
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["Date_Time"] = pd.to_datetime(df["Date_Time"])
            df = df.sort_values(["Ticker", "Date_Time"])
            before = len(df)
            df = df.drop_duplicates(subset=["Ticker", "Date_Time"], keep="first")
            df = df.reset_index(drop=True)
            df.to_parquet(out_dir / f.name, index=False)
            log.info("ticks %s -> %s (%d rows, dropped %d dupes)", f.name, out_dir, len(df), before - len(df))
        except Exception as e:
            log.error("ticks %s FAILED: %s", f.name, e)
    log.info("processed %d/%d tick files", len(files), len(files))


def process_derived_signal() -> None:
    """btc_eth_prep_ethbtc_spot_data.csv: pre-derived funding/Kalman-spread/zscore
    dataset for the P3 funding-arb paper. Not raw OHLCV -- kept as-is."""
    src = RAW / "btc_eth_prep_ethbtc_spot_data.csv"
    out_dir = PROCESSED / "crypto" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    out_path = out_dir / "eth_btc_funding_arb.parquet"
    df.to_parquet(out_path, index=False)
    log.info("derived %s -> %s (%d rows)", src.name, out_path, len(df))


def main() -> None:
    failures = []
    for spec in MANIFEST:
        try:
            _process_source(spec)
        except Exception as e:
            log.error("%s FAILED: %s", spec["name"], e)
            failures.append(spec["name"])
    process_ticks()
    process_derived_signal()
    if failures:
        log.error("FAILED sources: %s", failures)
        sys.exit(1)
    log.info("ETL complete: %d OHLCV sources + ticks + derived", len(MANIFEST))


if __name__ == "__main__":
    main()
