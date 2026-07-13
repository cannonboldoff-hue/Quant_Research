"""Unified data loaders + resampling.

Consolidates ``resample_1hr_custom`` / ``process_ticker`` patterns that were
repeated in ~96/57 notebooks. Canonical schema: long format with columns
``Date, Ticker, open, high, low, close, volume`` (see config.constants).
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable
import pandas as pd

from ..config.constants import TIMEFRAMES, OHLCV, COL_DATE, COL_TICKER
from ..utils.logging_config import get_logger

_log = get_logger("qresearch.data")

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """Resample a multi-ticker long-format OHLCV frame to ``timeframe``.

    Generalizes the notebooks' hard-coded ``resample_1hr_custom``.
    """
    rule = TIMEFRAMES.get(timeframe, timeframe)
    df = df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    parts = []
    for tkr, group in df.groupby(COL_TICKER):
        out = (group.set_index(COL_DATE)[OHLCV]
                    .resample(rule).agg(_AGG).dropna().reset_index())
        out[COL_TICKER] = tkr
        parts.append(out[[COL_DATE, COL_TICKER, *OHLCV]])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def process_by_ticker(df: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame],
                      tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Apply ``fn`` per ticker and concatenate — replaces repeated ``process_ticker`` loops."""
    if tickers is not None:
        df = df[df[COL_TICKER].isin(list(tickers))]
    parts = [fn(g.copy()) for _, g in df.groupby(COL_TICKER)]
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def load_crypto_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 1000,
                      exchange: str = "binance") -> pd.DataFrame:
    """Fetch crypto OHLCV via ccxt. Requires ``pip install ccxt``."""
    import ccxt  # local import: optional dependency
    ex = getattr(ccxt, exchange)()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", *OHLCV])
    df[COL_DATE] = pd.to_datetime(df["ts"], unit="ms")
    df[COL_TICKER] = symbol
    return df[[COL_DATE, COL_TICKER, *OHLCV]]


def save_parquet_by_ticker(df: pd.DataFrame, output_dir: str, ticker_col: str = COL_TICKER) -> None:
    """Split a long-format frame into one parquet file per ticker — ``save_parquet_by_ticker``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ticker, group in df.groupby(ticker_col):
        safe_name = str(ticker).replace(".", "_")
        group.to_parquet(out / f"{safe_name}.parquet", index=False)
        _log.info("saved %s -> %s", ticker, out / f"{safe_name}.parquet")


def load_yfinance(ticker: str, start=None, end=None, interval: str = "1d") -> pd.DataFrame:
    """Fetch equities via yfinance into the canonical long schema."""
    import yfinance as yf  # local import: optional dependency
    raw = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        # newer yfinance returns ("Close", "<ticker>") column tuples for a
        # single-ticker download -- drop the per-ticker level, keep the field name.
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower).reset_index()
    raw = raw.rename(columns={"index": COL_DATE, "date": COL_DATE, "datetime": COL_DATE})
    raw[COL_TICKER] = ticker
    keep = [COL_DATE, COL_TICKER] + [c for c in OHLCV if c in raw.columns]
    return raw[keep]
