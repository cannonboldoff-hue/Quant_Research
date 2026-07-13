"""Data preprocessing utilities — OKX-safe fetchers (generalized to take an
explicit ccxt ``exchange`` instead of an ambient notebook global), funding-rate
merge, quarterly feature aggregation, and column-name auto-resolution for
messy tradebook exports. Consolidates ``fetch_ohlcv_okx_safe``,
``fetch_funding_okx_safe``, ``merge_funding``, ``preprocess_ohlc``,
``create_quarterly_features``, ``add_tradebook_features``, ``resolve_columns``.

Not ported: ``remove_pps_outliers``/``is_float``/``convert``
(``data-cleaning_006``) — turned out to be a real-estate price-per-sqft
tutorial (site_location/price_per_sqft columns), not trading data.
``load_data``/``_is_colab`` — Google Colab environment glue, moot outside
Colab.
"""
from __future__ import annotations
import time
import pandas as pd

from ..utils.logging_config import get_logger

_log = get_logger("qresearch.data.preprocess")


def fetch_ohlcv_okx_safe(exchange, symbol: str, timeframe: str = "1m",
                         start: str = "2025-01-01", batch_limit: int = 100) -> pd.DataFrame:
    """Paginated OHLCV fetch via a ccxt exchange instance, retrying on error."""
    since = exchange.parse8601(start + "T00:00:00Z")
    rows = []
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=batch_limit)
            if not candles:
                break
            rows.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < batch_limit:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            _log.warning("fetch_ohlcv retry: %s", e)
            time.sleep(2)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_funding_okx_safe(exchange, symbol: str) -> pd.DataFrame:
    """Funding-rate history via a ccxt exchange instance, retrying on error."""
    while True:
        try:
            data = exchange.fetch_funding_rate_history(symbol)
            break
        except Exception as e:
            _log.warning("fetch_funding retry: %s", e)
            time.sleep(2)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["funding"] = df["fundingRate"].astype(float)
    return df[["timestamp", "funding"]]


def merge_funding(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Left-merge funding rate onto OHLCV by timestamp, forward-filling gaps."""
    df = ohlcv.merge(funding, on="timestamp", how="left")
    df["funding"] = df["funding"].ffill().fillna(0.0)
    return df


def preprocess_ohlc(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Parse and set the date column as index."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col)


def create_quarterly_features(df: pd.DataFrame, ticker_col: str = "Ticker") -> pd.DataFrame:
    """Daily -> monthly -> quarterly return/volatility/Sharpe aggregation per
    ticker, joined with a forward-quarter return target for ML feature sets."""
    parts = []
    for ticker, group in df.groupby(ticker_col):
        daily = group.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                          "close": "last", "volume": "sum"}).dropna()
        daily["return"] = daily["close"].pct_change()
        daily["volatility"] = daily["return"].rolling(21).std()
        daily["rolling_sharpe"] = daily["return"].rolling(21).mean() / (daily["volatility"] + 1e-6)

        monthly = daily.resample("ME").agg({"return": ["mean", "std"], "rolling_sharpe": "mean"})
        monthly.columns = ["monthly_return_mean", "monthly_return_std", "monthly_rolling_sharpe"]

        quarterly = daily.resample("QE").agg({"close": "last", "return": ["mean", "std"],
                                              "rolling_sharpe": "mean", "volume": "mean"})
        quarterly.columns = ["_".join(c).strip() for c in quarterly.columns]
        quarterly["ticker"] = ticker
        quarterly["forward_return"] = quarterly["close_last"].pct_change().shift(-1)

        monthly["quarter"] = monthly.index.to_period("Q")
        quarterly["quarter"] = quarterly.index.to_period("Q")
        quarterly = quarterly.join(monthly.groupby("quarter").mean(), on="quarter")
        parts.append(quarterly)

    return pd.concat(parts).dropna().reset_index() if parts else df.iloc[0:0]


def add_tradebook_features(features: pd.DataFrame, tradebook: pd.DataFrame,
                           ticker_col: str = "Ticker", entry_col: str = "Entry Time") -> pd.DataFrame:
    """Join per-(ticker, quarter) trade stats (mean/std PnL, long ratio, trade
    count) onto a quarterly feature frame from :func:`create_quarterly_features`."""
    tb = tradebook.copy()
    tb[entry_col] = pd.to_datetime(tb[entry_col])
    tb["quarter"] = tb[entry_col].dt.to_period("Q")

    agg = tb.groupby([ticker_col, "quarter"]).agg(
        pnl_mean=("PnL", "mean"), pnl_std=("PnL", "std"),
        long_ratio=("Direction", lambda x: (x.str.lower() == "long").mean()),
        trade_count=(entry_col, "count"),
    ).reset_index()
    agg["Date"] = agg["quarter"].dt.end_time
    agg = agg.drop(columns="quarter")

    out = features.merge(agg, left_on=["ticker", "Date"], right_on=[ticker_col, "Date"], how="left")
    return out.drop(columns=ticker_col, errors="ignore").fillna(0)


_COLUMN_ALIASES = {
    "entry_time": ["Entry Time", "Entry Date", "EntryTime"],
    "exit_time": ["Exit Time", "Exit Date", "ExitTime"],
    "pnl": ["PnL", "pnl", "P&L"],
    "ticker": ["Ticker", "ticker"],
    "direction": ["Direction", "direction"],
    "exit_reason": ["Exit Reason", "exit_reason"],
    "entry_price": ["Entry Price", "entry_price"],
    "exit_price": ["Exit Price", "exit_price"],
}


def resolve_columns(df: pd.DataFrame, overrides: dict | None = None) -> dict:
    """Map logical trade-column names to whatever they're actually called in
    a given tradebook export, via alias lookup (or an explicit override)."""
    overrides = overrides or {}
    resolved = {}
    for key, aliases in _COLUMN_ALIASES.items():
        if key in overrides:
            if overrides[key] not in df.columns:
                raise ValueError(f"override '{key}'='{overrides[key]}' not in DataFrame columns")
            resolved[key] = overrides[key]
            continue
        resolved[key] = next((a for a in aliases if a in df.columns), None)
    missing = [k for k in ("entry_time", "exit_time", "pnl") if resolved[k] is None]
    if missing:
        raise ValueError(f"required columns not found/aliased: {missing}")
    return resolved
