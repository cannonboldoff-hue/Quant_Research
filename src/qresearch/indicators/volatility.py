"""Volatility indicators — consolidates ``compute_atr``/``compute_atr_max``
(27+/22+ notebooks), ``rogers_satchell_vol``/``compute_rs_volatility_fast``,
``compute_hv``, ``rolling_zscore``, ``rolling_r2``."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range on an OHLC frame (columns: high, low, close)."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()


def atr_bands(df: pd.DataFrame, length: int = 14, mult: float = 2.0):
    """Return (upper, lower) ATR bands around close — basis for ATR stops/targets."""
    a = atr(df, length)
    return df["close"] + mult * a, df["close"] - mult * a


def rogers_satchell_vol(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Rogers-Satchell volatility (drift-independent), scaled by 1e6.
    Uses the *previous* bar's OHLC to avoid lookahead."""
    o = df["open"].shift(1).replace(0, np.nan)
    h = df["high"].shift(1)
    l = df["low"].shift(1)
    c = df["close"].shift(1).replace(0, np.nan)
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return rs.rolling(length).mean() * 1e6


def historical_volatility(close: pd.Series, length: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualized rolling std of returns — ``compute_hv``."""
    return close.pct_change().rolling(length).std() * np.sqrt(periods_per_year)


def rolling_zscore(x: pd.Series, window: int, min_periods: int = 10) -> pd.Series:
    """Z-score of the latest value in each rolling window."""
    def _z(w):
        if len(w) == 0:
            return 0.0
        return (w[-1] - w.mean()) / (w.std() + 1e-8)
    return x.rolling(window, min_periods=min_periods).apply(lambda w: _z(w.to_numpy()), raw=False)


def rolling_r2(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling R^2 of an OLS fit against a linear time trend — trend-fit quality."""
    r2 = np.full(len(series), np.nan)
    x = np.arange(window).reshape(-1, 1)
    values = series.to_numpy()
    for i in range(window, len(series)):
        y = values[i - window:i].reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        r2[i] = model.score(x, y)
    return pd.Series(r2, index=series.index)
