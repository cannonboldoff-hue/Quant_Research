"""Trend indicators — SuperTrend, HalfTrend, DEMA/normalized-DEMA, KAMA,
zero-lag MA, weighted MA. Consolidates ``calculate_supertrend``, ``halftrend``
/``halftrend_with_session_reset``, ``dema``/``compute_normalized_dema``,
``compute_kama_indicator``, ``zero_lag_ma``, ``weighted_moving_average``."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .moving_averages import ema
from .volatility import atr as _atr

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(f):
        return f


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend on a real OHLC frame. The original ``calculate_supertrend``
    fed the same LTP series in as high/low/close, which makes true range
    collapse to zero (h-l is always 0) — a bug fixed here by using the
    frame's actual high/low/close columns."""
    a = _atr(df, period)
    mid = (df["high"] + df["low"]) / 2
    upper = mid + multiplier * a
    lower = mid - multiplier * a

    close = df["close"].to_numpy(copy=True)
    upper_v, lower_v = upper.to_numpy(copy=True), lower.to_numpy(copy=True)
    n = len(df)
    st = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int8)  # 1 = uptrend, -1 = downtrend
    trend[0] = 1
    st[0] = lower_v[0]
    for i in range(1, n):
        if close[i] > upper_v[i - 1]:
            trend[i] = 1
        elif close[i] < lower_v[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
            if trend[i] == 1:
                lower_v[i] = max(lower_v[i], lower_v[i - 1])
            else:
                upper_v[i] = min(upper_v[i], upper_v[i - 1])
        st[i] = lower_v[i] if trend[i] == 1 else upper_v[i]
    return pd.DataFrame({"supertrend": st, "trend": trend}, index=df.index)


def halftrend(df: pd.DataFrame, amplitude: int = 2, channel_deviation: int = 2, atr_period: int = 100) -> pd.DataFrame:
    """HalfTrend indicator for one continuous series (single ticker/session)."""
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(df)

    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    tr = np.maximum(hl, np.maximum(hc, lc))
    tr[0] = hl[0]

    atr_v = np.zeros(n)
    alpha = 2.0 / (atr_period + 1)
    atr_v[0] = tr[0]
    for i in range(1, n):
        atr_v[i] = alpha * tr[i] + (1 - alpha) * atr_v[i - 1]
    dev = channel_deviation * (atr_v / 2)

    trend = np.zeros(n, dtype=int)
    next_trend = np.zeros(n, dtype=int)
    max_low = np.zeros(n)
    min_high = np.zeros(n)
    up, down, ht = np.zeros(n), np.zeros(n), np.zeros(n)
    max_low[0], min_high[0] = low[0], high[0]
    up[0], down[0], ht[0] = low[0], high[0], close[0]

    for i in range(1, n):
        start = max(0, i - amplitude)
        high_price = np.max(high[start:i + 1])
        low_price = np.min(low[start:i + 1])
        high_ma = np.mean(high[i - amplitude + 1:i + 1]) if i >= amplitude else np.mean(high[:i + 1])
        low_ma = np.mean(low[i - amplitude + 1:i + 1]) if i >= amplitude else np.mean(low[:i + 1])

        if next_trend[i - 1] == 0:
            max_low[i] = max(low_price, max_low[i - 1])
            if high_ma < max_low[i] and close[i] < low[i - 1]:
                trend[i] = next_trend[i] = 1
                min_high[i] = high_price
            else:
                trend[i], next_trend[i] = trend[i - 1], next_trend[i - 1]
                min_high[i] = min_high[i - 1]
        else:
            min_high[i] = min(high_price, min_high[i - 1])
            if low_ma > min_high[i] and close[i] > high[i - 1]:
                trend[i] = next_trend[i] = 0
                max_low[i] = low_price
            else:
                trend[i], next_trend[i] = trend[i - 1], next_trend[i - 1]
                max_low[i] = max_low[i - 1]

        if trend[i] == 0:
            up[i] = down[i - 1] if trend[i - 1] == 1 else max(max_low[i], up[i - 1])
            down[i] = down[i - 1]
            ht[i] = up[i]
        else:
            down[i] = up[i - 1] if trend[i - 1] == 0 else min(min_high[i], down[i - 1])
            up[i] = up[i - 1]
            ht[i] = down[i]

    signal = np.zeros(n, dtype=int)
    flip_to_up = (trend[:-1] == 1) & (trend[1:] == 0)
    flip_to_down = (trend[:-1] == 0) & (trend[1:] == 1)
    signal[1:][flip_to_up] = 1
    signal[1:][flip_to_down] = 2

    return pd.DataFrame({
        "half_trend": ht, "atr_high": ht + dev, "atr_low": ht - dev,
        "signal": signal, "trend": trend, "atr": atr_v,
    }, index=df.index)


def halftrend_session_reset(df: pd.DataFrame, ticker_col: str = "Ticker", date_col: str = "Date",
                             amplitude: int = 2, channel_deviation: int = 2, atr_period: int = 100) -> pd.DataFrame:
    """Apply :func:`halftrend` per (ticker, session-date) group to avoid
    leaking ATR/trend state across sessions or tickers."""
    session = pd.to_datetime(df[date_col]).dt.date
    parts = [halftrend(g, amplitude, channel_deviation, atr_period)
             for _, g in df.groupby([df[ticker_col], session])]
    return pd.concat(parts).sort_index()


def dema(close: pd.Series, length: int) -> pd.Series:
    """Double EMA."""
    e1 = ema(close, length)
    return 2 * e1 - ema(e1, length)


def normalized_dema(close: pd.Series, len_dema: int = 20, base_len: int = 50) -> pd.DataFrame:
    """DEMA normalized into a 0-100 band via a rolling mean +/- 2*std envelope
    (``compute_normalized_dema`` stripped of its strategy-specific long/short
    thresholds, which depended on ambient notebook globals)."""
    d = dema(close, len_dema)
    base = d.rolling(base_len).mean()
    sd = d.rolling(base_len).std() * 2
    upper, lower = base + sd, base - sd
    norm = 100 * (d - lower) / (upper - lower)
    return pd.DataFrame({"dema": d, "base": base, "upper": upper, "lower": lower, "norm": norm}, index=close.index)


@njit
def _kama_core(price: np.ndarray, sc: np.ndarray) -> np.ndarray:
    out = np.empty(len(price))
    out[0] = price[0]
    for i in range(1, len(price)):
        if np.isnan(sc[i]):
            out[i] = out[i - 1]
        else:
            out[i] = out[i - 1] + sc[i] * (price[i] - out[i - 1])
    return out


def kama(close: pd.Series, length: int = 3, fast_ema: int = 2, slow_ema: int = 50) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    price = close.to_numpy(dtype=float)
    change = np.abs(np.concatenate([[np.nan] * length, price[length:] - price[:-length]]))
    volatility = pd.Series(np.abs(np.diff(price, prepend=np.nan))).rolling(length).sum().to_numpy()
    er = change / np.where(volatility == 0, np.nan, volatility)
    fast_sc, slow_sc = 2 / (fast_ema + 1), 2 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    return pd.Series(_kama_core(price, sc), index=close.index)


def zero_lag_ma(series: pd.Series, period: int) -> pd.Series:
    """Double linearly-weighted MA — reduced-lag smoother."""
    weights = np.arange(period, 0, -1)
    lwma1 = series.rolling(period).apply(lambda x: np.dot(weights, x) / weights.sum(), raw=True)
    return lwma1.rolling(period).apply(lambda x: np.dot(weights, x) / weights.sum(), raw=True)


def weighted_moving_average(series: pd.Series, length: int) -> np.ndarray:
    """Linearly-weighted moving average (most-recent bar weighted highest)."""
    weights = np.arange(1, length + 1)
    wma = np.convolve(series.to_numpy(), weights[::-1], mode="valid") / weights.sum()
    return np.concatenate([np.full(length - 1, np.nan), wma])
