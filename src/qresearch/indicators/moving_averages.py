"""Moving averages, including the Jurik Moving Average (JMA) that underpins
the repository's core signal family (``compute_jma_indicator`` appeared in
32+ notebooks)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ohlc4(df: pd.DataFrame) -> pd.Series:
    """Average of open/high/low/close — a common smoothed price input."""
    return (df["open"] + df["high"] + df["low"] + df["close"]) / 4


def sma(s: pd.Series, length: int = 20) -> pd.Series:
    return s.rolling(length, min_periods=length).mean()


def ema(s: pd.Series, length: int = 20) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def jma(price: pd.Series, length: int = 7, phase: float = 0.0, power: float = 2.0) -> pd.Series:
    """Jurik Moving Average — low-lag adaptive smoother.

    Consolidated, vectorized-per-step reference implementation of the JMA logic
    that was re-implemented across many notebooks. Parameters follow Jurik's
    convention: ``length`` (period), ``phase`` in [-100, 100], ``power`` (>=1).
    """
    price = price.astype(float).to_numpy()
    n = len(price)
    if n == 0:
        return pd.Series(dtype=float)

    phase_ratio = np.clip(phase / 100.0 + 1.5, 0.5, 2.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power

    ma1 = det0 = det1 = jm = price[0]
    out = np.empty(n)
    for i in range(n):
        p = price[i]
        ma1 = (1 - alpha) * p + alpha * ma1
        det0 = (p - ma1) * (1 - beta) + beta * det0
        ma2 = ma1 + phase_ratio * det0
        det1 = (ma2 - jm) * (1 - alpha) ** 2 + (alpha ** 2) * det1
        jm = jm + det1
        out[i] = jm
    return pd.Series(out)
