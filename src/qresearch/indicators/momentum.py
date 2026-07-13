"""Momentum indicators. Thin wrappers preferring TA-Lib when installed
(TA-Lib appeared in 178 notebooks) with a pure-pandas fallback."""
from __future__ import annotations
import pandas as pd

try:
    import talib  # type: ignore
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    if _HAS_TALIB:
        return pd.Series(talib.RSI(close.astype(float).values, timeperiod=length), index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-12)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef, es = close.ewm(span=fast).mean(), close.ewm(span=slow).mean()
    line = ef - es
    sig = line.ewm(span=signal).mean()
    return line, sig, line - sig


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """%K/%D stochastic oscillator — core calc behind the notebooks'
    ``stochastic_oscillator``/``calculate_quad_stochastic`` signal wrappers
    (those layer entry/exit logic on top; that belongs in ``signals``)."""
    low_n = df["low"].rolling(k_period).min()
    high_n = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_n) / (high_n - low_n)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d}, index=df.index)
