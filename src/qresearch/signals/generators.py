"""Signal generation utilities.

Consolidates ``compute_jma_and_signals`` (25 notebooks) and
``map_signals_to_1min`` (7 notebooks): generate signals on a higher timeframe,
then map them onto a finer timeframe for realistic exit simulation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..indicators.moving_averages import jma


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True where ``a`` crosses above ``b``."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """True where ``a`` crosses below ``b``."""
    return (a < b) & (a.shift(1) >= b.shift(1))


def jma_signals(df: pd.DataFrame, fast: int = 7, slow: int = 21, price_col: str = "close") -> pd.DataFrame:
    """Fast/slow JMA crossover signals. Returns df with ``jma_fast``, ``jma_slow``,
    and ``signal`` in {1: long, -1: short, 0: flat}."""
    out = df.copy()
    out["jma_fast"] = jma(out[price_col], length=fast).values
    out["jma_slow"] = jma(out[price_col], length=slow).values
    long_c = crossover(out["jma_fast"], out["jma_slow"])
    short_c = crossunder(out["jma_fast"], out["jma_slow"])
    out["signal"] = np.select([long_c, short_c], [1, -1], default=0)
    return out


def map_signals_to_timeframe(signals: pd.DataFrame, fine: pd.DataFrame,
                             date_col: str = "Date") -> pd.DataFrame:
    """Forward-fill higher-timeframe signals onto a finer-timeframe frame so
    exits can be simulated at (e.g.) 1-minute resolution."""
    s = signals[[date_col, "signal"]].copy().sort_values(date_col)
    f = fine.sort_values(date_col).copy()
    merged = pd.merge_asof(f, s, on=date_col, direction="backward")
    merged["signal"] = merged["signal"].fillna(0)
    return merged
