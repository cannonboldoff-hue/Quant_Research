"""Risk management: ATR-based stop-loss / take-profit and trailing stops.

Consolidates the stop/target logic embedded in ``backtest_trades`` /
``close_position`` across the strategy notebooks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..indicators.volatility import atr


def atr_stop_levels(df: pd.DataFrame, entry_price: float, direction: int,
                    length: int = 14, sl_mult: float = 1.5, tp_mult: float = 3.0):
    """Return (stop_loss, take_profit) for a single entry using ATR distance.

    ``direction``: 1 for long, -1 for short. Default 1.5xATR stop / 3xATR target
    gives a 1:2 reward:risk, a common default in the notebooks.
    """
    a = atr(df, length).iloc[-1]
    sl = entry_price - direction * sl_mult * a
    tp = entry_price + direction * tp_mult * a
    return float(sl), float(tp)


def apply_stop_take(prices: pd.Series, entry: float, direction: int, sl: float, tp: float):
    """Scan forward prices; return (exit_index, exit_price, reason)."""
    for i, p in enumerate(prices):
        if direction == 1:
            if p <= sl: return i, sl, "stop"
            if p >= tp: return i, tp, "target"
        else:
            if p >= sl: return i, sl, "stop"
            if p <= tp: return i, tp, "target"
    return len(prices) - 1, float(prices.iloc[-1]), "eod"


def trailing_stop(prices: pd.Series, entry: float, direction: int, atr_series: pd.Series,
                  mult: float = 2.0):
    """ATR trailing stop. Returns (exit_index, exit_price)."""
    stop = entry - direction * mult * atr_series.iloc[0]
    for i, p in enumerate(prices):
        stop = max(stop, p - direction * mult * atr_series.iloc[min(i, len(atr_series) - 1)]) \
            if direction == 1 else min(stop, p + direction * -mult * atr_series.iloc[min(i, len(atr_series) - 1)])
        if (direction == 1 and p <= stop) or (direction == -1 and p >= stop):
            return i, float(stop)
    return len(prices) - 1, float(prices.iloc[-1])
