"""VWAP — consolidates ``compute_vwap`` / ``calculate_vwap`` (04_indicators, 09_risk_management)."""
from __future__ import annotations
import pandas as pd


def vwap(df: pd.DataFrame, anchor: pd.Series | None = None) -> pd.Series:
    """Volume-weighted average price on an OHLCV frame (typical-price basis).

    ``anchor``: optional grouping key (e.g. a date column) to reset the
    cumulative sum per session. Original notebooks ran one unbroken
    cumulative VWAP with no session reset — pass ``anchor`` to fix that.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    if anchor is not None:
        return pv.groupby(anchor).cumsum() / df["volume"].groupby(anchor).cumsum()
    return pv.cumsum() / df["volume"].cumsum()


def tick_vwap(price: pd.Series, qty: pd.Series) -> pd.Series:
    """Tick-data VWAP (last-traded-price/quantity basis) — ``compute_vwap`` variant."""
    return (price * qty).cumsum() / qty.cumsum()
