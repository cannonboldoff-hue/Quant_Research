"""Volume indicators — OBV, MFI, TMF, PVS/VMS/SAMX, breadth, ADX.
Consolidates ``compute_obv``, ``compute_mfi``/``calculate_mfi``, ``calculate_tmf``,
``calculate_pvs_vms``/``calculate_vms``, ``calculate_samx``, ``compute_breadth``,
``calculate_adx`` (two variants — the naive-diff DM variant was replaced by the
more correct Wilder directional-movement comparison from the second)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .trend import weighted_moving_average


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Money Flow Index on a typical-price/volume basis."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical * df["volume"]
    up = money_flow.where(typical > typical.shift(1), 0.0)
    down = money_flow.where(typical < typical.shift(1), 0.0)
    ratio = up.rolling(length, min_periods=1).sum() / (down.rolling(length, min_periods=1).sum() + 1e-10)
    return 100 - 100 / (1 + ratio)


def tmf(df: pd.DataFrame, length: int) -> pd.Series:
    """Twiggs Money Flow."""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    true_high = np.maximum(close.shift(1), high)
    true_low = np.minimum(close.shift(1), low)
    true_range = true_high - true_low
    adv = (volume * ((close - true_low) - (true_high - close)) / true_range.replace(0, np.nan)).fillna(0)

    wv = weighted_moving_average(volume, length)
    wa = weighted_moving_average(adv, length)
    return pd.Series(np.where(wv == 0, 0, wa / wv), index=df.index)


def pvs_vms(df: pd.DataFrame, atr_period: int = 14, volume_period: int = 14) -> pd.DataFrame:
    """Price Volatility Strength / Volume Momentum Strength for a single
    ticker's OHLCV frame (group by ticker upstream, e.g. via
    ``data.loaders.process_by_ticker``, before calling this)."""
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                     np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()))
    atr_v = tr.rolling(atr_period).mean()
    avg_volume = df["volume"].rolling(volume_period).mean()

    pvs = (df["close"] - prev_close) / atr_v
    vms = (df["volume"] - avg_volume) / avg_volume
    return pd.DataFrame({"pvs": pvs, "vms": vms}, index=df.index)


def samx(df: pd.DataFrame, atr_period: int = 14, volume_period: int = 14) -> pd.Series:
    """Simplified AMX: price-momentum-over-ATR times volume-momentum-strength."""
    pv = pvs_vms(df, atr_period, volume_period)
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                     np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()))
    atr_v = tr.rolling(atr_period, min_periods=1).mean().replace(0, np.finfo(float).eps)
    pms = df["close"].diff() / atr_v
    return pms * pv["vms"]


def breadth(close: pd.Series) -> pd.DataFrame:
    """Cumulative advancing/declining tick counts — ``compute_breadth``."""
    advancing = (close.diff() > 0).astype(int).cumsum()
    declining = (close.diff() < 0).astype(int).cumsum()
    return pd.DataFrame({"advancing": advancing, "declining": declining}, index=close.index)


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI/-DI, using proper Wilder
    directional-movement comparison (``high[i]-high[i-1]`` vs
    ``low[i-1]-low[i]``, whichever is larger and positive)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_s = pd.Series(tr).rolling(length, min_periods=1).mean()
    plus_s = pd.Series(plus_dm, index=df.index).rolling(length, min_periods=1).mean()
    minus_s = pd.Series(minus_dm, index=df.index).rolling(length, min_periods=1).mean()

    plus_di = 100 * plus_s / tr_s
    minus_di = 100 * minus_s / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_v = dx.rolling(length, min_periods=1).mean()
    return pd.DataFrame({"adx": adx_v, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)
