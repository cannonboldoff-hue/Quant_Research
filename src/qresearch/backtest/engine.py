"""Vectorized event backtester.

Consolidates ``backtest_trades`` (88 notebooks), ``run_backtest`` (25) and the
numba/cython variants (``cy_backtest_trades``/``py_backtest_trades``) into one
clean, documented engine. Signal-in / OHLCV-in, trades-out.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config.settings import get_settings
from ..risk.stops import atr_stop_levels, apply_stop_take
from ..indicators.volatility import atr
from .engine_numba import _scan_trades, _scan_trades_trailing, STOP, TARGET


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: int
    entry_price: float
    exit_price: float
    reason: str
    pnl: float
    ret: float


def backtest_trades(df: pd.DataFrame, sl_mult: float = 1.5, tp_mult: float = 3.0,
                    fee_bps: float | None = None, price_col: str = "close") -> pd.DataFrame:
    """Run an ATR stop/target backtest over a frame that already contains a
    ``signal`` column (1 long / -1 short / 0 flat) and OHLC columns.

    Returns a trades DataFrame. Costs applied as round-trip bps.
    """
    s = get_settings()
    fee = (s.fee_bps + s.slippage_bps if fee_bps is None else fee_bps) / 1e4
    df = df.reset_index(drop=True)
    trades: list[Trade] = []
    i, n = 0, len(df)
    while i < n - 1:
        sig = int(df.at[i, "signal"])
        if sig == 0:
            i += 1
            continue
        entry = float(df.at[i, price_col])
        sl, tp = atr_stop_levels(df.iloc[: i + 1], entry, sig, sl_mult=sl_mult, tp_mult=tp_mult)
        if not (np.isfinite(sl) and np.isfinite(tp)):
            # ATR not warmed up yet (first `length` bars) -- sl/tp would be
            # NaN, and NaN comparisons are always False, so this "trade"
            # would never close and swallow the rest of the series. Skip it.
            i += 1
            continue
        fwd = df[price_col].iloc[i + 1:]
        j, exit_px, reason = apply_stop_take(fwd, entry, sig, sl, tp)
        exit_idx = i + 1 + j
        gross = sig * (exit_px - entry) / entry
        ret = gross - fee
        trades.append(Trade(df.at[i, "Date"] if "Date" in df else i,
                            df.at[exit_idx, "Date"] if "Date" in df else exit_idx,
                            sig, entry, exit_px, reason, ret * entry, ret))
        i = exit_idx + 1
    return pd.DataFrame([t.__dict__ for t in trades])


def backtest_trades_fast(df: pd.DataFrame, sl_mult: float = 1.5, tp_mult: float = 3.0,
                          fee_bps: float | None = None, price_col: str = "close",
                          atr_length: int = 14) -> pd.DataFrame:
    """numba-accelerated equivalent of ``backtest_trades`` — same signal-in/
    trades-out contract and identical exit semantics, for the campaign
    runner's data volume. ``backtest_trades`` is left untouched; this is
    the entry point large-scale/parallel runs should call instead."""
    s = get_settings()
    fee = (s.fee_bps + s.slippage_bps if fee_bps is None else fee_bps) / 1e4
    df = df.reset_index(drop=True)

    price = df[price_col].to_numpy(dtype=np.float64)
    signal = df["signal"].to_numpy(dtype=np.int64)
    atr_arr = atr(df, atr_length).to_numpy(dtype=np.float64)

    entry_idx, exit_idx, entry_px, exit_px, direction, reason = _scan_trades(
        price, signal, atr_arr, sl_mult, tp_mult
    )
    if len(entry_idx) == 0:
        return pd.DataFrame([], columns=["entry_time", "exit_time", "direction",
                                          "entry_price", "exit_price", "reason", "pnl", "ret"])

    gross = direction * (exit_px - entry_px) / entry_px
    ret = gross - fee
    reason_str = np.where(reason == STOP, "stop", np.where(reason == TARGET, "target", "eod"))

    dates = df["Date"].to_numpy() if "Date" in df.columns else None
    return pd.DataFrame({
        "entry_time": dates[entry_idx] if dates is not None else entry_idx,
        "exit_time": dates[exit_idx] if dates is not None else exit_idx,
        "direction": direction,
        "entry_price": entry_px,
        "exit_price": exit_px,
        "reason": reason_str,
        "pnl": ret * entry_px,
        "ret": ret,
    })


def backtest_trades_trailing_fast(df: pd.DataFrame, sl_mult: float = 1.5, trail_mult: float = 2.0,
                                   fee_bps: float | None = None, price_col: str = "close",
                                   atr_length: int = 14) -> pd.DataFrame:
    """Same contract as ``backtest_trades_fast``, but exits via a ratcheting
    ATR trailing stop (``risk.stops.trailing_stop``'s logic, numba-jitted)
    instead of a fixed take-profit -- locks in gains before a winner
    round-trips into a loss, rather than only capping the downside."""
    s = get_settings()
    fee = (s.fee_bps + s.slippage_bps if fee_bps is None else fee_bps) / 1e4
    df = df.reset_index(drop=True)

    price = df[price_col].to_numpy(dtype=np.float64)
    signal = df["signal"].to_numpy(dtype=np.int64)
    atr_arr = atr(df, atr_length).to_numpy(dtype=np.float64)

    entry_idx, exit_idx, entry_px, exit_px, direction, reason = _scan_trades_trailing(
        price, signal, atr_arr, sl_mult, trail_mult
    )
    if len(entry_idx) == 0:
        return pd.DataFrame([], columns=["entry_time", "exit_time", "direction",
                                          "entry_price", "exit_price", "reason", "pnl", "ret"])

    gross = direction * (exit_px - entry_px) / entry_px
    ret = gross - fee
    reason_str = np.where(reason == STOP, "stop", np.where(reason == TARGET, "target", "eod"))

    dates = df["Date"].to_numpy() if "Date" in df.columns else None
    return pd.DataFrame({
        "entry_time": dates[entry_idx] if dates is not None else entry_idx,
        "exit_time": dates[exit_idx] if dates is not None else exit_idx,
        "direction": direction,
        "entry_price": entry_px,
        "exit_price": exit_px,
        "reason": reason_str,
        "pnl": ret * entry_px,
        "ret": ret,
    })


def run_backtest(df, **kw) -> dict:
    """Convenience wrapper: run backtest and attach summary metrics."""
    from .metrics import calculate_metrics
    trades = backtest_trades(df, **kw)
    return {"trades": trades, "metrics": calculate_metrics(trades)}
