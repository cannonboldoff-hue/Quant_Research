"""numba-jitted core for the bar-scanning loop in ``backtest_trades``.

Same exit semantics as ``backtest.engine.backtest_trades`` /
``risk.stops.apply_stop_take`` (close-only touch on ``price_col``, no
intrabar high/low), just operating on plain numpy arrays instead of
per-bar pandas ``.at[]`` access. ATR must be precomputed on the *whole*
series before calling this (rolling(length).mean() at position i only
depends on bars i-length+1..i, so this is numerically identical to the
original per-trade ``atr(df.iloc[:i+1])`` call, just O(n) instead of
O(n^2)). numba nopython mode can't touch pandas/TA-Lib, so nothing
pandas-shaped crosses this boundary.
"""
from __future__ import annotations

import numba
import numpy as np

STOP, TARGET, EOD = 0, 1, 2


@numba.njit(cache=True)
def _scan_trades(price: np.ndarray, signal: np.ndarray, atr_arr: np.ndarray,
                  sl_mult: float, tp_mult: float):
    n = price.shape[0]
    entry_idx = np.empty(n, dtype=np.int64)
    exit_idx = np.empty(n, dtype=np.int64)
    entry_px = np.empty(n, dtype=np.float64)
    exit_px = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=np.int64)
    reason = np.empty(n, dtype=np.int64)
    count = 0
    i = 0
    while i < n - 1:
        sig = signal[i]
        if sig == 0:
            i += 1
            continue
        a = atr_arr[i]
        if np.isnan(a):
            # ATR not warmed up yet -- sl/tp would be NaN and never trigger,
            # swallowing the rest of the series into one degenerate trade.
            i += 1
            continue
        entry = price[i]
        sl = entry - sig * sl_mult * a
        tp = entry + sig * tp_mult * a

        ex_i = n - 1
        ex_px = price[n - 1]
        r = EOD
        j = i + 1
        while j < n:
            p = price[j]
            if sig == 1:
                if p <= sl:
                    ex_i, ex_px, r = j, sl, STOP
                    break
                if p >= tp:
                    ex_i, ex_px, r = j, tp, TARGET
                    break
            else:
                if p >= sl:
                    ex_i, ex_px, r = j, sl, STOP
                    break
                if p <= tp:
                    ex_i, ex_px, r = j, tp, TARGET
                    break
            j += 1

        entry_idx[count] = i
        exit_idx[count] = ex_i
        entry_px[count] = entry
        exit_px[count] = ex_px
        direction[count] = sig
        reason[count] = r
        count += 1
        i = ex_i + 1

    return (entry_idx[:count], exit_idx[:count], entry_px[:count],
            exit_px[:count], direction[:count], reason[:count])


@numba.njit(cache=True)
def _scan_trades_trailing(price: np.ndarray, signal: np.ndarray, atr_arr: np.ndarray,
                           sl_mult: float, trail_mult: float):
    """Same entry/warmup semantics as ``_scan_trades``, but no fixed take-profit
    -- the stop only ever ratchets in the trade's favor (``trail_mult * ATR``
    behind the best price seen), locking in gains before a winner round-trips
    back into a loss. Exit reasons collapse to STOP/EOD (TARGET never fires)."""
    n = price.shape[0]
    entry_idx = np.empty(n, dtype=np.int64)
    exit_idx = np.empty(n, dtype=np.int64)
    entry_px = np.empty(n, dtype=np.float64)
    exit_px = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=np.int64)
    reason = np.empty(n, dtype=np.int64)
    count = 0
    i = 0
    while i < n - 1:
        sig = signal[i]
        if sig == 0:
            i += 1
            continue
        a = atr_arr[i]
        if np.isnan(a):
            i += 1
            continue
        entry = price[i]
        stop = entry - sig * sl_mult * a

        ex_i = n - 1
        ex_px = price[n - 1]
        r = EOD
        j = i + 1
        while j < n:
            p = price[j]
            aj = atr_arr[j]
            if np.isnan(aj):
                aj = a
            if sig == 1:
                candidate = p - trail_mult * aj
                if candidate > stop:
                    stop = candidate
                if p <= stop:
                    ex_i, ex_px, r = j, stop, STOP
                    break
            else:
                candidate = p + trail_mult * aj
                if candidate < stop:
                    stop = candidate
                if p >= stop:
                    ex_i, ex_px, r = j, stop, STOP
                    break
            j += 1

        entry_idx[count] = i
        exit_idx[count] = ex_i
        entry_px[count] = entry
        exit_px[count] = ex_px
        direction[count] = sig
        reason[count] = r
        count += 1
        i = ex_i + 1

    return (entry_idx[:count], exit_idx[:count], entry_px[:count],
            exit_px[:count], direction[:count], reason[:count])
