"""Pattern-based signals — divergence detection, quad-stochastic consensus,
z-score mean reversion, range/box breakout, and a Lyapunov-style mean-reversion
signal. Consolidates ``find_divergences``/``_divergences``, ``add_stoch_signals``/
``generate_quad_stochastic_signals``, ``calculate_z_score`` (2 near-identical
variants — one took a raw spread, one built the spread from 2-leg prices; the
2-leg version is just "build a spread, then z-score it" so only the z-score
half is kept here), ``range_detector``, ``mean_reversion_signal``."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..indicators.momentum import stochastic_oscillator


def find_divergences(oscillator: pd.Series, price: pd.Series, left: int = 5, search: int = 20):
    """Bullish/bearish divergence bar indices: oscillator makes a local
    extreme while price doesn't confirm it. Returns (bullish_idx, bearish_idx)
    as integer position lists (not index labels)."""
    osc, px = oscillator.to_numpy(), price.to_numpy()
    bull, bear = [], []
    for i in range(left, len(osc)):
        if (osc[i] < osc[i - left:i]).all():
            for j in range(1, search + 1):
                if i - j < 0:
                    break
                if osc[i] > osc[i - j] and px[i] < px[i - j]:
                    bull.append(i)
                    break
        if (osc[i] > osc[i - left:i]).all():
            for j in range(1, search + 1):
                if i - j < 0:
                    break
                if osc[i] < osc[i - j] and px[i] > px[i - j]:
                    bear.append(i)
                    break
    return bull, bear


def quad_stochastic_signals(df: pd.DataFrame, periods: list[tuple[int, int, int]] | None = None,
                            left: int = 5, search: int = 20, min_confirm: int = 2) -> pd.Series:
    """Consensus signal across N stochastic configs (default 4): needs
    ``min_confirm`` of them to show both a divergence and a %K/%D crossover
    in the same direction. Generalizes ``add_stoch_signals`` +
    ``generate_quad_stochastic_signals`` (which relied on ambient
    ``STOCHS``/``LEFT`` globals) into one explicit-params function."""
    periods = periods or [(14, 3, 3), (21, 5, 3), (9, 3, 3), (28, 5, 5)]
    n = len(df)
    bull_div_tot = np.zeros(n, dtype=int)
    bear_div_tot = np.zeros(n, dtype=int)
    bull_cross_tot = np.zeros(n, dtype=int)
    bear_cross_tot = np.zeros(n, dtype=int)

    for k_period, d_period, smooth in periods:
        stoch = stochastic_oscillator(df, k_period=k_period, d_period=d_period)
        k_smoothed = stoch["k"].rolling(smooth).mean()
        d_line = k_smoothed.rolling(d_period).mean()

        bull_idx, bear_idx = find_divergences(d_line.fillna(0), df["close"], left, search)
        bull_div_tot[bull_idx] += 1
        bear_div_tot[bear_idx] += 1

        k_shift, d_shift = k_smoothed.shift(1), d_line.shift(1)
        bull_cross_tot += ((k_shift < d_shift) & (k_smoothed > d_line)).fillna(False).astype(int).to_numpy()
        bear_cross_tot += ((k_shift > d_shift) & (k_smoothed < d_line)).fillna(False).astype(int).to_numpy()

    signal = np.where(
        (bull_div_tot >= min_confirm) & (bull_cross_tot >= min_confirm), 1,
        np.where((bear_div_tot >= min_confirm) & (bear_cross_tot >= min_confirm), 2, 0),
    )
    return pd.Series(signal, index=df.index)


def zscore_spread(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of a spread series — ``calculate_z_score``, the core
    of both notebook variants (build your spread, then call this)."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def range_detector(df: pd.DataFrame, length: int = 5, mult: float = 2.0,
                   atr_len: int = 500, volume_mult: float = 1.5) -> pd.DataFrame:
    """Detects consolidation "boxes" (price within an ATR-scaled band around
    its SMA with no outliers for ``length`` bars) and signals breakouts on
    above-average volume. Single-ticker frame — group upstream via
    ``data.loaders.process_by_ticker`` for multi-ticker use."""
    close, high, low, vol = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["volume"].to_numpy()
    n = len(df)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr_v = pd.Series(tr).rolling(atr_len).mean().to_numpy()
    sma_v = pd.Series(close).rolling(length).mean().to_numpy()
    upper, lower = sma_v + atr_v * mult, sma_v - atr_v * mult

    in_range = np.full(n, False)
    if n >= length:
        close_win = np.lib.stride_tricks.sliding_window_view(close, length)
        sma_win = np.lib.stride_tricks.sliding_window_view(sma_v, length)
        atr_win = np.lib.stride_tricks.sliding_window_view(atr_v, length)
        outliers = (np.abs(close_win - sma_win) > atr_win * mult).sum(axis=1)
        in_range[length - 1:] = outliers == 0

    avg_vol = pd.Series(vol).rolling(length).mean().to_numpy()
    volume_ok = vol > avg_vol * volume_mult

    signal = np.zeros(n, dtype=np.int8)
    box_top = box_bottom = None
    in_box = False
    for i in range(n):
        if in_range[i] and not in_box:
            in_box, box_top, box_bottom = True, upper[i], lower[i]
        elif not in_range[i] and in_box:
            in_box = False
        elif in_box:
            box_top, box_bottom = max(box_top, upper[i]), min(box_bottom, lower[i])

        if box_top is not None and volume_ok[i]:
            if close[i] > box_top:
                signal[i] = 1
            elif close[i] < box_bottom:
                signal[i] = 2

    return pd.DataFrame({"atr": atr_v, "sma": sma_v, "upper": upper, "lower": lower,
                         "in_range": in_range, "signal": signal}, index=df.index)


def mean_reversion_signal(df: pd.DataFrame, smooth_length: int = 172, lyap_window: int = 148,
                          rs_window: int = 24, rs_threshold: float = 0.15) -> pd.Series:
    """Mean-reversion signal: fade deviations from an EMA-style baseline that
    exceed a rolling stddev ("Lyapunov" band), gated to the lowest-volatility
    quantile via Rogers-Satchell. Signal: 0=neutral, 1=buy, 2=sell."""
    close, high, low, open_ = (df[c].to_numpy(dtype=float) for c in ("close", "high", "low", "open"))
    n = len(close)
    if n < max(smooth_length, lyap_window, rs_window):
        return pd.Series(0, index=df.index, dtype=np.int8)

    alpha = 2.0 / (smooth_length + 1)
    baseline = np.empty(n)
    baseline[0] = close[0]
    for i in range(1, n):
        baseline[i] = baseline[i - 1] + (close[i] - baseline[i - 1]) * alpha

    deviations = close - baseline
    lyap = pd.Series(deviations).rolling(lyap_window, min_periods=1).std().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        log_hc = np.log(np.divide(high, close, out=np.zeros_like(close), where=close > 0))
        log_ho = np.log(np.divide(high, open_, out=np.zeros_like(open_), where=open_ > 0))
        log_lc = np.log(np.divide(low, close, out=np.zeros_like(close), where=close > 0))
        log_lo = np.log(np.divide(low, open_, out=np.zeros_like(open_), where=open_ > 0))
    rs_vol = pd.Series(log_hc * log_ho + log_lc * log_lo).rolling(rs_window, min_periods=1).mean().to_numpy()
    rs_vol = np.sqrt(np.clip(rs_vol, 0, None))
    vol_cutoff = np.nanquantile(rs_vol, rs_threshold)

    signal = np.zeros(n, dtype=np.int8)
    filt = rs_vol < vol_cutoff
    signal[(deviations < -lyap) & filt] = 1
    signal[(deviations > lyap) & filt] = 2
    return pd.Series(signal, index=df.index)
