"""Cycle-detection indicators — ``goertzel``/``detect_strongest_cycle``,
``rainflow_hilbert``, ``cfba_jdmx_histogram`` (multi_1h_strat_035/036)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def goertzel(samples: np.ndarray, sample_rate: float, target_freq: float) -> tuple[float, float]:
    """Goertzel algorithm: magnitude/phase of one DFT frequency bin."""
    n = len(samples)
    k = int(0.5 + (n * target_freq) / sample_rate)
    omega = (2 * np.pi * k) / n
    coeff = 2 * np.cos(omega)
    s_prev = s_prev2 = 0.0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    real = s_prev - s_prev2 * np.cos(omega)
    imag = s_prev2 * np.sin(omega)
    return float(np.hypot(real, imag)), float(np.arctan2(imag, real))


def detect_strongest_cycle(series: pd.Series, max_cycle: int = 120, window: int = 100) -> tuple[float, float, float]:
    """Strongest cycle (magnitude, frequency, phase) over the trailing window,
    swept via Goertzel across candidate cycle lengths 5..max_cycle."""
    best_mag = best_freq = best_phase = 0.0
    segment = series.tail(window).dropna().to_numpy()
    for cycle_len in range(5, max_cycle + 1):
        freq = 1 / cycle_len
        mag, phase = goertzel(segment, 1, freq)
        if mag > best_mag:
            best_mag, best_freq, best_phase = mag, freq, phase
    return best_mag, best_freq, best_phase


def rainflow_hilbert(close: pd.Series, smooth_len: int = 42, amplitude_thresh: float = 0.5) -> pd.Series:
    """Hilbert-transform cycle detector on a fast/slow-MA detail series;
    signals on phase-inflection points. Signal: 0=neutral, 1=buy, 2=sell."""
    price = close.to_numpy(dtype=float)
    n = len(price)
    kernel_fast = np.ones(smooth_len) / smooth_len
    kernel_slow = np.ones(smooth_len * 2) / (smooth_len * 2)
    fast_ma = np.convolve(price, kernel_fast, mode="same")
    slow_ma = np.convolve(price, kernel_slow, mode="same")
    detail = fast_ma - slow_ma

    h = np.zeros(n)
    odd = np.arange(1, n, 2)
    h[odd] = 2 / (np.pi * odd)
    imag = np.convolve(detail, h, mode="same")

    amplitude = np.hypot(detail, imag)
    phase = np.arctan2(imag, detail)
    d1 = np.diff(phase, prepend=phase[0])
    d2 = np.diff(d1, prepend=d1[0])

    signal = np.zeros(n, dtype=np.int8)
    buy = (amplitude > amplitude_thresh) & (d1 > 0) & (d2 > 0) & ((phase % (2 * np.pi)) < np.pi / 2)
    sell = (amplitude > amplitude_thresh) & (d1 < 0) & (d2 < 0) & ((phase % (2 * np.pi)) > np.pi)
    signal[buy] = 1
    signal[sell] = 2
    return pd.Series(signal, index=close.index)


def cfba_jdmx_histogram(
    close: pd.Series,
    length: int = 32,
    siglen: int = 5,
    nlen: int = 50,
    cfb_len: int = 8,
    llim: float = 60,
    kama_fast: float = 0.666,
    kama_slow: float = 0.0645,
) -> pd.DataFrame:
    """CFB-adaptive Jurik-style DMX histogram: KAMA trend line, a
    clamped/normalized CFB filter of it, and a signal-line crossover
    histogram. Returns columns [kama, filt, signal_line, histogram, signal]."""
    price = close.to_numpy(dtype=float)
    n = len(price)

    change = close.diff(length).abs()
    volatility = close.diff().abs().rolling(length).sum()
    er = (change / volatility).replace([np.inf, -np.inf], 0).fillna(0).to_numpy()

    fast_sc = 2 / (kama_fast * 100 + 1)
    slow_sc = 2 / (kama_slow * 100 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = np.empty(n)
    kama[0] = price[0]
    for i in range(1, n):
        kama[i] = kama[i - 1] + sc[i] * (price[i] - kama[i - 1])

    filt = np.empty(n)
    filt[0] = kama[0]
    for i in range(1, n):
        diff = np.clip(kama[i] - filt[i - 1], -llim, llim)
        filt[i] = filt[i - 1] + diff / nlen
        if i >= cfb_len:
            filt[i] = filt[i - cfb_len + 1:i + 1].mean()

    filt_s = pd.Series(filt, index=close.index)
    signal_line = filt_s.rolling(siglen, min_periods=1).mean()
    histogram = filt_s - signal_line
    signal = np.where(histogram > 0, 1, np.where(histogram < 0, 2, 0))

    return pd.DataFrame({
        "kama": kama, "filt": filt, "signal_line": signal_line,
        "histogram": histogram, "signal": signal,
    }, index=close.index)
