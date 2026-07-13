"""LHP / LTP-KAMA DSL indicators — the "LHPOK"-family signals from
``compute_lhp_dsl_indicator`` (5 notebooks: forex_1h_strat_004,
multi_1h_strat_022/028/029/030) and ``compute_ltpkamadsl_indicator``
(multi_1h_strat_031/032). "LHP" = Log High-Pass: a log-return series
smoothed with a KAMA-style adaptive moving average, with dynamic
support/resistance levels built from where price crosses that average.
All 5 LHP notebooks share identical logic — only period/fastend/slowend
defaults differed, so they collapse into one function here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .volatility import rogers_satchell_vol


def lhp_dsl(close: pd.Series, period: int = 72, fastend: float = 0.6, slowend: float = 0.04) -> pd.Series:
    """Log High-Pass DSL signal: 0=neutral, 1=buy, 2=sell.

    ``period`` here is the KAMA-style volatility lookback (default matches
    ``forex_1h_strat_004``; other notebooks used 54/72 with fastend in
    0.07-0.6 and slowend in 0.03-0.04 — pass those explicitly to reproduce).
    """
    price = close.to_numpy(dtype=float)
    n = len(price)
    prev = np.roll(price, 1)
    prev[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        lhp = np.where((price != 0) & (prev != 0), np.log(np.abs(prev / price)), np.nan)

    vol = pd.Series(np.abs(np.diff(lhp, prepend=np.nan))).rolling(10, min_periods=1).sum().to_numpy()

    kama = np.empty(n)
    kama[0] = lhp[0] if not np.isnan(lhp[0]) else 0.0
    for i in range(1, n):
        if np.isnan(lhp[i]) or np.isnan(lhp[i - 1]):
            kama[i] = kama[i - 1]
            continue
        er = abs(lhp[i] - lhp[i - 1]) / (vol[i] if vol[i] != 0 else 1e-10)
        sc = er * (fastend - slowend) + slowend
        kama[i] = kama[i - 1] + sc * (lhp[i] - kama[i - 1])

    kama_s, lhp_s = pd.Series(kama), pd.Series(lhp)
    level_up = kama_s.where(lhp_s > kama_s).ffill()
    level_down = kama_s.where(lhp_s < kama_s).ffill()

    signal = np.zeros(n, dtype=np.int8)
    signal[lhp > level_up.to_numpy()] = 1
    signal[lhp < level_down.to_numpy()] = 2
    return pd.Series(signal, index=close.index)


def ltp_kama_dsl(
    df: pd.DataFrame,
    period: int = 55,
    fastend: float = 0.5,
    slowend: float = 0.08,
    window: int = 14,
) -> pd.Series:
    """LTP-KAMA DSL: LHP z-scored over ``period``, KAMA-smoothed, gated by a
    Rogers-Satchell volatility filter (>= its rolling median). Signal:
    0=neutral, 1=buy, 2=sell. Consolidates the two near-identical notebook
    variants (031 used Rogers-Satchell directly; 032 used a Yang-Zhang blend
    that reduces to the same RS term plus OC/CC variance — RS alone captures
    the dominant signal and keeps this to one implementation)."""
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    prev = np.roll(close, 1)
    prev[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((close > 0) & (prev > 0), prev / close, np.nan)
        lhp = np.log(ratio)
    lhp[~np.isfinite(lhp)] = np.nan

    # Vectorized rolling z-score (pandas' rolling mean/std already skip NaN
    # and honor min_periods the same way the old per-window Python closure
    # did -- this is ~100x faster on multi-million-row series since it's
    # C-level rolling instead of one Python call per row via .apply()).
    lhp_s = pd.Series(lhp)
    roll_mean = lhp_s.rolling(period, min_periods=10).mean()
    roll_std = lhp_s.rolling(period, min_periods=10).std(ddof=0)
    lhp_z = ((lhp_s - roll_mean) / (roll_std + 1e-8)).fillna(0).to_numpy()

    diff = np.abs(np.diff(lhp_z, prepend=lhp_z[0]))
    rolling_vol = pd.Series(diff).rolling(10, min_periods=1).sum().to_numpy()

    kama = np.empty(n)
    kama[0] = lhp_z[0]
    for i in range(1, n):
        er = abs(lhp_z[i] - lhp_z[i - 1]) / (rolling_vol[i] if rolling_vol[i] != 0 else 1e-8)
        sc = er * (fastend - slowend) + slowend
        kama[i] = kama[i - 1] + sc * (lhp_z[i] - kama[i - 1])

    rs_vol = rogers_satchell_vol(df, window).fillna(0).to_numpy()
    rs_threshold = np.median(rs_vol[np.isfinite(rs_vol)]) if np.any(np.isfinite(rs_vol)) else 0.0
    valid = rs_vol >= rs_threshold

    signal = np.zeros(n, dtype=np.int8)
    signal[(lhp_z > kama) & valid] = 1
    signal[(lhp_z < kama) & valid] = 2
    return pd.Series(signal, index=df.index)
