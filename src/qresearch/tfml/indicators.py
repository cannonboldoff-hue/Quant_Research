"""Causal technical-indicator primitives (value at bar t uses bars <= t only).

All functions take/return pandas objects aligned to the input index. Stateful
recursions are numba-jitted. These primitives back both the rule-based
strategies (``strategies.py``) and the ML feature set (``features.py``).

References are given per indicator; parameter defaults follow the originators.
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from ..indicators.moving_averages import _jma_core


# --------------------------------------------------------------------------- moving averages
def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA), alpha = 1/n."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def wma(x: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    vals = x.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) >= n:
        conv = np.convolve(vals, w[::-1], mode="valid") / w.sum()
        out[n - 1:] = conv
    return pd.Series(out, index=x.index)


def hma(x: pd.Series, n: int) -> pd.Series:
    """Hull Moving Average (Hull 2005)."""
    return wma(2 * wma(x, max(n // 2, 1)) - wma(x, n), max(int(np.sqrt(n)), 1))


def dema(x: pd.Series, n: int) -> pd.Series:
    """Double EMA (Mulloy 1994)."""
    e1 = ema(x, n)
    return 2 * e1 - ema(e1, n)


def tema(x: pd.Series, n: int) -> pd.Series:
    """Triple EMA (Mulloy 1994)."""
    e1 = ema(x, n); e2 = ema(e1, n); e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def t3(x: pd.Series, n: int, v: float = 0.7) -> pd.Series:
    """Tillson T3 (Tillson 1998)."""
    def gd(s):
        e = ema(s, n)
        return e * (1 + v) - ema(e, n) * v
    return gd(gd(gd(x)))


def zlema(x: pd.Series, n: int) -> pd.Series:
    """Zero-lag EMA (Ehlers & Way 2010)."""
    lag = (n - 1) // 2
    return ema(x + (x - x.shift(lag)), n)


@numba.njit(cache=True)
def _adaptive_core(price, alpha):
    out = np.full(price.shape[0], np.nan)
    started = False
    for i in range(price.shape[0]):
        a = alpha[i]
        if not started:
            if not np.isnan(a) and not np.isnan(price[i]):
                out[i] = price[i]
                started = True
            continue
        if np.isnan(a) or np.isnan(price[i]):
            out[i] = out[i - 1]
        else:
            out[i] = out[i - 1] + a * (price[i] - out[i - 1])
    return out


def efficiency_ratio(x: pd.Series, n: int) -> pd.Series:
    """Kaufman efficiency ratio |x_t - x_{t-n}| / sum|dx| over n bars."""
    change = (x - x.shift(n)).abs()
    vol = x.diff().abs().rolling(n, min_periods=n).sum()
    return change / vol.replace(0, np.nan)


def kama(x: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average (Kaufman 1995)."""
    er = efficiency_ratio(x, n)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    return pd.Series(_adaptive_core(x.to_numpy(float), sc.to_numpy(float)), index=x.index)


def cmo(x: pd.Series, n: int) -> pd.Series:
    """Chande Momentum Oscillator in [-100, 100]."""
    d = x.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).sum()
    dn = (-d.clip(upper=0)).rolling(n, min_periods=n).sum()
    return 100 * (up - dn) / (up + dn).replace(0, np.nan)


def vidya(x: pd.Series, n: int = 14, cmo_n: int = 9) -> pd.Series:
    """Variable Index Dynamic Average (Chande 1992)."""
    alpha = 2 / (n + 1) * cmo(x, cmo_n).abs() / 100
    return pd.Series(_adaptive_core(x.to_numpy(float), alpha.to_numpy(float)), index=x.index)


@numba.njit(cache=True)
def _mcginley_core(price, n):
    out = np.full(price.shape[0], np.nan)
    for i in range(price.shape[0]):
        if i == 0 or np.isnan(out[i - 1]):
            out[i] = price[i]
            continue
        prev = out[i - 1]
        ratio = price[i] / prev if prev != 0 else 1.0
        denom = n * ratio ** 4
        # The textbook recursion overshoots (and can diverge to inf) when price falls far
        # below the average, because (P/MD)^4 -> 0. Cap the step at the full distance to
        # the price; inactive in normal regimes (binds only if P/MD < n**-0.25).
        step = 1.0 if denom <= 1.0 else 1.0 / denom
        out[i] = prev + (price[i] - prev) * step
    return out


def mcginley(x: pd.Series, n: int = 14) -> pd.Series:
    """McGinley Dynamic (McGinley 1990), with the step capped for numerical stability."""
    return pd.Series(_mcginley_core(x.to_numpy(float), float(n)), index=x.index)


def frama(df: pd.DataFrame, n: int = 16, slow: int = 198) -> pd.Series:
    """Fractal Adaptive Moving Average (Ehlers 2005). n must be even."""
    n = max(2 * (n // 2), 4)
    h, l = df["high"], df["low"]
    half = n // 2
    n1 = (h.rolling(half).max() - l.rolling(half).min()) / half
    n2 = (h.shift(half).rolling(half).max() - l.shift(half).rolling(half).min()) / half
    n3 = (h.rolling(n).max() - l.rolling(n).min()) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        dim = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
    w = np.log(2 / (slow + 1))
    alpha = np.exp(w * (dim - 1)).clip(2 / (slow + 1), 1.0)
    price = ((df["high"] + df["low"]) / 2).to_numpy(float)
    return pd.Series(_adaptive_core(price, alpha.to_numpy(float)), index=df.index)


def jma(x: pd.Series, n: int = 7, phase: float = 0.0, power: float = 2.0) -> pd.Series:
    """Jurik-style moving average -- reuses the repository's numba core
    (``qresearch.indicators.moving_averages``) but keeps the input index (the
    original wrapper returned a RangeIndex, misaligning date-indexed inputs)."""
    phase_ratio = float(np.clip(phase / 100.0 + 1.5, 0.5, 2.5))
    beta = 0.45 * (n - 1) / (0.45 * (n - 1) + 2)
    alpha = beta ** power
    vals = x.to_numpy(float)
    out = _jma_core(vals, phase_ratio, beta, alpha)
    out[: n] = np.nan  # warm-up
    return pd.Series(out, index=x.index)


# --------------------------------------------------------------------------- range / volatility
def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                     axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range, Wilder smoothing (Wilder 1978)."""
    return wilder(true_range(df), n)


def realized_vol(close: pd.Series, n: int = 20, ann: int = 252) -> pd.Series:
    return np.log(close).diff().rolling(n, min_periods=n).std() * np.sqrt(ann)


def parkinson_vol(df: pd.DataFrame, n: int = 20, ann: int = 252) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(n, min_periods=n).mean() / (4 * np.log(2)) * ann)


def garman_klass_vol(df: pd.DataFrame, n: int = 20, ann: int = 252) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    v = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(v.clip(lower=0).rolling(n, min_periods=n).mean() * ann)


def rogers_satchell_vol(df: pd.DataFrame, n: int = 20, ann: int = 252) -> pd.Series:
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    v = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(v.clip(lower=0).rolling(n, min_periods=n).mean() * ann)


def yang_zhang_vol(df: pd.DataFrame, n: int = 20, ann: int = 252) -> pd.Series:
    """Yang & Zhang (2000) drift-independent estimator with opening jumps."""
    o, c = df["open"], df["close"]
    oc = np.log(o / c.shift(1))
    co = np.log(c / o)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    rs = rogers_satchell_vol(df, n, ann=1) ** 2
    v = oc.rolling(n).var() + k * co.rolling(n).var() + (1 - k) * rs
    return np.sqrt(v.clip(lower=0) * ann)


# --------------------------------------------------------------------------- directional / oscillators
def dmi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder's Directional Movement: +DI, -DI, ADX (Wilder 1978)."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    pdi = 100 * wilder(plus_dm, n) / tr.replace(0, np.nan)
    mdi = 100 * wilder(minus_dm, n) / tr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": wilder(dx, n)}, index=df.index)


def aroon(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Aroon Up/Down (Chande 1995)."""
    hi_idx = df["high"].rolling(n + 1, min_periods=n + 1).apply(np.argmax, raw=True)
    lo_idx = df["low"].rolling(n + 1, min_periods=n + 1).apply(np.argmin, raw=True)
    return pd.DataFrame({"up": 100 * hi_idx / n, "down": 100 * lo_idx / n}, index=df.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = wilder(d.clip(lower=0), n)
    l_ = wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + g / l_.replace(0, np.nan))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig}, index=close.index)


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3) -> pd.DataFrame:
    lo = df["low"].rolling(n, min_periods=n).min()
    hi = df["high"].rolling(n, min_periods=n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pd.DataFrame({"k": k, "d": k.rolling(d).mean()}, index=df.index)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Commodity Channel Index (Lambert 1980)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def trix(close: pd.Series, n: int = 15) -> pd.Series:
    """TRIX (Hutson 1983): 1-bar % change of triple-smoothed EMA of log price."""
    e = ema(ema(ema(np.log(close), n), n), n)
    return 1e4 * e.diff()


def vortex(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Vortex Indicator (Botes & Siepman 2010)."""
    vmp = (df["high"] - df["low"].shift(1)).abs().rolling(n).sum()
    vmm = (df["low"] - df["high"].shift(1)).abs().rolling(n).sum()
    tr = true_range(df).rolling(n).sum().replace(0, np.nan)
    return pd.DataFrame({"vip": vmp / tr, "vim": vmm / tr}, index=df.index)


def kst(close: pd.Series, scale: float = 1.0) -> pd.DataFrame:
    """Know Sure Thing (Pring 1992), daily parameterisation; ``scale`` stretches lookbacks."""
    s = lambda n: max(2, int(round(n * scale)))
    roc = lambda n: 100 * (close / close.shift(s(n)) - 1)
    k = (sma(roc(10), s(10)) + 2 * sma(roc(15), s(10)) + 3 * sma(roc(20), s(10)) + 4 * sma(roc(30), s(15)))
    return pd.DataFrame({"kst": k, "signal": sma(k, 9)}, index=close.index)


def tsi(close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    """True Strength Index (Blau 1991)."""
    m = close.diff()
    return 100 * ema(ema(m, r), s) / ema(ema(m.abs(), r), s).replace(0, np.nan)


def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Choppiness Index (Dreiss): 100 = choppy, 0 = trending."""
    s = true_range(df).rolling(n).sum()
    rng = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    return 100 * np.log10(s / rng.replace(0, np.nan)) / np.log10(n)


def linreg_stats(y: pd.Series, n: int) -> pd.DataFrame:
    """Rolling OLS of y on time: slope, R^2, slope t-stat (vectorised)."""
    t = np.arange(n, dtype=float)
    tm = t.mean()
    sxx = ((t - tm) ** 2).sum()
    ym = y.rolling(n, min_periods=n).mean()
    # sum(t*y) over window via rolling dot product with weights
    vals = y.to_numpy(float)
    sty = np.full(len(vals), np.nan)
    if len(vals) >= n:
        sty[n - 1:] = np.convolve(vals, t[::-1], mode="valid")
    sty = pd.Series(sty, index=y.index)
    slope = (sty - tm * n * ym) / sxx
    syy = y.rolling(n, min_periods=n).var(ddof=0) * n
    r2 = (slope ** 2 * sxx / syy.replace(0, np.nan)).clip(0, 1)
    resid_var = (syy - slope ** 2 * sxx).clip(lower=0) / max(n - 2, 1)
    tstat = slope / np.sqrt(resid_var / sxx).replace(0, np.nan)
    return pd.DataFrame({"slope": slope, "r2": r2, "tstat": tstat}, index=y.index)


# --------------------------------------------------------------------------- channels / stops
def donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Prior-n-bar channel (excludes the current bar, so a breakout is close > upper)."""
    return pd.DataFrame({"upper": df["high"].rolling(n, min_periods=n).max().shift(1),
                         "lower": df["low"].rolling(n, min_periods=n).min().shift(1)}, index=df.index)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    m = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": m, "upper": m + k * sd, "lower": m - k * sd}, index=close.index)


def keltner(df: pd.DataFrame, n: int = 20, atr_n: int = 10, k: float = 2.0) -> pd.DataFrame:
    m = ema(df["close"], n)
    a = atr(df, atr_n)
    return pd.DataFrame({"mid": m, "upper": m + k * a, "lower": m - k * a}, index=df.index)


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    """Ichimoku Kinko Hyo (Hosoda). Spans are displaced forward by ``kijun`` bars,
    so the cloud at bar t uses data from t-kijun (causal)."""
    mid = lambda n: (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2
    t, k = mid(tenkan), mid(kijun)
    a = ((t + k) / 2).shift(kijun)
    b = mid(senkou).shift(kijun)
    return pd.DataFrame({"tenkan": t, "kijun": k, "span_a": a, "span_b": b}, index=df.index)


@numba.njit(cache=True)
def _psar_core(high, low, af0, af_max):
    n = high.shape[0]
    sar = np.full(n, np.nan)
    trend = np.zeros(n)
    if n < 3:
        return sar, trend
    up = high[1] >= high[0]
    ep = high[1] if up else low[1]
    s = low[0] if up else high[0]
    af = af0
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < s:
                up = False; s = ep; ep = low[i]; af = af0
            elif high[i] > ep:
                ep = high[i]; af = min(af + af0, af_max)
        else:
            s = max(s, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > s:
                up = True; s = ep; ep = high[i]; af = af0
            elif low[i] < ep:
                ep = low[i]; af = min(af + af0, af_max)
        sar[i] = s
        trend[i] = 1.0 if up else -1.0
    return sar, trend


def psar(df: pd.DataFrame, af: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """Parabolic SAR (Wilder 1978)."""
    s, tr = _psar_core(df["high"].to_numpy(float), df["low"].to_numpy(float), af, af_max)
    return pd.DataFrame({"sar": s, "trend": tr}, index=df.index)


@numba.njit(cache=True)
def _band_flip_core(close, upper, lower):
    """Generic stop-and-reverse on ratcheting bands (SuperTrend / Chandelier-style).
    upper: short-side stop (price above -> go long); lower: long-side stop."""
    n = close.shape[0]
    trend = np.zeros(n)
    line = np.full(n, np.nan)
    fu = np.nan
    fl = np.nan
    t = 0.0
    for i in range(n):
        u, l = upper[i], lower[i]
        if np.isnan(u) or np.isnan(l):
            continue
        if np.isnan(fu):
            fu, fl = u, l
            t = 1.0 if close[i] >= (u + l) / 2 else -1.0
        else:
            fu = u if (u < fu or close[i - 1] > fu) else fu
            fl = l if (l > fl or close[i - 1] < fl) else fl
            if t <= 0 and close[i] > fu:
                t = 1.0
            elif t >= 0 and close[i] < fl:
                t = -1.0
        trend[i] = t
        line[i] = fl if t > 0 else fu
    return trend, line


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """SuperTrend (Seban): ATR bands around median price with ratcheting."""
    mid = (df["high"] + df["low"]) / 2
    a = atr(df, n)
    tr, line = _band_flip_core(df["close"].to_numpy(float), (mid + mult * a).to_numpy(float),
                               (mid - mult * a).to_numpy(float))
    return pd.DataFrame({"trend": tr, "line": line}, index=df.index)


def chandelier(df: pd.DataFrame, n: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """Chandelier Exit (LeBeau): long stop = HH(n) - k*ATR, short stop = LL(n) + k*ATR,
    used here as a stop-and-reverse system."""
    a = atr(df, n)
    long_stop = df["high"].rolling(n).max() - mult * a
    short_stop = df["low"].rolling(n).min() + mult * a
    tr, line = _band_flip_core(df["close"].to_numpy(float), short_stop.to_numpy(float),
                               long_stop.to_numpy(float))
    return pd.DataFrame({"trend": tr, "line": line}, index=df.index)


def chande_kroll(df: pd.DataFrame, p: int = 10, x: float = 1.0, q: int = 9) -> pd.DataFrame:
    """Chande-Kroll stop (Chande & Kroll 1994)."""
    a = atr(df, p)
    first_high = df["high"].rolling(p).max() - x * a
    first_low = df["low"].rolling(p).min() + x * a
    return pd.DataFrame({"stop_short": first_high.rolling(q).max(),
                         "stop_long": first_low.rolling(q).min()}, index=df.index)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    o = np.empty(len(df))
    hc = ha_close.to_numpy(float)
    o[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        o[i] = (o[i - 1] + hc[i - 1]) / 2
    return pd.DataFrame({"ha_open": o, "ha_close": hc}, index=df.index)


# --------------------------------------------------------------------------- volume
def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    vol = df["volume"]
    return (mfm.fillna(0) * vol).rolling(n).sum() / vol.rolling(n).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))
