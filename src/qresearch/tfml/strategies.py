"""Established trend-following strategies, implemented as faithful rule-based baselines.

Each strategy maps an OHLCV frame to a *target position* series p_t in [-1, 1],
decided with information up to and including the close of bar t. Execution
(next open / next close) and costs are applied by ``backtest.py``, never here.

Conventions:
- Long/short ("stop-and-reverse") unless the originating source is long-only
  (Faber 2007; Antonacci 2014), flagged ``long_only=True``.
- ``scale`` lists the lookback parameters stretched/compressed to build the
  slow (x2) / fast (x0.5) variants used by the parameter-adaptation ML stage;
  entries with a ``-`` prefix are scaled inversely (e.g. PSAR acceleration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as I


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    family: str
    reference: str
    fn: Callable
    params: dict
    indicators: tuple
    scale: tuple = ()
    long_only: bool = False

    def positions(self, df: pd.DataFrame, ctx: dict | None = None, **override) -> pd.Series:
        p = {**self.params, **override}
        pos = self.fn(df, ctx or {}, **p)
        pos = pd.Series(pos, index=df.index, dtype=float).fillna(0.0).clip(-1, 1)
        if self.long_only:
            pos = pos.clip(lower=0)
        return pos

    def variant_params(self, factor: float) -> dict:
        """Lookback-scaled parameters (factor 2 = slower, 0.5 = faster)."""
        out = dict(self.params)
        for name in self.scale:
            inv = name.startswith("-")
            key = name.lstrip("-")
            v = self.params[key]
            f = 1 / factor if inv else factor
            if isinstance(v, int):
                out[key] = max(2, int(round(v * f)))
            else:
                out[key] = float(v * f)
        return out


# --------------------------------------------------------------------------- helpers
def _hold(long_cond: pd.Series, short_cond: pd.Series, exit_long: pd.Series | None = None,
          exit_short: pd.Series | None = None) -> np.ndarray:
    """State machine: enter on conditions, hold until an exit/opposite entry."""
    lc = long_cond.fillna(False).to_numpy(bool)
    sc = short_cond.fillna(False).to_numpy(bool)
    el = exit_long.fillna(False).to_numpy(bool) if exit_long is not None else np.zeros(len(lc), bool)
    es = exit_short.fillna(False).to_numpy(bool) if exit_short is not None else np.zeros(len(lc), bool)
    return _hold_core(lc, sc, el, es)


try:
    import numba

    @numba.njit(cache=True)
    def _hold_core(lc, sc, el, es):
        n = lc.shape[0]
        out = np.zeros(n)
        p = 0.0
        for i in range(n):
            if lc[i] and not sc[i]:
                p = 1.0
            elif sc[i] and not lc[i]:
                p = -1.0
            elif p > 0 and el[i]:
                p = 0.0
            elif p < 0 and es[i]:
                p = 0.0
            out[i] = p
        return out
except Exception:  # pragma: no cover
    def _hold_core(lc, sc, el, es):
        out = np.zeros(len(lc)); p = 0.0
        for i in range(len(lc)):
            if lc[i] and not sc[i]: p = 1.0
            elif sc[i] and not lc[i]: p = -1.0
            elif p > 0 and el[i]: p = 0.0
            elif p < 0 and es[i]: p = 0.0
            out[i] = p
        return out


def _sign(x: pd.Series) -> pd.Series:
    return np.sign(x).where(x.notna(), 0.0)


def _month_end_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """True on the last bar of each calendar month. The final bar is flagged only if the
    next business day falls in a new month (known from the calendar, so this stays causal:
    a truncated history never treats a mid-month bar as a month end)."""
    m = idx.to_period("M")
    last = idx[-1]
    nxt = last + (pd.Timedelta(days=1) if (idx.dayofweek >= 5).any() else pd.offsets.BDay(1))
    return np.r_[m[1:] != m[:-1], nxt.month != last.month]


# --------------------------------------------------------------------------- strategy rules
def s_ma_cross(df, ctx, fast, slow, kind="sma"):
    f = getattr(I, kind)
    return _sign(f(df["close"], fast) - f(df["close"], slow))


def s_bll_vma(df, ctx, n, band):
    """Brock, Lakonishok & LeBaron (1992) variable-length MA with a 1% band."""
    c, m = df["close"], I.sma(df["close"], n)
    return _hold(c > m * (1 + band), c < m * (1 - band))


def s_faber(df, ctx, months):
    """Faber (2007) 10-month SMA timing: month-end close vs 10-month SMA, long/cash."""
    me = _month_end_mask(df.index)
    mc = df["close"][me]
    sig = (mc > mc.rolling(months, min_periods=months).mean()).astype(float)
    sig[mc.rolling(months, min_periods=months).mean().isna()] = 0.0
    return sig.reindex(df.index).ffill().fillna(0.0)


def s_triple_ma(df, ctx, a, b, c_):
    c = df["close"]
    x, y, z = I.sma(c, a), I.sma(c, b), I.sma(c, c_)
    return pd.Series(np.where((x > y) & (y > z), 1.0, np.where((x < y) & (y < z), -1.0, 0.0)), index=df.index)


def s_donchian(df, ctx, n):
    ch = I.donchian(df, n)
    return _hold(df["close"] > ch["upper"], df["close"] < ch["lower"])


def s_turtle(df, ctx, entry, exit):
    """Turtle system (Dennis & Eckhardt; Faith 2007): enter on N-day breakout,
    exit on M-day opposite breakout. (Unit pyramiding / N-sizing left to the sizing layer.)"""
    en, ex = I.donchian(df, entry), I.donchian(df, exit)
    c = df["close"]
    return _hold(c > en["upper"], c < en["lower"], exit_long=c < ex["lower"], exit_short=c > ex["upper"])


def s_macd(df, ctx, fast, slow, signal):
    m = I.macd(df["close"], fast, slow, signal)
    return _sign(m["hist"])


def s_tsmom(df, ctx, n):
    """Time-series momentum (Moskowitz, Ooi & Pedersen 2012): sign of past n-bar return."""
    return _sign(df["close"] / df["close"].shift(n) - 1)


def s_tsmom_multi(df, ctx, n1, n2, n3):
    """Multi-horizon TSMOM, equal-weight of 1/3/12-month signs (Hurst, Ooi & Pedersen 2017)."""
    c = df["close"]
    return (_sign(c / c.shift(n1) - 1) + _sign(c / c.shift(n2) - 1) + _sign(c / c.shift(n3) - 1)) / 3


def s_abs_momentum(df, ctx, n):
    """Antonacci (2014) absolute momentum: long if n-bar return exceeds T-bill return, else cash.
    Evaluated monthly."""
    c = df["close"]
    rf = ctx.get("rf")
    rf_n = (rf.reindex(df.index).ffill().fillna(0.0) * n / 252) if rf is not None else 0.0
    sig = ((c / c.shift(n) - 1) > rf_n).astype(float).where(c.shift(n).notna(), 0.0)
    me = _month_end_mask(df.index)
    return sig.where(me).ffill().fillna(0.0)


def s_bollinger(df, ctx, n, k):
    bb = I.bollinger(df["close"], n, k)
    c = df["close"]
    return _hold(c > bb["upper"], c < bb["lower"], exit_long=c < bb["mid"], exit_short=c > bb["mid"])


def s_keltner(df, ctx, n, atr_n, k):
    kc = I.keltner(df, n, atr_n, k)
    c = df["close"]
    return _hold(c > kc["upper"], c < kc["lower"], exit_long=c < kc["mid"], exit_short=c > kc["mid"])


def s_supertrend(df, ctx, n, mult):
    return I.supertrend(df, n, mult)["trend"]


def s_psar(df, ctx, af, af_max):
    return I.psar(df, af, af_max)["trend"]


def s_adx(df, ctx, n, threshold):
    d = I.dmi(df, n)
    strong = d["adx"] > threshold
    return pd.Series(np.where(strong & (d["pdi"] > d["mdi"]), 1.0,
                              np.where(strong & (d["mdi"] > d["pdi"]), -1.0, 0.0)), index=df.index)


def s_aroon(df, ctx, n):
    ar = I.aroon(df, n)
    return _sign(ar["up"] - ar["down"])


def s_ichimoku(df, ctx, tenkan, kijun, senkou):
    ich = I.ichimoku(df, tenkan, kijun, senkou)
    c = df["close"]
    top = ich[["span_a", "span_b"]].max(axis=1)
    bot = ich[["span_a", "span_b"]].min(axis=1)
    lc = (c > top) & (ich["tenkan"] > ich["kijun"])
    sc = (c < bot) & (ich["tenkan"] < ich["kijun"])
    return pd.Series(np.where(lc, 1.0, np.where(sc, -1.0, 0.0)), index=df.index)


def s_price_vs_ma(df, ctx, n, kind, **kw):
    f = getattr(I, kind)
    return _sign(df["close"] - f(df["close"], n, **kw))


def s_kama(df, ctx, n, fast, slow):
    return _sign(df["close"] - I.kama(df["close"], n, fast, slow))


def s_frama(df, ctx, n):
    return _sign(df["close"] - I.frama(df, n))


def s_trix(df, ctx, n, signal):
    t = I.trix(df["close"], n)
    return _sign(t - I.ema(t, signal))


def s_linreg(df, ctx, n):
    return _sign(I.linreg_stats(np.log(df["close"]), n)["slope"])


def s_vortex(df, ctx, n):
    v = I.vortex(df, n)
    return _sign(v["vip"] - v["vim"])


def s_hma(df, ctx, n):
    h = I.hma(df["close"], n)
    return _sign(h - h.shift(1))


def s_elder(df, ctx, ema_n, fast, slow, signal):
    """Elder (2002) Impulse System: green = EMA rising & MACD-hist rising (long),
    red = both falling (short), otherwise neutral."""
    e = I.ema(df["close"], ema_n)
    h = I.macd(df["close"], fast, slow, signal)["hist"]
    up = (e > e.shift(1)) & (h > h.shift(1))
    dn = (e < e.shift(1)) & (h < h.shift(1))
    return pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=df.index)


def s_heikin(df, ctx, n):
    """Heikin-Ashi trend: hold the direction of n consecutive same-colour HA candles."""
    ha = I.heikin_ashi(df)
    bull = (ha["ha_close"] > ha["ha_open"]).astype(int)
    bear = (ha["ha_close"] < ha["ha_open"]).astype(int)
    return _hold(bull.rolling(n).sum() == n, bear.rolling(n).sum() == n)


def s_kst(df, ctx, scale=1.0):
    k = I.kst(df["close"], scale)
    return _sign(k["kst"] - k["signal"])


def s_chandelier(df, ctx, n, mult):
    return I.chandelier(df, n, mult)["trend"]


def s_chande_kroll(df, ctx, p, x, q):
    ck = I.chande_kroll(df, p, x, q)
    c = df["close"]
    return _hold(c > ck["stop_short"], c < ck["stop_long"])


def s_cci(df, ctx, n, level):
    v = I.cci(df, n)
    return _hold(v > level, v < -level)


def s_gmma(df, ctx, scale=1.0):
    """Guppy Multiple Moving Average: all short EMAs above all long EMAs -> long."""
    c = df["close"]
    short = [I.ema(c, max(2, int(round(n * scale)))) for n in (3, 5, 8, 10, 12, 15)]
    long_ = [I.ema(c, max(2, int(round(n * scale)))) for n in (30, 35, 40, 45, 50, 60)]
    smin = pd.concat(short, axis=1).min(axis=1); smax = pd.concat(short, axis=1).max(axis=1)
    lmin = pd.concat(long_, axis=1).min(axis=1); lmax = pd.concat(long_, axis=1).max(axis=1)
    return pd.Series(np.where(smin > lmax, 1.0, np.where(smax < lmin, -1.0, 0.0)), index=df.index)


def s_jma_cross(df, ctx, fast, slow):
    """JMA fast/slow crossover -- the repository's original signal family
    (``qresearch.signals.generators.jma_signals``), held as a position."""
    return _sign(I.jma(df["close"], fast) - I.jma(df["close"], slow))


def s_tsi(df, ctx, r, s, signal):
    t = I.tsi(df["close"], r, s)
    return _sign(t - I.ema(t, signal))


# --------------------------------------------------------------------------- registry
S = Strategy
STRATEGIES: list[Strategy] = [
    S("sma_50_200", "SMA 50/200 crossover (golden/death cross)", "ma_crossover",
      "Brock, Lakonishok & LeBaron (1992); Gartley (1935)", s_ma_cross,
      {"fast": 50, "slow": 200, "kind": "sma"}, ("SMA",), ("fast", "slow")),
    S("ema_12_26", "EMA 12/26 crossover", "ma_crossover", "Appel (1979); standard EMA crossover",
      s_ma_cross, {"fast": 12, "slow": 26, "kind": "ema"}, ("EMA",), ("fast", "slow")),
    S("dema_20_50", "DEMA 20/50 crossover", "ma_crossover", "Mulloy (1994)", s_ma_cross,
      {"fast": 20, "slow": 50, "kind": "dema"}, ("DEMA",), ("fast", "slow")),
    S("zlema_10_30", "Zero-lag EMA 10/30 crossover", "ma_crossover", "Ehlers & Way (2010)", s_ma_cross,
      {"fast": 10, "slow": 30, "kind": "zlema"}, ("ZLEMA",), ("fast", "slow")),
    S("jma_7_21", "Jurik MA 7/21 crossover (repository baseline)", "ma_crossover",
      "Jurik Research; this repository's JMA signal family", s_jma_cross,
      {"fast": 7, "slow": 21}, ("JMA",), ("fast", "slow")),
    S("triple_ma_4_9_18", "Triple MA 4/9/18", "ma_crossover", "Allen (1972) 4-9-18 day method",
      s_triple_ma, {"a": 4, "b": 9, "c_": 18}, ("SMA",), ("a", "b", "c_")),
    S("gmma", "Guppy multiple moving average", "ma_crossover", "Guppy (2004)", s_gmma,
      {"scale": 1.0}, ("EMA x12",), ("scale",)),
    S("bll_vma_200", "Price vs 200-day MA with 1% band", "price_vs_ma",
      "Brock, Lakonishok & LeBaron (1992)", s_bll_vma, {"n": 200, "band": 0.01}, ("SMA",), ("n",)),
    S("faber_10m", "Faber 10-month SMA timing (long/cash)", "price_vs_ma", "Faber (2007)", s_faber,
      {"months": 10}, ("monthly SMA",), ("months",), long_only=True),
    S("kama_10", "Price vs KAMA(10,2,30)", "adaptive_ma", "Kaufman (1995)", s_kama,
      {"n": 10, "fast": 2, "slow": 30}, ("KAMA", "efficiency ratio"), ("n", "slow")),
    S("frama_16", "Price vs FRAMA(16)", "adaptive_ma", "Ehlers (2005)", s_frama, {"n": 16},
      ("FRAMA",), ("n",)),
    S("vidya_14", "Price vs VIDYA(14, CMO 9)", "adaptive_ma", "Chande (1992)", s_price_vs_ma,
      {"n": 14, "kind": "vidya"}, ("VIDYA", "CMO"), ("n",)),
    S("mcginley_14", "Price vs McGinley Dynamic(14)", "adaptive_ma", "McGinley (1990)", s_price_vs_ma,
      {"n": 14, "kind": "mcginley"}, ("McGinley Dynamic",), ("n",)),
    S("t3_20", "Price vs Tillson T3(20)", "adaptive_ma", "Tillson (1998)", s_price_vs_ma,
      {"n": 20, "kind": "t3"}, ("T3",), ("n",)),
    S("hma_55_slope", "Hull MA(55) slope", "adaptive_ma", "Hull (2005)", s_hma, {"n": 55}, ("HMA",), ("n",)),
    S("donchian_20", "Donchian 20-day channel breakout (stop-and-reverse)", "breakout",
      "Donchian (1960); 4-week rule", s_donchian, {"n": 20}, ("Donchian channel",), ("n",)),
    S("turtle_20_10", "Turtle System 1 (20/10)", "breakout", "Dennis & Eckhardt; Faith (2007)", s_turtle,
      {"entry": 20, "exit": 10}, ("Donchian channel",), ("entry", "exit")),
    S("turtle_55_20", "Turtle System 2 (55/20)", "breakout", "Dennis & Eckhardt; Faith (2007)", s_turtle,
      {"entry": 55, "exit": 20}, ("Donchian channel",), ("entry", "exit")),
    S("bollinger_20_2", "Bollinger band breakout, exit at mid", "breakout", "Bollinger (2001)",
      s_bollinger, {"n": 20, "k": 2.0}, ("Bollinger bands",), ("n",)),
    S("keltner_20", "Keltner channel breakout, exit at mid", "breakout", "Keltner (1960); Raschke ATR version",
      s_keltner, {"n": 20, "atr_n": 10, "k": 2.0}, ("Keltner channel", "ATR"), ("n", "atr_n")),
    S("supertrend_10_3", "SuperTrend(10,3)", "volatility_stop", "Seban (SuperTrend); ATR bands",
      s_supertrend, {"n": 10, "mult": 3.0}, ("ATR", "SuperTrend"), ("n",)),
    S("psar", "Parabolic SAR (0.02/0.2)", "volatility_stop", "Wilder (1978)", s_psar,
      {"af": 0.02, "af_max": 0.2}, ("Parabolic SAR",), ("-af", "-af_max")),
    S("chandelier_22_3", "Chandelier exit stop-and-reverse (22,3)", "volatility_stop", "LeBeau & Lucas (1992)",
      s_chandelier, {"n": 22, "mult": 3.0}, ("ATR", "highest high/lowest low"), ("n",)),
    S("chande_kroll", "Chande-Kroll stop (10,1,9)", "volatility_stop", "Chande & Kroll (1994)",
      s_chande_kroll, {"p": 10, "x": 1.0, "q": 9}, ("ATR",), ("p", "q")),
    S("tsmom_252", "Time-series momentum 12m", "time_series_momentum", "Moskowitz, Ooi & Pedersen (2012)",
      s_tsmom, {"n": 252}, ("ROC",), ("n",)),
    S("tsmom_multi", "Multi-horizon TSMOM (1/3/12m)", "time_series_momentum",
      "Hurst, Ooi & Pedersen (2017)", s_tsmom_multi, {"n1": 21, "n2": 63, "n3": 252}, ("ROC",),
      ("n1", "n2", "n3")),
    S("abs_mom_12m", "Absolute momentum vs T-bills (long/cash)", "time_series_momentum",
      "Antonacci (2014)", s_abs_momentum, {"n": 252}, ("ROC", "T-bill rate"), ("n",), long_only=True),
    S("linreg_63", "63-day regression slope sign", "time_series_momentum",
      "Clenow (2015); regression trend", s_linreg, {"n": 63}, ("linear regression",), ("n",)),
    S("macd_12_26_9", "MACD signal-line crossover", "oscillator_trend", "Appel (1979)", s_macd,
      {"fast": 12, "slow": 26, "signal": 9}, ("MACD",), ("fast", "slow", "signal")),
    S("trix_15", "TRIX(15) signal crossover", "oscillator_trend", "Hutson (1983)", s_trix,
      {"n": 15, "signal": 9}, ("TRIX",), ("n", "signal")),
    S("kst", "Know Sure Thing signal crossover", "oscillator_trend", "Pring (1992)", s_kst, {"scale": 1.0}, ("KST", "ROC"), ("scale",)),
    S("tsi_25_13", "True Strength Index signal crossover", "oscillator_trend", "Blau (1991)", s_tsi,
      {"r": 25, "s": 13, "signal": 7}, ("TSI",), ("r", "s")),
    S("cci_20_100", "CCI(20) +/-100 trend rule", "oscillator_trend", "Lambert (1980)", s_cci,
      {"n": 20, "level": 100.0}, ("CCI",), ("n",)),
    S("elder_impulse", "Elder Impulse system", "oscillator_trend", "Elder (2002)", s_elder,
      {"ema_n": 13, "fast": 12, "slow": 26, "signal": 9}, ("EMA", "MACD"), ("ema_n", "fast", "slow")),
    S("adx_dmi_14", "ADX/DMI (ADX>25, DI direction)", "directional", "Wilder (1978)", s_adx,
      {"n": 14, "threshold": 25.0}, ("ADX", "+DI/-DI"), ("n",)),
    S("aroon_25", "Aroon up/down crossover", "directional", "Chande (1995)", s_aroon, {"n": 25}, ("Aroon",), ("n",)),
    S("vortex_14", "Vortex indicator crossover", "directional", "Botes & Siepman (2010)", s_vortex,
      {"n": 14}, ("Vortex",), ("n",)),
    S("ichimoku", "Ichimoku cloud + TK cross", "directional", "Hosoda (1969)", s_ichimoku,
      {"tenkan": 9, "kijun": 26, "senkou": 52}, ("Ichimoku",), ("tenkan", "kijun", "senkou")),
    S("heikin_ashi_3", "Heikin-Ashi 3-candle trend", "directional", "Heikin-Ashi (Homma tradition)",
      s_heikin, {"n": 3}, ("Heikin-Ashi",), ("n",)),
]
BY_ID = {s.id: s for s in STRATEGIES}


def get(strategy_id: str) -> Strategy:
    return BY_ID[strategy_id]
