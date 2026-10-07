"""ML feature set: scale-free, causal transforms of technical indicators.

Every feature is dimensionless (ratios, oscillators, z-scores, vol-normalised
distances) so one pooled model can be trained across instruments with very
different price levels and volatilities. Each feature at bar t uses bars <= t.

``FEATURE_SPECS`` documents each feature's indicator family and source; the
paper's indicator table is generated from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as I

# name -> (family, description / reference)
FEATURE_SPECS: dict[str, tuple[str, str]] = {
    "mom_5": ("momentum", "5-bar log return / (vol*sqrt(5))"),
    "mom_21": ("momentum", "21-bar vol-scaled log return"),
    "mom_63": ("momentum", "63-bar vol-scaled log return"),
    "mom_126": ("momentum", "126-bar vol-scaled log return"),
    "mom_252": ("momentum", "252-bar vol-scaled log return (Moskowitz et al. 2012)"),
    "mom_252_21": ("momentum", "12-1 momentum, vol-scaled (Jegadeesh & Titman 1993)"),
    "vol_20": ("volatility", "20-bar realised vol (annualised)"),
    "vol_ratio": ("volatility", "realised vol 20 / realised vol 120"),
    "vol_of_vol": ("volatility", "std of 20-bar vol over 60 bars / mean"),
    "parkinson_ratio": ("volatility", "Parkinson (1980) vol / close-close vol"),
    "gk_ratio": ("volatility", "Garman-Klass (1980) vol / close-close vol"),
    "rs_ratio": ("volatility", "Rogers-Satchell (1991) vol / close-close vol"),
    "yz_ratio": ("volatility", "Yang-Zhang (2000) vol / close-close vol"),
    "atr_pct": ("volatility", "ATR(14) / close (Wilder 1978)"),
    "adx": ("trend_strength", "ADX(14) / 100 (Wilder 1978)"),
    "di_diff": ("trend_direction", "(+DI - -DI)/(+DI + -DI)"),
    "aroon_osc": ("trend_direction", "(Aroon up - down)/100, n=25 (Chande 1995)"),
    "rsi": ("oscillator", "RSI(14)/100 - 0.5 (Wilder 1978)"),
    "macd_hist": ("trend_direction", "MACD(12,26,9) histogram / ATR (Appel)"),
    "macd_line": ("trend_direction", "MACD line / ATR"),
    "stoch_k": ("oscillator", "Stochastic %K(14)/100 - 0.5 (Lane)"),
    "cci": ("oscillator", "CCI(20)/100 (Lambert 1980)"),
    "trix": ("trend_direction", "TRIX(15) z-scored by 252-bar std (Hutson 1983)"),
    "vortex_diff": ("trend_direction", "VI+ - VI- (Botes & Siepman 2010)"),
    "er_10": ("trend_strength", "Kaufman efficiency ratio, 10 bars"),
    "er_63": ("trend_strength", "Kaufman efficiency ratio, 63 bars"),
    "chop": ("trend_strength", "Choppiness index(14)/100 (Dreiss)"),
    "lr_tstat_63": ("trend_strength", "t-stat of 63-bar log-price regression slope / 10"),
    "lr_r2_63": ("trend_strength", "R^2 of 63-bar log-price regression"),
    "lr_r2_126": ("trend_strength", "R^2 of 126-bar log-price regression"),
    "dist_sma50": ("trend_direction", "(close - SMA50)/ATR"),
    "dist_sma200": ("trend_direction", "(close - SMA200)/ATR"),
    "sma50_200": ("trend_direction", "SMA50/SMA200 - 1, vol-scaled"),
    "bb_pctb": ("channel", "Bollinger %b(20,2) - 0.5"),
    "bb_width": ("volatility", "Bollinger bandwidth / its 252-bar median"),
    "donch_pos_20": ("channel", "position in 20-bar Donchian channel - 0.5"),
    "donch_pos_55": ("channel", "position in 55-bar Donchian channel - 0.5"),
    "keltner_pos": ("channel", "(close - EMA20)/(2*ATR10)"),
    "ichimoku_dist": ("trend_direction", "(close - cloud mid)/ATR (Hosoda)"),
    "psar_dist": ("trend_direction", "(close - PSAR)/ATR (Wilder 1978)"),
    "supertrend_dist": ("trend_direction", "(close - SuperTrend line)/ATR"),
    "kama_dist": ("trend_direction", "(close - KAMA(10,2,30))/ATR (Kaufman)"),
    "hma_slope": ("trend_direction", "5-bar change of HMA(55)/ATR (Hull)"),
    "tsi": ("momentum", "TSI(25,13)/100 (Blau 1991)"),
    "kst": ("momentum", "KST - signal, z-scored (Pring)"),
    "var_ratio_5": ("serial_dependence", "variance ratio VR(5) over 126 bars - 1 (Lo & MacKinlay 1988)"),
    "autocorr_1": ("serial_dependence", "lag-1 return autocorrelation over 63 bars"),
    "skew_63": ("distribution", "63-bar return skewness"),
    "kurt_63": ("distribution", "63-bar return excess kurtosis / 10"),
    "dd_252": ("drawdown", "close / 252-bar max close - 1, vol-scaled"),
    "days_since_high": ("drawdown", "bars since 252-bar high / 252"),
    "ulcer_14": ("drawdown", "Ulcer index(14) / vol"),
    "vol_z": ("volume", "log-volume z-score (20); 0 if no volume"),
    "obv_slope": ("volume", "20-bar OBV change / 20-bar volume sum"),
    "cmf": ("volume", "Chaikin money flow(20)"),
    "mfi": ("volume", "MFI(14)/100 - 0.5"),
}
FEATURE_NAMES = list(FEATURE_SPECS)


def _z(x: pd.Series, n: int = 252) -> pd.Series:
    return (x - x.rolling(n, min_periods=60).mean()) / x.rolling(n, min_periods=60).std().replace(0, np.nan)


def compute_features(df: pd.DataFrame, ann: int = 252) -> pd.DataFrame:
    """Feature matrix for one instrument's OHLCV frame (DatetimeIndex)."""
    c = df["close"]
    lc = np.log(c)
    lr = lc.diff()
    dvol = lr.ewm(span=60, adjust=False, min_periods=20).std()  # daily vol, EWMA (no look-ahead)
    a14 = I.atr(df, 14).replace(0, np.nan)
    f = {}
    for h in (5, 21, 63, 126, 252):
        f[f"mom_{h}"] = (lc - lc.shift(h)) / (dvol * np.sqrt(h))
    f["mom_252_21"] = (lc.shift(21) - lc.shift(252)) / (dvol * np.sqrt(231))
    rv20 = I.realized_vol(c, 20, ann)
    rv120 = I.realized_vol(c, 120, ann)
    f["vol_20"] = rv20
    f["vol_ratio"] = rv20 / rv120
    f["vol_of_vol"] = rv20.rolling(60).std() / rv20.rolling(60).mean()
    f["parkinson_ratio"] = I.parkinson_vol(df, 20, ann) / rv20
    f["gk_ratio"] = I.garman_klass_vol(df, 20, ann) / rv20
    f["rs_ratio"] = I.rogers_satchell_vol(df, 20, ann) / rv20
    f["yz_ratio"] = I.yang_zhang_vol(df, 20, ann) / rv20
    f["atr_pct"] = a14 / c
    d = I.dmi(df, 14)
    f["adx"] = d["adx"] / 100
    f["di_diff"] = (d["pdi"] - d["mdi"]) / (d["pdi"] + d["mdi"]).replace(0, np.nan)
    ar = I.aroon(df, 25)
    f["aroon_osc"] = (ar["up"] - ar["down"]) / 100
    f["rsi"] = I.rsi(c, 14) / 100 - 0.5
    m = I.macd(c)
    f["macd_hist"] = m["hist"] / a14
    f["macd_line"] = m["macd"] / a14
    f["stoch_k"] = I.stochastic(df, 14)["k"] / 100 - 0.5
    f["cci"] = I.cci(df, 20) / 100
    f["trix"] = _z(I.trix(c, 15))
    vx = I.vortex(df, 14)
    f["vortex_diff"] = vx["vip"] - vx["vim"]
    f["er_10"] = I.efficiency_ratio(c, 10)
    f["er_63"] = I.efficiency_ratio(c, 63)
    f["chop"] = I.choppiness(df, 14) / 100
    lr63 = I.linreg_stats(lc, 63)
    f["lr_tstat_63"] = lr63["tstat"] / 10
    f["lr_r2_63"] = lr63["r2"]
    f["lr_r2_126"] = I.linreg_stats(lc, 126)["r2"]
    s50, s200 = I.sma(c, 50), I.sma(c, 200)
    f["dist_sma50"] = (c - s50) / a14
    f["dist_sma200"] = (c - s200) / a14
    f["sma50_200"] = (s50 / s200 - 1) / (dvol * np.sqrt(150))
    bb = I.bollinger(c, 20, 2)
    width = (bb["upper"] - bb["lower"])
    f["bb_pctb"] = (c - bb["lower"]) / width.replace(0, np.nan) - 0.5
    bw = width / bb["mid"]
    f["bb_width"] = bw / bw.rolling(252, min_periods=60).median()
    for n in (20, 55):
        hi = df["high"].rolling(n).max(); lo = df["low"].rolling(n).min()
        f[f"donch_pos_{n}"] = (c - lo) / (hi - lo).replace(0, np.nan) - 0.5
    kel = I.keltner(df, 20, 10, 2)
    f["keltner_pos"] = (c - kel["mid"]) / (kel["upper"] - kel["mid"]).replace(0, np.nan)
    ich = I.ichimoku(df)
    f["ichimoku_dist"] = (c - (ich["span_a"] + ich["span_b"]) / 2) / a14
    f["psar_dist"] = (c - I.psar(df)["sar"]) / a14
    f["supertrend_dist"] = (c - I.supertrend(df, 10, 3)["line"]) / a14
    f["kama_dist"] = (c - I.kama(c, 10, 2, 30)) / a14
    h55 = I.hma(c, 55)
    f["hma_slope"] = (h55 - h55.shift(5)) / a14
    f["tsi"] = I.tsi(c) / 100
    k = I.kst(c)
    f["kst"] = _z(k["kst"] - k["signal"])
    v1 = lr.rolling(126).var()
    v5 = (lc - lc.shift(5)).rolling(126).var()
    f["var_ratio_5"] = v5 / (5 * v1) - 1
    f["autocorr_1"] = lr.rolling(63).corr(lr.shift(1))
    f["skew_63"] = lr.rolling(63).skew()
    f["kurt_63"] = lr.rolling(63).kurt() / 10
    mx = c.rolling(252, min_periods=60).max()
    f["dd_252"] = (c / mx - 1) / (dvol * np.sqrt(252))
    f["days_since_high"] = c.rolling(252, min_periods=60).apply(lambda w: len(w) - 1 - np.argmax(w), raw=True) / 252
    dd14 = 100 * (c / c.rolling(14).max() - 1)
    f["ulcer_14"] = np.sqrt((dd14 ** 2).rolling(14).mean()) / (100 * dvol * np.sqrt(14))
    vol = df["volume"].astype(float)
    has_vol = vol.rolling(60, min_periods=1).sum() > 0
    lv = np.log1p(vol)
    f["vol_z"] = ((lv - lv.rolling(20).mean()) / lv.rolling(20).std().replace(0, np.nan)).where(has_vol, 0.0)
    f["obv_slope"] = ((I.obv(df) - I.obv(df).shift(20)) / vol.rolling(20).sum().replace(0, np.nan)).where(has_vol, 0.0)
    f["cmf"] = I.cmf(df, 20).where(has_vol, 0.0)
    f["mfi"] = (I.mfi(df, 14) / 100 - 0.5).where(has_vol, 0.0)
    out = pd.DataFrame(f, index=df.index)[FEATURE_NAMES]
    out = out.replace([np.inf, -np.inf], np.nan)
    # winsorise extreme values (pooled models are sensitive to a few bad prints)
    return out.clip(-20, 20).astype(np.float32)
