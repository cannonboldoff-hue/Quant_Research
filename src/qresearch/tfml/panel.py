"""Stacked multi-instrument panel (rows sorted by instrument, then time) with
precomputed features and execution arrays, so strategies, labels, decision
rules and backtests run vectorised over the whole universe at once."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import data as D
from .features import FEATURE_NAMES, compute_features


@dataclass
class Panel:
    frequency: str
    meta: pd.DataFrame            # one row per instrument (index = panel instrument number)
    inst: np.ndarray              # (n,) instrument number per row
    date: np.ndarray              # (n,) datetime64[ns]
    inst_pos: np.ndarray          # (n,) bar number within instrument
    inst_start: np.ndarray        # (n,) bool, first row of instrument
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    cost_rate: np.ndarray         # (n,) per-side cost as a fraction, per row
    X: np.ndarray                 # (n, F) float32 features
    ann: float                    # bars per year
    bounds: np.ndarray            # (n_inst+1,) row offsets

    @property
    def n(self) -> int:
        return len(self.close)

    def frame(self, k: int) -> pd.DataFrame:
        s, e = self.bounds[k], self.bounds[k + 1]
        return pd.DataFrame({"open": self.open[s:e], "high": self.high[s:e], "low": self.low[s:e],
                             "close": self.close[s:e], "volume": self.volume[s:e]},
                            index=pd.DatetimeIndex(self.date[s:e], name="Date"))

    def lag(self, x: np.ndarray, k: int = 1, fill: float = 0.0) -> np.ndarray:
        """Lag x by k rows within each instrument."""
        out = np.empty_like(x, dtype=float)
        out[k:] = x[:-k]
        out[:k] = fill
        mask = self.inst_pos < k
        out[mask] = fill
        return out

    def lead(self, x: np.ndarray, k: int) -> np.ndarray:
        """x shifted k rows forward within instrument (NaN past the end)."""
        out = np.full(len(x), np.nan)
        out[:-k] = x[k:]
        n_rows = np.diff(self.bounds)[self.inst]
        out[self.inst_pos >= n_rows - k] = np.nan
        return out


def _features_cached(inst_id: str, frequency: str, ann: int) -> pd.DataFrame:
    cache = D.PROCESSED / "features" / frequency / f"{inst_id}.parquet"
    src = D.PROCESSED / frequency / f"{inst_id}.parquet"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return pd.read_parquet(cache)
    df = D.load_bars(inst_id, frequency)
    f = compute_features(df, ann=ann)
    cache.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(cache)
    return f


def build_panel(frequency: str = "daily", groups: list[str] | None = None, benchmark_only: bool = True,
                start: str | None = None, ann: int | None = None, n_jobs: int = 8) -> Panel:
    uni = D.universe_frame()
    if groups is not None:
        uni = uni[uni["group"].isin(groups)]
    elif benchmark_only:
        uni = uni[uni["benchmark"]]
    avail = [i for i in uni["id"] if (D.PROCESSED / frequency / f"{i}.parquet").exists()]
    uni = uni[uni["id"].isin(avail)].reset_index(drop=True)
    ann = ann or {"daily": 252, "1h": 24 * 365, "4h": 6 * 365, "daily_binance": 365, "weekly": 52}[frequency]
    feats = Parallel(n_jobs=n_jobs)(delayed(_features_cached)(i, frequency, ann) for i in uni["id"])
    parts, Xs = [], []
    for k, (iid, f) in enumerate(zip(uni["id"], feats)):
        df = D.load_bars(iid, frequency)
        if start:
            keep = df.index >= pd.Timestamp(start)
            df, f = df[keep], f[keep]
        df = df.assign(inst=k)
        parts.append(df)
        Xs.append(f.to_numpy(np.float32))
    big = pd.concat(parts)
    lens = np.array([len(p) for p in parts])
    bounds = np.r_[0, np.cumsum(lens)]
    inst = big["inst"].to_numpy(np.int32)
    inst_pos = np.concatenate([np.arange(l) for l in lens])
    rate = ((uni["cost_bps"] + uni["slippage_bps"]) / 1e4).to_numpy()[inst]
    uni["first_date"] = [p.index[0] for p in parts]
    uni["last_date"] = [p.index[-1] for p in parts]
    uni["n_bars"] = lens
    return Panel(frequency=frequency, meta=uni, inst=inst, date=big.index.to_numpy(),
                 inst_pos=inst_pos, inst_start=inst_pos == 0,
                 open=big["open"].to_numpy(float), high=big["high"].to_numpy(float),
                 low=big["low"].to_numpy(float), close=big["close"].to_numpy(float),
                 volume=big["volume"].to_numpy(float), cost_rate=rate,
                 X=np.concatenate(Xs).astype(np.float32), ann=float(ann), bounds=bounds)


def subset_panel(P: Panel, inst_idx) -> Panel:
    """Panel restricted to the given instrument numbers (renumbered 0..k-1)."""
    inst_idx = list(inst_idx)
    rows = np.concatenate([np.arange(P.bounds[k], P.bounds[k + 1]) for k in inst_idx])
    remap = np.full(len(P.meta), -1, np.int32)
    remap[inst_idx] = np.arange(len(inst_idx), dtype=np.int32)
    lens = np.diff(P.bounds)[inst_idx]
    return Panel(P.frequency, P.meta.loc[inst_idx].reset_index(drop=True), remap[P.inst[rows]], P.date[rows],
                 P.inst_pos[rows], P.inst_start[rows], P.open[rows], P.high[rows], P.low[rows], P.close[rows],
                 P.volume[rows], P.cost_rate[rows], P.X[rows], P.ann, np.r_[0, np.cumsum(lens)])


# --------------------------------------------------------------------------- panel backtest
def simulate_panel(P: Panel, p: np.ndarray, execution: str = "next_open", cost_mult: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised version of backtest.simulate over the stacked panel.
    Returns (net, held, cost) per row."""
    c_prev = P.lag(P.close, 1, np.nan)
    if execution == "next_open":
        h = P.lag(p, 1); h_prev = P.lag(p, 2)
        gross = np.nan_to_num(h_prev * (P.open / c_prev - 1)) + np.nan_to_num(h * (P.close / P.open - 1))
        trade = np.abs(h - h_prev)
    elif execution == "next_close":
        h = P.lag(p, 2)
        gross = np.nan_to_num(h * (P.close / c_prev - 1))
        trade = np.abs(P.lag(p, 1) - P.lag(p, 2))
    elif execution == "same_close":
        h = P.lag(p, 1)
        gross = np.nan_to_num(h * (P.close / c_prev - 1))
        trade = np.abs(p - P.lag(p, 1))
    else:
        raise ValueError(execution)
    gross[P.inst_start] = 0.0
    cost = trade * P.cost_rate * cost_mult
    return gross - cost, h, cost


def vol_weights(P: Panel, target_vol: float = 0.10, span: int = 60, cap: float = 3.0) -> np.ndarray:
    """Per-row inverse-vol position multiplier, EWMA of returns up to t (causal)."""
    out = np.zeros(P.n)
    lr = np.log(P.close / P.lag(P.close, 1, np.nan))
    for k in range(len(P.meta)):
        s, e = P.bounds[k], P.bounds[k + 1]
        sig = pd.Series(lr[s:e]).ewm(span=span, adjust=False, min_periods=20).std().to_numpy() * np.sqrt(P.ann)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[s:e] = np.nan_to_num(np.minimum(target_vol / sig, cap))
    return out


def portfolio_returns(P: Panel, net: np.ndarray, mask: np.ndarray | None = None,
                      inst_subset: np.ndarray | None = None) -> pd.Series:
    """Equal-capital portfolio: each date's return = sum of instrument returns / number
    of instruments live (first_date <= t <= last_date) at that date."""
    m = np.ones(P.n, bool) if mask is None else mask.copy()
    if inst_subset is not None:
        m &= np.isin(P.inst, inst_subset)
    df = pd.DataFrame({"d": P.date[m], "r": net[m], "i": P.inst[m]})
    s = df.groupby("d")["r"].sum()
    # live count per date
    meta = P.meta if inst_subset is None else P.meta.loc[inst_subset]
    dates = s.index.values
    first = meta["first_date"].values.astype("datetime64[ns]")
    last = meta["last_date"].values.astype("datetime64[ns]")
    live = ((first[None, :] <= dates[:, None]) & (last[None, :] >= dates[:, None])).sum(1)
    return pd.Series(s.values / np.maximum(live, 1), index=pd.DatetimeIndex(dates))
