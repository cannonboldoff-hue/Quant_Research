"""Performance metrics computed from per-bar net returns and held positions.

Return convention: strategy returns are returns on notional exposure
(excess-return convention used in the managed-futures literature); Sharpe and
Sortino are therefore computed without subtracting a risk-free rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES = [
    "cagr", "total_return", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown",
    "win_rate_days", "win_rate_trades", "profit_factor", "profit_factor_trades", "turnover",
    "cost_drag", "exposure", "n_trades", "avg_hold_bars", "stability_r2", "pct_pos_years",
    "skew", "n_bars", "years",
]


def bars_per_year(index: pd.DatetimeIndex) -> float:
    years = (index[-1] - index[0]).days / 365.25
    return len(index) / years if years > 0 else 252.0


def trade_returns(net: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Per-trade compounded returns; a trade = maximal run of constant non-zero position sign."""
    sgn = np.sign(held)
    change = np.r_[True, sgn[1:] != sgn[:-1]]
    tid = np.cumsum(change)
    mask = sgn != 0
    if not mask.any():
        return np.array([])
    s = pd.Series(np.log1p(net[mask])).groupby(tid[mask]).sum()
    return np.expm1(s.to_numpy())


def compute_metrics(net: pd.Series, held: pd.Series | None = None, cost: pd.Series | None = None,
                    turnover: pd.Series | None = None, ann: float | None = None) -> dict:
    r = net.dropna()
    n = len(r)
    if n < 20:
        return {k: np.nan for k in METRIC_NAMES}
    ann = ann or bars_per_year(r.index)
    years = (r.index[-1] - r.index[0]).days / 365.25
    eq = np.cumprod(1 + r.to_numpy())
    tr = eq[-1] - 1
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
    sd = r.std(ddof=1)
    mu = r.mean()
    sharpe = mu / sd * np.sqrt(ann) if sd > 0 else 0.0
    dd = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0) ** 2))
    sortino = mu / dd * np.sqrt(ann) if dd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    calmar = cagr / abs(mdd) if mdd < 0 and np.isfinite(cagr) else np.nan
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    pf = pos / neg if neg > 0 else np.nan
    out = {"cagr": cagr, "total_return": tr, "ann_vol": sd * np.sqrt(ann), "sharpe": sharpe,
           "sortino": sortino, "calmar": calmar, "max_drawdown": mdd, "profit_factor": pf,
           "skew": float(r.skew()), "n_bars": n, "years": years}
    if held is not None:
        h = held.reindex(r.index).fillna(0).to_numpy()
        active = h != 0
        out["exposure"] = float(active.mean())
        out["win_rate_days"] = float((r.to_numpy()[active] > 0).mean()) if active.any() else np.nan
        trs = trade_returns(r.to_numpy(), h)
        out["n_trades"] = int(len(trs))
        out["win_rate_trades"] = float((trs > 0).mean()) if len(trs) else np.nan
        tp, tn = trs[trs > 0].sum(), -trs[trs < 0].sum()
        out["profit_factor_trades"] = tp / tn if tn > 0 else np.nan
        out["avg_hold_bars"] = float(active.sum() / len(trs)) if len(trs) else np.nan
    if turnover is not None:
        out["turnover"] = float(turnover.reindex(r.index).fillna(0).sum() / years) if years > 0 else np.nan
    if cost is not None:
        out["cost_drag"] = float(cost.reindex(r.index).fillna(0).sum() / years) if years > 0 else np.nan
    # stability: R^2 of log-equity on time, and share of positive calendar years
    le = np.log(np.maximum(eq, 1e-12))
    t = np.arange(n)
    c = np.corrcoef(t, le)[0, 1] if np.std(le) > 0 else 0.0
    out["stability_r2"] = float(c ** 2) * (1 if c >= 0 else -1)
    yr = r.groupby(r.index.year).apply(lambda x: np.prod(1 + x) - 1)
    out["pct_pos_years"] = float((yr > 0).mean())
    return {k: out.get(k, np.nan) for k in METRIC_NAMES}
