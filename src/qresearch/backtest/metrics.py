"""Performance metrics + strategy ranking.

Consolidates ``calculate_metrics``/``calculate_metrics_fast`` (64+24 notebooks),
``calculate_composite_score``, ``normalize_metric`` and ``rank_and_score``.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..config.settings import get_settings


def sharpe(returns: pd.Series, periods: int | None = None) -> float:
    s = get_settings()
    periods = periods or s.bars_per_year
    r = pd.Series(returns).dropna()
    if r.std(ddof=0) == 0 or len(r) < 2:
        return 0.0
    excess = r - s.risk_free_rate / periods
    return float(np.sqrt(periods) * excess.mean() / r.std(ddof=0))


def sortino(returns: pd.Series, periods: int | None = None) -> float:
    s = get_settings()
    periods = periods or s.bars_per_year
    r = pd.Series(returns).dropna()
    downside = r[r < 0].std(ddof=0)
    if downside == 0 or len(r) < 2:
        return 0.0
    return float(np.sqrt(periods) * r.mean() / downside)


def max_drawdown(equity: pd.Series) -> float:
    """Max peak-to-trough drawdown of an equity curve (level series, e.g.
    ``(1 + returns).cumprod()`` — not a returns series). Returns a fraction
    (e.g. -0.2 for a 20% drawdown)."""
    eq = pd.Series(equity)
    return float((eq / eq.cummax() - 1).min())


def profit_factor(returns: pd.Series) -> float:
    """Gross wins / gross losses — ``safe_profit_factor``. 0 if both are ~0
    (no activity), inf if wins with no losses."""
    wins, losses = returns[returns > 0].sum(), abs(returns[returns < 0].sum())
    if losses < 1e-10:
        return float("inf") if wins > 1e-10 else 0.0
    return float(wins / losses)


def win_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean()) if len(returns) else 0.0


def quarterly_hpr(returns: pd.Series, dates) -> pd.Series:
    """Holding-period return by calendar quarter — ``calculate_quarterly_hpr``."""
    equity = (1 + returns).cumprod()
    q = pd.PeriodIndex(pd.to_datetime(dates), freq="Q")
    return equity.groupby(q).apply(lambda e: e.iloc[-1] / e.iloc[0] - 1 if len(e) else 0.0)


def calculate_metrics(trades: pd.DataFrame) -> dict:
    """Summary stats for a trades frame with a ``ret`` column."""
    if trades is None or trades.empty:
        return {"n_trades": 0, "win_rate": 0, "total_return": 0, "sharpe": 0,
                "sortino": 0, "max_drawdown": 0, "profit_factor": 0, "avg_ret": 0}
    r = trades["ret"]
    equity = (1 + r).cumprod()
    return {
        "n_trades": int(len(r)),
        "win_rate": win_rate(r),
        "total_return": float(equity.iloc[-1] - 1),
        "avg_ret": float(r.mean()),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(r),
    }


def _normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else pd.Series(0.5, index=s.index)


def rank_and_score(metrics_df: pd.DataFrame,
                   weights: dict | None = None) -> pd.DataFrame:
    """Rank strategies/parameter sets by a weighted composite of normalized metrics.
    Consolidates the ``composite_score``/``rank_and_score`` pattern."""
    weights = weights or {"sharpe": 0.4, "total_return": 0.3, "win_rate": 0.2, "max_drawdown": 0.1}
    df = metrics_df.copy()
    score = pd.Series(0.0, index=df.index)
    for k, w in weights.items():
        if k not in df:
            continue
        col = -df[k] if k == "max_drawdown" else df[k]  # smaller drawdown is better
        score += w * _normalize(col)
    df["composite_score"] = score
    return df.sort_values("composite_score", ascending=False)


# Backwards-compatible alias used in the notebooks.
composite_score = rank_and_score
