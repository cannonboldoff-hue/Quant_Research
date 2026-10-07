"""Position-based, vectorised backtester with explicit execution timing and costs.

Target position p_t is decided at the close of bar t. Execution models:

- ``next_open`` (benchmark): the new position is established at the next bar's
  open.  Bar-t return = h_{t-1} * (O_t/C_{t-1} - 1) + h_t * (C_t/O_t - 1), where
  h_t = p_{t-1} is the position held during bar t. The overnight gap is earned by
  the *old* position, so a signal can never earn the gap that produced it.
- ``next_close``: the trade happens at the close of bar t+1 (one full bar of
  delay); return of bar t = p_{t-2} * (C_t/C_{t-1} - 1).
- ``same_close``: optimistic textbook convention (trade at the very close that
  generated the signal); r_t = p_{t-1} * (C_t/C_{t-1} - 1). Reported only to
  quantify execution-assumption bias.

Costs: per-side ``(cost_bps + slippage_bps) * cost_mult`` on every unit of
|position change|, charged on the bar the trade executes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EXECUTIONS = ("next_open", "next_close", "same_close")


@dataclass(frozen=True)
class BacktestConfig:
    execution: str = "next_open"
    cost_mult: float = 1.0
    sizing: str = "unit"          # unit | vol_target
    target_vol: float = 0.10      # only for vol_target sizing (annualised)
    max_leverage: float = 3.0

    def id(self) -> str:
        s = f"{self.execution}|cost{self.cost_mult:g}|{self.sizing}"
        return s + (f"{self.target_vol:g}" if self.sizing == "vol_target" else "")


def vol_target_weights(df: pd.DataFrame, target_vol: float = 0.10, ann: int = 252,
                       span: int = 60, max_leverage: float = 3.0) -> pd.Series:
    """Inverse-volatility scaling (Moskowitz et al. 2012 use 40%/sigma); EWMA
    vol from returns up to t only."""
    lr = np.log(df["close"]).diff()
    sig = lr.ewm(span=span, adjust=False, min_periods=20).std() * np.sqrt(ann)
    return (target_vol / sig).clip(upper=max_leverage).fillna(0.0)


def simulate(df: pd.DataFrame, target: pd.Series, cost_per_side_bps: float,
             execution: str = "next_open", cost_mult: float = 1.0) -> pd.DataFrame:
    """Return per-bar frame: held, gross, cost, net."""
    p = target.reindex(df.index).fillna(0.0).to_numpy(float)
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)
    c_prev = np.r_[np.nan, c[:-1]]
    gap = o / c_prev - 1
    intra = c / o - 1
    cc = c / c_prev - 1
    rate = cost_per_side_bps * cost_mult / 1e4

    def lag(x, k):
        return np.r_[np.zeros(k), x[:-k]] if k > 0 else x

    if execution == "next_open":
        h = lag(p, 1)
        h_prev = lag(p, 2)
        gross = np.nan_to_num(h_prev * gap) + np.nan_to_num(h * intra)
        trade = np.abs(h - h_prev)
    elif execution == "next_close":
        h = lag(p, 2)
        gross = np.nan_to_num(h * cc)
        trade = np.abs(lag(p, 1) - lag(p, 2))
    elif execution == "same_close":
        h = lag(p, 1)
        gross = np.nan_to_num(h * cc)
        trade = np.abs(p - lag(p, 1))
    else:
        raise ValueError(execution)
    gross[0] = 0.0
    cost = trade * rate
    return pd.DataFrame({"held": h, "gross": gross, "cost": cost, "net": gross - cost,
                         "turnover": trade}, index=df.index)


def forward_strategy_return(sim: pd.DataFrame, h: int) -> pd.Series:
    """Sum of net strategy returns over bars t+2..t+h+1, i.e. the P&L earned by
    the position decided at close t (executed at open t+1, first full bar t+1).
    Uses the next_open convention: bar t+1's return already belongs to p_t
    except the overnight gap, which we include since it is small and it keeps
    labels simple; the label horizon is ``h`` bars."""
    net = sim["net"]
    return net[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
