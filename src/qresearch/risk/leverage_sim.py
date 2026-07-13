"""Leverage / margin account simulation over an existing trade list.

Deliberately decoupled from ``backtest.engine`` (which computes per-trade %
returns): leverage/margin/position-sizing is an account-level concern layered
on top of trades that already exist, not something signal evaluation needs to
know about.

Two sizing philosophies, since "what does leverage do" depends entirely on
how it's used:
- ``disciplined``: fixed-fractional risk per trade (``risk_pct`` of current
  equity), position size derived from the strategy's own realized stop
  distance. Leverage mostly just determines whether enough margin is
  available -- with disciplined sizing that's rarely binding.
- ``aggressive``: position scaled to a fixed fraction of the account's
  leveraged buying power (``leverage * utilization``), regardless of stop
  distance -- how retail accounts overleverage. Expect much higher ruin risk,
  especially at higher leverage.

Both approximate a trade's risk from its REALIZED outcome (entry/exit only)
rather than the intrabar path the engine doesn't retain -- an extreme
leveraged position could in reality be liquidated mid-trade before the
strategy's own stop/target triggers. ``structural_ruin`` flags that instant-
liquidation risk from sizing alone, separate from the sequential equity walk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import monte_carlo_simulation


@dataclass
class LeverageSimResult:
    sizing_mode: str
    leverage: float
    position_fraction: float  # notional / equity used per trade
    typical_stop_pct: float
    structural_ruin: bool  # a single normal stop-sized move alone wipes the account
    final_equity: float
    max_drawdown: float
    ruined: bool  # equity hit 0 somewhere in the sequential walk
    n_trades: int
    ruin_probability_mc: float  # fraction of bootstrap paths breaching -100% cumulative


def _typical_stop_pct(trades: pd.DataFrame) -> float:
    stopped = trades.loc[trades["reason"] == "stop", "ret"].abs()
    return float(stopped.mean()) if len(stopped) else float(trades["ret"].abs().mean())


def simulate(trades: pd.DataFrame, leverage: float, initial_capital: float = 100_000.0,
             sizing_mode: str = "disciplined", risk_pct: float = 0.01,
             utilization: float = 0.5, num_mc_sims: int = 5000) -> LeverageSimResult:
    trades = trades.sort_values("entry_time")
    stop_pct = _typical_stop_pct(trades)

    if sizing_mode == "disciplined":
        position_fraction = min(risk_pct / stop_pct, leverage)
    elif sizing_mode == "aggressive":
        position_fraction = leverage * utilization
    else:
        raise ValueError(f"unknown sizing_mode: {sizing_mode}")

    structural_ruin = position_fraction * stop_pct >= 1.0

    rets = trades["ret"].to_numpy(dtype=np.float64) * position_fraction
    equity = initial_capital
    curve = [equity]
    ruined = False
    for r in rets:
        if ruined:
            curve.append(0.0)
            continue
        equity *= (1 + r)
        if equity <= 0:
            equity, ruined = 0.0, True
        curve.append(equity)
    curve = np.array(curve)
    running_max = np.maximum.accumulate(curve)
    drawdown = np.divide(curve - running_max, running_max,
                          out=np.zeros_like(curve), where=running_max > 0)

    mc_paths = monte_carlo_simulation(rets, num_simulations=num_mc_sims, num_days=len(rets))
    ruin_probability_mc = float((mc_paths.min(axis=1) <= -1.0).mean())

    return LeverageSimResult(
        sizing_mode=sizing_mode, leverage=leverage, position_fraction=float(position_fraction),
        typical_stop_pct=stop_pct, structural_ruin=structural_ruin,
        final_equity=float(curve[-1]), max_drawdown=float(drawdown.min()),
        ruined=ruined, n_trades=len(trades), ruin_probability_mc=ruin_probability_mc,
    )
