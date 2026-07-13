"""Optimization & walk-forward testing.

Consolidates ``split_data_by_periods`` and ``walk_forward_optimize`` (11 notebooks)
plus the ``itertools.product`` grid patterns.
"""
from __future__ import annotations
from itertools import product
from typing import Callable, Iterable
import pandas as pd

from ..backtest.engine import backtest_trades
from ..backtest.metrics import calculate_metrics


def split_data_by_periods(df: pd.DataFrame, n_splits: int = 5, date_col: str = "Date"):
    """Chronological (no-shuffle) splits for walk-forward analysis."""
    df = df.sort_values(date_col).reset_index(drop=True)
    size = len(df) // n_splits
    return [df.iloc[i * size:(i + 1) * size] for i in range(n_splits)]


def grid_search(df: pd.DataFrame, signal_fn: Callable[..., pd.DataFrame],
                param_grid: dict) -> pd.DataFrame:
    """Exhaustive grid search. ``signal_fn(df, **params)`` must return a frame
    with a ``signal`` column; each combo is backtested and scored."""
    keys = list(param_grid)
    rows = []
    for combo in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        sig = signal_fn(df, **params)
        m = calculate_metrics(backtest_trades(sig))
        rows.append({**params, **m})
    return pd.DataFrame(rows)


def walk_forward_optimize(df: pd.DataFrame, signal_fn: Callable, param_grid: dict,
                          n_splits: int = 5, objective: str = "sharpe") -> pd.DataFrame:
    """Anchored walk-forward: optimize params on split *t*, evaluate on *t+1*."""
    splits = split_data_by_periods(df, n_splits)
    results = []
    for t in range(len(splits) - 1):
        insample = grid_search(splits[t], signal_fn, param_grid)
        if insample.empty:
            continue
        best = insample.sort_values(objective, ascending=False).iloc[0]
        best_params = {k: best[k] for k in param_grid}
        oos = calculate_metrics(backtest_trades(signal_fn(splits[t + 1], **best_params)))
        results.append({"fold": t, **best_params, **{f"oos_{k}": v for k, v in oos.items()}})
    return pd.DataFrame(results)
