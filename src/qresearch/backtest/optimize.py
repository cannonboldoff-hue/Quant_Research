"""Robustness testing across tickers x parameter combinations — consolidates
``robustness_testing``/``test_combination`` (``risk-management_002``), adding
a ticker dimension on top of the existing ``optimize.walk_forward.grid_search``.
(``evaluate_params``/``grid_search_params`` from ``multi_strat_049`` were not
ported: hardcoded to one Colab notebook's SAMX-strategy 6-param signature and
a `/content/drive` log path — the generic ``grid_search`` already covers the
same pattern.)"""
from __future__ import annotations
from itertools import product
from typing import Callable
import pandas as pd

from .metrics import calculate_metrics


def robustness_testing(df: pd.DataFrame, signal_fn: Callable[..., pd.DataFrame], param_grid: dict,
                       ticker_col: str = "Ticker", tickers: list | None = None) -> pd.DataFrame:
    """Backtest every (ticker, param-combo) pair. ``signal_fn(ticker_df, **params)``
    must return a frame with a ``signal`` column."""
    from .engine import backtest_trades

    tickers = tickers or df[ticker_col].unique().tolist()
    keys = list(param_grid)
    rows = []
    for ticker in tickers:
        ticker_df = df[df[ticker_col] == ticker]
        for combo in product(*(param_grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            trades = backtest_trades(signal_fn(ticker_df, **params))
            m = calculate_metrics(trades)
            rows.append({"ticker": ticker, **params, **m})
    return pd.DataFrame(rows)
