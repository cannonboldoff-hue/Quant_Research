"""Shared walk-forward selection logic for JMA+ATR finalist sleeves.

Extracted from ``scripts/walk_forward_finalists.py`` so ``scripts/build_portfolio.py``
(portfolio-combination phase) can reuse the exact same fold-by-fold param
selection instead of re-deriving it -- the OOS gate and the portfolio need to
run over identical trades, or the combined Sharpe wouldn't reflect what the
gate actually validated.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.costs import realistic_net_ret
from ..backtest.engine import backtest_trades_fast
from ..campaign.runner import STOP_GRID
from ..config.settings import get_settings
from ..signals.generators import jma_signals
from ..signals.regime import apply_regime_filter, trend_regime
from ..signals.volume_filter import apply_volume_filter
from .walk_forward import split_data_by_periods

SIGNAL_GRID = {"fast": [5, 7, 9], "slow": [21, 28]}
N_SPLITS = 5
MIN_TRADES = 10  # in-sample combos with fewer trades than this are too noisy to trust as "best"

_FLAT_FEE = (get_settings().fee_bps + get_settings().slippage_bps) / 1e4


def _combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in product(*grid.values())]


def cost_adjust(trades: pd.DataFrame, ticker: str, market: str) -> pd.DataFrame:
    """Forex trades come out of backtest_trades_fast with the flat 7bps fee
    baked in -- recompute net return with the realistic per-pair spread+swap,
    same as apply_realistic_forex_costs.py. No-op for every other market."""
    if market != "forex" or trades.empty:
        return trades
    trades = trades.copy()
    trades["ticker"] = ticker
    trades["gross"] = trades["ret"] + _FLAT_FEE
    trades["ret"] = realistic_net_ret(trades)
    return trades


def _align_market_mask(df: pd.DataFrame, market_mask: pd.Series) -> pd.Series:
    """Reindex a Date-indexed broad-market mask onto df's row order (positional,
    not Date) via df['Date'] lookup -- df/sig may carry a plain RangeIndex, so
    the mask has to be looked up by calendar date, not by .index alignment."""
    aligned = market_mask.reindex(df["Date"].to_numpy()).fillna(False).to_numpy()
    return pd.Series(aligned, index=df.index)


def _gated_signal(df: pd.DataFrame, sparams: dict, signal_fn, regime_filter: bool, adx_threshold: float,
                   volume_filter: bool, mfi_midpoint: float,
                   market_mask: pd.Series | None = None) -> pd.DataFrame:
    sig = signal_fn(df, **sparams)
    if regime_filter:
        sig = apply_regime_filter(sig, trend_regime(df, adx_threshold=adx_threshold))
    if volume_filter:
        sig = apply_volume_filter(sig, mfi_midpoint=mfi_midpoint)
    if market_mask is not None:
        sig = apply_regime_filter(sig, _align_market_mask(df, market_mask))
    return sig


def best_insample(df: pd.DataFrame, ticker: str, market: str,
                   regime_filter: bool = False, adx_threshold: float = 25.0,
                   volume_filter: bool = False, mfi_midpoint: float = 50.0,
                   signal_fn=jma_signals, signal_grid: dict | None = None,
                   stop_grid: dict | None = None,
                   market_mask: pd.Series | None = None,
                   backtest_fn=backtest_trades_fast) -> tuple[dict, dict] | None:
    """Grid-search signal x risk params on one fold; return (signal_params, risk_params) of the
    best-by-mean-net-return combo with at least MIN_TRADES trades, or None if nothing qualifies.
    ``regime_filter``: gate entries to ADX-confirmed trend bars (signals/regime.py).
    ``volume_filter``: gate entries to MFI-confirmed money-flow direction (signals/volume_filter.py).
    Both selected the same way every other param is, from this fold's own in-sample data, no OOS peeking.
    ``signal_fn``/``signal_grid``: swap in a different signal generator (e.g.
    ``signals.dsl_signals.kama_dsl_signals`` with its own param grid) -- defaults to
    ``jma_signals``/``SIGNAL_GRID`` so every existing caller is unaffected.
    ``stop_grid``: swap in a different sl_mult/tp_mult grid (e.g. tighter stops
    to trade off return for drawdown) -- defaults to ``STOP_GRID``.
    ``market_mask``: a Date-indexed boolean Series (True = broad market
    trending), shared across every ticker -- zeroes NEW entries on days the
    broad market isn't trending, unlike ``regime_filter`` which gates on each
    ticker's OWN ADX. Default None leaves every existing caller unaffected.
    ``backtest_fn``: swap in a different signal-in/trades-out engine (e.g.
    ``backtest_trades_trailing_fast`` + a ``{"sl_mult":...,"trail_mult":...}``
    ``stop_grid``) -- defaults to ``backtest_trades_fast``, every existing
    caller unaffected."""
    grid = signal_grid if signal_grid is not None else SIGNAL_GRID
    sgrid = stop_grid if stop_grid is not None else STOP_GRID
    best, best_ret = None, -np.inf
    for sparams in _combos(grid):
        sig = _gated_signal(df, sparams, signal_fn, regime_filter, adx_threshold, volume_filter, mfi_midpoint,
                             market_mask)
        for rparams in _combos(sgrid):
            trades = cost_adjust(backtest_fn(sig, **rparams), ticker, market)
            if len(trades) < MIN_TRADES:
                continue
            ret = trades["ret"].mean()
            if ret > best_ret:
                best_ret, best = ret, (sparams, rparams)
    return best


def oos_trades(df: pd.DataFrame, sparams: dict, rparams: dict, ticker: str, market: str,
                regime_filter: bool = False, adx_threshold: float = 25.0,
                volume_filter: bool = False, mfi_midpoint: float = 50.0,
                signal_fn=jma_signals, market_mask: pd.Series | None = None,
                backtest_fn=backtest_trades_fast) -> pd.DataFrame:
    """OOS trades for one fold under the winning params -- the full trades
    frame (entry_time/exit_time/ret/...), not just a summary dict, so callers
    can build daily return series (build_portfolio.py) or a per-fold scalar
    summary (walk_forward_finalists.py)."""
    sig = _gated_signal(df, sparams, signal_fn, regime_filter, adx_threshold, volume_filter, mfi_midpoint,
                         market_mask)
    return cost_adjust(backtest_fn(sig, **rparams), ticker, market)


def run_finalist_trades(strategy_id: str, market: str, timeframe: str, processed_dir: Path,
                         regime_filter: bool = False, adx_threshold: float = 25.0,
                         volume_filter: bool = False, mfi_midpoint: float = 50.0,
                         signal_fn=jma_signals, signal_grid: dict | None = None,
                         tickers: set[str] | None = None,
                         stop_grid: dict | None = None,
                         market_mask: pd.Series | None = None,
                         backtest_fn=backtest_trades_fast) -> pd.DataFrame:
    """Anchored walk-forward over every ticker's parquet under
    ``processed_dir/market/timeframe``: select the best in-sample (signal,
    risk) params per fold, evaluate OOS on the next fold, and return the
    concatenated OOS trades across all ticker-folds -- one row per trade,
    tagged with ``ticker``, ``fold``, and the winning params.
    ``signal_fn``/``signal_grid`` default to ``jma_signals``/``SIGNAL_GRID`` --
    pass a different pair (e.g. ``kama_dsl_signals`` + its own grid) to
    walk-forward validate a different signal family through the same gate.
    ``tickers``: restrict to this subset of parquet stems (e.g. a sector) --
    default None walks every ticker on disk, unchanged for every existing caller.
    ``stop_grid``: swap in a different sl_mult/tp_mult grid -- default None uses
    ``STOP_GRID`` (via ``best_insample``), unchanged for every existing caller.
    ``market_mask``: a Date-indexed boolean Series shared across every ticker
    (True = broad market trending) -- gates NEW entries at the portfolio level
    instead of per-ticker. Default None, unchanged for every existing caller.
    ``backtest_fn``: swap in a different exit engine (e.g. a trailing-stop
    variant) -- default ``backtest_trades_fast``, unchanged for every existing
    caller."""
    grid = signal_grid if signal_grid is not None else SIGNAL_GRID
    src_dir = processed_dir / market / timeframe
    frames = []
    for path in sorted(src_dir.glob("*.parquet")):
        if tickers is not None and path.stem not in tickers:
            continue
        ticker = path.stem
        df = pd.read_parquet(path)
        splits = split_data_by_periods(df, N_SPLITS)
        for t in range(len(splits) - 1):
            best = best_insample(splits[t], ticker, market, regime_filter, adx_threshold,
                                  volume_filter, mfi_midpoint, signal_fn, grid, stop_grid, market_mask, backtest_fn)
            if best is None:
                continue
            sparams, rparams = best
            trades = oos_trades(splits[t + 1], sparams, rparams, ticker, market,
                                 regime_filter, adx_threshold, volume_filter, mfi_midpoint, signal_fn,
                                 market_mask, backtest_fn)
            if trades.empty:
                continue
            trades = trades.copy()
            trades["ticker"] = ticker
            trades["fold"] = t
            for k, v in {**sparams, **rparams}.items():
                trades[k] = v
            frames.append(trades)
    if not frames:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "entry_price",
                                      "exit_price", "reason", "pnl", "ret", "ticker", "fold",
                                      *grid.keys(), *STOP_GRID.keys()])
    return pd.concat(frames, ignore_index=True)
