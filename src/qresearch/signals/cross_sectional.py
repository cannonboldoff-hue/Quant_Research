"""Cross-sectional long/short ranking signal.

Every JMA+ATR sleeve is time-series trend-following on ONE instrument at a
time -- structurally similar regardless of asset class (each is "is this
instrument trending"), which is why they're only weakly correlated with each
other but still share a common bet type. A cross-sectional rank -- long the
strongest performers, short the weakest, within the SAME universe at the
SAME time -- is a genuinely different bet ("which names are relatively
strong right now"), so it should sit at a different, lower correlation to
the trend book -- the actual reason to add it (see
``qresearch.portfolio.combine``, which needs low correlation to raise the
combined Sharpe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_return_rank(wide_close: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Percentile rank (0=worst, 1=best) of each ticker's trailing
    ``lookback``-bar return, computed cross-sectionally per row/date. NaN
    where a ticker has no valid trailing return that day (``rank`` handles
    ties/NaNs directly -- ``na_option="keep"`` so warm-up rows stay NaN
    rather than getting an arbitrary rank)."""
    trailing_ret = wide_close.pct_change(lookback)
    return trailing_ret.rank(axis=1, pct=True, na_option="keep")


def long_short_weights(ranks: pd.DataFrame, top_frac: float = 0.2, bottom_frac: float = 0.2) -> pd.DataFrame:
    """Dollar-neutral long/short weights from a cross-sectional rank frame:
    equal-weight long the top ``top_frac``, equal-weight short the bottom
    ``bottom_frac``, flat everything else. Each row's weights sum to ~0
    (exactly 0 whenever the long and short legs have equal names)."""
    long_mask = ranks.ge(1 - top_frac)
    short_mask = ranks.le(bottom_frac)
    n_long = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)
    return long_mask.div(n_long, axis=0).fillna(0.0) - short_mask.div(n_short, axis=0).fillna(0.0)


def cross_sectional_returns(wide_close: pd.DataFrame, lookback: int = 20, hold: int = 5,
                            top_frac: float = 0.2, bottom_frac: float = 0.2) -> pd.Series:
    """Daily portfolio return of a rebalance-every-``hold``-bars long/short
    momentum book: rank on trailing ``lookback``-bar return, form
    dollar-neutral long/short weights, hold those weights for ``hold`` bars
    (not re-ranked daily -- lower turnover, matches "periodic rebalance"),
    applied with a one-bar lag (today's weight was set from data available
    at yesterday's close -- no lookahead)."""
    ranks = trailing_return_rank(wide_close, lookback)
    weights = long_short_weights(ranks, top_frac, bottom_frac)

    rebalance_mask = pd.Series(np.arange(len(weights)) % hold == 0, index=weights.index)
    weights = weights.where(rebalance_mask, np.nan).ffill().fillna(0.0)

    daily_ret = wide_close.pct_change()
    port_ret = (weights.shift(1) * daily_ret).sum(axis=1, min_count=1)
    return port_ret.dropna()
