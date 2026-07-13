"""Regime filter for JMA+ATR entries.

The daily sleeves' walk-forward win rates are modest (33-45%, per
walk_forward_finalists_summary.csv) -- a lot of JMA crossover entries fire in
low-ADX chop rather than an actual trend. This gates entries to bars where
ADX confirms a trend, so a crossover only becomes a trade when the market is
plausibly trending.

ponytail: single ADX threshold first (per the alpha-search plan) -- add a
realized-vol-percentile regime (indicators.volatility.historical_volatility /
rolling_zscore) only if ADX alone doesn't move the walk-forward gate.
"""
from __future__ import annotations

import pandas as pd

from ..indicators.volume import adx


def trend_regime(df: pd.DataFrame, adx_length: int = 14, adx_threshold: float = 25.0) -> pd.Series:
    """Boolean mask, True where ADX(adx_length) >= adx_threshold (a
    confirmed trend, not chop). NaN during ADX warm-up counts as False."""
    return (adx(df, length=adx_length)["adx"] >= adx_threshold).fillna(False)


def apply_regime_filter(sig_df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Zero out signal entries where the regime mask is False. Returns a
    copy; doesn't mutate the input or touch bars that were already flat."""
    out = sig_df.copy()
    gated = mask.reindex(out.index).fillna(False)
    out["signal"] = out["signal"].where(gated, 0)
    return out
