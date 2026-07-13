"""Regime filter: ADX-gated entries -- trending bars keep their JMA
crossover signal, chop bars get zeroed out."""
import numpy as np
import pandas as pd

from qresearch.signals.regime import apply_regime_filter, trend_regime


def _trending_ohlc(n=100, seed=0):
    rng = np.random.default_rng(seed)
    trend = np.cumsum(np.full(n, 1.0)) + rng.normal(0, 0.1, n)  # steady uptrend
    return pd.DataFrame({
        "open": trend, "high": trend + 0.5, "low": trend - 0.5, "close": trend, "volume": 1000,
    })


def _choppy_ohlc(n=100, seed=0):
    rng = np.random.default_rng(seed)
    price = 100 + rng.normal(0, 0.05, n).cumsum() * 0  # flat + tiny noise, no drift
    price = 100 + rng.normal(0, 0.2, n)
    return pd.DataFrame({
        "open": price, "high": price + 0.1, "low": price - 0.1, "close": price, "volume": 1000,
    })


def test_trend_regime_true_in_steady_trend():
    df = _trending_ohlc()
    mask = trend_regime(df, adx_length=14, adx_threshold=25.0)
    assert mask.dtype == bool
    # after warmup, a steady one-directional trend should read as trending
    assert mask.iloc[30:].mean() > 0.7


def test_trend_regime_false_in_flat_chop():
    df = _choppy_ohlc()
    mask = trend_regime(df, adx_length=14, adx_threshold=25.0)
    assert mask.iloc[30:].mean() < 0.3


def test_trend_regime_no_nan_leaks_through():
    df = _trending_ohlc(n=20)  # shorter than adx_length warmup
    mask = trend_regime(df, adx_length=14, adx_threshold=25.0)
    assert mask.isna().sum() == 0


def test_apply_regime_filter_zeroes_signal_outside_mask():
    sig_df = pd.DataFrame({
        "signal": [1, -1, 1, 0, -1],
    }, index=pd.date_range("2024-01-01", periods=5))
    mask = pd.Series([True, False, True, False, True], index=sig_df.index)
    out = apply_regime_filter(sig_df, mask)
    assert list(out["signal"]) == [1, 0, 1, 0, -1]


def test_apply_regime_filter_does_not_mutate_input():
    sig_df = pd.DataFrame({"signal": [1, -1]}, index=pd.date_range("2024-01-01", periods=2))
    mask = pd.Series([False, False], index=sig_df.index)
    apply_regime_filter(sig_df, mask)
    assert list(sig_df["signal"]) == [1, -1]  # original untouched
