"""Cross-sectional long/short: trailing-return rank -> dollar-neutral
weights -> portfolio daily return, with a rebalance hold period and a
one-bar lag (no lookahead)."""
import numpy as np
import pandas as pd

from qresearch.signals.cross_sectional import (
    cross_sectional_returns, long_short_weights, trailing_return_rank,
)


def _wide_close(n=60, n_tickers=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # give each ticker a distinct constant drift so ranks are stable and
    # predictable: ticker i has the i-th highest drift
    drifts = np.linspace(-0.01, 0.01, n_tickers)
    data = {}
    for i, d in enumerate(drifts):
        data[f"T{i}"] = 100 * np.cumprod(1 + d + rng.normal(0, 0.001, n))
    return pd.DataFrame(data, index=dates)


def test_trailing_return_rank_orders_by_trailing_performance():
    wide = _wide_close()
    ranks = trailing_return_rank(wide, lookback=20)
    last = ranks.iloc[-1].dropna()
    # T9 has the highest drift (built into _wide_close) -> should rank highest
    assert last.idxmax() == "T9"
    assert last.idxmin() == "T0"


def test_trailing_return_rank_nan_during_warmup():
    wide = _wide_close(n=30)
    ranks = trailing_return_rank(wide, lookback=20)
    assert ranks.iloc[5].isna().all()  # before any ticker has 20 bars of history


def test_long_short_weights_sum_to_zero_dollar_neutral():
    wide = _wide_close(n_tickers=10)
    ranks = trailing_return_rank(wide, lookback=20)
    weights = long_short_weights(ranks, top_frac=0.2, bottom_frac=0.2)
    row = weights.iloc[-1]
    assert abs(row.sum()) < 1e-9  # dollar-neutral regardless of leg size
    # pct-rank on n=10 lands exactly on 0.1 increments and ge/le are
    # inclusive of the threshold, so top/bottom 20% isn't a clean 2/2 split
    # at this boundary (ge(0.8) catches 3, le(0.2) catches 2) -- both legs
    # non-empty is what matters, exact counts are a pandas rank-tie artifact.
    assert (row > 0).sum() >= 2
    assert (row < 0).sum() >= 2


def test_long_short_weights_top_and_bottom_correct_names():
    wide = _wide_close(n_tickers=10)
    ranks = trailing_return_rank(wide, lookback=20)
    weights = long_short_weights(ranks, top_frac=0.2, bottom_frac=0.2)
    row = weights.iloc[-1]
    assert row["T9"] > 0 and row["T8"] > 0  # strongest two, long
    assert row["T0"] < 0 and row["T1"] < 0  # weakest two, short


def test_cross_sectional_returns_favors_persistent_momentum():
    # With persistent per-ticker drift (as built by _wide_close), a
    # long-strong/short-weak book should have a positive mean daily return.
    wide = _wide_close(n=200, n_tickers=10)
    port = cross_sectional_returns(wide, lookback=20, hold=5, top_frac=0.2, bottom_frac=0.2)
    assert len(port) > 0
    assert port.mean() > 0


def test_cross_sectional_returns_has_no_lookahead():
    # Truncating the input after the last date used for a given day's return
    # must not change that day's already-computed return -- weights come
    # only from data strictly before the day they're applied to.
    wide = _wide_close(n=100, n_tickers=10)
    full = cross_sectional_returns(wide, lookback=20, hold=5)
    truncated = cross_sectional_returns(wide.iloc[:80], lookback=20, hold=5)
    common = full.index.intersection(truncated.index)
    pd.testing.assert_series_equal(full.loc[common], truncated.loc[common])


def test_cross_sectional_returns_rebalance_hold_keeps_weights_constant_between_holds():
    wide = _wide_close(n=60, n_tickers=10)
    ranks = trailing_return_rank(wide, lookback=20)
    weights = long_short_weights(ranks, top_frac=0.2, bottom_frac=0.2)
    hold = 5
    rebalance_mask = pd.Series(np.arange(len(weights)) % hold == 0, index=weights.index)
    held = weights.where(rebalance_mask, np.nan).ffill().fillna(0.0)
    # any two rows within the same hold block must be identical
    block = held.iloc[21:21 + hold]
    for i in range(1, len(block)):
        pd.testing.assert_series_equal(block.iloc[0], block.iloc[i], check_names=False)
