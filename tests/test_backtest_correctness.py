"""Golden-value regression: backtest_trades() (pure Python) and
backtest_trades_fast() (numba) must produce identical trades. Required gate
before backtest_trades_fast() is trusted anywhere in the campaign runner."""
import numpy as np
import pandas as pd

from qresearch.signals import jma_signals, lhp_dsl_signals
from qresearch.backtest.engine import backtest_trades, backtest_trades_fast


def _synth(n=1000, seed=1):
    rng = pd.date_range("2024-01-01", periods=n, freq="1h")
    np.random.seed(seed)
    p = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({"Date": rng, "Ticker": "TEST", "open": p, "high": p + 1,
                         "low": p - 1, "close": p, "volume": 1000})


def _assert_matches(df):
    py = backtest_trades(df, sl_mult=1.5, tp_mult=3.0)
    fast = backtest_trades_fast(df, sl_mult=1.5, tp_mult=3.0)
    assert len(py) == len(fast)
    if len(py) == 0:
        return  # empty-trades shape differs (no columns vs named empty columns) -- not a correctness issue
    cols = ["entry_price", "exit_price", "ret", "reason"]
    pd.testing.assert_frame_equal(
        py[cols].reset_index(drop=True), fast[cols].reset_index(drop=True),
        rtol=1e-9, atol=1e-9,
    )


def test_numba_matches_python_jma():
    _assert_matches(jma_signals(_synth(1000), fast=7, slow=21))


def test_numba_matches_python_early_firing_signal():
    """lhp_dsl fires from bar 0 -- exercises the ATR-warmup NaN-skip path
    that once caused a degenerate single trade swallowing the whole series."""
    _assert_matches(lhp_dsl_signals(_synth(1000)))


def test_numba_matches_python_no_trades():
    df = _synth(50)
    df["signal"] = 0
    _assert_matches(df)
