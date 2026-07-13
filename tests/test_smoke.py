"""Import + end-to-end smoke test for the qresearch package."""
import numpy as np, pandas as pd, pytest


def _synth(n=400):
    rng = pd.date_range("2024-01-01", periods=n, freq="1h")
    np.random.seed(1)
    p = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({"Date": rng, "Ticker": "TEST", "open": p, "high": p + 1,
                         "low": p - 1, "close": p, "volume": 1000})


def test_imports():
    import qresearch
    from qresearch.data import resample_ohlcv
    from qresearch.signals import jma_signals
    from qresearch.backtest import run_backtest, calculate_metrics, rank_and_score
    from qresearch.indicators import jma, atr, rsi
    from qresearch.risk import atr_stop_levels
    from qresearch.optimize import walk_forward_optimize
    from qresearch.portfolio import kelly_fraction
    from qresearch.execution import PaperBroker


def test_backtest_runs():
    from qresearch.signals import jma_signals
    from qresearch.backtest import run_backtest
    res = run_backtest(jma_signals(_synth(), fast=7, slow=21))
    assert set(["n_trades", "sharpe", "max_drawdown"]).issubset(res["metrics"])


def test_resample():
    from qresearch.data import resample_ohlcv
    out = resample_ohlcv(_synth(), "4h")
    assert {"Date", "Ticker", "close"}.issubset(out.columns)


def test_sizing():
    from qresearch.portfolio import kelly_fraction
    assert 0 <= kelly_fraction(0.55, 1.5) <= 1
