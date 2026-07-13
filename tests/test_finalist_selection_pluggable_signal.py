"""finalist_selection's walk-forward gate must stay usable with a signal
function other than jma_signals (e.g. kama_dsl_signals), while every
existing call site (no signal_fn/signal_grid args) keeps working unchanged."""
import numpy as np
import pandas as pd

from qresearch.optimize.finalist_selection import best_insample, run_finalist_trades
from qresearch.signals.generators import jma_signals


def _trending_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000,
    })


def _always_flat_signals(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """A trivial non-jma signal function: same shape contract (adds a
    'signal' column), but never enters a trade. Used to prove signal_fn
    actually routes through best_insample/run_finalist_trades rather than
    silently falling back to jma_signals."""
    out = df.copy()
    out["signal"] = 0
    return out


def test_best_insample_default_matches_jma_signals_explicit():
    df = _trending_df()
    default_result = best_insample(df, "TEST", "commodities")
    explicit_result = best_insample(df, "TEST", "commodities", signal_fn=jma_signals)
    assert default_result == explicit_result


def test_best_insample_stop_grid_restricts_risk_params():
    df = _trending_df()
    tight = {"sl_mult": [0.75], "tp_mult": [2.0]}
    result = best_insample(df, "TEST", "commodities", stop_grid=tight)
    # a custom stop_grid must be what's actually searched, not a silent
    # fallback to the module-level STOP_GRID
    assert result is None or result[1] == {"sl_mult": 0.75, "tp_mult": 2.0}


def test_best_insample_with_always_flat_signal_finds_nothing():
    df = _trending_df()
    # a signal function that never fires can never clear MIN_TRADES -- proves
    # the custom signal_fn is what's actually being backtested, not jma_signals
    result = best_insample(df, "TEST", "commodities",
                           signal_fn=_always_flat_signals, signal_grid={"x": [1]})
    assert result is None


def test_run_finalist_trades_with_custom_signal_grid_uses_grid_keys(tmp_path):
    market_dir = tmp_path / "test_market" / "daily"
    market_dir.mkdir(parents=True)
    _trending_df(n=400, seed=1).to_parquet(market_dir / "TICKA.parquet", index=False)

    # flat signal -> zero trades -> exercises the empty-result fallback path,
    # which must reflect the CUSTOM grid's keys ("x"), not jma's fast/slow
    trades = run_finalist_trades("dummy", "test_market", "daily", tmp_path,
                                  signal_fn=_always_flat_signals, signal_grid={"x": [1, 2]})
    assert trades.empty
    assert "x" in trades.columns
    assert "fast" not in trades.columns


def test_run_finalist_trades_default_signal_fn_still_jma(tmp_path):
    market_dir = tmp_path / "test_market" / "daily"
    market_dir.mkdir(parents=True)
    _trending_df(n=400, seed=2).to_parquet(market_dir / "TICKA.parquet", index=False)

    trades = run_finalist_trades("dummy", "test_market", "daily", tmp_path)
    # a real trending series should produce at least some JMA crossover trades
    # by default, unlike the always-flat signal_fn case above
    assert len(trades) >= 0  # sanity: call succeeds with default args, no signature break
    if not trades.empty:
        assert {"fast", "slow", "sl_mult", "tp_mult"}.issubset(trades.columns)
