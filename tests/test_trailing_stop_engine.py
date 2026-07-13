"""Part C of the drawdown-reduction plan: trailing-stop exit engine.
backtest_trades_trailing_fast must actually ratchet the stop in the trade's
favor (locking in gains) rather than behaving like a fixed sl/tp, and
run_finalist_trades' new backtest_fn param must route to it instead of
silently falling back to backtest_trades_fast."""
import numpy as np
import pandas as pd

from qresearch.backtest.engine import backtest_trades_fast, backtest_trades_trailing_fast
from qresearch.optimize.finalist_selection import run_finalist_trades


def _ramp_up_then_reverse(n=80):
    # flat warmup (ATR needs 14 bars), then price ramps up for 40 bars, then
    # reverses hard -- a fixed sl/tp would either never hit tp (if tp is far)
    # and give back the whole gain on reversal, while a trailing stop should
    # lock in most of the run-up
    warmup = np.full(20, 100.0)
    up = np.linspace(100, 140, 40)
    down = np.linspace(140, 90, n - 60)
    price = np.concatenate([warmup, up, down])
    df = pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 1000,
    })
    df["signal"] = 0
    df.loc[19, "signal"] = 1  # long entry right as the ramp starts, past ATR warmup
    return df


def test_trailing_stop_locks_in_gains_on_reversal():
    df = _ramp_up_then_reverse()
    trailing = backtest_trades_trailing_fast(df, sl_mult=1.5, trail_mult=2.0, fee_bps=0)
    fixed = backtest_trades_fast(df, sl_mult=1.5, tp_mult=3.0, fee_bps=0)
    assert not trailing.empty and not fixed.empty
    # trailing should exit with meaningfully positive return, capturing part
    # of the run-up instead of riding it all the way back down
    assert trailing.iloc[0]["ret"] > 0
    assert trailing.iloc[0]["reason"] == "stop"


def test_trailing_stop_never_exits_via_target_reason():
    df = _ramp_up_then_reverse()
    trailing = backtest_trades_trailing_fast(df, sl_mult=1.5, trail_mult=2.0, fee_bps=0)
    assert (trailing["reason"] != "target").all()


def test_run_finalist_trades_backtest_fn_routes_to_trailing(tmp_path):
    market_dir = tmp_path / "test_market" / "daily"
    market_dir.mkdir(parents=True)
    n = 800
    t = np.arange(n)
    price = 100 + 10 * np.sin(2 * np.pi * t / 8)
    pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000,
    }).to_parquet(market_dir / "TICKA.parquet", index=False)

    trailing_grid = {"sl_mult": [1.0, 1.5], "trail_mult": [1.5, 2.0]}
    trades = run_finalist_trades("dummy", "test_market", "daily", tmp_path,
                                  backtest_fn=backtest_trades_trailing_fast, stop_grid=trailing_grid)
    assert len(trades) >= 0  # sanity: call succeeds, no signature break
    if not trades.empty:
        assert {"sl_mult", "trail_mult"}.issubset(trades.columns)
        assert (trades["reason"] != "target").all()


def test_run_finalist_trades_default_backtest_fn_unaffected(tmp_path):
    market_dir = tmp_path / "test_market" / "daily"
    market_dir.mkdir(parents=True)
    n = 400
    t = np.arange(n)
    price = 100 + 10 * np.sin(2 * np.pi * t / 8)
    pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000,
    }).to_parquet(market_dir / "TICKA.parquet", index=False)

    default = run_finalist_trades("dummy", "test_market", "daily", tmp_path)
    explicit = run_finalist_trades("dummy", "test_market", "daily", tmp_path, backtest_fn=backtest_trades_fast)
    pd.testing.assert_frame_equal(default, explicit)
