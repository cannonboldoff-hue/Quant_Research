"""max_concurrent admission control on leverage_sim.simulate(): a real book
has a finite number of risk "slots" -- this caps how many trades can be open
at once, rejecting new entries once the cap is hit, rather than letting an
unbounded number of simultaneously-open positions stack risk with no limit
(the diagnostic behind this fix found ~30 of 48 tickers open concurrently on
average in the real trade book)."""
import pandas as pd

from qresearch.risk.leverage_sim import simulate


def _overlapping_trades():
    # 3 trades, all open 2016-01-01..2016-01-10 (fully overlapping), each -1%
    return pd.DataFrame({
        "entry_time": pd.to_datetime(["2016-01-01", "2016-01-01", "2016-01-01"]),
        "exit_time": pd.to_datetime(["2016-01-10", "2016-01-10", "2016-01-10"]),
        "reason": ["stop", "stop", "stop"],
        "ret": [-0.01, -0.01, -0.01],
    })


def test_max_concurrent_none_admits_every_trade():
    trades = _overlapping_trades()
    result = simulate(trades, leverage=100, max_concurrent=None)
    assert result.n_trades == 3
    assert result.n_skipped_capacity == 0


def test_max_concurrent_caps_admission():
    trades = _overlapping_trades()
    result = simulate(trades, leverage=100, max_concurrent=1)
    # only the first of 3 fully-overlapping trades fits under a cap of 1
    assert result.n_trades == 1
    assert result.n_skipped_capacity == 2


def test_max_concurrent_admits_sequential_non_overlapping_trades():
    # two trades that don't overlap in time both fit even under cap=1
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime(["2016-01-01", "2016-02-01"]),
        "exit_time": pd.to_datetime(["2016-01-10", "2016-02-10"]),
        "reason": ["stop", "stop"],
        "ret": [-0.01, -0.01],
    })
    result = simulate(trades, leverage=100, max_concurrent=1)
    assert result.n_trades == 2
    assert result.n_skipped_capacity == 0
