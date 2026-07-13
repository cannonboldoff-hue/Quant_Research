"""Portfolio combiner: trades -> daily returns -> combined sleeve, and the
first annualized Sharpe/CAGR/max-DD in the repo (backtest/metrics.py only
reports per-trade stats)."""
import numpy as np
import pandas as pd
import pytest

from qresearch.portfolio.combine import (
    apply_weights, combine_sleeves, portfolio_metrics, trades_to_daily_returns,
)


def _trades(exit_dates, rets):
    n = len(exit_dates)
    return pd.DataFrame({
        "entry_time": pd.to_datetime(exit_dates),
        "exit_time": pd.to_datetime(exit_dates),
        "direction": [1] * n,
        "entry_price": [100.0] * n,
        "exit_price": [100.0] * n,
        "reason": ["target"] * n,
        "pnl": rets,
        "ret": rets,
    })


def test_trades_to_daily_returns_averages_same_day_exits():
    trades = _trades(
        ["2024-01-02", "2024-01-02", "2024-01-03"],
        [0.02, 0.04, -0.01],
    )
    daily = trades_to_daily_returns(trades)
    assert list(daily.index.date.astype(str)) == ["2024-01-02", "2024-01-03"]
    assert daily.iloc[0] == 0.03  # mean of two same-day exits
    assert daily.iloc[1] == -0.01


def test_trades_to_daily_returns_empty():
    empty = pd.DataFrame(columns=["exit_time", "ret"])
    daily = trades_to_daily_returns(empty)
    assert daily.empty


def test_combine_sleeves_diversifies_anticorrelated_sleeves():
    # Two sleeves with equal individual vol but perfectly anti-correlated
    # daily returns -- combining them should reduce vol below either sleeve's
    # own vol (the whole point of a portfolio combiner).
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 0.01, 100), index=dates)
    b = -a  # perfectly anti-correlated, identical vol
    combo = combine_sleeves({"A": a, "B": b}, mode="equal_weight")
    assert combo["combined"].std(ddof=0) < a.std(ddof=0)
    assert combo["combined"].std(ddof=0) < 1e-9  # anti-correlated + equal weight -> near-zero vol
    assert set(combo["weights"]) == {"A", "B"}
    assert combo["corr"].loc["A", "B"] == -1.0


def test_combine_sleeves_inverse_vol_downweights_volatile_sleeve():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(1)
    calm = pd.Series(rng.normal(0.0005, 0.005, 200), index=dates)
    wild = pd.Series(rng.normal(0.0005, 0.05, 200), index=dates)  # 10x the vol
    combo = combine_sleeves({"calm": calm, "wild": wild}, mode="inverse_vol")
    assert combo["weights"]["calm"] > combo["weights"]["wild"]


def test_combine_sleeves_fills_missing_days_as_flat():
    a = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    b = pd.Series([0.01], index=pd.to_datetime(["2024-01-01"]))
    combo = combine_sleeves({"a": a, "b": b}, mode="equal_weight")
    assert len(combo["aligned"]) == 2
    assert combo["aligned"].loc["2024-01-02", "b"] == 0.0


def test_combine_sleeves_tangency_beats_best_single_sleeve_when_uncorrelated():
    # Deterministic (not RNG-sampled) uncorrelated sleeves with DIFFERENT
    # Sharpes, built from orthogonal sinusoids so mean/var/near-zero-corr are
    # exact rather than at the mercy of a random seed. This is the case that
    # broke naive inverse-vol on the real 5-sleeve run: it overweighted the
    # low-vol/lower-Sharpe sleeve (forex) and the combined Sharpe fell below
    # the best single sleeve (commodities). Tangency weighting should combine
    # into a Sharpe that exceeds either individual sleeve.
    i = np.arange(1000)
    dates = pd.date_range("2024-01-01", periods=1000, freq="D")
    a = pd.Series(0.001 + 0.01 * np.sin(2 * np.pi * i / 50), index=dates)
    b = pd.Series(0.0003 + 0.003 * np.cos(2 * np.pi * i / 40), index=dates)
    assert abs(a.corr(b)) < 1e-6  # sanity: construction is (numerically) uncorrelated

    combo = combine_sleeves({"a": a, "b": b}, mode="tangency")
    sr_a = portfolio_metrics(a)["sharpe"]
    sr_b = portfolio_metrics(b)["sharpe"]
    sr_combined = portfolio_metrics(combo["combined"])["sharpe"]
    assert sr_a > 0 and sr_b > 0  # sanity: both legit positive-Sharpe inputs
    assert sr_combined > max(sr_a, sr_b)

    # naive inverse-vol on the same data should NOT beat tangency (it's the
    # exact failure mode tangency fixes: it ignores return quality)
    inv_combo = combine_sleeves({"a": a, "b": b}, mode="inverse_vol")
    sr_inv = portfolio_metrics(inv_combo["combined"])["sharpe"]
    assert sr_combined > sr_inv


def test_combine_sleeves_tangency_is_long_only():
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    rng = np.random.default_rng(3)
    good = pd.Series(rng.normal(0.002, 0.01, 300), index=dates)
    # strongly anti-correlated with `good` and near-zero mean -- an
    # unconstrained mean-variance solve could want to short this sleeve.
    bad = -good * 0.9 + rng.normal(0.0, 0.001, 300)
    combo = combine_sleeves({"good": good, "bad": bad}, mode="tangency")
    assert all(w >= 0 for w in combo["weights"].values())
    assert abs(sum(combo["weights"].values()) - 1.0) < 1e-9


def test_apply_weights_matches_manual_weighted_sum():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    aligned = pd.DataFrame({"a": [0.01, 0.02, -0.01, 0.0, 0.03],
                            "b": [0.00, 0.01, 0.01, 0.02, -0.02]}, index=dates)
    combined = apply_weights(aligned, {"a": 0.7, "b": 0.3})
    expected = aligned["a"] * 0.7 + aligned["b"] * 0.3
    pd.testing.assert_series_equal(combined, expected, check_names=False)


def test_apply_weights_can_fit_on_one_window_and_eval_on_another():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0.001, 0.01, 200), index=dates)
    b = pd.Series(rng.normal(0.0005, 0.02, 200), index=dates)
    fit_dates, eval_dates = dates[:140], dates[140:]

    fit_combo = combine_sleeves({"a": a[fit_dates], "b": b[fit_dates]}, mode="tangency")
    eval_aligned = pd.DataFrame({"a": a[eval_dates], "b": b[eval_dates]})
    eval_combined = apply_weights(eval_aligned, fit_combo["weights"])

    assert len(eval_combined) == len(eval_dates)
    # weights came from the fit window only -- eval-window combined series
    # must be a plain linear combination of the eval returns, not re-derived
    assert eval_combined.iloc[0] == pytest.approx(
        eval_aligned["a"].iloc[0] * fit_combo["weights"]["a"]
        + eval_aligned["b"].iloc[0] * fit_combo["weights"]["b"]
    )


def test_portfolio_metrics_constant_positive_returns():
    r = pd.Series([0.001] * 252)
    m = portfolio_metrics(r, periods=252)
    assert m["n_days"] == 252
    assert m["max_drawdown"] == 0.0  # monotonically increasing equity
    assert m["cagr"] == pytest.approx((1.001 ** 252) - 1, rel=1e-9)
    assert m["sharpe"] > 0


def test_portfolio_metrics_empty():
    m = portfolio_metrics(pd.Series(dtype=float))
    assert m == {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                 "annual_vol": 0.0, "n_days": 0}
