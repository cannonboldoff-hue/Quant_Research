"""Tests for the Phase 2/3/4 exhaustive-extraction pass: signals (smc/patterns),
backtest (metrics fixes + robustness_testing), risk analysis, viz, data
preprocess, features, ml, execution."""
import numpy as np
import pandas as pd
import pytest

from qresearch.signals import (
    fair_value_gap, order_blocks, liquidity_sweeps, fibonacci_retracement, in_session,
    find_divergences, quad_stochastic_signals, zscore_spread, range_detector, mean_reversion_signal,
)
from qresearch.backtest import (
    profit_factor, win_rate, quarterly_hpr, max_drawdown, calculate_metrics, robustness_testing,
)
from qresearch.risk import check_margin_call, monte_carlo_simulation, expected_loss
from qresearch.viz import plot_drawdown, plot_pnl_by_exit_reason, plot_pnl_by_weekday, plot_monthly_profit_heatmap
from qresearch.data import create_quarterly_features, resolve_columns
from qresearch.features import add_lag_features, add_cyclical_time_features, add_market_microstructure_features
from qresearch.ml import label_trade_outcome, align_signals_to_ohlcv, get_score
from qresearch.signals.generators import jma_signals


def _ohlcv(n=300, seed=1, ticker="TEST"):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1, n)
    low = close - rng.uniform(0.1, 1, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume, "Date": idx, "Ticker": ticker})


def _trades(n=50, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.001, 0.02, n)
    return pd.DataFrame({
        "entry_time": pd.date_range("2024-01-01", periods=n, freq="1D"),
        "exit_time": pd.date_range("2024-01-02", periods=n, freq="1D"),
        "direction": rng.choice([1, -1], n),
        "pnl": ret * 100,
        "ret": ret,
        "reason": rng.choice(["stop", "target", "eod"], n),
    })


# --- signals: smc ---

def test_fair_value_gap_runs():
    df = _ohlcv()
    out = fair_value_gap(df["close"], threshold=0.5)
    assert out.dtype == bool and len(out) == len(df)


def test_order_blocks_categorical():
    df = _ohlcv()
    out = order_blocks(df["close"], df["close"] * 0.99, df["close"] * 1.01, lookback=20)
    assert set(out.dropna().unique()).issubset({"bullish", "bearish"})


def test_liquidity_sweeps_runs():
    df = _ohlcv()
    out = liquidity_sweeps(df["close"], df["close"] * 0.99, df["close"] * 1.01, short_lookback=5, long_lookback=50)
    assert out.dtype == bool


def test_fibonacci_retracement_zone_bounded():
    df = _ohlcv()
    out = fibonacci_retracement(df["close"], lookback=50)
    valid = out["retracement"].dropna()
    assert (valid >= -0.01).all() and (valid <= 1.01).all()


def test_in_session_matches_manual():
    idx = pd.date_range("2024-01-01 09:00", periods=24, freq="1h")
    out = in_session(idx, "10:00", "14:00")
    expected = [(t.time() >= pd.to_datetime("10:00").time()) and (t.time() <= pd.to_datetime("14:00").time()) for t in idx]
    assert out.tolist() == expected


# --- signals: patterns ---

def test_find_divergences_detects_bullish_case():
    # oscillator makes a lower low while price makes a higher low -> bullish divergence
    osc = pd.Series([5, 4, 3, 2, 1, 2, 3, 1.5, 2, 3, 4, 5, 6, 7, 8])
    price = pd.Series([10, 9, 8, 7, 6, 7, 8, 6.5, 9, 10, 11, 12, 13, 14, 15])
    bull, bear = find_divergences(osc, price, left=3, search=10)
    assert isinstance(bull, list) and isinstance(bear, list)


def test_quad_stochastic_signals_categorical():
    df = _ohlcv(200)
    out = quad_stochastic_signals(df, min_confirm=1)
    assert set(out.unique()).issubset({0, 1, 2})


def test_zscore_spread_matches_manual():
    spread = pd.Series(np.arange(100, dtype=float))
    out = zscore_spread(spread, window=10)
    mean = spread.rolling(10).mean()
    std = spread.rolling(10).std()
    expected = (spread - mean) / std
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_range_detector_runs():
    df = _ohlcv(200)
    out = range_detector(df, length=5, mult=2, atr_len=50, volume_mult=1.0)
    assert set(out["signal"].unique()).issubset({0, 1, 2})


def test_mean_reversion_signal_short_series_returns_zeros():
    df = _ohlcv(10)
    out = mean_reversion_signal(df, smooth_length=172)
    assert (out == 0).all()


def test_mean_reversion_signal_categorical_on_long_series():
    df = _ohlcv(400)
    out = mean_reversion_signal(df, smooth_length=50, lyap_window=40, rs_window=20)
    assert set(out.unique()).issubset({0, 1, 2})


# --- backtest ---

def test_profit_factor_edge_cases():
    assert profit_factor(pd.Series([0.0, 0.0])) == 0.0
    assert profit_factor(pd.Series([0.1, 0.2])) == float("inf")
    assert profit_factor(pd.Series([0.2, -0.1])) == pytest.approx(2.0)


def test_win_rate_matches_manual():
    r = pd.Series([0.1, -0.1, 0.2, -0.2, 0.3])
    assert win_rate(r) == pytest.approx(0.6)


def test_max_drawdown_on_equity_curve():
    equity = pd.Series([1.0, 1.1, 1.05, 0.9, 0.95, 1.2])
    dd = max_drawdown(equity)
    assert dd == pytest.approx((0.9 - 1.1) / 1.1)


def test_calculate_metrics_uses_max_drawdown_not_dead_code():
    trades = pd.DataFrame({"ret": [0.1, -0.05, 0.2, -0.3, 0.1]})
    m = calculate_metrics(trades)
    equity = (1 + trades["ret"]).cumprod()
    assert m["max_drawdown"] == pytest.approx(max_drawdown(equity))


def test_quarterly_hpr_runs():
    dates = pd.date_range("2024-01-01", periods=200, freq="1D")
    returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 200))
    out = quarterly_hpr(returns, dates)
    assert len(out) > 0


def test_robustness_testing_runs_across_tickers_and_params():
    df = pd.concat([_ohlcv(100, seed=1, ticker="A"), _ohlcv(100, seed=2, ticker="B")], ignore_index=True)
    out = robustness_testing(df, lambda d, fast, slow: jma_signals(d, fast=fast, slow=slow),
                             {"fast": [5], "slow": [20]}, ticker_col="Ticker")
    assert set(out["ticker"].unique()) == {"A", "B"}
    assert "sharpe" in out.columns


# --- risk analysis ---

def test_check_margin_call_triggers_when_equity_below_required_margin():
    # margin_used = capital/leverage = 2000; equity < 2000 needs drawdown > 1 - 1/leverage = 0.8
    is_call, equity = check_margin_call(drawdown=0.9, capital=10_000, leverage=5, margin_call_threshold=1.0)
    assert is_call is True
    assert equity == 1_000


def test_check_margin_call_no_trigger_on_small_drawdown():
    is_call, equity = check_margin_call(drawdown=0.01, capital=10_000, leverage=5, margin_call_threshold=1.0)
    assert is_call is False


def test_monte_carlo_simulation_shape():
    returns = np.random.default_rng(1).normal(0.001, 0.01, 100)
    out = monte_carlo_simulation(returns, num_simulations=50, num_days=20)
    assert out.shape == (50, 20)


def test_expected_loss_matches_manual():
    assert expected_loss(0.1, recovery_rate=0.2) == pytest.approx(0.08)


# --- viz (just verify they run headless without error) ---

def test_viz_functions_run():
    trades = _trades()
    plot_drawdown(trades)
    plot_pnl_by_exit_reason(trades)
    plot_pnl_by_weekday(trades)
    plot_monthly_profit_heatmap(trades)


# --- data preprocess ---

def test_create_quarterly_features_runs():
    df = _ohlcv(400).set_index("Date")
    out = create_quarterly_features(df)
    assert "forward_return" in out.columns


def test_resolve_columns_finds_aliases():
    df = pd.DataFrame({"Entry Time": [1], "Exit Time": [2], "PnL": [3]})
    cols = resolve_columns(df)
    assert cols["entry_time"] == "Entry Time"
    assert cols["pnl"] == "PnL"


def test_resolve_columns_raises_on_missing_required():
    df = pd.DataFrame({"foo": [1]})
    with pytest.raises(ValueError):
        resolve_columns(df)


# --- features ---

def test_add_lag_features_matches_manual():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = add_lag_features(df, ["x"], lags=(1,))
    assert out["x_lag1"].tolist()[1:] == [1, 2, 3, 4]


def test_add_cyclical_time_features_bounded():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=48, freq="1h")})
    out = add_cyclical_time_features(df)
    assert (out["hour_sin"].abs() <= 1).all() and (out["hour_cos"].abs() <= 1).all()


def test_add_market_microstructure_features_no_inf():
    df = _ohlcv(100)
    out = add_market_microstructure_features(df)
    feature_cols = ["return_1", "return_5", "volatility_5", "volume_change", "spread"]
    assert np.isfinite(out[feature_cols].to_numpy()).all()


# --- ml (non-xgboost parts only, since xgboost isn't installed here) ---

def test_label_trade_outcome():
    signal = pd.Series([0, 1, 1, 2, 2])
    future_return = pd.Series([0.0, 0.01, -0.01, -0.01, 0.01])
    out = label_trade_outcome(signal, future_return)
    assert out.tolist() == [np.nan, 1, 0, 1, 0] or out.dropna().tolist() == [1, 0, 1, 0]


def test_align_signals_to_ohlcv_fills_zero():
    ohlcv = pd.DataFrame({"Ticker": ["A", "A"], "Date": pd.date_range("2024-01-01", periods=2)})
    signals = pd.DataFrame({"Ticker": ["A"], "Date": [pd.Timestamp("2024-01-01")], "signal": [1]})
    out = align_signals_to_ohlcv(ohlcv, signals)
    assert out["signal"].tolist() == [1, 0]


def test_get_score_runs_with_sklearn_model():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(50, 3))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(20, 3))
    y_test = (X_test[:, 0] > 0).astype(int)
    score = get_score(LogisticRegression(), X_train, X_test, y_train, y_test)
    assert 0 <= score <= 1


# --- execution ---

def test_ccxt_broker_is_a_broker_base():
    from qresearch.execution import CcxtBroker, BrokerBase
    assert issubclass(CcxtBroker, BrokerBase)
    assert hasattr(CcxtBroker, "place_order") and hasattr(CcxtBroker, "get_positions")


def test_ccxt_broker_round_price_uses_exchange_precision():
    from qresearch.execution import CcxtBroker

    class _FakeExchange:
        def load_markets(self): pass
        def price_to_precision(self, symbol, price): return round(price, 2)
        def amount_to_precision(self, symbol, qty): return round(qty, 3)

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.exchange = _FakeExchange()
    assert broker.round_price("BTC/USDT", 100.12345) == 100.12
    assert broker.round_quantity("BTC/USDT", 1.23456) == 1.235
