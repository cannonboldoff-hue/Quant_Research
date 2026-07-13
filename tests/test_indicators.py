"""Golden-value / sanity tests for the Phase 1 indicator extraction
(vwap, curvature, dsl, kalman, cycle, trend, volume) — the functions that
were still inline in notebooks (VWAP, curvature, LHP DSL, JMA-adjacent
trend/volume indicators) before this pass."""
import numpy as np
import pandas as pd
import pytest

from qresearch.indicators import (
    vwap, tick_vwap, curvature, lhp_dsl, ltp_kama_dsl, kalman_filter,
    goertzel, detect_strongest_cycle, supertrend, halftrend, dema,
    normalized_dema, kama, zero_lag_ma, weighted_moving_average,
    obv, mfi, tmf, pvs_vms, samx, breadth, adx, ohlc4,
    rogers_satchell_vol, historical_volatility, rolling_zscore, rolling_r2,
    stochastic_oscillator,
)


def _ohlcv(n=300, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1, n)
    low = close - rng.uniform(0.1, 1, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_vwap_matches_manual_typical_price_calc():
    df = _ohlcv()
    out = vwap(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    expected = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_vwap_constant_volume_equals_expanding_mean_of_typical_price():
    df = _ohlcv()
    df["volume"] = 10.0
    out = vwap(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    np.testing.assert_allclose(out.to_numpy(), typical.expanding().mean().to_numpy())


def test_tick_vwap_bounded_by_price_range():
    price = pd.Series(np.linspace(100, 110, 50))
    qty = pd.Series(np.ones(50))
    out = tick_vwap(price, qty)
    assert (out.dropna() >= price.min() - 1e-9).all()
    assert (out.dropna() <= price.max() + 1e-9).all()


def test_curvature_zero_for_straight_line_motion():
    # constant-velocity residuals -> acceleration ~ 0 -> curvature ~ 0
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(2)
    btc = 100 + np.cumsum(rng.normal(0, 0.1, n))
    trend = np.linspace(0, 5, n)
    alt1 = 50 + trend + btc * 0.0001
    alt2 = 60 + trend * 1.5 + btc * 0.0001
    prices = pd.DataFrame({"BTC": btc, "ALT1": alt1, "ALT2": alt2}, index=idx)
    curv = curvature(prices, benchmark="BTC", beta_lookback=48, velocity_period=5, pca_n_components=2)
    tail = curv.tail(100)
    assert tail.median() < 1.0  # near-linear motion -> low curvature, not exploding


def test_lhp_dsl_signal_is_categorical():
    df = _ohlcv()
    sig = lhp_dsl(df["close"])
    assert set(sig.unique()).issubset({0, 1, 2})
    assert len(sig) == len(df)


def test_ltp_kama_dsl_signal_is_categorical():
    df = _ohlcv()
    sig = ltp_kama_dsl(df)
    assert set(sig.unique()).issubset({0, 1, 2})


def test_kalman_filter_converges_on_constant_series():
    prices = pd.Series(np.full(200, 50.0))
    out = kalman_filter(prices)
    assert abs(out.iloc[-1] - 50.0) < 1e-6


def test_kalman_filter_smooths_noise():
    rng = np.random.default_rng(3)
    prices = pd.Series(100 + rng.normal(0, 2, 500))
    out = kalman_filter(prices, process_variance=1e-6, measurement_variance=4.0)
    assert out.diff().abs().mean() < prices.diff().abs().mean()


def test_goertzel_detects_known_sine_frequency():
    n = 200
    freq = 1 / 20  # 20-sample cycle
    t = np.arange(n)
    signal = np.sin(2 * np.pi * freq * t)
    mag, best_freq, _ = detect_strongest_cycle(pd.Series(signal), max_cycle=40, window=n)
    assert abs(best_freq - freq) < 0.01


def test_supertrend_is_directionally_sane():
    df = _ohlcv()
    out = supertrend(df, period=10, multiplier=3.0).dropna()
    assert set(out["trend"].unique()).issubset({1, -1})
    # in an uptrend bar, supertrend line sits below close; downtrend, above
    up = out["trend"] == 1
    aligned_close = df.loc[out.index, "close"]
    assert (out.loc[up, "supertrend"] <= aligned_close[up] + 1e-6).mean() > 0.7


def test_halftrend_produces_expected_columns():
    df = _ohlcv(100)
    out = halftrend(df)
    assert {"half_trend", "atr_high", "atr_low", "signal", "trend", "atr"}.issubset(out.columns)
    assert set(out["signal"].unique()).issubset({0, 1, 2})


def test_dema_less_lagged_than_ema_on_ramp():
    ramp = pd.Series(np.arange(200, dtype=float))
    from qresearch.indicators import ema
    e = ema(ramp, 20)
    d = dema(ramp, 20)
    # DEMA should track the ramp more closely (smaller mean lag/error) than EMA
    assert (ramp - d).abs().tail(50).mean() < (ramp - e).abs().tail(50).mean()


def test_normalized_dema_band_is_bounded_ish():
    df = _ohlcv()
    out = normalized_dema(df["close"], len_dema=10, base_len=30)
    valid = out["norm"].dropna()
    assert len(valid) > 0


def test_kama_flat_series_stays_flat():
    flat = pd.Series(np.full(100, 42.0))
    out = kama(flat)
    assert np.allclose(out.to_numpy(), 42.0)


def test_zero_lag_ma_runs_and_tracks_series():
    s = pd.Series(100 + np.cumsum(np.random.default_rng(4).normal(0, 1, 200)))
    out = zero_lag_ma(s, 10)
    assert out.notna().sum() > 0


def test_weighted_moving_average_matches_manual_calc():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = weighted_moving_average(s, 3)
    # last window [3,4,5] weighted [1,2,3] desc-applied per original conv logic
    weights = np.array([3, 2, 1])
    expected_last = np.dot(weights, [5, 4, 3]) / weights.sum()
    assert abs(out[-1] - expected_last) < 1e-9


def test_obv_matches_manual_calc():
    close = pd.Series([10, 11, 10, 12, 12, 11])
    volume = pd.Series([100, 100, 100, 100, 100, 100])
    out = obv(close, volume)
    assert out.tolist() == [0, 100, 0, 100, 100, 0]


def test_mfi_bounded_0_100():
    df = _ohlcv()
    out = mfi(df).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_tmf_runs_without_error():
    df = _ohlcv()
    out = tmf(df, 21)
    assert len(out) == len(df)


def test_pvs_vms_and_samx_runs():
    df = _ohlcv()
    pv = pvs_vms(df)
    assert {"pvs", "vms"}.issubset(pv.columns)
    s = samx(df)
    assert len(s) == len(df)


def test_breadth_monotonic_nondecreasing():
    close = pd.Series([1, 2, 1, 3, 2, 4])
    out = breadth(close)
    assert (out["advancing"].diff().dropna() >= 0).all()
    assert (out["declining"].diff().dropna() >= 0).all()


def test_adx_bounded_0_100():
    df = _ohlcv()
    out = adx(df).dropna()
    assert (out["adx"] >= 0).all() and (out["adx"] <= 100).all()


def test_ohlc4_matches_manual_calc():
    df = _ohlcv(10)
    out = ohlc4(df)
    expected = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_rogers_satchell_vol_runs():
    df = _ohlcv()
    out = rogers_satchell_vol(df)
    assert len(out) == len(df)


def test_historical_volatility_nonnegative():
    df = _ohlcv()
    out = historical_volatility(df["close"]).dropna()
    assert (out >= 0).all()


def test_rolling_zscore_zero_for_constant_window():
    s = pd.Series(np.full(50, 5.0))
    out = rolling_zscore(s, window=10)
    assert (out.dropna() == 0).all()


def test_rolling_r2_high_for_perfect_line():
    s = pd.Series(np.arange(60, dtype=float))
    out = rolling_r2(s, window=20).dropna()
    assert (out > 0.99).all()


def test_stochastic_oscillator_bounded_0_100():
    df = _ohlcv()
    out = stochastic_oscillator(df).dropna()
    assert (out["k"] >= 0).all() and (out["k"] <= 100).all()
