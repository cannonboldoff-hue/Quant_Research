"""Volume/money-flow confirmation filter: long entries kept only when MFI
confirms buying pressure, shorts only when MFI confirms selling pressure."""
import numpy as np
import pandas as pd

from qresearch.signals.volume_filter import apply_volume_filter


def _ohlcv_with_mfi_pattern(n=60):
    # first half: rising price + rising volume (MFI should trend high)
    # second half: falling price + rising volume (MFI should trend low)
    rng = np.random.default_rng(0)
    up = 100 + np.arange(n // 2) * 0.5 + rng.normal(0, 0.05, n // 2)
    down = up[-1] - np.arange(n // 2) * 0.5 + rng.normal(0, 0.05, n // 2)
    close = np.concatenate([up, down])
    volume = np.full(n, 1000) + rng.normal(0, 10, n)
    return pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": volume,
    })


def test_apply_volume_filter_keeps_long_when_mfi_confirms():
    df = _ohlcv_with_mfi_pattern()
    sig_df = df.copy()
    sig_df["signal"] = 0
    sig_df.loc[10, "signal"] = 1  # long entry during the uptrend (high MFI expected)
    out = apply_volume_filter(sig_df, mfi_length=14, mfi_midpoint=50.0)
    assert out.loc[10, "signal"] == 1


def test_apply_volume_filter_zeroes_long_when_mfi_does_not_confirm():
    df = _ohlcv_with_mfi_pattern()
    sig_df = df.copy()
    sig_df["signal"] = 0
    sig_df.loc[45, "signal"] = 1  # long entry during the downtrend (low MFI expected)
    out = apply_volume_filter(sig_df, mfi_length=14, mfi_midpoint=50.0)
    assert out.loc[45, "signal"] == 0


def test_apply_volume_filter_keeps_short_when_mfi_confirms():
    df = _ohlcv_with_mfi_pattern()
    sig_df = df.copy()
    sig_df["signal"] = 0
    sig_df.loc[45, "signal"] = -1  # short entry during the downtrend (low MFI expected)
    out = apply_volume_filter(sig_df, mfi_length=14, mfi_midpoint=50.0)
    assert out.loc[45, "signal"] == -1


def test_apply_volume_filter_zeroes_short_when_mfi_does_not_confirm():
    df = _ohlcv_with_mfi_pattern()
    sig_df = df.copy()
    sig_df["signal"] = 0
    sig_df.loc[10, "signal"] = -1  # short entry during the uptrend (high MFI expected)
    out = apply_volume_filter(sig_df, mfi_length=14, mfi_midpoint=50.0)
    assert out.loc[10, "signal"] == 0


def test_apply_volume_filter_leaves_flat_bars_alone():
    df = _ohlcv_with_mfi_pattern()
    sig_df = df.copy()
    sig_df["signal"] = 0
    out = apply_volume_filter(sig_df)
    assert (out["signal"] == 0).all()


def test_apply_volume_filter_does_not_mutate_input():
    df = _ohlcv_with_mfi_pattern()
    sig_df = df.copy()
    sig_df["signal"] = 0
    sig_df.loc[10, "signal"] = 1
    apply_volume_filter(sig_df)
    assert sig_df.loc[10, "signal"] == 1  # original untouched
