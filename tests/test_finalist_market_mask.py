"""market_mask must actually gate entries at the portfolio level (Part B of
the drawdown-reduction plan) -- a broad-market trend gate shared across every
ticker, distinct from the per-ticker regime_filter. Default None must leave
every existing caller unaffected."""
import numpy as np
import pandas as pd

from qresearch.optimize.finalist_selection import best_insample, run_finalist_trades


def _oscillating_df(n=1200, phase=0.0):
    t = np.arange(n)
    price = 100 + 10 * np.sin(2 * np.pi * t / 8 + phase)
    return pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000,
    })


def test_market_mask_all_false_blocks_every_entry():
    df = _oscillating_df()
    all_false = pd.Series(False, index=df["Date"])
    result = best_insample(df, "TEST", "commodities", market_mask=all_false)
    # every entry gated off -> can never clear MIN_TRADES -> no combo qualifies
    assert result is None


def test_market_mask_all_true_matches_no_mask():
    df = _oscillating_df()
    all_true = pd.Series(True, index=df["Date"])
    with_mask = best_insample(df, "TEST", "commodities", market_mask=all_true)
    without_mask = best_insample(df, "TEST", "commodities")
    assert with_mask == without_mask


def test_run_finalist_trades_market_mask_default_none_unaffected(tmp_path):
    market_dir = tmp_path / "test_market" / "daily"
    market_dir.mkdir(parents=True)
    _oscillating_df(n=400, phase=0).to_parquet(market_dir / "TICKA.parquet", index=False)

    default = run_finalist_trades("dummy", "test_market", "daily", tmp_path)
    explicit_none = run_finalist_trades("dummy", "test_market", "daily", tmp_path, market_mask=None)
    pd.testing.assert_frame_equal(default, explicit_none)
