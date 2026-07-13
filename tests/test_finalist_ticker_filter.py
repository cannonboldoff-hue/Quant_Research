"""run_finalist_trades' tickers= filter must restrict the walk-forward
universe to the given subset, while default (tickers=None) still walks
every ticker on disk -- proves the sector-analysis campaign script's
subsetting mechanism actually filters, not just accepts the arg."""
import numpy as np
import pandas as pd

from qresearch.optimize.finalist_selection import run_finalist_trades


def _oscillating_df(n=1200, phase=0.0):
    """Deterministic sine-wave price -- frequent, regular JMA fast/slow
    crossovers per fold (unlike a random walk, whose crossover count is too
    noisy to reliably clear MIN_TRADES=10 within an 80-200 bar fold)."""
    t = np.arange(n)
    price = 100 + 10 * np.sin(2 * np.pi * t / 8 + phase)
    return pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000,
    })


def _write_market(tmp_path, tickers):
    market_dir = tmp_path / "test_market" / "daily"
    market_dir.mkdir(parents=True, exist_ok=True)
    for i, ticker in enumerate(tickers):
        _oscillating_df(phase=i).to_parquet(market_dir / f"{ticker}.parquet", index=False)
    return market_dir


def test_tickers_filter_restricts_universe(tmp_path):
    _write_market(tmp_path, ["AAA", "BBB", "CCC"])

    filtered = run_finalist_trades("dummy", "test_market", "daily", tmp_path, tickers={"AAA"})
    assert set(filtered["ticker"].unique()) <= {"AAA"}

    unfiltered = run_finalist_trades("dummy", "test_market", "daily", tmp_path)
    assert set(unfiltered["ticker"].unique()) == {"AAA", "BBB", "CCC"}


def test_tickers_default_none_matches_prior_behavior(tmp_path):
    _write_market(tmp_path, ["AAA", "BBB"])

    explicit_none = run_finalist_trades("dummy", "test_market", "daily", tmp_path, tickers=None)
    default = run_finalist_trades("dummy", "test_market", "daily", tmp_path)
    pd.testing.assert_frame_equal(explicit_none, default)
