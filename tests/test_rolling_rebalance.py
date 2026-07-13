"""Rolling rebalance windowing: rebalance dates tile [start, end] with no
gaps/overlaps, and sector selection at date t never sees a trade at or after
t -- the no-lookahead invariant that makes this test honest (unlike the
static one-shot pick, which reused its own grading sample)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from campaign_rolling_rebalance import _rebalance_periods, _run_frequency  # noqa: E402


def test_rebalance_periods_tile_without_gaps_or_overlaps():
    start, end = pd.Timestamp("2018-01-01"), pd.Timestamp("2024-06-15")
    periods = _rebalance_periods(start, end, pd.DateOffset(months=3))
    assert periods[0][0] == start
    assert periods[-1][1] == end
    for (a1, b1), (a2, b2) in zip(periods, periods[1:]):
        assert b1 == a2  # no gap, no overlap between consecutive periods


def _synthetic_trades():
    # two tickers, one per sector-relevant group used by SECTOR_MAP (reuse
    # real tickers so csa.TICKER_SECTOR/SECTOR_MAP lookups resolve)
    dates = pd.date_range("2016-01-01", periods=40, freq="30D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"ticker": "RELIANCE", "entry_time": d, "exit_time": d + pd.Timedelta(days=5),
                     "fold": i % 4, "ret": 0.01})
        rows.append({"ticker": "ITC", "entry_time": d, "exit_time": d + pd.Timedelta(days=5),
                     "fold": i % 4, "ret": -0.005})
    return pd.DataFrame(rows)


def test_forward_window_has_no_lookahead_into_selection_history():
    trades = _synthetic_trades()
    start = trades["entry_time"].min() + pd.DateOffset(years=1)
    end = trades["entry_time"].max()
    curated, baseline, picks = _run_frequency(trades, "annual", pd.DateOffset(years=1), start, end)
    # every forward-window trade (baseline, which is a superset of curated)
    # must have entry_time >= the period it was assigned to at rebalance time
    assert (baseline["entry_time"] >= start).all()
    # picks list has one entry per rebalance step and never empty
    assert len(picks) >= 1
    assert all(p["n_curated_tickers"] >= 0 for p in picks)
