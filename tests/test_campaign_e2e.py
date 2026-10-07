"""End-to-end: registry -> parallel runner -> provenance store -> stats
module. Synthetic test always runs; the real-slice test only runs once
data/processed exists (Phase 1's ETL output -- gitignored, so absent on a
fresh checkout/CI)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qresearch.campaign import ProvenanceStore, run_campaign
from qresearch.campaign.runner import STOP_GRID, _param_combos
from qresearch.signals import jma_signals
from qresearch.stats import deflated_sharpe

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PROCESSED = REPO_ROOT / "data" / "processed"


def _synth(n=300, seed=1):
    rng = pd.date_range("2024-01-01", periods=n, freq="1h")
    np.random.seed(seed)
    p = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({"Date": rng, "Ticker": "TEST", "open": p, "high": p + 1,
                         "low": p - 1, "close": p, "volume": 1000})


def test_campaign_e2e_synthetic(tmp_path):
    processed_dir = tmp_path / "processed"
    out_dir = processed_dir / "test_market" / "1h"
    out_dir.mkdir(parents=True)
    for i, ticker in enumerate(["A", "B"]):
        _synth(300, seed=i).to_parquet(out_dir / f"{ticker}.parquet", index=False)

    registry = [
        ("jma_test", {"signal_fn": jma_signals, "market": "test_market", "timeframe": "1h",
                       "default_params": {"fast": 7, "slow": 21}, "param_grid": {}}),
    ]

    result = run_campaign(registry, campaign_id="e2e_synth", processed_dir=processed_dir,
                           provenance_dir=tmp_path / "provenance", n_jobs=1)
    assert len(result) == 2  # one task per ticker
    assert (result["errors"].apply(len) == 0).all()
    # one run per (ticker x signal-param combo x STOP_GRID risk combo)
    n_risk = len(_param_combos(STOP_GRID, {}))
    assert result["n_runs"].sum() == 2 * n_risk

    store = ProvenanceStore(tmp_path / "provenance")
    trades = store.read(campaign_id="e2e_synth")
    assert len(trades) > 0
    assert {"strategy_id", "market", "ticker", "ret"}.issubset(trades.columns)

    dsr = deflated_sharpe(trades["ret"], n_trials=1)
    assert 0.0 <= dsr <= 1.0


@pytest.mark.skipif(not any((REAL_PROCESSED / "indices").glob("*/*.parquet")), reason="data/processed not populated (run scripts/process_raw_data.py)")
def test_campaign_e2e_real_slice(tmp_path):
    """Same pipeline against a small real data slice from Phase 1's ETL."""
    from qresearch.campaign import StrategyRegistry

    registry = StrategyRegistry()
    result = run_campaign(registry, campaign_id="e2e_real_smoke", processed_dir=REAL_PROCESSED,
                           provenance_dir=tmp_path / "provenance", markets=["indices"], n_jobs=2)
    assert len(result) > 0
    assert (result["errors"].apply(len) == 0).all()

    store = ProvenanceStore(tmp_path / "provenance")
    trades = store.read(campaign_id="e2e_real_smoke")
    assert len(trades) > 0
