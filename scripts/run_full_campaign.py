"""Run the Unified Campaign: every registered strategy x its param grid
across every ticker in data/processed, then summarize per-strategy DSR/PBO
over the provenance store. Scopeable to a subset of markets so a single
invocation stays short.

Run: python scripts/run_full_campaign.py [market1,market2,...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from qresearch.campaign import ProvenanceStore, StrategyRegistry, run_campaign
from qresearch.stats import deflated_sharpe, pbo
from qresearch.utils.logging_config import get_logger

log = get_logger("qresearch.full_campaign")
CAMPAIGN_ID = "campaign_20260713_v1"


def main() -> None:
    markets = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    registry = StrategyRegistry()
    log.info("registry loaded: %d strategies, markets=%s", len(registry), markets or "ALL")
    sys.stdout.flush()

    t0 = time.perf_counter()
    # n_jobs=1: several signal generators (jma, lhp_dsl) are plain-Python
    # loops that hold the GIL -- threading across them causes contention,
    # not parallelism (measured: one ticker went from 4.6s standalone to
    # 315s under 4 threads). Sequential matches the standalone benchmarks.
    # Upgrade path: numba-jit those inner loops (same pattern as
    # engine_numba.py) before raising n_jobs back up.
    result = run_campaign(registry, campaign_id=CAMPAIGN_ID, markets=markets, n_jobs=1)
    elapsed = time.perf_counter() - t0
    log.info("campaign run finished in %.1fs (%.1f min)", elapsed, elapsed / 60)
    sys.stdout.flush()

    if result.empty:
        log.error("no tasks ran -- check data/processed and configs/strategies.yaml")
        sys.exit(1)

    total_errors = sum(len(e) for e in result["errors"])
    if total_errors:
        log.warning("%d task-level errors -- sample:", total_errors)
        for e in result["errors"]:
            for msg in e[:3]:
                log.warning("  %s", msg)

    store = ProvenanceStore()
    trades = store.read(campaign_id=CAMPAIGN_ID)
    log.info("provenance store: %d trades across %d strategies, %d tickers",
              len(trades), trades["strategy_id"].nunique(), trades["ticker"].nunique())

    summary_rows = []
    for strategy_id, g in trades.groupby("strategy_id"):
        r = g["ret"].to_numpy(dtype=np.float64)
        n_trials = g.groupby("params").ngroups
        dsr = deflated_sharpe(r, n_trials=n_trials)
        summary_rows.append({
            "strategy_id": strategy_id, "n_trades": len(g), "n_tickers": g["ticker"].nunique(),
            "n_param_combos": n_trials, "win_rate": float((r > 0).mean()),
            "mean_ret": float(r.mean()), "deflated_sharpe": dsr,
        })
    summary = pd.DataFrame(summary_rows).sort_values("deflated_sharpe", ascending=False)

    out_path = ROOT / "data" / "campaign_summaries" / f"{CAMPAIGN_ID}_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    log.info("per-strategy summary -> %s", out_path)
    print(summary.to_string(index=False))

    # PBO across strategies with >=2 param combos each and a shared best-params return series
    pivot_candidates = [sid for sid in trades["strategy_id"].unique()
                         if trades[trades["strategy_id"] == sid]["params"].nunique() >= 2]
    if len(pivot_candidates) >= 2:
        mat_rows = []
        for sid in pivot_candidates:
            g = trades[trades["strategy_id"] == sid].sort_values("entry_time")
            mat_rows.append(g["ret"].to_numpy(dtype=np.float64))
        min_len = min(len(r) for r in mat_rows)
        if min_len >= 20:
            mat = np.vstack([r[:min_len] for r in mat_rows])
            p = pbo(mat, n_splits=min(10, min_len // 2 * 2))
            log.info("PBO across %d strategies (CSCV, %d obs each): %.3f", len(pivot_candidates), min_len, p)


if __name__ == "__main__":
    main()
