"""Parallel campaign orchestrator.

Task granularity is one ``(market, timeframe, ticker)`` data load -- not one
task per ``(strategy, params)`` combo, which would reload the same ticker
data once per param variant in a strategy's grid. Each load fans out
in-process across every registered strategy for that market/timeframe and
its full param grid before the data is released.

Threading backend: ``backtest_trades_fast`` is numba nopython, which
releases the GIL, so threads genuinely parallelize without multiprocessing's
DataFrame-pickling cost.
"""
from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .provenance import ProvenanceStore
from .registry import StrategyRegistry
from ..backtest.engine import backtest_trades_fast
from ..utils.logging_config import get_logger

_log = get_logger("qresearch.campaign.runner")


def _list_tickers(processed_dir: Path, market: str, timeframe: str) -> list[str]:
    d = processed_dir / market / timeframe
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def _param_combos(param_grid: dict, default_params: dict) -> list[dict]:
    if not param_grid:
        return [default_params]
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*param_grid.values())]


def _run_ticker_task(market: str, timeframe: str, ticker: str, processed_dir: Path,
                      entries: dict, campaign_id: str, store: ProvenanceStore) -> dict:
    path = processed_dir / market / timeframe / f"{ticker}.parquet"
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {"ticker": ticker, "market": market, "timeframe": timeframe, "n_runs": 0, "errors": [str(e)]}

    n_runs, errors = 0, []
    for strategy_id, entry in entries.items():
        for params in _param_combos(entry["param_grid"], entry["default_params"]):
            try:
                sig_df = entry["signal_fn"](df, **params)
                trades = backtest_trades_fast(sig_df)
                store.write(campaign_id, strategy_id, market, timeframe, ticker, params, trades)
                n_runs += 1
            except Exception as e:
                errors.append(f"{strategy_id}/{params}: {e}")
    return {"ticker": ticker, "market": market, "timeframe": timeframe, "n_runs": n_runs, "errors": errors}


def run_campaign(registry: StrategyRegistry, campaign_id: str,
                  processed_dir: str | Path = "data/processed",
                  provenance_dir: str | Path = "data/provenance",
                  markets: list[str] | None = None, n_jobs: int = -1) -> pd.DataFrame:
    """Run every registered strategy x its param grid across every ticker
    found under ``processed_dir`` for each (market, timeframe) the registry
    references. Returns one row per (market, timeframe, ticker) task."""
    processed_dir = Path(processed_dir)
    store = ProvenanceStore(provenance_dir)

    groups: dict[tuple[str, str], dict] = {}
    for sid, entry in registry:
        if markets and entry["market"] not in markets:
            continue
        groups.setdefault((entry["market"], entry["timeframe"]), {})[sid] = entry

    tasks = [
        (market, timeframe, ticker, entries)
        for (market, timeframe), entries in groups.items()
        for ticker in _list_tickers(processed_dir, market, timeframe)
    ]
    if not tasks:
        _log.warning("no (market,timeframe,ticker) tasks matched the registry against %s", processed_dir)
        return pd.DataFrame([])

    t0 = time.perf_counter()
    job = Parallel(n_jobs=n_jobs, backend="threading", return_as="generator_unordered")(
        delayed(_run_ticker_task)(market, timeframe, ticker, processed_dir, entries, campaign_id, store)
        for market, timeframe, ticker, entries in tasks
    )
    results = []
    bar = tqdm(job, total=len(tasks), desc=f"campaign {campaign_id}", unit="ticker")
    for res in bar:
        results.append(res)
        bar.set_postfix(ticker=res["ticker"], runs=res["n_runs"], errs=len(res["errors"]))
    elapsed = time.perf_counter() - t0

    out = pd.DataFrame(results)
    n_runs = int(out["n_runs"].sum()) if "n_runs" in out else 0
    n_errors = sum(len(r["errors"]) for r in results)
    _log.info("campaign %s: %d tasks, %d strategy-runs, %d errors, %.1fs",
               campaign_id, len(tasks), n_runs, n_errors, elapsed)
    return out
