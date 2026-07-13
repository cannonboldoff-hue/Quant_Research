"""Walk-forward gate for the v2 alpha-search finalists (Lever 3).

v2 campaign showed 4 (market, timeframe) JMA configs with a positive net
mean_ret for their BEST individual (fast, slow, sl_mult, tp_mult) combo:
commodities/indices at daily and 4h. That's still in-sample -- this script
re-optimizes per chronological fold and evaluates on the next (anchored
walk-forward, reusing optimize.walk_forward.split_data_by_periods), so a
"lead" is only real if it survives out-of-sample.

walk_forward_optimize() isn't reused directly: it only sweeps the signal's
param_grid and hardcodes sl/tp at 1.5/3.0, so it can't validate the sl/tp
grid finding from Lever 2. This sweeps both, same grids as the campaign
runner (STOP_GRID) and registry (fast/slow), via the fast numba engine.

Run: python scripts/walk_forward_finalists.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.backtest.engine import backtest_trades_fast  # noqa: E402
from qresearch.campaign.runner import STOP_GRID  # noqa: E402
from qresearch.optimize.walk_forward import split_data_by_periods  # noqa: E402
from qresearch.signals.generators import jma_signals  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.walk_forward_finalists")
PROCESSED = ROOT / "data" / "processed"

SIGNAL_GRID = {"fast": [5, 7, 9], "slow": [21, 28]}
N_SPLITS = 5
MIN_TRADES = 10  # in-sample combos with fewer trades than this are too noisy to trust as "best"

FINALISTS = [
    ("commodities_jma_atr_daily", "commodities", "daily"),
    ("indices_jma_atr_daily", "indices", "daily"),
    ("commodities_jma_atr_4h", "commodities", "4h"),
    ("indices_jma_atr_4h", "indices", "4h"),
    # v2.1 (2026-07-13): crypto_jma_atr_daily's blended mean_ret (+156bps) is
    # the strongest of any strategy/timeframe so far; crypto_jma_atr_4h's best
    # combo (+61bps, n=1653) also clears cost with room. Both added here.
    ("crypto_jma_atr_daily", "crypto", "daily"),
    ("crypto_jma_atr_4h", "crypto", "4h"),
]


def _combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in product(*grid.values())]


def _best_insample(df: pd.DataFrame) -> tuple[dict, dict] | None:
    """Grid-search signal x risk params on one fold; return (signal_params, risk_params) of the
    best-by-mean-net-return combo with at least MIN_TRADES trades, or None if nothing qualifies."""
    best, best_ret = None, -np.inf
    for sparams in _combos(SIGNAL_GRID):
        sig = jma_signals(df, **sparams)
        for rparams in _combos(STOP_GRID):
            trades = backtest_trades_fast(sig, **rparams)
            if len(trades) < MIN_TRADES:
                continue
            ret = trades["ret"].mean()
            if ret > best_ret:
                best_ret, best = ret, (sparams, rparams)
    return best


def _oos_eval(df: pd.DataFrame, sparams: dict, rparams: dict) -> dict | None:
    sig = jma_signals(df, **sparams)
    trades = backtest_trades_fast(sig, **rparams)
    if trades.empty:
        return None
    r = trades["ret"].to_numpy(dtype=np.float64)
    return {"n_trades": len(r), "mean_ret": float(r.mean()), "win_rate": float((r > 0).mean())}


def run_finalist(strategy_id: str, market: str, timeframe: str) -> pd.DataFrame:
    src_dir = PROCESSED / market / timeframe
    rows = []
    for path in sorted(src_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        splits = split_data_by_periods(df, N_SPLITS)
        for t in range(len(splits) - 1):
            best = _best_insample(splits[t])
            if best is None:
                continue
            sparams, rparams = best
            oos = _oos_eval(splits[t + 1], sparams, rparams)
            if oos is None:
                continue
            rows.append({"ticker": path.stem, "fold": t, **sparams, **rparams, **oos})
    return pd.DataFrame(rows)


def main() -> None:
    summary_rows = []
    for strategy_id, market, timeframe in FINALISTS:
        log.info("walk-forward: %s (%s/%s)", strategy_id, market, timeframe)
        res = run_finalist(strategy_id, market, timeframe)
        if res.empty:
            log.warning("%s: no fold produced a qualifying OOS result", strategy_id)
            continue
        pos_frac = (res["mean_ret"] > 0).mean()
        summary_rows.append({
            "strategy_id": strategy_id, "n_ticker_folds": len(res),
            "oos_positive_frac": pos_frac, "oos_mean_ret": res["mean_ret"].mean(),
            "oos_mean_win_rate": res["win_rate"].mean(),
        })
        print(f"\n--- {strategy_id}: per ticker-fold OOS ---")
        print(res.to_string(index=False))

    summary = pd.DataFrame(summary_rows).sort_values("oos_positive_frac", ascending=False)
    print("\n=== walk-forward gate summary ===")
    print(summary.to_string(index=False))
    out_path = ROOT / "data" / "campaign_summaries" / "walk_forward_finalists_summary.csv"
    summary.to_csv(out_path, index=False)
    log.info("summary -> %s", out_path)


if __name__ == "__main__":
    main()
