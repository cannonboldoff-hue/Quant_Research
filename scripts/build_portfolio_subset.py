"""Portfolio combination for a narrower mix: commodities + indices + forex
only (drops crypto and indian_equities from the full 5-sleeve book in
scripts/build_portfolio.py). Compares three weighting methods -- tangency,
equal-weight, inverse-vol -- across three fit/eval split points (0.6/0.7/0.8)
so it's clear how much the optimizer adds over a naive split on this
3-market book, and whether the winning mode's edge (and its weights) holds
up or is an artifact of one split point. Each mode gets its own
held-out-CI gate check (bootstrap CI on Sharpe vs. best single sleeve),
not just the mode used in the full 5-sleeve build.

Reuses the exact combine API build_portfolio.py uses; reads the per-sleeve
daily returns it already persisted (portfolio_daily_combined.csv) instead of
re-running the slow walk-forward trade extraction.

Run: python scripts/build_portfolio_subset.py
       (default: uses the persisted short-history CSV)
     python scripts/build_portfolio_subset.py --recompute
       (recomputes from data/processed_yf -- the longer Yahoo-sourced history
       fetched by scripts/fetch_yahoo_daily.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.portfolio.combine import (  # noqa: E402
    apply_weights, combine_sleeves, fit_eval_split, portfolio_metrics,
)
from qresearch.stats.risk import paired_bootstrap_ci  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.build_portfolio_subset")
OUT = ROOT / "data" / "campaign_summaries"
DAILY_CSV = OUT / "portfolio_daily_combined.csv"

FIT_FRACS = [0.6, 0.7, 0.8]  # robustness check: does the split point change the picture?
SLEEVES = ["commodities_jma_atr_daily", "indices_jma_atr_daily", "forex_jma_atr_daily"]
MARKETS = {"commodities_jma_atr_daily": "commodities",
           "indices_jma_atr_daily": "indices",
           "forex_jma_atr_daily": "forex"}
MODES = ["tangency", "equal_weight", "inverse_vol"]


def _load_daily_by_sleeve(processed_dir: Path | None = None) -> dict[str, pd.Series]:
    if processed_dir is not None:
        # Recompute from raw OOS trades on the given processed_dir (e.g. the
        # longer-history data/processed_yf tree), same as
        # build_portfolio.py's loop but for just these 3 sleeves.
        from qresearch.optimize.finalist_selection import run_finalist_trades
        from qresearch.portfolio.combine import trades_to_daily_returns
        out = {}
        for sid in SLEEVES:
            trades = run_finalist_trades(sid, MARKETS[sid], "daily", processed_dir)
            out[sid] = trades_to_daily_returns(trades)
        return out

    df = pd.read_csv(DAILY_CSV, index_col="exit_time", parse_dates=True)
    return {sid: df[sid].dropna() for sid in SLEEVES}


def _run_split(daily_by_sleeve: dict[str, pd.Series], fit_frac: float) -> dict:
    """Fit weights on `fit_frac` of dates, eval on the rest. Returns per-mode
    metrics/weights plus a gate check (CI excludes 0 AND beats best single)
    for EVERY mode, not just tangency -- a mode that only "looks" better on
    point-estimate Sharpe may not clear significance on 564 eval days."""
    fit_returns, eval_returns, split_date = fit_eval_split(daily_by_sleeve, fit_frac)
    eval_aligned = pd.DataFrame(eval_returns).sort_index().fillna(0.0)
    eval_metrics = {sid: portfolio_metrics(eval_aligned[sid]) for sid in eval_aligned.columns}

    best_single = max(fit_returns, key=lambda sid: portfolio_metrics(fit_returns[sid])["sharpe"])
    best_single_eval_sharpe = eval_metrics[best_single]["sharpe"]

    mode_rows = {}
    for mode in MODES:
        combo = combine_sleeves(fit_returns, mode=mode)
        weights = combo["weights"]
        combined = apply_weights(eval_aligned, weights)
        row = dict(portfolio_metrics(combined))
        for sid in SLEEVES:
            row[f"weight_{sid}"] = weights.get(sid, 0.0)
        ci_lo, ci_hi = paired_bootstrap_ci(combined, eval_aligned[best_single], statistic="sharpe")
        row["ci_lo"], row["ci_hi"] = ci_lo, ci_hi
        row["gate_pass"] = ci_lo > 0 and row["sharpe"] > best_single_eval_sharpe
        mode_rows[mode] = row

    return {"split_date": split_date, "eval_metrics": eval_metrics,
            "best_single": best_single, "best_single_eval_sharpe": best_single_eval_sharpe,
            "mode_rows": mode_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default=None,
                         help="OHLC parquet tree to recompute sleeves from "
                              "(e.g. data/processed_yf). Implies --recompute.")
    parser.add_argument("--recompute", action="store_true",
                         help="Recompute daily returns from raw trades instead of "
                              "the persisted portfolio_daily_combined.csv shortcut. "
                              "Uses --processed-dir (default data/processed_yf).")
    parser.add_argument("--out", default=None, help="Output CSV filename under "
                         "data/campaign_summaries (default derived from the source).")
    args = parser.parse_args()

    recompute = args.recompute or args.processed_dir is not None
    processed_dir = Path(args.processed_dir) if args.processed_dir else (ROOT / "data" / "processed_yf")
    out_name = args.out or ("portfolio_subset_cif_yf_metrics.csv" if recompute
                             else "portfolio_subset_cif_metrics.csv")

    daily_by_sleeve = _load_daily_by_sleeve(processed_dir if recompute else None)
    for sid, d in daily_by_sleeve.items():
        log.info("%s: %d trading days", sid, len(d))

    full_rows = [dict(portfolio_metrics(d), strategy_id=sid) for sid, d in daily_by_sleeve.items()]
    full_df = pd.DataFrame(full_rows).set_index("strategy_id")
    full_corr = pd.DataFrame(daily_by_sleeve).fillna(0.0).corr()

    print("\n=== full-period per-sleeve metrics (commodities+indices+forex, informational) ===")
    print(full_df.to_string())
    print(f"\nfull-period correlation matrix:\n{full_corr.to_string()}")

    results = {}
    for fit_frac in FIT_FRACS:
        r = _run_split(daily_by_sleeve, fit_frac)
        results[fit_frac] = r

        print(f"\n=== fit_frac={fit_frac} | eval window >= {r['split_date'].date()} "
              f"({len(r['eval_metrics'])} sleeves, best single on fit window: {r['best_single']} "
              f"eval Sharpe {r['best_single_eval_sharpe']:.3f}) ===")
        print("solo sleeves (no combination -- 'focus on just this market'):")
        print(pd.DataFrame(r["eval_metrics"]).T.to_string())
        print("combined by weighting method:")
        mode_df = pd.DataFrame(r["mode_rows"]).T
        print(mode_df.to_string())

    print("\n=== robustness: inverse_vol across split points ===")
    robustness = pd.DataFrame({
        fit_frac: {
            "sharpe": r["mode_rows"]["inverse_vol"]["sharpe"],
            "max_drawdown": r["mode_rows"]["inverse_vol"]["max_drawdown"],
            "weight_forex": r["mode_rows"]["inverse_vol"]["weight_forex_jma_atr_daily"],
            "gate_pass": r["mode_rows"]["inverse_vol"]["gate_pass"],
        }
        for fit_frac, r in results.items()
    }).T
    print(robustness.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    out_df = pd.concat(
        {fit_frac: pd.concat([pd.DataFrame(r["eval_metrics"]).T, pd.DataFrame(r["mode_rows"]).T])
         for fit_frac, r in results.items()},
        names=["fit_frac", "mode"],
    )
    out_df.to_csv(OUT / out_name)
    log.info("wrote %s", OUT / out_name)


if __name__ == "__main__":
    main()
