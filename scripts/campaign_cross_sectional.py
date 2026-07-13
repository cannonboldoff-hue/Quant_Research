"""Phase 3 of the alpha-search plan: cross-sectional long/short on the
indian_equities 48-name daily universe.

Every sleeve combined so far (Phase 1/2) is time-series trend-following on
ONE instrument at a time -- structurally similar even across asset classes.
A cross-sectional rank (long the strongest names, short the weakest, within
the SAME universe at the SAME time) is a genuinely different bet, so it
should sit at low correlation to the trend book -- the actual point of
adding it, not just "another sleeve." indian_equities is the only universe
here large enough for this to be meaningful (48 names vs 5-8 elsewhere).

xgboost isn't installed (ml.walk_forward_topn needs it, tests already skip
those paths) -- this uses the plan's literal fallback instead: rank by
trailing return, no ML. See signals/cross_sectional.py.

Pipeline:
1. grid-search (lookback, hold) on the FIT window (same 70/30 split
   convention as Phase 1/2), picking the combo with the best fit-window
   Sharpe; PBO computed across the whole grid (Bailey et al. CSCV) as the
   overfitting check the plan requires alongside DSR.
2. DSR of the chosen combo's EVAL-window (held-out) returns, corrected for
   the number of grid combos tried.
3. correlation of the eval-window cross-sectional returns against the same
   eval-window trend book from Phase 2 (indices regime-filtered, others
   baseline) -- must be low for this to be worth adding.
4. add it as a 6th sleeve to the Phase-1/2 tangency combiner; gate is
   whether the resulting combined eval Sharpe beats Phase 2's 1.780.

Run: python scripts/campaign_cross_sectional.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.optimize.finalist_selection import run_finalist_trades  # noqa: E402
from qresearch.portfolio.combine import (  # noqa: E402
    apply_weights, combine_sleeves, fit_eval_split, portfolio_metrics, trades_to_daily_returns,
)
from qresearch.signals.cross_sectional import cross_sectional_returns  # noqa: E402
from qresearch.stats.risk import deflated_sharpe, pbo  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.campaign_cross_sectional")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"

FIT_FRAC = 0.7
PHASE2_COMBINED_EVAL_SHARPE = 1.780  # from scripts/apply_regime_filter.py's held-out gate

CS_GRID = {"lookback": [10, 20, 40], "hold": [3, 5, 10]}
TOP_FRAC = BOTTOM_FRAC = 0.2  # quintiles: with n=48 names, deciles (0.1) would be ~5 names/leg,
# still workable, but quintiles trade a bit of concentration for stability -- see cross_sectional.py

# The trend book Phase 2 actually kept -- indices regime-filtered, everything
# else baseline (data/campaign_summaries/phase2_regime_filter_decisions.csv).
TREND_SLEEVES = [
    ("commodities_jma_atr_daily", "commodities", "daily", False),
    ("indices_jma_atr_daily", "indices", "daily", True),
    ("crypto_jma_atr_daily", "crypto", "daily", False),
    ("indian_equities_jma_atr_daily", "indian_equities", "daily", False),
    ("forex_jma_atr_daily", "forex", "daily", False),
]


def _load_wide_close(market: str, timeframe: str) -> pd.DataFrame:
    src_dir = PROCESSED / market / timeframe
    data = {}
    for path in sorted(src_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        data[path.stem] = df.set_index("Date")["close"]
    return pd.DataFrame(data).sort_index()


def _combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in product(*grid.values())]


def main() -> None:
    log.info("loading indian_equities daily wide close (48 tickers)")
    wide_close = _load_wide_close("indian_equities", "daily")

    grid = _combos(CS_GRID)
    # string keys throughout (not raw (lookback, hold) tuples) -- a dict with
    # ALL-tuple keys silently becomes a MultiIndex when passed to
    # pd.DataFrame(...), which would corrupt both the PBO matrix columns and
    # any indexing done via those keys.
    key_of = lambda c: f"lb{c['lookback']}_hold{c['hold']}"
    params_by_key = {key_of(c): c for c in grid}
    log.info("computing cross-sectional returns for %d (lookback, hold) combos", len(grid))
    returns_by_combo = {
        key_of(c): cross_sectional_returns(
            wide_close, lookback=c["lookback"], hold=c["hold"],
            top_frac=TOP_FRAC, bottom_frac=BOTTOM_FRAC,
        )
        for c in grid
    }

    fit_by_combo, eval_by_combo, split_date = fit_eval_split(returns_by_combo, FIT_FRAC)

    fit_sharpes = {k: portfolio_metrics(v)["sharpe"] for k, v in fit_by_combo.items()}
    best_key = max(fit_sharpes, key=fit_sharpes.get)
    best_params = params_by_key[best_key]
    print(f"\n=== cross-sectional grid, fit-window (< {split_date.date()}) Sharpe ===")
    for k, v in sorted(fit_sharpes.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v:.3f}")
    print(f"chosen combo: lookback={best_params['lookback']}, hold={best_params['hold']}")

    # PBO across the grid (Bailey et al. CSCV) -- overfitting check on the
    # SELECTION itself, not just the chosen combo's own significance.
    aligned_fit = pd.DataFrame(fit_by_combo).fillna(0.0)
    pbo_score = pbo(aligned_fit.to_numpy().T, n_splits=10)
    print(f"PBO across {len(grid)} grid combos (fit window): {pbo_score:.3f} (0=no overfitting signal, 1=worst)")

    eval_returns_chosen = eval_by_combo[best_key]
    dsr = deflated_sharpe(eval_returns_chosen, n_trials=len(grid))
    eval_metrics = portfolio_metrics(eval_returns_chosen)
    print(f"\nchosen combo eval-window (held-out) Sharpe: {eval_metrics['sharpe']:.3f}, "
          f"CAGR: {eval_metrics['cagr']:.3f}, DSR (n_trials={len(grid)}): {dsr:.3f}")

    # Cheap gate first (DSR/PBO, already computed) -- only pay for rebuilding
    # the Phase-2 trend book (a ~2-3min walk-forward re-run across 5 sleeves)
    # if the cross-sectional sleeve is even a candidate worth correlating.
    cheap_gate = dsr > 0.5 and pbo_score < 0.5
    print(f"\ncheap gate (DSR>0.5 AND PBO<0.5), before correlation check: {'PASS' if cheap_gate else 'FAIL'}")

    if not cheap_gate:
        print("\ncross-sectional sleeve fails the standalone DSR/PBO gate on its own -- "
              "not worth the trend-book rebuild for a correlation check. Not added to the portfolio.")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "chosen_lookback": best_params["lookback"], "chosen_hold": best_params["hold"],
            "eval_sharpe": eval_metrics["sharpe"], "eval_dsr": dsr, "pbo": pbo_score,
            "corr_vs_trend_book": None, "standalone_gate": False,
            "combined_with_cs_eval_sharpe": None, "phase3_gate": False,
        }]).to_csv(OUT / "phase3_cross_sectional_decision.csv", index=False)
        return

    # correlation vs the Phase-2 trend book, same eval window
    log.info("rebuilding Phase-2 trend book for correlation comparison")
    trend_daily = {}
    for strategy_id, market, timeframe, regime in TREND_SLEEVES:
        trades = run_finalist_trades(strategy_id, market, timeframe, PROCESSED, regime_filter=regime)
        trend_daily[strategy_id] = trades_to_daily_returns(trades)
    trend_fit, trend_eval, _ = fit_eval_split(trend_daily, FIT_FRAC)
    trend_weights = combine_sleeves(trend_fit, mode="tangency")["weights"]
    trend_eval_aligned = pd.DataFrame(trend_eval).sort_index().fillna(0.0)
    trend_eval_combined = apply_weights(trend_eval_aligned, trend_weights)

    common = eval_returns_chosen.index.intersection(trend_eval_combined.index)
    corr_vs_trend_book = float(eval_returns_chosen.loc[common].corr(trend_eval_combined.loc[common]))
    print(f"correlation vs Phase-2 combined trend book (eval window, n={len(common)} overlapping days): "
          f"{corr_vs_trend_book:.3f}")

    standalone_gate = cheap_gate and abs(corr_vs_trend_book) < 0.3
    print(f"\nstandalone gate (DSR>0.5 AND PBO<0.5 AND |corr|<0.3): {'PASS' if standalone_gate else 'FAIL'}")

    if not standalone_gate:
        print("\ncross-sectional sleeve does not clear the standalone gate -- not added to the portfolio.")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "chosen_lookback": best_params["lookback"], "chosen_hold": best_params["hold"],
            "eval_sharpe": eval_metrics["sharpe"], "eval_dsr": dsr, "pbo": pbo_score,
            "corr_vs_trend_book": corr_vs_trend_book, "standalone_gate": standalone_gate,
            "combined_with_cs_eval_sharpe": None, "phase3_gate": False,
        }]).to_csv(OUT / "phase3_cross_sectional_decision.csv", index=False)
        return

    # add as a 6th sleeve and rebuild the combined portfolio
    all_daily = dict(trend_daily)
    all_daily["cross_sectional_indian_equities"] = returns_by_combo[best_key]
    fit_all, eval_all, _ = fit_eval_split(all_daily, FIT_FRAC)
    combo = combine_sleeves(fit_all, mode="tangency")
    eval_aligned_all = pd.DataFrame(eval_all).sort_index().fillna(0.0)
    combined_eval = apply_weights(eval_aligned_all, combo["weights"])
    combined_metrics = portfolio_metrics(combined_eval)

    phase3_gate = combined_metrics["sharpe"] > PHASE2_COMBINED_EVAL_SHARPE
    print(f"\n6-sleeve tangency weights (fit): {combo['weights']}")
    print(f"6-sleeve combined eval Sharpe: {combined_metrics['sharpe']:.3f} "
          f"(Phase 2 baseline was {PHASE2_COMBINED_EVAL_SHARPE:.3f})")
    print(f"\nPhase 3 gate (6-sleeve combined eval Sharpe beats Phase 2 baseline): "
          f"{'PASS' if phase3_gate else 'FAIL'}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "chosen_lookback": best_params["lookback"], "chosen_hold": best_params["hold"],
        "eval_sharpe": eval_metrics["sharpe"], "eval_dsr": dsr, "pbo": pbo_score,
        "corr_vs_trend_book": corr_vs_trend_book, "standalone_gate": standalone_gate,
        "combined_with_cs_eval_sharpe": combined_metrics["sharpe"], "phase3_gate": phase3_gate,
    }]).to_csv(OUT / "phase3_cross_sectional_decision.csv", index=False)
    log.info("wrote %s", OUT / "phase3_cross_sectional_decision.csv")


if __name__ == "__main__":
    main()
