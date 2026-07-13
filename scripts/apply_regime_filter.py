"""Phase 2 of the alpha-search plan: does an ADX regime filter improve the
JMA+ATR daily sleeves?

The daily sleeves' walk-forward win rates are modest (33-45%) -- a lot of
crossover entries fire in chop. This re-runs the SAME walk-forward selection
Phase 1 used (qresearch.optimize.finalist_selection.run_finalist_trades) once
per sleeve WITHOUT and WITH an ADX>=25 entry filter (signals/regime.py), then:

1. keeps the regime-filtered version only for sleeves where it improves
   OOS mean-ret (never on 4h/other -- these are the same 5 daily sleeves
   Phase 1 combined), discarding it otherwise
2. rebuilds the Phase-1 combined portfolio from whichever version (baseline
   or regime-filtered) won per sleeve, using the identical fit/eval split
   and tangency combiner Phase 1 already validated
3. reports whether the resulting combined Sharpe beats Phase 1's baseline
   combined eval Sharpe (1.436, held-out) -- the actual Phase 2 gate.

Run: python scripts/apply_regime_filter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.optimize.finalist_selection import run_finalist_trades  # noqa: E402
from qresearch.portfolio.combine import (  # noqa: E402
    apply_weights, combine_sleeves, fit_eval_split, portfolio_metrics, trades_to_daily_returns,
)
from qresearch.stats.risk import paired_bootstrap_ci  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.apply_regime_filter")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"

FIT_FRAC = 0.7
PHASE1_COMBINED_EVAL_SHARPE = 1.436243  # from scripts/build_portfolio.py's held-out gate

# Same 5 sleeves Phase 1 combined -- see build_portfolio.py for why these
# and not the 4h variants.
SLEEVES = [
    ("commodities_jma_atr_daily", "commodities", "daily"),
    ("indices_jma_atr_daily", "indices", "daily"),
    ("crypto_jma_atr_daily", "crypto", "daily"),
    ("indian_equities_jma_atr_daily", "indian_equities", "daily"),
    ("forex_jma_atr_daily", "forex", "daily"),
]


def main() -> None:
    chosen_trades: dict[str, pd.DataFrame] = {}
    decisions = []
    for strategy_id, market, timeframe in SLEEVES:
        log.info("baseline OOS trades: %s", strategy_id)
        baseline = run_finalist_trades(strategy_id, market, timeframe, PROCESSED)
        log.info("regime-filtered OOS trades: %s", strategy_id)
        filtered = run_finalist_trades(strategy_id, market, timeframe, PROCESSED, regime_filter=True)

        base_mean = float(baseline["ret"].mean()) if not baseline.empty else float("-inf")
        filt_mean = float(filtered["ret"].mean()) if not filtered.empty else float("-inf")
        keep_filtered = filt_mean > base_mean and len(filtered) >= 10

        decisions.append({
            "strategy_id": strategy_id,
            "baseline_n_trades": len(baseline), "baseline_mean_ret": base_mean,
            "filtered_n_trades": len(filtered), "filtered_mean_ret": filt_mean,
            "kept": "regime_filtered" if keep_filtered else "baseline",
        })
        chosen_trades[strategy_id] = filtered if keep_filtered else baseline
        log.info("%s: baseline mean_ret=%.5f (n=%d) vs regime mean_ret=%.5f (n=%d) -> keep %s",
                  strategy_id, base_mean, len(baseline), filt_mean, len(filtered),
                  "regime_filtered" if keep_filtered else "baseline")

    decisions_df = pd.DataFrame(decisions).set_index("strategy_id")
    print("\n=== per-sleeve: baseline vs regime-filtered OOS mean_ret ===")
    print(decisions_df.to_string())

    daily_by_sleeve = {sid: trades_to_daily_returns(t) for sid, t in chosen_trades.items()}
    fit_returns, eval_returns, split_date = fit_eval_split(daily_by_sleeve, FIT_FRAC)
    fit_combo = combine_sleeves(fit_returns, mode="tangency")
    weights = fit_combo["weights"]

    eval_aligned = pd.DataFrame(eval_returns).sort_index().fillna(0.0)
    eval_combined = apply_weights(eval_aligned, weights)
    combined_metrics = portfolio_metrics(eval_combined)

    best_single = max(fit_returns, key=lambda sid: portfolio_metrics(fit_returns[sid])["sharpe"])
    ci_lo, ci_hi = paired_bootstrap_ci(eval_combined, eval_aligned[best_single], statistic="sharpe")
    beats_phase1 = combined_metrics["sharpe"] > PHASE1_COMBINED_EVAL_SHARPE

    print(f"\nPhase 2 tangency weights (fit < {split_date.date()}): {weights}")
    print(f"Phase 2 combined eval Sharpe: {combined_metrics['sharpe']:.3f} "
          f"(Phase 1 baseline was {PHASE1_COMBINED_EVAL_SHARPE:.3f})")
    print(f"bootstrap 95% CI vs best single sleeve (eval): [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"\nPhase 2 gate (combined eval Sharpe beats Phase 1 baseline): "
          f"{'PASS' if beats_phase1 else 'FAIL'}")

    OUT.mkdir(parents=True, exist_ok=True)
    decisions_df.to_csv(OUT / "phase2_regime_filter_decisions.csv")
    log.info("wrote %s", OUT / "phase2_regime_filter_decisions.csv")


if __name__ == "__main__":
    main()
