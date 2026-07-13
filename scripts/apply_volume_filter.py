"""Phase 4 of the alpha-search plan: one focused new-signal spike.

No real alt-data (funding rates, options IV) exists locally
(data/external/ is empty) and this repo has no reliable way to pull live
external data into a reproducible backtest. Substituting the plan's literal
example for the nearest zero-new-data equivalent: a Money Flow Index
volume/money-flow confirmation filter (signals/volume_filter.py) as a filter
on the crypto trend sleeve, same spirit as Phase 2's ADX filter but gating
on volume-confirmed direction instead of trend strength.

Same structure as apply_regime_filter.py: compare baseline vs filtered OOS
mean_ret for crypto_jma_atr_daily; only swap it into the Phase-2 5-sleeve
book if it wins, then check whether the resulting combined eval Sharpe beats
Phase 2's 1.780 (same bar every sleeve addition has been held to).

Run: python scripts/apply_volume_filter.py
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
from qresearch.stats.risk import deflated_sharpe  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.apply_volume_filter")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"

FIT_FRAC = 0.7
PHASE2_COMBINED_EVAL_SHARPE = 1.780

TARGET_SLEEVE = ("crypto_jma_atr_daily", "crypto", "daily")

# The Phase-2 trend book (indices regime-filtered, everything else baseline)
# minus crypto, which this script re-decides.
OTHER_SLEEVES = [
    ("commodities_jma_atr_daily", "commodities", "daily", False),
    ("indices_jma_atr_daily", "indices", "daily", True),
    ("indian_equities_jma_atr_daily", "indian_equities", "daily", False),
    ("forex_jma_atr_daily", "forex", "daily", False),
]


def main() -> None:
    strategy_id, market, timeframe = TARGET_SLEEVE
    log.info("baseline OOS trades: %s", strategy_id)
    baseline = run_finalist_trades(strategy_id, market, timeframe, PROCESSED)
    log.info("volume-filtered OOS trades: %s", strategy_id)
    filtered = run_finalist_trades(strategy_id, market, timeframe, PROCESSED, volume_filter=True)

    base_mean = float(baseline["ret"].mean()) if not baseline.empty else float("-inf")
    filt_mean = float(filtered["ret"].mean()) if not filtered.empty else float("-inf")
    base_dsr = deflated_sharpe(baseline["ret"], n_trials=1) if not baseline.empty else 0.0
    filt_dsr = deflated_sharpe(filtered["ret"], n_trials=1) if not filtered.empty else 0.0

    print(f"\n=== {strategy_id}: baseline vs volume-filtered OOS ===")
    print(f"baseline:         n_trades={len(baseline)}, mean_ret={base_mean:.5f}, dsr={base_dsr:.3f}")
    print(f"volume_filtered:  n_trades={len(filtered)}, mean_ret={filt_mean:.5f}, dsr={filt_dsr:.3f}")

    keep_filtered = filt_mean > base_mean and len(filtered) >= 10
    print(f"\nkeep: {'volume_filtered' if keep_filtered else 'baseline'}")

    if not keep_filtered:
        print("\nvolume filter does not improve crypto's OOS mean_ret -- not swapped into the portfolio. "
              "Phase 4 spike fails (consistent with every prior non-JMA/non-ADX signal tried in this repo).")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "target_sleeve": strategy_id, "baseline_n_trades": len(baseline), "baseline_mean_ret": base_mean,
            "filtered_n_trades": len(filtered), "filtered_mean_ret": filt_mean, "kept": "baseline",
            "combined_eval_sharpe": None, "phase4_gate": False,
        }]).to_csv(OUT / "phase4_volume_filter_decision.csv", index=False)
        return

    log.info("volume filter improved crypto -- rebuilding combined portfolio")
    daily_by_sleeve = {strategy_id: trades_to_daily_returns(filtered)}
    for sid, mkt, tf, regime in OTHER_SLEEVES:
        trades = run_finalist_trades(sid, mkt, tf, PROCESSED, regime_filter=regime)
        daily_by_sleeve[sid] = trades_to_daily_returns(trades)

    fit_returns, eval_returns, split_date = fit_eval_split(daily_by_sleeve, FIT_FRAC)
    weights = combine_sleeves(fit_returns, mode="tangency")["weights"]
    eval_aligned = pd.DataFrame(eval_returns).sort_index().fillna(0.0)
    combined_metrics = portfolio_metrics(apply_weights(eval_aligned, weights))

    phase4_gate = combined_metrics["sharpe"] > PHASE2_COMBINED_EVAL_SHARPE
    print(f"\ntangency weights (fit < {split_date.date()}): {weights}")
    print(f"combined eval Sharpe: {combined_metrics['sharpe']:.3f} "
          f"(Phase 2 baseline was {PHASE2_COMBINED_EVAL_SHARPE:.3f})")
    print(f"\nPhase 4 gate (combined eval Sharpe beats Phase 2 baseline): {'PASS' if phase4_gate else 'FAIL'}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "target_sleeve": strategy_id, "baseline_n_trades": len(baseline), "baseline_mean_ret": base_mean,
        "filtered_n_trades": len(filtered), "filtered_mean_ret": filt_mean, "kept": "volume_filtered",
        "combined_eval_sharpe": combined_metrics["sharpe"], "phase4_gate": phase4_gate,
    }]).to_csv(OUT / "phase4_volume_filter_decision.csv", index=False)
    log.info("wrote %s", OUT / "phase4_volume_filter_decision.csv")


if __name__ == "__main__":
    main()
