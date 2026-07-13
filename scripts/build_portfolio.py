"""Portfolio-level combination of the five walk-forward-validated daily
JMA+ATR sleeves (Phase 1 of the alpha-search plan).

Every prior campaign artifact (campaign_20260713_v2_summary.csv,
walk_forward_finalists_summary.csv) reports *per-trade mean return*; none
report an annualized, portfolio-level Sharpe/CAGR/max-DD, and the five
DSR-clean sleeves have never been combined -- so the free diversification
across asset classes has never been harvested and the strategy's true
risk-adjusted performance is unknown. This script fills both gaps.

Reuses the exact walk-forward-selected trades scripts/walk_forward_finalists.py
already validates (qresearch.optimize.finalist_selection.run_finalist_trades),
so "the portfolio" is built from the same OOS trades the gate approved --
not a second, possibly-diverging selection.

Combination weights: naive inverse-vol weighting equalizes each sleeve's risk
*contribution* but ignores return quality -- with these sleeves' Sharpes
ranging 1.06-2.31, that overweights low-vol/mediocre-Sharpe sleeves (it
handed forex 58% of the book) and produced a combined Sharpe BELOW the best
single sleeve, defeating the point. Tangency (mean-variance-optimal,
w ~ Sigma^-1 mu) weighting is what actually harvests near-zero correlation
into a higher combined Sharpe. But fitting those weights and grading the
combined Sharpe on the same sample is in-sample optimization -- the same
overfitting risk stats/risk.py exists to guard against elsewhere in this
repo. So weights are fit on the first 70% of the combined date range and the
gate is evaluated ONLY on the held-out last 30%, walk-forward style.

Run: python scripts/build_portfolio.py
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
from qresearch.risk.leverage_sim import simulate  # noqa: E402
from qresearch.stats.risk import paired_bootstrap_ci  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.build_portfolio")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"

FIT_FRAC = 0.7  # fraction of the combined date range used to fit weights

# The 5 daily sleeves that clear BOTH the in-sample DSR gate
# (campaign_20260713_v2_summary.csv: DSR 1.0/1.0/1.0/1.0/0.85) and the
# walk-forward OOS gate (walk_forward_finalists_summary.csv:
# oos_positive_frac 0.85/0.80/0.69/0.64/0.60) -- the only strategies this
# repo has actually validated end to end. 4h sleeves are excluded: crypto_4h
# fails the OOS gate outright (0.43) and the rest are marginal.
SLEEVES = [
    ("commodities_jma_atr_daily", "commodities", "daily"),
    ("indices_jma_atr_daily", "indices", "daily"),
    ("crypto_jma_atr_daily", "crypto", "daily"),
    ("indian_equities_jma_atr_daily", "indian_equities", "daily"),
    ("forex_jma_atr_daily", "forex", "daily"),
]


def main() -> None:
    trades_by_sleeve: dict[str, pd.DataFrame] = {}
    daily_by_sleeve: dict[str, pd.Series] = {}
    for strategy_id, market, timeframe in SLEEVES:
        log.info("collecting OOS trades: %s (%s/%s)", strategy_id, market, timeframe)
        trades = run_finalist_trades(strategy_id, market, timeframe, PROCESSED)
        trades_by_sleeve[strategy_id] = trades
        daily = trades_to_daily_returns(trades)
        daily_by_sleeve[strategy_id] = daily
        log.info("%s: %d trades -> %d trading days", strategy_id, len(trades), len(daily))

    # Full-period per-sleeve metrics + correlation are informational context
    # (not gated on) -- they describe the strategies, not the combiner.
    full_rows = [dict(portfolio_metrics(d), strategy_id=sid) for sid, d in daily_by_sleeve.items()]
    full_df = pd.DataFrame(full_rows).set_index("strategy_id")
    full_corr = pd.DataFrame(daily_by_sleeve).fillna(0.0).corr()

    fit_returns, eval_returns, split_date = fit_eval_split(daily_by_sleeve, FIT_FRAC)
    fit_combo = combine_sleeves(fit_returns, mode="tangency")
    weights = fit_combo["weights"]

    eval_aligned = pd.DataFrame(eval_returns).sort_index().fillna(0.0)
    eval_combined = apply_weights(eval_aligned, weights)

    eval_metrics = {sid: portfolio_metrics(eval_aligned[sid]) for sid in eval_aligned.columns}
    combined_metrics = portfolio_metrics(eval_combined)

    # best single sleeve chosen on the FIT window (matches how weights were
    # picked) then graded on the eval window, same as it's graded for the
    # combination -- no peeking at eval-period performance to pick a winner.
    best_single = max(fit_returns, key=lambda sid: portfolio_metrics(fit_returns[sid])["sharpe"])
    best_single_eval_sharpe = eval_metrics[best_single]["sharpe"]
    ci_lo, ci_hi = paired_bootstrap_ci(eval_combined, eval_aligned[best_single], statistic="sharpe")
    gate_pass = ci_lo > 0 and combined_metrics["sharpe"] > best_single_eval_sharpe

    print("\n=== full-period per-sleeve metrics (informational, not gated) ===")
    print(full_df.to_string())
    print(f"\nfull-period correlation matrix:\n{full_corr.to_string()}")

    print(f"\n=== fit window (< {split_date.date()}): tangency weights ===")
    print(weights)

    print(f"\n=== eval window (>= {split_date.date()}): held-out gate ===")
    eval_df = pd.DataFrame(eval_metrics).T
    eval_df.loc["COMBINED_PORTFOLIO"] = combined_metrics
    print(eval_df.to_string())
    print(f"\nbest single sleeve (picked on fit window): {best_single} "
          f"(eval Sharpe {best_single_eval_sharpe:.3f})")
    print(f"combined eval Sharpe: {combined_metrics['sharpe']:.3f}")
    print(f"bootstrap 95% CI for (combined - best_single) eval Sharpe: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"\nPhase 1 gate (held-out CI excludes 0 AND combined beats best single): "
          f"{'PASS' if gate_pass else 'FAIL'}")

    # Disciplined-sizing/ruin overlay on the combined trade sequence -- same
    # sizing philosophy already validated in campaign_20260713_v2_leverage_simulation.csv,
    # applied across the whole combined book instead of one forex sleeve.
    all_trades = pd.concat(trades_by_sleeve.values(), ignore_index=True).sort_values("entry_time")
    sim = simulate(all_trades, leverage=100, sizing_mode="disciplined")
    print(f"\ndisciplined-sizing overlay (leverage=100x, full combined book): "
          f"final_equity={sim.final_equity:,.0f} max_drawdown={sim.max_drawdown:.1%} "
          f"ruined={sim.ruined} ruin_prob_mc={sim.ruin_probability_mc:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    daily_out = pd.DataFrame(daily_by_sleeve).sort_index().fillna(0.0)
    daily_out["COMBINED_eval_only"] = eval_combined.reindex(daily_out.index)
    daily_out.to_csv(OUT / "portfolio_daily_combined.csv")

    metrics_out = full_df.copy()
    metrics_out["tangency_weight"] = pd.Series(weights)
    metrics_out.loc["COMBINED_PORTFOLIO_eval"] = pd.Series(combined_metrics)
    metrics_out.to_csv(OUT / "portfolio_metrics.csv")
    log.info("wrote %s and %s", OUT / "portfolio_daily_combined.csv", OUT / "portfolio_metrics.csv")


if __name__ == "__main__":
    main()
