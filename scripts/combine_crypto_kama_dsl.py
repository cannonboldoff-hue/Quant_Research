"""Phase 5 of the alpha-search plan: walk-forward validate crypto_kama_dsl_daily
and add it to the portfolio if it genuinely diversifies crypto_jma_atr_daily.

A concurrent process's campaign run found crypto_kama_dsl_daily (LTP-KAMA-DSL --
a different indicator family from JMA) clears the IN-SAMPLE DSR gate cleanly on
the same 7-ticker crypto daily universe crypto_jma_atr_daily uses (DSR=1.0,
mean_ret +0.0165/trade, slightly ahead of jma_atr's +0.0156). That's the same
"wrong-timeframe, not a dead signal" pattern this session found for JMA --
kama_dsl was previously only tested (and killed) at 1m.

But in-sample DSR isn't the bar the 5 combined sleeves were held to -- this
walk-forward validates it for real (qresearch.optimize.finalist_selection, now
generalized to accept a pluggable signal_fn/grid instead of being jma-only),
checks its correlation to crypto_jma_atr_daily specifically (the whole point of
adding it -- low correlation raises the combined Sharpe even at modest
standalone Sharpe), and adds it as a 6th sleeve only if the combined eval
Sharpe beats Phase 2's 1.780, the same bar Phase 3/4 were held to (and failed).

Reuses configs/strategies.yaml's crypto_kama_dsl_daily grid (already cleared
the in-sample gate with these exact ranges) rather than inventing a new search
space: period in [40,55,72], fastend in [0.3,0.5], slowend/window at their
defaults 0.08/14.

Run: python scripts/combine_crypto_kama_dsl.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.optimize.finalist_selection import _combos, run_finalist_trades  # noqa: E402
from qresearch.portfolio.combine import (  # noqa: E402
    apply_weights, combine_sleeves, fit_eval_split, portfolio_metrics, trades_to_daily_returns,
)
from qresearch.signals.dsl_signals import kama_dsl_signals  # noqa: E402
from qresearch.stats.risk import deflated_sharpe  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.combine_crypto_kama_dsl")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "campaign_summaries"

FIT_FRAC = 0.7
PHASE2_COMBINED_EVAL_SHARPE = 1.780

KAMA_GRID = {"period": [40, 55, 72], "fastend": [0.3, 0.5]}

# The Phase-2 trend book minus crypto_jma_atr_daily -- kama_dsl is added
# ALONGSIDE jma, not instead of it, so both crypto strategies stay in if kama
# clears the gate (that's the diversification bet: two signal families on the
# same market).
OTHER_SLEEVES = [
    ("commodities_jma_atr_daily", "commodities", "daily", False),
    ("indices_jma_atr_daily", "indices", "daily", True),
    ("indian_equities_jma_atr_daily", "indian_equities", "daily", False),
    ("forex_jma_atr_daily", "forex", "daily", False),
]
CRYPTO_JMA_SLEEVE = ("crypto_jma_atr_daily", "crypto", "daily")


def main() -> None:
    log.info("walk-forward validating crypto_kama_dsl_daily (period x fastend grid)")
    kama_trades = run_finalist_trades("crypto_kama_dsl_daily", "crypto", "daily", PROCESSED,
                                       signal_fn=kama_dsl_signals, signal_grid=KAMA_GRID)

    if kama_trades.empty:
        print("\ncrypto_kama_dsl_daily produced no qualifying OOS trades -- fails immediately.")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "n_trades": 0, "mean_ret": None, "win_rate": None, "dsr": None,
            "corr_vs_crypto_jma": None, "standalone_gate": False,
            "combined_eval_sharpe": None, "phase5_gate": False,
        }]).to_csv(OUT / "phase5_kama_dsl_decision.csv", index=False)
        return

    n_trades = len(kama_trades)
    mean_ret = float(kama_trades["ret"].mean())
    win_rate = float((kama_trades["ret"] > 0).mean())
    n_trials = len(_combos(KAMA_GRID))
    dsr = deflated_sharpe(kama_trades["ret"], n_trials=n_trials)
    print(f"\n=== crypto_kama_dsl_daily: walk-forward OOS ===")
    print(f"n_trades={n_trades}, mean_ret={mean_ret:.5f}, win_rate={win_rate:.3f}, "
          f"DSR (n_trials={n_trials})={dsr:.3f}")

    # cheap gate first -- same discipline as campaign_cross_sectional.py: don't
    # pay for the trend-book rebuild on a foregone conclusion.
    cheap_gate = mean_ret > 0 and dsr > 0.5
    print(f"\ncheap gate (mean_ret>0 AND DSR>0.5): {'PASS' if cheap_gate else 'FAIL'}")
    if not cheap_gate:
        print("\ncrypto_kama_dsl_daily does not clear its own walk-forward OOS bar -- "
              "not worth the correlation/portfolio rebuild. Not added.")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "n_trades": n_trades, "mean_ret": mean_ret, "win_rate": win_rate, "dsr": dsr,
            "corr_vs_crypto_jma": None, "standalone_gate": False,
            "combined_eval_sharpe": None, "phase5_gate": False,
        }]).to_csv(OUT / "phase5_kama_dsl_decision.csv", index=False)
        return

    log.info("clears standalone gate -- checking correlation vs crypto_jma_atr_daily")
    kama_daily = trades_to_daily_returns(kama_trades)
    jma_id, jma_market, jma_tf = CRYPTO_JMA_SLEEVE
    jma_trades = run_finalist_trades(jma_id, jma_market, jma_tf, PROCESSED)
    jma_daily = trades_to_daily_returns(jma_trades)

    common = kama_daily.index.intersection(jma_daily.index)
    corr = float(kama_daily.loc[common].corr(jma_daily.loc[common])) if len(common) > 1 else float("nan")
    print(f"\ncorrelation vs crypto_jma_atr_daily (n={len(common)} overlapping days): {corr:.3f}")

    log.info("rebuilding 6-sleeve portfolio (Phase-2 trend book + crypto_jma + crypto_kama_dsl)")
    daily_by_sleeve = {jma_id: jma_daily, "crypto_kama_dsl_daily": kama_daily}
    for sid, market, timeframe, regime in OTHER_SLEEVES:
        trades = run_finalist_trades(sid, market, timeframe, PROCESSED, regime_filter=regime)
        daily_by_sleeve[sid] = trades_to_daily_returns(trades)

    fit_returns, eval_returns, split_date = fit_eval_split(daily_by_sleeve, FIT_FRAC)
    weights = combine_sleeves(fit_returns, mode="tangency")["weights"]
    eval_aligned = pd.DataFrame(eval_returns).sort_index().fillna(0.0)
    combined_metrics = portfolio_metrics(apply_weights(eval_aligned, weights))

    phase5_gate = combined_metrics["sharpe"] > PHASE2_COMBINED_EVAL_SHARPE
    print(f"\n6-sleeve tangency weights (fit < {split_date.date()}): {weights}")
    print(f"6-sleeve combined eval Sharpe: {combined_metrics['sharpe']:.3f} "
          f"(Phase 2 baseline was {PHASE2_COMBINED_EVAL_SHARPE:.3f})")
    print(f"\nPhase 5 gate (6-sleeve combined eval Sharpe beats Phase 2 baseline): "
          f"{'PASS' if phase5_gate else 'FAIL'}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "n_trades": n_trades, "mean_ret": mean_ret, "win_rate": win_rate, "dsr": dsr,
        "corr_vs_crypto_jma": corr, "standalone_gate": cheap_gate,
        "combined_eval_sharpe": combined_metrics["sharpe"], "phase5_gate": phase5_gate,
    }]).to_csv(OUT / "phase5_kama_dsl_decision.csv", index=False)
    log.info("wrote %s", OUT / "phase5_kama_dsl_decision.csv")


if __name__ == "__main__":
    main()
