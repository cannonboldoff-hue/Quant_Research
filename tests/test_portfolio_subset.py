"""Sanity check for the commodities+indices+forex subset wiring in
scripts/build_portfolio_subset.py: same fit/combine/apply/metrics chain as
the full 5-sleeve portfolio, run on 3 synthetic sleeves."""
import numpy as np
import pandas as pd

from qresearch.portfolio.combine import apply_weights, combine_sleeves, fit_eval_split, portfolio_metrics

SLEEVES = ["commodities_jma_atr_daily", "indices_jma_atr_daily", "forex_jma_atr_daily"]


def test_subset_equal_weight_gives_one_third_each():
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    rng = np.random.default_rng(0)
    daily = {sid: pd.Series(rng.normal(0.0005, 0.01, 300), index=dates) for sid in SLEEVES}

    fit_returns, eval_returns, _ = fit_eval_split(daily, 0.7)
    combo = combine_sleeves(fit_returns, mode="equal_weight")
    for sid in SLEEVES:
        assert combo["weights"][sid] == 1 / 3

    eval_aligned = pd.DataFrame(eval_returns).sort_index().fillna(0.0)
    combined = apply_weights(eval_aligned, combo["weights"])
    metrics = portfolio_metrics(combined)
    assert np.isfinite(metrics["sharpe"])
    assert metrics["n_days"] == len(eval_aligned)


def test_subset_tangency_weights_are_long_only_and_sum_to_one():
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    rng = np.random.default_rng(1)
    daily = {sid: pd.Series(rng.normal(0.0005, 0.01 * (i + 1), 300), index=dates)
             for i, sid in enumerate(SLEEVES)}

    fit_returns, _, _ = fit_eval_split(daily, 0.7)
    combo = combine_sleeves(fit_returns, mode="tangency")
    assert all(w >= 0 for w in combo["weights"].values())
    assert abs(sum(combo["weights"].values()) - 1.0) < 1e-9
