"""Portfolio-level combination of independently-validated sleeve trade streams.

Fills the gap flagged in ``docs/ROADMAP.md`` ("Portfolio-level backtesting
... combine strategies, correlation, capital allocation"): every campaign
artifact to date reports *per-trade mean return*; none report an annualized,
portfolio-level Sharpe/CAGR/max-DD, and the DSR-clean daily JMA+ATR sleeves
(commodities/indices/crypto/indian_equities/forex) have never been combined
to harvest cross-asset-class diversification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.metrics import max_drawdown, sharpe
from ..config.settings import get_settings


def trades_to_daily_returns(trades: pd.DataFrame) -> pd.Series:
    """Collapse a sleeve's trade stream into a daily return series: each
    trade's ``ret`` is realized on its exit date, averaged across whatever
    instruments/folds exit that day (a sleeve holds several instruments at
    once even though each instrument trades one position at a time).

    ponytail: realized-at-exit bucketing, not daily mark-to-market -- a trade
    open for 10 days shows up as one lump on its exit day, not smoothed
    across the holding period. Fine for Sharpe/CAGR (path-independent at the
    trade level); upgrade to MTM only if drawdown *timing* needs to be
    trustworthy (e.g. cross-sleeve drawdown-clustering analysis).
    """
    if trades.empty:
        return pd.Series(dtype=float)
    exit_date = pd.to_datetime(trades["exit_time"]).dt.normalize()
    return trades.groupby(exit_date)["ret"].mean().sort_index()


def _long_only_tangency(cov: pd.DataFrame, excess_mu: pd.Series) -> pd.Series:
    """Long-only mean-variance-optimal (tangency) weights, ``w ~ Sigma^-1 mu``,
    via iterative active-set removal: solve unconstrained, drop the most
    negative-weight sleeve, re-solve on the rest, repeat until every weight
    is non-negative. A plain elementwise clip of the unconstrained solution
    can zero out every asset in a degenerate case (e.g. two strongly
    anti-correlated sleeves can both come out negative before clipping);
    this converges to the actual long-only-optimal active set instead."""
    active = list(cov.columns)
    w = np.zeros(0)
    while active:
        sub_cov = cov.loc[active, active].to_numpy()
        sub_mu = excess_mu.loc[active].to_numpy()
        try:
            w = np.linalg.solve(sub_cov, sub_mu)
        except np.linalg.LinAlgError:
            var = np.diag(sub_cov)
            w = np.divide(sub_mu, var, out=np.zeros_like(sub_mu), where=var > 0)
        if (w >= 0).all():
            break
        active.pop(int(np.argmin(w)))

    raw_w = pd.Series(0.0, index=cov.columns)
    if active:
        raw_w.loc[active] = np.clip(w, 0.0, None)
    return raw_w


def fit_eval_split(daily_returns: dict[str, pd.Series], fit_frac: float = 0.7):
    """Chronological fit/eval split across a shared date range: fit combiner
    weights (or select params) on the FIRST ``fit_frac`` of the union of all
    sleeves' dates, evaluate on the rest. Same walk-forward discipline as the
    rest of this repo -- weights/params picked on fit-window data only,
    graded on a disjoint eval window.

    Returns ``(fit_returns, eval_returns, split_date)`` -- two dicts with the
    same keys as ``daily_returns``, sliced by date, plus the split boundary.
    """
    all_dates = sorted(set().union(*(d.index for d in daily_returns.values())))
    split_date = all_dates[int(len(all_dates) * fit_frac)]
    fit = {sid: d[d.index < split_date] for sid, d in daily_returns.items()}
    eval_ = {sid: d[d.index >= split_date] for sid, d in daily_returns.items()}
    return fit, eval_, split_date


def apply_weights(aligned: pd.DataFrame, weights: dict[str, float],
                   target_vol: float | None = None) -> pd.Series:
    """Apply a fixed weight vector to an (already date-aligned) returns
    frame. Split out from ``combine_sleeves`` so weights fit on one window
    (e.g. a walk-forward "fit" period) can be evaluated on a disjoint window
    without refitting -- see ``scripts/build_portfolio.py``."""
    w = pd.Series(weights).reindex(aligned.columns).fillna(0.0)
    combined = aligned.mul(w, axis=1).sum(axis=1)
    if target_vol is not None and combined.std(ddof=0) > 0:
        combined = combined * (target_vol / combined.std(ddof=0))
    return combined


def combine_sleeves(daily_returns: dict[str, pd.Series], mode: str = "tangency",
                     target_vol: float | None = None) -> dict:
    """Align sleeves on date, weight, and sum into one combined daily return
    series. Days a sleeve had no exit are treated as flat (0 return) for that
    sleeve, not dropped -- otherwise sparser sleeves (fewer instruments/trades)
    would look artificially less correlated with the others.

    ``mode``:
      - ``"tangency"`` (default) -- mean-variance-optimal (Sharpe-maximizing)
        weights, ``w ~ Sigma^-1 mu`` (Markowitz), long-only clipped. This is
        the mode that actually harvests diversification: for near-uncorrelated
        assets the achievable combined Sharpe is ``sqrt(sum(SR_i^2))``, which
        inverse-vol weighting does NOT reach whenever Sharpes differ across
        sleeves (it equalizes risk contribution, not return-per-risk).
        Falls back to inverse-variance-of-mean (``mu_i / var_i``, ignoring
        cross terms) if the covariance matrix is singular.
      - ``"equal_weight"`` -- 1/n per sleeve.
      - ``"inverse_vol"`` / ``"equal_risk"`` -- weight inversely proportional
        to each sleeve's own daily volatility. Robust (no return estimate
        needed) but not Sharpe-optimal when sleeve Sharpes differ; use when
        you don't trust the mean-return estimates enough to fit tangency
        weights (e.g. very short history). ``"equal_risk"`` is an alias here
        -- ponytail: true cross-sleeve risk-parity needs a numeric solve for
        pairwise covariance contributions; with 5 sleeves inverse-vol
        approximates it.
      - ``target_vol``: if given, rescale the combined series to this daily
        vol (e.g. to match a single sleeve's vol for an apples-to-apples
        Sharpe comparison). Leave ``None`` to just sum the weighted sleeves.

    Fitting tangency weights and evaluating the combined Sharpe on the SAME
    sample is in-sample optimization -- it will inflate the combined Sharpe
    the same way any unvalidated backtest does. Callers proving a real
    diversification benefit must fit weights on one window and evaluate on a
    held-out window (``apply_weights`` on the eval window's aligned frame),
    the same walk-forward discipline the rest of this repo already applies.

    Returns ``{"combined": Series, "weights": dict, "corr": DataFrame,
    "aligned": DataFrame}``.
    """
    aligned = pd.DataFrame(daily_returns).sort_index().fillna(0.0)
    if aligned.empty:
        raise ValueError("no overlapping data across sleeves")

    vols = aligned.std(ddof=0)
    if mode == "equal_weight":
        raw_w = pd.Series(1.0, index=aligned.columns)
    elif mode in ("inverse_vol", "equal_risk"):
        raw_w = 1.0 / vols.replace(0, np.nan)
    elif mode == "tangency":
        # Solve on EXCESS mean (mu - daily rf), matching backtest.metrics.sharpe's
        # own excess-return convention -- solving on raw mu would maximize
        # mean/variance, not risk-free-adjusted Sharpe, and the two only agree
        # when rf is negligible next to the sleeves' mean returns.
        daily_rf = get_settings().risk_free_rate / get_settings().bars_per_year
        excess_mu = aligned.mean() - daily_rf
        cov = aligned.cov()
        raw_w = _long_only_tangency(cov, excess_mu)
        if raw_w.sum() <= 0:
            raise ValueError("tangency solve produced no positive-weight sleeve")
    else:
        raise ValueError(f"unknown mode: {mode}")
    weights = (raw_w / raw_w.sum()).fillna(0.0)

    combined = apply_weights(aligned, weights.to_dict(), target_vol=target_vol)

    return {"combined": combined, "weights": weights.to_dict(),
            "corr": aligned.corr(), "aligned": aligned}


def portfolio_metrics(daily: pd.Series, periods: int | None = None) -> dict:
    """Annualized Sharpe/CAGR/max-DD/Calmar/vol for a daily return series --
    the number missing from every existing campaign artifact (those report
    per-trade means, not annualized portfolio performance). Reuses
    ``backtest.metrics.sharpe``/``max_drawdown`` rather than reimplementing."""
    s = get_settings()
    periods = periods or s.bars_per_year
    r = pd.Series(daily).dropna()
    if r.empty:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                "annual_vol": 0.0, "n_days": 0}

    equity = (1 + r).cumprod()
    years = len(r) / periods
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    dd = max_drawdown(equity)
    calmar = float(cagr / abs(dd)) if dd != 0 and not np.isnan(cagr) else 0.0

    return {
        "sharpe": sharpe(r, periods=periods),
        "cagr": cagr,
        "max_drawdown": dd,
        "calmar": calmar,
        "annual_vol": float(r.std(ddof=0) * np.sqrt(periods)),
        "n_days": int(len(r)),
    }
