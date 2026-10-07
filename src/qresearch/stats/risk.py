"""Overfitting-aware statistics for the campaign's multi-strategy/multi-market
results: Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV),
Romano-Wolf stepdown, and a paired bootstrap CI. Fills the gap flagged in
``docs/ROADMAP.md`` ("formalize walk-forward reports ... deflated Sharpe,
PBO"). Pure numpy/scipy -- no new dependency.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

_EULER_GAMMA = 0.5772156649015329


def _expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """E[max SR] under the null across ``n_trials`` independent trials
    (Bailey & Lopez de Prado 2014, eq. 6-7)."""
    if n_trials <= 1:
        return 0.0
    return sr_std * ((1 - _EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / n_trials)
                      + _EULER_GAMMA * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))


def deflated_sharpe(returns: pd.Series | np.ndarray, n_trials: int = 1,
                     risk_free_rate: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio deflated for ``n_trials`` of multiple
    testing (Bailey, Borwein, Lopez de Prado & Zhu 2014). Operates on
    per-period returns directly (not annualized) per the original formula.

    Returns: P(true Sharpe > 0 | observed), in [0, 1]. Above ~0.95 is the
    conventional "survives multiple testing" bar.
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=np.float64)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return 0.0
    sr = (r.mean() - risk_free_rate) / r.std(ddof=1)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # non-excess, normal == 3
    sr_std = np.sqrt(max((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1), 1e-12))
    sr_benchmark = _expected_max_sharpe(sr_std, n_trials)
    return float(stats.norm.cdf((sr - sr_benchmark) / sr_std))


def pbo(returns_matrix: np.ndarray, n_splits: int = 10) -> float:
    """Probability of Backtest Overfitting via combinatorially symmetric
    cross-validation (Bailey, Borwein, Lopez de Prado & Zhu 2015).

    Args:
        returns_matrix: (n_strategies, n_periods) array -- one row per
            strategy/param-combo being compared, one column per time period.
        n_splits: number of contiguous blocks to partition the timeline
            into (rounded down to even).

    Returns: PBO in [0, 1] -- fraction of train/test splits where the
    in-sample winner ranked below the OOS median. 0 = no overfitting signal,
    1 = the in-sample winner is consistently an OOS loser.
    """
    m = np.asarray(returns_matrix, dtype=np.float64)
    n_strat, n_periods = m.shape
    n_splits = max(n_splits - (n_splits % 2), 2)
    blocks = np.array_split(np.arange(n_periods), n_splits)
    half = n_splits // 2

    below_median, total = 0, 0
    for is_blocks in combinations(range(n_splits), half):
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_blocks])

        is_mean = m[:, is_idx].mean(axis=1)
        oos_mean = m[:, oos_idx].mean(axis=1)
        best = int(np.argmax(is_mean))

        rank = int((oos_mean < oos_mean[best]).sum()) + 1  # 1..n_strat, higher = better OOS
        omega = rank / (n_strat + 1)
        logit = np.log(omega / (1 - omega)) if 0 < omega < 1 else (10.0 if omega >= 1 else -10.0)
        below_median += logit <= 0
        total += 1
    return float(below_median / total) if total else 0.0


def romano_wolf_stepdown(returns_list: list[pd.Series | np.ndarray], alpha: float = 0.05,
                          n_boot: int = 1000, seed: int = 0) -> list[bool]:
    """Romano-Wolf stepdown FWER control (Romano & Wolf 2005), bootstrap-based.

    Args:
        returns_list: one 1-D return array per strategy/hypothesis being
            tested jointly (e.g. every strategy surviving a screen).
        alpha: family-wise error rate to control.

    Returns: list[bool], True = null rejected (mean return significantly
    > 0 after correcting for testing ``len(returns_list)`` hypotheses
    jointly) for that index.
    """
    arrs = [np.asarray(pd.Series(r).dropna(), dtype=np.float64) for r in returns_list]
    k = len(arrs)
    t_stats = np.array([
        a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 and a.std(ddof=1) > 0 else 0.0
        for a in arrs
    ])

    rng = np.random.default_rng(seed)
    boot_t = np.zeros((n_boot, k))
    for j, a in enumerate(arrs):
        n = len(a)
        centered = a - a.mean()
        for b in range(n_boot):
            sample = rng.choice(centered, size=n, replace=True)
            s = sample.std(ddof=1)
            boot_t[b, j] = sample.mean() / (s / np.sqrt(n)) if s > 0 else 0.0

    rejected = np.zeros(k, dtype=bool)
    active = list(range(k))
    changed = True
    while active and changed:
        changed = False
        max_boot = boot_t[:, active].max(axis=1)
        crit = np.quantile(max_boot, 1 - alpha)
        newly = [idx for idx in active if t_stats[idx] > crit]
        if newly:
            for idx in newly:
                rejected[idx] = True
            active = [idx for idx in active if idx not in newly]
            changed = True
    return rejected.tolist()


def paired_bootstrap_ci(returns_a: pd.Series | np.ndarray, returns_b: pd.Series | np.ndarray,
                         statistic: str = "mean", n_resamples: int = 2000, ci: float = 0.95,
                         seed: int = 0, block: int = 20) -> tuple[float, float]:
    """Bootstrap CI for the difference in a statistic between two strategies'
    return series. ``statistic``: "mean", "sharpe", or a callable(np.ndarray)->float.

    Returns: (ci_low, ci_high) for (stat(a) - stat(b)).
    """
    a = np.asarray(pd.Series(returns_a).dropna(), dtype=np.float64)
    b = np.asarray(pd.Series(returns_b).dropna(), dtype=np.float64)

    if callable(statistic):
        stat_fn = statistic
    elif statistic == "sharpe":
        stat_fn = lambda x: x.mean() / x.std(ddof=1) if len(x) > 1 and x.std(ddof=1) > 0 else 0.0
    else:
        stat_fn = lambda x: x.mean()

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    paired = len(a) == len(b)
    block = max(int(block), 1)
    for i in range(n_resamples):
        if paired:
            # Paired circular block bootstrap: same time indices for both series, so the
            # cross-correlation and serial dependence of the returns are preserved. (The
            # original version resampled a and b independently and i.i.d., which ignores
            # both and overstates significance for overlapping strategies.)
            n = len(a)
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
            sa, sb = a[idx], b[idx]
        else:
            sa = rng.choice(a, size=len(a), replace=True)
            sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = stat_fn(sa) - stat_fn(sb)

    alpha = 1 - ci
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
