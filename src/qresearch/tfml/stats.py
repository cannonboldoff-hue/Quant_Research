"""Inference for ML-vs-baseline comparisons.

- ``sharpe_diff_hac``: Ledoit & Wolf (2008) delta-method test of SR_a - SR_b with a
  Newey-West (Bartlett) HAC covariance of the first/second moments. Fast and
  robust to fat tails / serial correlation; used for the many instrument-level tests.
- ``sharpe_diff_bootstrap``: paired circular block bootstrap (Politis & Romano
  1992) of the Sharpe difference; used for portfolio-level headline tests.
- ``holm``, ``bh``, ``by``: family-wise / false-discovery corrections.
- ``spa_test``: Hansen (2005) Superior Predictive Ability test (and White's 2000
  Reality Check p-value) of "does any ML configuration beat the baseline?".
- ``romano_wolf``: stepdown max-t control of the FWER (Romano & Wolf 2005),
  block-bootstrap version for dependent data.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def _nw_cov(m: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the mean of moment matrix m (T x k)."""
    T = m.shape[0]
    u = m - m.mean(0)
    S = u.T @ u / T
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = u[l:].T @ u[:-l] / T
        S += w * (G + G.T)
    return S


def sharpe_diff_hac(a: np.ndarray, b: np.ndarray, lags: int | None = None) -> tuple[float, float, float]:
    """Return (SR_a - SR_b per-period, z-stat, two-sided p-value). NaNs dropped pairwise."""
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    T = len(a)
    if T < 30 or a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, np.nan
    lags = lags if lags is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
    m = np.column_stack([a, b, a ** 2, b ** 2])
    mu1, mu2, g1, g2 = m.mean(0)
    v1, v2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    d = mu1 / np.sqrt(v1) - mu2 / np.sqrt(v2)
    grad = np.array([g1 / v1 ** 1.5, -g2 / v2 ** 1.5, -mu1 / (2 * v1 ** 1.5), mu2 / (2 * v2 ** 1.5)])
    S = _nw_cov(m, lags)
    se = np.sqrt(max(grad @ S @ grad / T, 1e-300))
    z = d / se
    return float(d), float(z), float(2 * st.norm.sf(abs(z)))


def _cbb_indices(T: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    return idx.reshape(n_boot, -1)[:, :T]


def sharpe(x: np.ndarray, axis=-1) -> np.ndarray:
    sd = x.std(axis=axis, ddof=1)
    return np.where(sd > 0, x.mean(axis=axis) / np.where(sd > 0, sd, 1), 0.0)


def sharpe_diff_bootstrap(a: np.ndarray, b: np.ndarray, block: int = 20, n_boot: int = 2000,
                          seed: int = 0, ann: float = 252.0) -> dict:
    """Paired circular-block bootstrap of the annualised Sharpe difference a - b."""
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    T = len(a)
    if T < 60:
        return {"d_sharpe": np.nan, "p_two": np.nan, "p_one": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    rng = np.random.default_rng(seed)
    idx = _cbb_indices(T, block, n_boot, rng)
    k = np.sqrt(ann)
    d_hat = (sharpe(a) - sharpe(b)) * k
    d_b = (sharpe(a[idx]) - sharpe(b[idx])) * k
    centered = d_b - d_hat
    return {"d_sharpe": float(d_hat),
            "p_two": float(np.mean(np.abs(centered) >= abs(d_hat))),
            "p_one": float(np.mean(centered >= d_hat)),   # H0: d <= 0 vs H1: d > 0
            "ci_lo": float(np.percentile(d_b, 2.5)), "ci_hi": float(np.percentile(d_b, 97.5))}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1)
    out[ok] = res
    return out


def bh(p: np.ndarray, dependent: bool = False) -> np.ndarray:
    """Benjamini-Hochberg (or Benjamini-Yekutieli if dependent=True) adjusted p-values."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    c = np.sum(1 / np.arange(1, m + 1)) if dependent else 1.0
    order = np.argsort(pv)
    ranked = pv[order] * m * c / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1)
    out[ok] = res
    return out


def spa_test(losses_diff: np.ndarray, block: int = 20, n_boot: int = 2000, seed: int = 0) -> dict:
    """Hansen (2005) SPA test. ``losses_diff``: (T x K) matrix of performance
    differentials d_k,t = r_model_k,t - r_base,t (higher = better). H0: max_k E[d_k] <= 0.
    Returns consistent SPA p-value and White's Reality Check p-value."""
    d = np.asarray(losses_diff, float)
    d = d[np.all(np.isfinite(d), axis=1)]
    T, K = d.shape
    if T < 60:
        return {"p_spa": np.nan, "p_rc": np.nan, "best_k": None}
    rng = np.random.default_rng(seed)
    idx = _cbb_indices(T, block, n_boot, rng)
    dbar = d.mean(0)
    # HAC-ish std of sqrt(T)*dbar via bootstrap
    boot_means = np.stack([d[i].mean(0) for i in idx])          # (B, K)
    omega = np.sqrt(T) * boot_means.std(0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    t_stat = np.max(np.sqrt(T) * dbar / omega)
    thresh = omega / np.sqrt(T) * np.sqrt(2 * np.log(np.log(T)))
    mu_c = np.where(dbar >= -thresh, dbar, 0.0)                    # g_c: consistent re-centring
    z_c = np.sqrt(T) * (boot_means - mu_c) / omega                 # Hansen: d* - g_c(dbar)
    t_spa = np.maximum(z_c.max(1), 0)
    z_rc = np.sqrt(T) * (boot_means - dbar)
    rc_stat = np.max(np.sqrt(T) * dbar)
    return {"p_spa": float(np.mean(t_spa >= max(t_stat, 0))),
            "p_rc": float(np.mean(z_rc.max(1) >= rc_stat)),
            "best_k": int(np.argmax(dbar))}


def romano_wolf(diffs: np.ndarray, block: int = 20, n_boot: int = 2000, seed: int = 0,
                alpha: float = 0.05) -> np.ndarray:
    """Stepdown FWER-adjusted p-values for H0_k: E[d_k] <= 0 (one-sided), d: (T x K)."""
    d = np.asarray(diffs, float)
    d = d[np.all(np.isfinite(d), axis=1)]
    T, K = d.shape
    rng = np.random.default_rng(seed)
    idx = _cbb_indices(T, block, n_boot, rng)
    dbar = d.mean(0)
    boot = np.stack([d[i].mean(0) for i in idx]) - dbar          # centred
    se = boot.std(0, ddof=1)
    se = np.where(se > 0, se, np.inf)
    t = dbar / se
    tb = boot / se
    order = np.argsort(-t)
    adj = np.empty(K)
    prev = 0.0
    for r, k in enumerate(order):
        remaining = order[r:]
        maxb = tb[:, remaining].max(1)
        p = float(np.mean(maxb >= t[k]))
        prev = max(prev, p)
        adj[k] = prev
    return adj


def deflated_sharpe(sr_ann: float, n_obs: int, n_trials: int, sr_var_trials: float,
                    skew: float = 0.0, kurt: float = 3.0, ann: float = 252.0) -> float:
    """Bailey & Lopez de Prado (2014) DSR from an annualised SR, the number of
    configurations tried and the variance of their (annualised) SRs."""
    sr = sr_ann / np.sqrt(ann)
    var_sr = sr_var_trials / ann
    g = 0.5772156649015329
    e_max = np.sqrt(var_sr) * ((1 - g) * st.norm.ppf(1 - 1 / max(n_trials, 2))
                               + g * st.norm.ppf(1 - 1 / (max(n_trials, 2) * np.e)))
    denom = np.sqrt(max((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n_obs - 1), 1e-12))
    return float(st.norm.cdf((sr - e_max) / denom))
