"""Kalman filter price smoother — consolidates ``kalman_filter`` / ``kalman_filter_fast``
/ ``kalman_filter_optimized`` (multi_strat_046/009, indian_equities_strat_004). All
three were the same scalar (1D state, 1D measurement) Kalman filter with different
hardcoded process/measurement variance defaults; unified into one function with those
as parameters. The ``pykalman``-based ``initialize_kalman_filter``/``apply_kalman_filter``
wrapper (multi_strat_010) is intentionally not ported — it adds a dependency not in
requirements.txt for the same math this function already covers."""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(f):
        return f


@njit
def _kalman_core(prices: np.ndarray, process_variance: float, measurement_variance: float) -> np.ndarray:
    n = len(prices)
    out = np.empty(n)
    estimate = prices[0]
    error_cov = 1.0
    for t in range(n):
        pred_cov = error_cov + process_variance
        gain = pred_cov / (pred_cov + measurement_variance)
        estimate = estimate + gain * (prices[t] - estimate)
        error_cov = (1 - gain) * pred_cov
        out[t] = estimate
    return out


def kalman_filter(prices: pd.Series, process_variance: float = 1e-5, measurement_variance: float = 3.0) -> pd.Series:
    """Smooth a price series with a scalar Kalman filter."""
    out = _kalman_core(prices.to_numpy(dtype=float), process_variance, measurement_variance)
    return pd.Series(out, index=prices.index)
