"""Curvature — PCA-based residual-curvature indicator from ``crypto_4h_strat_002``
(``calculate_strategy_signals``). Isolated from its strategy/signal wrapper: this
module returns the raw curvature series, not entry/exit signals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def curvature(
    prices: pd.DataFrame,
    benchmark: str,
    beta_lookback: int = 180,
    velocity_period: int = 12,
    pca_n_components: int = 3,
) -> pd.Series:
    """Curvature of beta-adjusted residual returns: ``||v x a|| / ||v||**3``.

    ``prices``: wide frame (columns = tickers, index = time, values = close).
    ``benchmark``: column name of the reference ticker (e.g. "BTC") used to
    beta-adjust the other columns' log returns before taking velocity
    (1st derivative) and acceleration (2nd derivative), then projecting to
    ``pca_n_components`` dims via PCA to make the cross product well-defined.
    """
    others = [c for c in prices.columns if c != benchmark]
    log_returns = np.log(prices).diff()
    bench_returns = log_returns[benchmark]

    covariance = log_returns[others].rolling(beta_lookback).cov(bench_returns)
    variance = bench_returns.rolling(beta_lookback).var()
    beta = covariance.div(variance, axis=0).ffill().fillna(1.0)

    residuals = (log_returns[others] - beta.mul(bench_returns, axis=0)).dropna()
    velocity = residuals.diff(velocity_period)
    acceleration = velocity.diff(velocity_period)

    valid_rows = ~(velocity.isna().any(axis=1) | acceleration.isna().any(axis=1))
    curv = pd.Series(0.0, index=velocity.index)
    if valid_rows.sum() < 10:
        return curv

    n_components = min(pca_n_components, velocity.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(velocity.loc[valid_rows].fillna(0))

    v_pca = pca.transform(velocity.fillna(0))
    a_pca = pca.transform(acceleration.fillna(0))
    if v_pca.shape[1] < 3:
        pad = 3 - v_pca.shape[1]
        v_pca = np.pad(v_pca, ((0, 0), (0, pad)))
        a_pca = np.pad(a_pca, ((0, 0), (0, pad)))

    v_norm = np.linalg.norm(v_pca, axis=1)
    cross_norm = np.linalg.norm(np.cross(v_pca, a_pca), axis=1)
    mask = v_norm > 1e-8
    curv.values[mask] = cross_norm[mask] / (v_norm[mask] ** 3 + 1e-8)
    return curv
