"""Risk analysis beyond stop/target sizing — margin calls and Monte Carlo
return-path simulation. Consolidates ``check_margin_call``, ``monte_carlo_simulation``,
``calculate_expected_loss``. (``calculate_drawdown``/``compute_drawdown``/
``calculate_max_drawdown`` were duplicates of the existing
``backtest.metrics.max_drawdown`` and are not re-added here.)"""
from __future__ import annotations
import numpy as np


def check_margin_call(drawdown: float, capital: float, leverage: float,
                      margin_call_threshold: float = 1.0) -> tuple[bool, float]:
    """True if a leveraged ``drawdown`` fraction breaches the margin
    maintenance threshold. Returns (is_margin_call, remaining_equity)."""
    leveraged_loss = capital * drawdown
    equity = capital - leveraged_loss
    margin_used = capital / leverage
    return equity < margin_call_threshold * margin_used, equity


def monte_carlo_simulation(returns: np.ndarray, num_simulations: int = 10_000,
                           num_days: int | None = None) -> np.ndarray:
    """Bootstrap-resample historical daily returns to build simulated
    cumulative-return paths — for VaR / risk-of-ruin analysis. Returns an
    ``(num_simulations, num_days)`` array of cumulative returns."""
    num_days = num_days or len(returns)
    sampled = np.random.choice(returns, size=(num_simulations, num_days), replace=True)
    return np.cumsum(sampled, axis=1)


def expected_loss(probability_of_default: float, recovery_rate: float = 0.1) -> float:
    """Expected loss = PD * (1 - recovery rate) — ``calculate_expected_loss``,
    stripped of its model/scaler coupling (compute the probability upstream)."""
    return float(probability_of_default * (1 - recovery_rate))
