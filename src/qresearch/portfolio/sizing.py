"""Position sizing: Kelly and fixed-fractional."""
from __future__ import annotations


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Kelly fraction f* = W - (1-W)/R. Clipped to [0, 1]."""
    if win_loss_ratio <= 0:
        return 0.0
    f = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0.0, min(1.0, f))


def fixed_fractional(capital: float, risk_per_trade: float, stop_distance: float) -> float:
    """Units to trade risking ``risk_per_trade`` fraction of capital to the stop."""
    if stop_distance <= 0:
        return 0.0
    return (capital * risk_per_trade) / stop_distance
