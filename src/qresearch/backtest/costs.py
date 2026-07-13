"""Realistic per-pair forex trading costs (spread + swap), replacing the flat
cross-asset ``fee_bps`` for forex specifically.

The engine's flat 7bps round-trip fee (``config/settings.py``) overcharges
forex majors by roughly 10x -- real retail cost is the bid/ask spread, paid
once per round trip, not a doubled fee+slippage assumption. Spread figures
below are representative standard-account industry averages (2026 web data),
not a specific broker's live sheet -- swap the ``FOREX_SPECS`` table for your
broker's MT4/MT5 symbol specification if you have one.

Costs are computed from trade columns already on the provenance store output
(entry_price, direction, entry_time, exit_time) -- this recomputes net return
post-hoc rather than re-running the backtest engine (see
scripts/apply_realistic_forex_costs.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STANDARD_LOT_UNITS = 100_000


@dataclass(frozen=True)
class ForexSpec:
    pip_size: float
    spread_pips: float  # representative standard-account round-trip spread
    swap_long_per_lot: float  # USD per standard lot per night (long side)
    swap_short_per_lot: float  # USD per standard lot per night (short side)


# Representative standard-account figures (2026 industry averages, not
# broker-verified) -- see plan for sourcing. JPY pairs use a 0.01 pip size;
# everything else here is 0.0001.
FOREX_SPECS: dict[str, ForexSpec] = {
    "EURUSD": ForexSpec(0.0001, 0.6, -6.0, 1.0),
    "GBPUSD": ForexSpec(0.0001, 1.2, -5.0, 0.5),
    "USDJPY": ForexSpec(0.01, 0.7, 3.0, -8.0),
    "USDCHF": ForexSpec(0.0001, 0.8, 2.0, -6.0),
    "USDCAD": ForexSpec(0.0001, 0.85, 1.0, -5.0),
    "AUDUSD": ForexSpec(0.0001, 0.8, -3.0, 0.5),
    "NZDUSD": ForexSpec(0.0001, 1.4, -3.0, 0.5),
    "EURGBP": ForexSpec(0.0001, 0.9, -2.0, -1.0),
    # Cross pairs (all remaining C(8,2) combos of EUR/GBP/USD/JPY/CHF/CAD/AUD/NZD)
    # -- wider spreads than the majors above (thinner liquidity), swap signs
    # follow the same funding-currency logic but are rougher estimates than
    # the 8 majors above: not sourced per-pair, just directionally plausible.
    "EURJPY": ForexSpec(0.01, 1.5, -5.0, 1.0),
    "EURCHF": ForexSpec(0.0001, 1.8, 1.0, -4.0),
    "EURCAD": ForexSpec(0.0001, 2.5, -3.0, 0.0),
    "EURAUD": ForexSpec(0.0001, 2.5, 2.0, -5.0),
    "EURNZD": ForexSpec(0.0001, 3.0, 2.0, -5.0),
    "GBPJPY": ForexSpec(0.01, 2.0, -4.0, 0.0),
    "GBPCHF": ForexSpec(0.0001, 2.5, 1.0, -4.0),
    "GBPCAD": ForexSpec(0.0001, 3.0, -2.0, -1.0),
    "GBPAUD": ForexSpec(0.0001, 3.0, 1.0, -4.0),
    "GBPNZD": ForexSpec(0.0001, 4.0, 1.0, -4.0),
    "AUDJPY": ForexSpec(0.01, 1.8, 1.0, -5.0),
    "AUDCHF": ForexSpec(0.0001, 2.5, 1.0, -4.0),
    "AUDCAD": ForexSpec(0.0001, 2.5, 0.0, -3.0),
    "AUDNZD": ForexSpec(0.0001, 3.0, -2.0, -1.0),
    "NZDJPY": ForexSpec(0.01, 2.5, 1.0, -5.0),
    "NZDCHF": ForexSpec(0.0001, 3.0, 1.0, -4.0),
    "NZDCAD": ForexSpec(0.0001, 3.0, -1.0, -2.0),
    "CADJPY": ForexSpec(0.01, 2.0, 0.0, -4.0),
    "CADCHF": ForexSpec(0.0001, 2.5, 0.0, -3.0),
    "CHFJPY": ForexSpec(0.01, 2.5, -3.0, -1.0),
}


def spread_cost_bps(ticker: str, entry_price: float) -> float:
    """Round-trip spread cost in bps, computed at this trade's entry price
    (not a static constant -- a fixed-pip spread's bps-equivalent moves with
    price, e.g. EURUSD ranged 0.95-1.23 over 2020-2025)."""
    spec = FOREX_SPECS[ticker]
    return spec.spread_pips * spec.pip_size / entry_price * 1e4


def _nights_and_wednesdays(entry_time: pd.Series, exit_time: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Vectorized (nights, wednesday_rollovers) between entry and exit, calendar-date based."""
    entry_date = pd.to_datetime(entry_time).dt.normalize()
    exit_date = pd.to_datetime(exit_time).dt.normalize()
    nights = (exit_date - entry_date).dt.days.clip(lower=0)

    # Rollovers post at the end of each night held, i.e. on each calendar date
    # in [entry_date+1, exit_date]. Count Wednesdays in that inclusive range
    # via ordinal-day arithmetic instead of a per-row pd.date_range (which is
    # what made the original row-wise .apply slow): epoch day 0 (1970-01-01)
    # was a Thursday (pandas weekday()==3), so weekday(d) = (3+d) % 7, and
    # Wednesday (weekday==2) <=> d % 7 == 6. Count of integers === 6 (mod 7)
    # in [a, b] is floor((b-6)/7) - floor((a-1-6)/7).
    entry_ord = entry_date.astype("int64") // 86_400_000_000_000
    exit_ord = exit_date.astype("int64") // 86_400_000_000_000
    a = entry_ord + 1
    b = exit_ord
    wednesdays = (b - 6).floordiv(7) - (a - 7).floordiv(7)
    wednesdays = wednesdays.where(nights > 0, 0)
    return nights, wednesdays


def swap_cost_bps(trades: pd.DataFrame) -> pd.Series:
    """Overnight swap cost in bps of notional, vectorized over a trades frame
    with ticker/direction/entry_price/entry_time/exit_time columns. Zero for
    same-day trades (the common case at 1m/4h)."""
    spec_per_lot_long = trades["ticker"].map(lambda t: FOREX_SPECS[t].swap_long_per_lot)
    spec_per_lot_short = trades["ticker"].map(lambda t: FOREX_SPECS[t].swap_short_per_lot)
    per_lot = spec_per_lot_long.where(trades["direction"] == 1, spec_per_lot_short)

    nights, wednesdays = _nights_and_wednesdays(trades["entry_time"], trades["exit_time"])
    regular_nights = nights - wednesdays
    total_usd_per_lot = per_lot * regular_nights + per_lot * 3 * wednesdays
    notional = STANDARD_LOT_UNITS * trades["entry_price"]
    return total_usd_per_lot / notional * 1e4


def realistic_net_ret(trades: pd.DataFrame, gross_col: str = "gross") -> pd.Series:
    """Recompute net return per trade using realistic per-pair spread + swap,
    given a trades frame with a ``ticker`` column and the columns produced by
    the backtest engine (entry_price, direction, entry_time, exit_time) plus
    a precomputed gross-return column."""
    pip_size = trades["ticker"].map(lambda t: FOREX_SPECS[t].pip_size)
    spread_pips = trades["ticker"].map(lambda t: FOREX_SPECS[t].spread_pips)
    spread = spread_pips * pip_size / trades["entry_price"] * 1e4
    swap = swap_cost_bps(trades)
    return trades[gross_col] - (spread + swap) / 1e4
