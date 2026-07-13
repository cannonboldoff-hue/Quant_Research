"""Smart-money-concepts signal primitives — consolidates ``identify_fvg``,
``find_order_blocks``, ``identify_liquidity_sweeps``, ``calculate_fibonacci_levels``,
``filter_session`` (``indian_equities_strat_001``), generalized off tick-data
column names (``LTP``/``BuyPrice``/``SellPrice``) onto the repo's OHLCV schema."""
from __future__ import annotations
import numpy as np
import pandas as pd


def fair_value_gap(close: pd.Series, threshold: float) -> pd.Series:
    """True where a 2-bar-forward gap exceeds ``threshold`` — ``identify_fvg``."""
    return (close.shift(1) < close.shift(-1)) & ((close.shift(-1) - close).abs() > threshold)


def order_blocks(close: pd.Series, buy_price: pd.Series, sell_price: pd.Series, lookback: int = 150) -> pd.Series:
    """'bullish' / 'bearish' / None per bar — ``find_order_blocks``."""
    bearish = (close < buy_price) & (close == close.rolling(lookback).min())
    bullish = (close > sell_price) & (close == close.rolling(lookback).max())
    return pd.Series(np.where(bearish, "bearish", np.where(bullish, "bullish", None)), index=close.index)


def liquidity_sweeps(close: pd.Series, buy_price: pd.Series, sell_price: pd.Series,
                     short_lookback: int = 5, long_lookback: int = 250) -> pd.Series:
    """True where price sweeps below recent lows / above longer-run highs —
    ``identify_liquidity_sweeps``."""
    return (close < buy_price.rolling(short_lookback).min()) | (close > sell_price.rolling(long_lookback).max())


def fibonacci_retracement(close: pd.Series, lookback: int = 500, zone: tuple[float, float] = (0.62, 0.79)) -> pd.DataFrame:
    """Rolling Fibonacci high/low/retracement + optimal-trade-entry zone flag —
    ``calculate_fibonacci_levels``."""
    fib_high = close.rolling(lookback).max()
    fib_low = close.rolling(lookback).min()
    retracement = (close - fib_low) / (fib_high - fib_low)
    in_zone = (retracement >= zone[0]) & (retracement <= zone[1])
    return pd.DataFrame({"fib_high": fib_high, "fib_low": fib_low,
                         "retracement": retracement, "ote_zone": in_zone}, index=close.index)


def in_session(index: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    """True where a datetime index's time-of-day falls in ``[start, end]`` — ``filter_session``."""
    t = index.time
    return pd.Series((t >= pd.to_datetime(start).time()) & (t <= pd.to_datetime(end).time()), index=index)


def smc_signals(df: pd.DataFrame, price_col: str = "close", high_col: str = "high",
                 low_col: str = "low", date_col: str = "Date", fvg_threshold: float | None = None,
                 order_block_lookback: int = 150, sweep_short: int = 5, sweep_long: int = 250,
                 fib_lookback: int = 500, session_start: str = "09:15",
                 session_end: str = "15:30") -> pd.DataFrame:
    """df-in/df-out wrapper joining the SMC primitives into the engine's
    ``signal`` column via the joint-AND condition ``RESEARCH_PAPERS.md`` P4
    tests (FVG + order block direction + liquidity sweep + Fibonacci OTE
    zone + session), the campaign-registry adapter for this module's
    primitives, same shape as ``generators.jma_signals``.

    ponytail: the source notebook (``indian_equities_strat_001``) ran these
    on tick bid/ask (LTP/BuyPrice/SellPrice -- see
    ``data/processed/indian_equities/ticks/``); plain OHLCV bars have no
    bid/ask, so this substitutes the *prior* bar's low/high as buy/sell-price
    proxies (using the same bar's low/high is self-referential and makes
    order_blocks/liquidity_sweeps degenerate to always-False). A
    research-grade replication of P4 needs a tick-native runner reading the
    ticks/ store directly, not this OHLCV path.
    """
    out = df.copy()
    close = out[price_col]
    buy_price = out[low_col].shift(1)
    sell_price = out[high_col].shift(1)
    threshold = fvg_threshold if fvg_threshold is not None else float(close.diff().abs().median())

    fvg = fair_value_gap(close, threshold)
    ob = order_blocks(close, buy_price, sell_price, order_block_lookback)
    sweep = liquidity_sweeps(close, buy_price, sell_price, sweep_short, sweep_long)
    ote = fibonacci_retracement(close, fib_lookback)["ote_zone"]
    sess = in_session(pd.DatetimeIndex(out[date_col]), session_start, session_end).to_numpy()

    long_cond = (fvg & (ob == "bullish") & sweep & ote).to_numpy() & sess
    short_cond = (fvg & (ob == "bearish") & sweep & ote).to_numpy() & sess
    out["signal"] = np.select([long_cond, short_cond], [1, -1], default=0)
    return out
