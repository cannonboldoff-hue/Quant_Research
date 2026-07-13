"""Feature engineering pipeline helpers (returns, rolling stats, lags, cyclical
time encodings). Consolidates ``add_returns``/``add_rolling_features`` with
``create_advanced_features`` (lags + cyclical hour/weekday encodings —
the module's original docstring promised lags but never had them) and
``generate_features`` (return/volatility/volume-change/spread features)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df[price_col].pct_change()
    df["log_ret_1"] = (df[price_col] / df[price_col].shift(1)).apply("log")
    return df


def add_rolling_features(df: pd.DataFrame, price_col: str = "close", windows=(5, 10, 20)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"mean_{w}"] = df[price_col].rolling(w).mean()
        df[f"std_{w}"] = df[price_col].rolling(w).std()
        df[f"zscore_{w}"] = (df[price_col] - df[f"mean_{w}"]) / df[f"std_{w}"]
    return df


def add_lag_features(df: pd.DataFrame, columns: list[str], lags=(1, 2, 3)) -> pd.DataFrame:
    """Lagged copies of ``columns`` — the "lags" the module docstring always
    promised. Part of ``create_advanced_features``."""
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_cyclical_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Sin/cos-encoded hour-of-day and day-of-week — part of ``create_advanced_features``."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    hour, weekday = dt.dt.hour, dt.dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return df


def add_market_microstructure_features(df: pd.DataFrame, ticker_col: str = "Ticker") -> pd.DataFrame:
    """Per-ticker short-horizon return/volatility/volume-change/spread
    features — ``generate_features``, generalized off its hardcoded
    ``Signal``-derived "book imbalance" placeholder (dropped: it wasn't a
    real feature, just a 1:1 remap of the signal column)."""
    df = df.copy()
    g = df.groupby(ticker_col)
    df["return_1"] = g["close"].pct_change(1)
    df["return_5"] = g["close"].pct_change(5)
    df["volatility_5"] = g["close"].transform(lambda x: x.pct_change().rolling(5).std())
    df["volume_change"] = g["volume"].pct_change(1)
    df["spread"] = (df["high"] - df["low"]) / df["close"]
    feature_cols = ["return_1", "return_5", "volatility_5", "volume_change", "spread"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
