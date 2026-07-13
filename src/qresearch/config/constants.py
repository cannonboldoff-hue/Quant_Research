"""Project-wide constants (timeframes, column schema, session windows)."""

# Canonical OHLCV long-format schema used across all datasets.
COL_DATE = "Date"
COL_TICKER = "Ticker"
OHLCV = ["open", "high", "low", "close", "volume"]

# Supported timeframes and their pandas offset aliases.
TIMEFRAMES = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "1h": "1h", "4h": "4h", "daily": "1D", "weekly": "1W",
}

# Indian market session windows (IST, HH:MM) for session-based studies.
SESSIONS = {
    "opening": ("09:15", "10:30"),
    "midday":  ("10:30", "14:00"),
    "closing": ("14:00", "15:30"),
    "overnight": ("15:30", "09:15"),
}

TRADING_STYLES = ["scalping", "intraday", "swing", "positional"]
MARKETS = ["indian_equities", "crypto", "commodities", "forex", "options", "futures", "us_equities", "indices"]
