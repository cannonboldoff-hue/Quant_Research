"""Central configuration. Replaces scattered hard-coded paths / API keys.

Secrets are read from environment variables (never commit them). See
``.env.example`` at the repo root.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # --- paths ---
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    data_raw: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "raw")
    data_processed: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "processed")

    # --- broker / exchange credentials (from env) ---
    binance_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    kite_key: str = field(default_factory=lambda: os.getenv("KITE_API_KEY", ""))
    kite_secret: str = field(default_factory=lambda: os.getenv("KITE_API_SECRET", ""))

    # --- backtest defaults ---
    initial_capital: float = 100_000.0
    fee_bps: float = 5.0          # round-trip fee assumption, basis points
    slippage_bps: float = 2.0
    risk_free_rate: float = 0.06  # annual, India ~6%
    bars_per_year: int = 252


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    return Settings()
