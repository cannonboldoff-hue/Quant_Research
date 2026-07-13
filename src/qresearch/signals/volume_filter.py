"""Volume/money-flow confirmation filter for JMA+ATR entries.

A different signal family from the ADX regime filter (signals/regime.py):
instead of gating on trend STRENGTH, this gates on money-flow DIRECTION -- a
long crossover only fires when the Money Flow Index
(indicators.volume.mfi) shows net buying pressure, a short only when it
shows net selling pressure. Zero new data ingest -- MFI is derived from the
OHLCV volume column every sleeve already has, unlike a real alt-data source
(funding rates, options IV) which this repo has no local data for and no
reliable network access to fetch live (data/external/ is empty).
"""
from __future__ import annotations

import pandas as pd

from ..indicators.volume import mfi


def apply_volume_filter(sig_df: pd.DataFrame, mfi_length: int = 14, mfi_midpoint: float = 50.0) -> pd.DataFrame:
    """Zero out long entries where MFI < ``mfi_midpoint`` (no net buying
    pressure) and short entries where MFI > ``mfi_midpoint`` (no net selling
    pressure). Returns a copy; doesn't mutate the input or touch bars that
    were already flat."""
    out = sig_df.copy()
    m = mfi(out, length=mfi_length).reindex(out.index)
    confirmed = ((out["signal"] > 0) & (m >= mfi_midpoint)) | ((out["signal"] < 0) & (m <= mfi_midpoint))
    out["signal"] = out["signal"].where(confirmed.fillna(False), 0)
    return out
