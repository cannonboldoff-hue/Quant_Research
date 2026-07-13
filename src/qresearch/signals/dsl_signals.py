"""df-in/df-out wrappers around ``indicators.dsl``'s LHP / LTP-KAMA-DSL
family, adapting their 0=neutral/1=buy/2=sell convention to the engine's
``signal`` column (1=long, -1=short, 0=flat). The indicator math itself
already lives in indicators/dsl.py -- this is just the campaign-registry
adapter layer, same shape as ``generators.jma_signals``.
"""
from __future__ import annotations

import pandas as pd

from ..indicators.dsl import lhp_dsl, ltp_kama_dsl

_TO_SIGNAL = {0: 0, 1: 1, 2: -1}


def lhp_dsl_signals(df: pd.DataFrame, price_col: str = "close", **kw) -> pd.DataFrame:
    out = df.copy()
    raw = lhp_dsl(out[price_col], **kw)
    out["signal"] = raw.map(_TO_SIGNAL).to_numpy()
    return out


def kama_dsl_signals(df: pd.DataFrame, **kw) -> pd.DataFrame:
    out = df.copy()
    raw = ltp_kama_dsl(out, **kw)
    out["signal"] = raw.map(_TO_SIGNAL).to_numpy()
    return out
