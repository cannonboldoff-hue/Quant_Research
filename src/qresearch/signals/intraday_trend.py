"""df-in/df-out wrapper around ``indicators.trend.halftrend_session_reset``,
adapting its 0=none/1=flip-to-up/2=flip-to-down convention to the engine's
``signal`` column. The session-reset HalfTrend math already lives in
indicators/trend.py -- this is just the campaign-registry adapter layer.
"""
from __future__ import annotations

import pandas as pd

from ..indicators.trend import halftrend_session_reset

_TO_SIGNAL = {0: 0, 1: 1, 2: -1}


def halftrend_signals_intraday(df: pd.DataFrame, ticker_col: str = "Ticker",
                                date_col: str = "Date", **kw) -> pd.DataFrame:
    out = df.copy()
    ht = halftrend_session_reset(out, ticker_col=ticker_col, date_col=date_col, **kw)
    out["signal"] = ht["signal"].reindex(out.index).map(_TO_SIGNAL).to_numpy()
    return out
