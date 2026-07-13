"""Resample existing per-ticker OHLCV parquets to higher timeframes.

Lever 1 of the v2 alpha search: the flat 7bps round-trip cost dominates the
1m/5m strategies but the 1h ones are close to clearing it gross -- giving the
JMA signal more room per trade (4h, daily) tests whether going coarser clears
cost by an even wider margin. Uses each market's finest full-ticker-coverage
source as the resample base (data/loaders.resample_ohlcv already does the
per-ticker OHLCV aggregation; this just wires it to the processed tree).

Run: python scripts/resample_timeframes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.data.loaders import resample_ohlcv  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
log = get_logger("qresearch.resample")

# (market, source_timeframe) -> target_timeframes. Source is each market's
# finest timeframe that already covers every ticker (commodities/indices only
# have full coverage at 1h; crypto/futures only at 1m).
JOBS = [
    ("commodities", "1h", ["4h", "daily"]),
    ("indices", "1h", ["4h", "daily"]),
    ("crypto", "1m", ["4h", "daily"]),
    ("futures", "1m", ["4h", "daily"]),
    ("forex", "1m", ["4h", "daily"]),
    # 5m is the full-48-ticker source (1m only has 1 ticker). NSE sessions run
    # ~09:15-15:30 IST; resample_ohlcv bins on calendar time, not session
    # boundaries, so 4h bins will be uneven/partial across the session --
    # accepted, same limitation as the other markets' resampling.
    ("indian_equities", "5m", ["4h", "daily"]),
]


def _resample_market(market: str, src_tf: str, target_tfs: list[str]) -> None:
    src_dir = PROCESSED / market / src_tf
    if not src_dir.exists():
        log.warning("%s/%s: no source dir, skipping", market, src_tf)
        return
    for target_tf in target_tfs:
        out_dir = PROCESSED / market / target_tf
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(src_dir.glob("*.parquet")):
            df = pd.read_parquet(path)
            out = resample_ohlcv(df, target_tf)
            out.to_parquet(out_dir / path.name, index=False)
        n = len(list(src_dir.glob("*.parquet")))
        log.info("%s: %s -> %s (%d tickers)", market, src_tf, target_tf, n)


def main() -> None:
    for market, src_tf, target_tfs in JOBS:
        _resample_market(market, src_tf, target_tfs)
    log.info("resample complete")


if __name__ == "__main__":
    main()
