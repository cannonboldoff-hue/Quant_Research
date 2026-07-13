"""Append-only provenance store for campaign trade logs.

Partitioned Parquet under ``{base_dir}/campaign_id=.../strategy_id=.../`` --
one partition per campaign run, queried later with Hive-partition filter
pushdown via ``pyarrow.dataset`` (no full-dataset scan needed to answer
"give me campaign X's trades for strategy Y").
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..utils.logging_config import get_logger

_log = get_logger("qresearch.campaign.provenance")


class ProvenanceStore:
    def __init__(self, base_dir: str | Path = "data/provenance"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, campaign_id: str, strategy_id: str, market: str, timeframe: str,
              ticker: str, params: dict, trades: pd.DataFrame) -> None:
        if trades is None or trades.empty:
            return
        out = trades.copy()
        out["campaign_id"] = campaign_id
        out["run_ts"] = datetime.now(timezone.utc).isoformat()
        out["strategy_id"] = strategy_id
        out["market"] = market
        out["timeframe"] = timeframe
        out["ticker"] = ticker
        out["params"] = str(sorted(params.items()))

        part_dir = self.base_dir / f"campaign_id={campaign_id}" / f"strategy_id={strategy_id}"
        part_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{ticker}_{abs(hash((ticker, str(params))))}.parquet"
        out.to_parquet(part_dir / fname, index=False)
        _log.debug("wrote %d trades -> %s", len(out), part_dir / fname)

    def read(self, campaign_id: str | None = None, strategy_id: str | None = None) -> pd.DataFrame:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        # Glob *.parquet explicitly rather than pointing at base_dir directly --
        # any stray non-parquet file dropped in this tree (e.g. a summary CSV)
        # would otherwise break dataset schema inference for the whole store.
        files = [str(p) for p in self.base_dir.rglob("*.parquet")]
        if not files:
            return pd.DataFrame([])
        try:
            dataset = ds.dataset(files, partitioning="hive")
        except (FileNotFoundError, ValueError):
            return pd.DataFrame([])

        filt = None
        if campaign_id is not None:
            filt = pc.field("campaign_id") == campaign_id
        if strategy_id is not None:
            f2 = pc.field("strategy_id") == strategy_id
            filt = f2 if filt is None else filt & f2

        table = dataset.to_table(filter=filt)
        return table.to_pandas()
