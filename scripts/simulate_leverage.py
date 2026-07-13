"""4-way leverage/margin comparison (disciplined/aggressive x 1:100/1:500) on
the best surviving realistic-cost forex config -- the "how dangerous is the
leverage I was given" answer.

Run AFTER scripts/apply_realistic_forex_costs.py has written
data/campaign_summaries/{campaign_id}_forex_realistic_costs.csv.

Run: python scripts/simulate_leverage.py [campaign_id]
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qresearch.backtest.costs import realistic_net_ret  # noqa: E402
from qresearch.campaign import ProvenanceStore  # noqa: E402
from qresearch.config.settings import get_settings  # noqa: E402
from qresearch.risk import simulate_leverage  # noqa: E402
from qresearch.utils.logging_config import get_logger  # noqa: E402

log = get_logger("qresearch.simulate_leverage")


def main() -> None:
    campaign_id = sys.argv[1] if len(sys.argv) > 1 else "campaign_20260713_v2"
    detail_path = ROOT / "data" / "campaign_summaries" / f"{campaign_id}_forex_realistic_costs.csv"
    if not detail_path.exists():
        log.error("missing %s -- run scripts/apply_realistic_forex_costs.py first", detail_path)
        sys.exit(1)
    detail = pd.read_csv(detail_path)
    best = detail.sort_values("realistic_mean_ret", ascending=False).iloc[0]
    strategy_id, params_str = best["strategy_id"], best["params"]
    params = dict(ast.literal_eval(params_str))
    log.info("best config: %s params=%s (realistic mean_ret=%.6f)",
              strategy_id, params, best["realistic_mean_ret"])

    store = ProvenanceStore()
    trades = store.read(campaign_id=campaign_id, strategy_id=strategy_id)
    trades = trades[trades["params"] == params_str].copy()
    if trades.empty:
        log.error("no trades matched strategy_id=%s params=%s", strategy_id, params_str)
        sys.exit(1)

    flat_fee = (get_settings().fee_bps + get_settings().slippage_bps) / 1e4
    trades["gross"] = trades["ret"] + flat_fee
    trades["ret"] = realistic_net_ret(trades)  # simulate() reads the "ret" column

    rows = []
    for leverage in (100, 500):
        for sizing_mode in ("disciplined", "aggressive"):
            res = simulate_leverage(trades, leverage=leverage, sizing_mode=sizing_mode)
            rows.append(vars(res))
    summary = pd.DataFrame(rows)
    print(f"\n=== leverage simulation: {strategy_id} {params} ({len(trades)} trades) ===")
    print(summary.to_string(index=False))

    out_path = ROOT / "data" / "campaign_summaries" / f"{campaign_id}_leverage_simulation.csv"
    summary.to_csv(out_path, index=False)
    log.info("summary -> %s", out_path)


if __name__ == "__main__":
    main()
