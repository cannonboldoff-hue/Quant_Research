"""Indicator-family ablation: which feature families carry the ML improvement?

Re-runs ML stages S1 (filter), S3 (sizing) and S4 (exit) with the feature matrix
restricted to one indicator family at a time (other features set to NaN, i.e. absent),
plus an all-features reference under the *same* lighter protocol: models
{logreg, lgbm}, retraining every 3 years. Results go to the registry with run_id
``ablation_<block>``; compare blocks with each other and with ``ablation_all`` only.

`vol_20` is kept in every block because the pipeline uses it to volatility-scale the
trade-context features (open P&L, previous trade return); this is disclosed in the paper.

    python scripts/tfml_ablation.py --block momentum --n-jobs 6
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfml_run import _worker, dataset_hash  # noqa: E402

from qresearch.tfml import data as D
from qresearch.tfml import registry as R
from qresearch.tfml import strategies as S
from qresearch.tfml.features import FEATURE_NAMES, FEATURE_SPECS
from qresearch.tfml.panel import build_panel
from qresearch.tfml.pipeline import RunConfig

BLOCKS = {
    "all": None,
    "momentum": {"momentum"},
    "volatility": {"volatility"},
    "trend": {"trend_strength", "trend_direction"},
    "oscillator_channel": {"oscillator", "channel"},
    "other": {"serial_dependence", "distribution", "drawdown", "volume"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=list(BLOCKS))
    ap.add_argument("--config", default="configs/tfml/benchmark.yaml")
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    panel_cfg = raw.pop("panel", {})
    valid = {f.name for f in fields(RunConfig)}
    cfg = RunConfig(**{k: (tuple(map(tuple, v)) if k == "subperiods" else tuple(v) if isinstance(v, list) else v)
                       for k, v in raw.items() if k in valid})
    cfg = replace(cfg, run_id=f"ablation_{args.block}", stages=("S1_filter", "S3_sizing", "S4_exit"),
                  models=("logreg", "lgbm"), retrain_every_years=3)
    P = build_panel(cfg.frequency, groups=panel_cfg.get("groups"), benchmark_only=panel_cfg.get("benchmark_only", True),
                    start=panel_cfg.get("start"))
    fams = BLOCKS[args.block]
    if fams is not None:
        keep = [i for i, f in enumerate(FEATURE_NAMES) if FEATURE_SPECS[f][0] in fams or f == "vol_20"]
        drop = [i for i in range(len(FEATURE_NAMES)) if i not in keep]
        P.X[:, drop] = np.nan
        print(f"block {args.block}: {len(keep)} features kept: {[FEATURE_NAMES[i] for i in keep]}")
    out_dir = R.RESULTS / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dh = dataset_hash(P)
    R.write("runs", [{"run_id": cfg.run_id, "started": time.ctime(), "git_commit": R.git_commit(), "code_hash": R.code_hash(),
                      "config": __import__("json").dumps({**asdict(cfg), "panel": panel_cfg, "feature_block": args.block}, default=str),
                      "n_instruments": len(P.meta), "n_rows": P.n, "dataset_hash": dh}], replace_keys={"run_id": cfg.run_id})
    ctx = {"rf": D.load_risk_free()}
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", max_nbytes="1M")(
        delayed(_worker)(s.id, P, cfg, None, ctx, dh, out_dir) for s in S.STRATEGIES)
    ok = 0
    for sid, status, dt, msg in res:
        ok += status == "ok"
        print(f"[{status}] {sid} {dt:.0f}s {'' if status == 'ok' else msg}", flush=True)
    print(f"ablation {args.block}: {ok}/{len(S.STRATEGIES)} ok in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
