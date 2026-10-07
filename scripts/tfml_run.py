"""Run the ML-vs-rule-based trend-following experiment grid and log everything
to the experiment registry (results/tfml/registry.sqlite).

    python scripts/tfml_run.py --config configs/tfml/benchmark.yaml
    python scripts/tfml_run.py --config configs/tfml/benchmark.yaml --strategies sma_50_200 donchian_20
    python scripts/tfml_run.py --config configs/tfml/robust_rolling.yaml --n-jobs 4

The config file sets run_id, frequency, universe groups and every RunConfig field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import traceback
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

from qresearch.tfml import data as D
from qresearch.tfml import registry as R
from qresearch.tfml import strategies as S
from qresearch.tfml.evaluate import evaluate
from qresearch.tfml.panel import build_panel, subset_panel
from qresearch.tfml.pipeline import RunConfig, fit_regime_models, run_strategy


def dataset_hash(P) -> str:
    man = json.loads((D.PROCESSED / "manifest.json").read_text())["instruments"]
    hs = [man.get(f"{i}@{P.frequency}", {}).get("raw_sha256", "?") for i in P.meta["id"]]
    return hashlib.sha256("".join(hs).encode()).hexdigest()[:16]


def _worker(sid, P, cfg, regime, ctx, dhash, out_dir):
    t0 = time.time()
    try:
        strat = S.get(sid)
        logs = []
        res = run_strategy(strat, P, cfg, regime, ctx, log=logs.append)
        rows, _ = evaluate(res, P, cfg, strat, dhash, out_dir)
        sel = [{"run_id": cfg.run_id, "strategy": sid, **r} for r in res.selection]
        R.write("experiments", rows, replace_keys={"run_id": cfg.run_id, "strategy": sid})
        R.write("ml_selection", sel, replace_keys={"run_id": cfg.run_id, "strategy": sid})
        return sid, "ok", time.time() - t0, "\n".join(logs[-3:])
    except Exception:
        return sid, "error", time.time() - t0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strategies", nargs="*")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-instruments", type=int, default=None, help="smoke-test subset")
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    panel_cfg = raw.pop("panel", {})
    valid = {f.name for f in fields(RunConfig)}
    cfg = RunConfig(**{k: (tuple(map(tuple, v)) if k == "subperiods" else tuple(v) if isinstance(v, list) else v)
                       for k, v in raw.items() if k in valid})
    out_dir = R.RESULTS / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "run.log", "a", buffering=1)
    say = lambda *a: (print(*a), print(*a, file=log))
    say(f"=== run {cfg.run_id} start {time.ctime()} config={args.config}")

    P = build_panel(cfg.frequency, groups=panel_cfg.get("groups"), benchmark_only=panel_cfg.get("benchmark_only", True),
                    start=panel_cfg.get("start"))
    if args.max_instruments:
        keep = P.meta.groupby("group").head(max(1, args.max_instruments // P.meta["group"].nunique())).index
        P = subset_panel(P, keep)
    say(f"panel: {len(P.meta)} instruments, {P.n} rows, groups={sorted(P.meta['group'].unique())}")
    dhash = dataset_hash(P)
    ctx = {"rf": D.load_risk_free()}

    regime_path = out_dir / "regime_preds.pkl"
    if "S2_regime" in cfg.stages:
        if regime_path.exists():
            regime = pickle.loads(regime_path.read_bytes())
        else:
            t0 = time.time()
            regime = fit_regime_models(P, cfg, log=say)
            regime_path.write_bytes(pickle.dumps(regime))
            say(f"regime models fitted in {time.time() - t0:.0f}s")
    else:
        regime = None

    R.write("runs", [{"run_id": cfg.run_id, "started": time.ctime(), "git_commit": R.git_commit(), "code_hash": R.code_hash(),
                      "config": json.dumps({**asdict(cfg), "panel": panel_cfg}, default=str),
                      "n_instruments": len(P.meta), "n_rows": P.n, "dataset_hash": dhash}],
            replace_keys={"run_id": cfg.run_id})
    sids = args.strategies or [s.id for s in S.STRATEGIES]
    say(f"strategies: {len(sids)}; n_jobs={args.n_jobs}")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", max_nbytes="1M")(
        delayed(_worker)(sid, P, cfg, regime, ctx, dhash, out_dir) for sid in sids)
    n_ok = 0
    for sid, status, dt, msg in results:
        n_ok += status == "ok"
        say(f"[{status}] {sid} {dt:.0f}s ({time.time() - t0:.0f}s elapsed)\n{msg if status != 'ok' else ''}")
    say(f"=== run {cfg.run_id} done: {n_ok}/{len(sids)} ok in {time.time() - t0:.0f}s")
    return 0 if n_ok == len(sids) else 1


if __name__ == "__main__":
    sys.exit(main())
