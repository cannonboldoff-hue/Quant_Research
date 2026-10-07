"""Placebo control for every ML stage: apply the *same* decision rule with the *same*
validation-chosen intensity, but replace the model's predictions with random numbers.

ΔSharpe(placebo) measures what the intervention alone (skipping trades, gating, resizing,
exiting early, switching parameter variants) does without any information. ML's genuine
incremental value is ΔSharpe(ML) − ΔSharpe(placebo); the empirical p-value is the share of
placebo draws that do at least as well as the ML-selected configuration.

    python scripts/tfml_placebo.py --config configs/tfml/benchmark.yaml --seeds 20 --n-jobs 6
Writes registry table ``placebo`` and nothing else (no model is trained).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from qresearch.tfml import data as D
from qresearch.tfml import ml
from qresearch.tfml import registry as R
from qresearch.tfml import strategies as S
from qresearch.tfml.panel import build_panel, portfolio_returns, simulate_panel, vol_weights
from qresearch.tfml.pipeline import RunConfig, STAGES, _decide, folds


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    ann = len(x) / max((x.index[-1] - x.index[0]).days / 365.25, 1e-9)
    return float(x.mean() / x.std() * np.sqrt(ann)) if x.std() > 0 else 0.0


def run_one(sid: str, P, cfg: RunConfig, sel: pd.DataFrame, seeds: int, ctx: dict) -> list[dict]:
    strat = S.get(sid)
    variants = []
    for fac in cfg.variant_factors:
        params = strat.variant_params(fac) if fac != 1.0 else dict(strat.params)
        pos = np.empty(P.n)
        for k in range(len(P.meta)):
            s, e = P.bounds[k], P.bounds[k + 1]
            pos[s:e] = strat.positions(P.frame(k), ctx, **params).to_numpy()
        variants.append(pos)
    p, pv = variants[1], np.column_stack(variants)
    w = vol_weights(P, cfg.target_vol)
    starts = ml.trade_starts(p, P.inst_start)
    H = cfg.h_variant
    rebalance = (P.inst_pos % H) == 0
    fl = folds(cfg, P.date.max())
    oos = np.zeros(P.n, bool)
    for f in fl:
        oos |= (P.date >= f["test_start"]) & (P.date < f["test_end"])
    base_net = simulate_panel(P, np.where(oos, p, 0.0) * w, cfg.execution, cfg.cost_mult)[0]  # as in evaluate.py
    base_sr = _sharpe(portfolio_returns(P, base_net, oos))
    out = []
    for stage in cfg.stages:
        ssel = sel[(sel.stage == stage) & (sel.selected == 1)].set_index("fold")["intensity"]
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            final = np.zeros(P.n)
            for f in fl:
                val_m = (P.date >= f["val_start"]) & (P.date < f["test_start"])
                test_m = (P.date >= f["test_start"]) & (P.date < f["test_end"])
                win = val_m | test_m
                inten = ssel.get(f["label"], np.nan)
                inten = None if pd.isna(inten) else float(inten)
                if stage == "S5_params":
                    q = np.full((P.n, 3), np.nan)
                    rows = np.flatnonzero(rebalance & win)
                    q[rows] = rng.dirichlet(np.ones(3), len(rows))
                else:
                    q = np.full(P.n, np.nan)
                    rows = np.flatnonzero({"S1_filter": starts & win, "S2_regime": win}.get(stage, (p != 0) & win))
                    q[rows] = rng.random(len(rows))
                    if stage != "S1_filter" and cfg.pred_smooth_span > 1:
                        q = ml.smooth_predictions(q, P.inst_start, cfg.pred_smooth_span)
                pos = _decide(stage, p, pv, starts, q, inten, val_m, P, rebalance, H)
                final[test_m] = pos[test_m]
            net = simulate_panel(P, final * w, cfg.execution, cfg.cost_mult)[0]
            sr = _sharpe(portfolio_returns(P, net, oos))
            out.append({"run_id": cfg.run_id, "strategy": sid, "stage": stage, "seed": seed,
                        "base_sharpe": base_sr, "placebo_sharpe": sr, "d_sharpe_placebo": sr - base_sr})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--strategies", nargs="*")
    args = ap.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    panel_cfg = raw.pop("panel", {})
    valid = {f.name for f in fields(RunConfig)}
    cfg = RunConfig(**{k: (tuple(map(tuple, v)) if k == "subperiods" else tuple(v) if isinstance(v, list) else v)
                       for k, v in raw.items() if k in valid})
    P = build_panel(cfg.frequency, groups=panel_cfg.get("groups"), benchmark_only=panel_cfg.get("benchmark_only", True),
                    start=panel_cfg.get("start"))
    sel = R.read("select strategy, fold, stage, model, intensity, selected from ml_selection where run_id = ?",
                 params=(cfg.run_id,))
    sids = args.strategies or sorted(sel.strategy.unique())
    ctx = {"rf": D.load_risk_free()}
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, max_nbytes="1M")(
        delayed(run_one)(sid, P, cfg, sel[sel.strategy == sid], args.seeds, ctx) for sid in sids)
    rows = [r for rr in res for r in rr]
    if args.strategies:   # partial rerun: replace only these strategies' rows
        for sid in sids:
            R.write("placebo", [r for r in rows if r["strategy"] == sid],
                    replace_keys={"run_id": cfg.run_id, "strategy": sid})
    else:
        R.write("placebo", rows, replace_keys={"run_id": cfg.run_id})
    print(f"placebo {cfg.run_id}: {len(rows)} rows, {len(sids)} strategies, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
