"""Leakage / false-discovery null test: run the full benchmark pipeline on synthetic
random-walk markets that contain NO predictable structure (i.i.d. Student-t innovations
with GARCH-style volatility clustering, zero return autocorrelation, realistic OHLC).

If any stage leaked future information (labels, features, execution, selection), ML would
appear to improve the rules here. Under no leakage, ML improvements must be ~0 or negative
(costs of intervening), and significance rates must be near the nominal level.

    python scripts/tfml_null_test.py --n-instruments 60 --n-jobs 6
Writes registry rows with run_id ``null_randomwalk``.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfml_run import _worker  # noqa: E402

from qresearch.tfml import registry as R
from qresearch.tfml.features import compute_features
from qresearch.tfml.panel import Panel
from qresearch.tfml.pipeline import RunConfig, fit_regime_models

STRATEGIES = ["sma_50_200", "ema_12_26", "jma_7_21", "donchian_20", "tsmom_252", "supertrend_10_3",
              "macd_12_26_9", "turtle_20_10"]


def synth(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1995-01-02", periods=n)
    # GARCH(1,1) variance with Student-t(5) shocks; unconditional daily vol ~1%
    a, b = 0.08, 0.90
    w = (0.01 ** 2) * (1 - a - b)
    h = np.empty(n); r = np.empty(n)
    h[0] = 0.01 ** 2
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    for t in range(n):
        if t:
            h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    c = 100 * np.exp(np.cumsum(r))
    gap = np.sqrt(h) * 0.3 * rng.standard_normal(n)
    o = np.r_[100.0, c[:-1]] * np.exp(gap)
    rng_hl = np.abs(rng.normal(0, 1, (2, n))) * np.sqrt(h) * 0.6
    hi = np.maximum(o, c) * np.exp(rng_hl[0])
    lo = np.minimum(o, c) * np.exp(-rng_hl[1])
    v = rng.lognormal(10, 0.3, n)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)


def make_panel(n_inst: int, n_bars: int) -> Panel:
    frames = [synth(n_bars, 10_000 + i) for i in range(n_inst)]
    X = np.concatenate([compute_features(f).to_numpy(np.float32) for f in frames])
    lens = np.array([len(f) for f in frames])
    big = pd.concat(frames)
    groups = ["null_a", "null_b", "null_c"]
    meta = pd.DataFrame({"id": [f"RW{i:03d}" for i in range(n_inst)], "group": [groups[i % 3] for i in range(n_inst)],
                         "asset_class": "synthetic", "region": "none",
                         "first_date": [f.index[0] for f in frames], "last_date": [f.index[-1] for f in frames]})
    inst = np.repeat(np.arange(n_inst), lens).astype(np.int32)
    pos = np.concatenate([np.arange(l) for l in lens])
    return Panel("daily", meta, inst, big.index.to_numpy(), pos, pos == 0, big.open.to_numpy(), big.high.to_numpy(),
                 big.low.to_numpy(), big.close.to_numpy(), big.volume.to_numpy(), np.full(len(big), 3e-4), X, 252.0,
                 np.r_[0, np.cumsum(lens)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-instruments", type=int, default=60)
    ap.add_argument("--n-bars", type=int, default=8000)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    ap.add_argument("--run-id", default="null_randomwalk")
    args = ap.parse_args()
    cfg = replace(RunConfig(), run_id=args.run_id, first_test_year=2006, last_test_year=2025)
    P = make_panel(args.n_instruments, args.n_bars)
    print(f"synthetic panel: {len(P.meta)} instruments x {args.n_bars} bars ({P.date.min()} .. {P.date.max()})")
    out_dir = R.RESULTS / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    R.write("runs", [{"run_id": cfg.run_id, "started": time.ctime(), "git_commit": R.git_commit(), "code_hash": R.code_hash(),
                      "config": __import__("json").dumps({**asdict(cfg), "panel": {"synthetic": True, "n_instruments": args.n_instruments,
                                                                                    "n_bars": args.n_bars}}, default=str),
                      "n_instruments": len(P.meta), "n_rows": P.n, "dataset_hash": "synthetic-garch-t5"}],
            replace_keys={"run_id": cfg.run_id})
    regime = fit_regime_models(P, cfg, log=lambda *a: None)
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", max_nbytes="1M")(
        delayed(_worker)(sid, P, cfg, regime, {}, "synthetic", out_dir) for sid in args.strategies)
    for sid, status, dt, msg in res:
        print(f"[{status}] {sid} {dt:.0f}s {'' if status == 'ok' else msg}", flush=True)
    print(f"null test done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
