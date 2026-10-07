"""Walk-forward experiment pipeline: rule-based baseline vs five ML stages.

For one strategy on one panel:

1. Baseline target positions p (and slow/fast parameter variants) for all rows.
2. For each fold (test period [T0, T1), validation [V0, T0), training < V0):
   - training rows: every label must be *known* (label row + embargo) before V0;
   - validation rows: labels known before T0;
   - every model in the zoo is fitted on training rows only;
   - decision rule + intervention intensity are evaluated on the validation
     period; the (model, intensity) with the best validation portfolio Sharpe
     is the *selected* configuration; each model also keeps its own best
     intensity so alternatives can be compared out of sample;
   - all configurations are then applied, frozen, to the test period.
3. Out-of-sample (concatenated test periods) results for the baseline, every
   model and the selected model are evaluated, never used for any choice.

ML stages (decision rules in ``ml.py``):
  S1_filter  meta-label filter on new trades           (signal filtering / trade selection)
  S2_regime  strategy-agnostic trend-regime gate        (trend / regime detection)
  S3_sizing  P(position profitable) -> position size    (position sizing)
  S4_exit    P(position profitable) -> early exit       (entry/exit decisions)
  S5_params  choose slow/base/fast parameter variant    (parameter adaptation)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numba
import numpy as np
import pandas as pd

from . import ml
from .metrics import compute_metrics
from .panel import Panel, simulate_panel, portfolio_returns, vol_weights
from .strategies import Strategy
from . import stats as ST

STAGES = ("S1_filter", "S2_regime", "S3_sizing", "S4_exit", "S5_params")
STAGE_DESCRIPTION = {
    "S1_filter": "signal filtering / trade selection (meta-labelling of new trades)",
    "S2_regime": "trend / regime detection gate (strategy-agnostic)",
    "S3_sizing": "position sizing from predicted probability of profit",
    "S4_exit": "ML exit decision inside open trades",
    "S5_params": "parameter adaptation (choose slow/base/fast variant monthly)",
}
# intervention intensities = quantiles of the validation-period predictions
INTENSITY = {"S1_filter": (0.2, 0.35, 0.5), "S2_regime": (0.2, 0.35, 0.5), "S4_exit": (0.1, 0.2, 0.3),
             "S3_sizing": (None,), "S5_params": (None,)}


@dataclass
class RunConfig:
    run_id: str = "benchmark"
    frequency: str = "daily"
    validation: str = "wf_expanding"     # wf_expanding | wf_rolling | holdout
    first_test_year: int = 2006
    last_test_year: int = 2026
    val_years: int = 2
    rolling_train_years: int = 10
    holdout_test_start: int = 2013
    retrain_every_years: int = 1
    execution: str = "next_open"
    cost_mult: float = 1.0
    target_vol: float = 0.10
    stages: tuple = STAGES
    models: tuple = ml.MODEL_NAMES
    h_bar: int = 10
    h_regime: int = 20
    h_variant: int = 21
    variant_factors: tuple = (2.0, 1.0, 0.5)   # slow, base, fast
    train_stride: int = 0          # 0 -> stride = label horizon (non-overlapping labels per instrument)
    max_train_rows: int = 150_000
    embargo_bars: int = 5
    pred_smooth_span: int = 5      # causal EMA of bar-level predictions before decisions
    seed: int = 0
    subperiods: tuple = ((2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026))

    def validation_id(self) -> str:
        if self.validation == "wf_rolling":
            return f"wf_rolling{self.rolling_train_years}y_val{self.val_years}y_step{self.retrain_every_years}y"
        if self.validation == "holdout":
            return f"holdout{self.holdout_test_start}_val{self.val_years}y"
        return f"wf_expanding_val{self.val_years}y_step{self.retrain_every_years}y"


def folds(cfg: RunConfig, date_max: np.datetime64) -> list[dict]:
    y = lambda yr: np.datetime64(f"{yr}-01-01", "ns")
    out = []
    if cfg.validation == "holdout":
        t0 = cfg.holdout_test_start
        out.append({"train_start": None, "val_start": y(t0 - cfg.val_years), "test_start": y(t0),
                    "test_end": date_max + np.timedelta64(1, "D"), "label": f"{t0}-end"})
        return out
    for yr in range(cfg.first_test_year, cfg.last_test_year + 1, cfg.retrain_every_years):
        ts = y(yr)
        if ts > date_max:
            break
        te = min(y(yr + cfg.retrain_every_years), date_max + np.timedelta64(1, "D"))
        vs = y(yr - cfg.val_years)
        trs = y(yr - cfg.val_years - cfg.rolling_train_years) if cfg.validation == "wf_rolling" else None
        out.append({"train_start": trs, "val_start": vs, "test_start": ts, "test_end": te, "label": str(yr)})
    return out


# --------------------------------------------------------------------------- numba helpers
@numba.njit(cache=True)
def _trade_context(p, close, inst_start, dvol):
    """Per-row trade context: bars since trade start, open P&L (vol-scaled), previous
    completed trade's vol-scaled return and length, trade start row."""
    n = p.shape[0]
    age = np.zeros(n); opnl = np.zeros(n); prev_ret = np.zeros(n); prev_len = np.zeros(n)
    start_row = np.full(n, -1)
    cur_start = -1; cur_sign = 0.0; last_ret = 0.0; last_len = 0.0
    for i in range(n):
        if inst_start[i]:
            cur_start = -1; cur_sign = 0.0; last_ret = 0.0; last_len = 0.0
        s = np.sign(p[i])
        if s != cur_sign:
            if cur_sign != 0 and cur_start >= 0:      # close previous trade at this bar's close
                v = dvol[i] if dvol[i] > 0 else 0.01
                ln = i - cur_start
                last_ret = cur_sign * np.log(close[i] / close[cur_start]) / (v * np.sqrt(ln + 1.0))
                last_len = ln
            cur_sign = s
            cur_start = i if s != 0 else -1
        if cur_sign != 0:
            a = i - cur_start
            v = dvol[i] if dvol[i] > 0 else 0.01
            age[i] = a
            opnl[i] = cur_sign * np.log(close[i] / close[cur_start]) / (v * np.sqrt(a + 1.0))
            start_row[i] = cur_start
        prev_ret[i] = last_ret
        prev_len[i] = last_len
    return age, opnl, prev_ret, prev_len, start_row


@numba.njit(cache=True)
def _trade_end_rows(p, inst_start, inst_end):
    """For each trade-start row, the last row of that trade (constant sign)."""
    n = p.shape[0]
    end = np.full(n, -1)
    i = 0
    while i < n:
        s = np.sign(p[i])
        if s != 0 and (inst_start[i] or np.sign(p[i - 1]) != s):
            j = i
            while j + 1 < n and not inst_start[j + 1] and np.sign(p[j + 1]) == s:
                j += 1
            end[i] = j
        i += 1
    return end


def _rolling_forward_sum(P: Panel, x: np.ndarray, h: int) -> np.ndarray:
    """sum_{k=1..h} x[t+k] within instrument (NaN where the window runs past the end)."""
    out = np.full(P.n, np.nan)
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(x))]
    idx = np.arange(P.n)
    end = idx + h
    n_rows = np.diff(P.bounds)[P.inst]
    ok = P.inst_pos + h < n_rows
    out[ok] = cs[end[ok] + 1] - cs[idx[ok] + 1]
    return out


# --------------------------------------------------------------------------- shared regime model (S2)
def regime_dataset(P: Panel, cfg: RunConfig):
    h = cfg.h_regime
    c = P.close
    absdiff = np.abs(c - P.lag(c, 1, np.nan))
    path = _rolling_forward_sum(P, absdiff, h)
    disp = np.abs(P.lead(c, h) - c)
    er_fwd = disp / np.where(path > 0, path, np.nan)
    label_row = np.arange(P.n) + h + cfg.embargo_bars
    return er_fwd, label_row


def _label_date(P: Panel, label_row: np.ndarray) -> np.ndarray:
    """Date at which a label becomes known (NaT if beyond the instrument's data)."""
    n_rows = np.diff(P.bounds)[P.inst]
    rel = label_row - P.bounds[P.inst]
    ok = (rel < n_rows) & (rel >= 0)
    out = np.full(P.n, np.datetime64("NaT"), dtype="datetime64[ns]")
    out[ok] = P.date[label_row[ok]]
    return out


def _train_rows(P, cfg, label_date, f, eligible):
    """Rows usable for training in fold f (labels known before val_start)."""
    m = eligible & ~np.isnat(label_date) & (label_date < f["val_start"])
    if f["train_start"] is not None:
        m &= P.date >= f["train_start"]
    return m


def _subsample(rows: np.ndarray, cfg: RunConfig, stride_mask: np.ndarray | None, rng) -> np.ndarray:
    idx = np.flatnonzero(rows & (stride_mask if stride_mask is not None else True))
    if len(idx) > cfg.max_train_rows:
        idx = np.sort(rng.choice(idx, cfg.max_train_rows, replace=False))
    return idx


def fit_regime_models(P: Panel, cfg: RunConfig, log=print) -> dict:
    """Strategy-agnostic S2 model: P(forward-h efficiency ratio > training median).
    Returns {fold_label: {model: q (n,) float32 with values on val+test rows}}."""
    er_fwd, label_row = regime_dataset(P, cfg)
    ld = _label_date(P, label_row)
    rng = np.random.default_rng(cfg.seed)
    stride = (P.inst_pos % (cfg.train_stride or cfg.h_bar)) == 0
    feat_ok = np.isfinite(P.X).mean(1) > 0.8
    out = {}
    dmax = P.date.max()
    for f in folds(cfg, dmax):
        tr = _subsample(_train_rows(P, cfg, ld, f, np.isfinite(er_fwd) & feat_ok), cfg, stride, rng)
        med = np.nanmedian(er_fwd[tr])
        y = (er_fwd[tr] > med).astype(int)
        pred_rows = np.flatnonzero((P.date >= f["val_start"]) & (P.date < f["test_end"]))
        out[f["label"]] = {"rows": pred_rows}
        for m in cfg.models:
            t0 = time.time()
            (q,) = ml.fit_predict(m, P.X[tr], y, [P.X[pred_rows]], seed=cfg.seed)
            out[f["label"]][m] = q.astype(np.float32)
            log(f"  regime fold {f['label']} {m}: n_train={len(tr)} {time.time() - t0:.1f}s")
    return out


# --------------------------------------------------------------------------- strategy run
@dataclass
class StrategyResult:
    strategy: str
    oos_mask: np.ndarray
    positions: dict = field(default_factory=dict)    # key -> (n,) final positions on OOS rows
    selection: list = field(default_factory=list)    # per-fold model/intensity choices
    timings: dict = field(default_factory=dict)


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x) / max((x.index[-1] - x.index[0]).days / 365.25, 1e-9))) \
        if len(x) > 20 and x.std() > 0 else -np.inf


def run_strategy(strategy: Strategy, P: Panel, cfg: RunConfig, regime: dict | None,
                 ctx: dict | None = None, log=print) -> StrategyResult:
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    ctx = ctx or {}
    # 1. baseline + variants
    variants = []
    for fac in cfg.variant_factors:
        params = strategy.variant_params(fac) if fac != 1.0 else dict(strategy.params)
        pos = np.empty(P.n)
        for k in range(len(P.meta)):
            s, e = P.bounds[k], P.bounds[k + 1]
            pos[s:e] = strategy.positions(P.frame(k), ctx, **params).to_numpy()
        variants.append(pos)
    p = variants[1]
    wvt = vol_weights(P, cfg.target_vol)
    dvol = P.X[:, list(_FEATS).index("vol_20")].astype(float) / np.sqrt(P.ann)
    age, opnl, prev_ret, prev_len, start_row = _trade_context(p, P.close, P.inst_start, np.nan_to_num(dvol))
    starts = ml.trade_starts(p, P.inst_start)
    feat_ok = np.isfinite(P.X).mean(1) > 0.8

    # 2. labels
    # S3/S4: direction of current position over next h bars, entry at next open
    h = cfg.h_bar
    o_next = P.lead(P.open, 1)
    fwd = np.sign(p) * np.log(P.lead(P.close, h) / o_next)
    bar_label_date = _label_date(P, np.arange(P.n) + h + cfg.embargo_bars)
    bar_ctx = np.column_stack([np.sign(p), np.abs(p), np.log1p(age), np.clip(opnl, -10, 10),
                               np.clip(prev_ret, -10, 10), np.log1p(prev_len)]).astype(np.float32)
    Xbar = lambda idx: np.hstack([P.X[idx], bar_ctx[idx]])   # built per row-subset (memory)
    # S1: trade outcome (open-to-open, net of round-trip cost)
    trade_end = _trade_end_rows(p, P.inst_start, None)
    tr_rows = np.flatnonzero(starts)
    exit_row = trade_end[tr_rows] + 1
    n_rows_inst = np.diff(P.bounds)[P.inst[tr_rows]]
    valid_exit = (exit_row - P.bounds[P.inst[tr_rows]]) < n_rows_inst
    trade_ret = np.full(len(tr_rows), np.nan)
    ok = valid_exit & ((tr_rows + 1 - P.bounds[P.inst[tr_rows]]) < n_rows_inst)
    trade_ret[ok] = (np.sign(p[tr_rows[ok]]) * np.log(P.open[exit_row[ok]] / P.open[tr_rows[ok] + 1])
                     - 2 * P.cost_rate[tr_rows[ok]] * cfg.cost_mult)
    trade_label_date = np.full(P.n, np.datetime64("NaT"), dtype="datetime64[ns]")
    lr_ = exit_row + cfg.embargo_bars
    okd = ok & ((lr_ - P.bounds[P.inst[tr_rows]]) < n_rows_inst)
    trade_label_date[tr_rows[okd]] = P.date[lr_[okd]]
    trade_y = np.full(P.n, np.nan)
    trade_y[tr_rows] = trade_ret
    # S5: best variant over next H bars
    H = cfg.h_variant
    var_net = [simulate_panel(P, v * wvt, cfg.execution, cfg.cost_mult)[0] for v in variants]
    var_fwd = np.column_stack([_rolling_forward_sum(P, x, H) for x in var_net])
    var_best = np.argmax(var_fwd, axis=1)
    tie = np.isclose(var_fwd.max(1), var_fwd.min(1))
    var_best[tie] = 1
    var_label_date = _label_date(P, np.arange(P.n) + H + cfg.embargo_bars)
    var_pos = np.column_stack(variants).astype(np.float32)
    Xvar = lambda idx: np.hstack([P.X[idx], var_pos[idx]])
    rebalance = (P.inst_pos % H) == 0
    pv = np.column_stack(variants)

    # 3. folds
    res = StrategyResult(strategy=strategy.id, oos_mask=np.zeros(P.n, bool))
    keys = [("base", None)] + [(s, m) for s in cfg.stages for m in (*cfg.models, "selected")]
    for k in keys:
        res.positions[k] = np.zeros(P.n, dtype=np.float32)
    stride = (P.inst_pos % (cfg.train_stride or cfg.h_bar)) == 0
    stride_var = (P.inst_pos % (cfg.train_stride or cfg.h_variant)) == 0
    dmax = P.date.max()
    for f in folds(cfg, dmax):
        val_m = (P.date >= f["val_start"]) & (P.date < f["test_start"])
        test_m = (P.date >= f["test_start"]) & (P.date < f["test_end"])
        if not test_m.any():
            continue
        res.oos_mask |= test_m
        res.positions[("base", None)][test_m] = p[test_m]
        win = val_m | test_m

        def val_score(pos_mod):
            net, _, _ = simulate_panel(P, pos_mod * wvt, cfg.execution, cfg.cost_mult)
            return _sharpe(portfolio_returns(P, net, val_m))

        base_val = val_score(p)
        shared = {}   # S4 reuses the S3 bar-level model predictions (same labels/features)
        for stage in cfg.stages:
            t0 = time.time()
            preds = {}
            # ---- fit models, predict on val+test rows
            if stage == "S1_filter":
                elig = starts & feat_ok & np.isfinite(trade_y)
                trm = elig & ~np.isnat(trade_label_date) & (trade_label_date < f["val_start"])
                if f["train_start"] is not None:
                    trm &= P.date >= f["train_start"]
                tr = _subsample(trm, cfg, None, rng)
                y = (trade_y[tr] > 0).astype(int)
                rows = np.flatnonzero(starts & win)
                Xtr, Xp = Xbar(tr), Xbar(rows)
            elif stage in ("S3_sizing", "S4_exit"):
                elig = (p != 0) & feat_ok & np.isfinite(fwd)
                tr = _subsample(_train_rows(P, cfg, bar_label_date, f, elig), cfg, stride, rng)
                y = (fwd[tr] > 0).astype(int)
                rows = np.flatnonzero((p != 0) & win)
                Xtr, Xp = Xbar(tr), Xbar(rows)
            elif stage == "S5_params":
                elig = feat_ok & np.isfinite(var_fwd).all(1)
                tr = _subsample(_train_rows(P, cfg, var_label_date, f, elig), cfg, stride_var, rng)
                y = var_best[tr]
                rows = np.flatnonzero(rebalance & win)
                Xtr, Xp = Xvar(tr), Xvar(rows)
            elif stage == "S2_regime":
                rows = regime[f["label"]]["rows"]
                tr = np.array([], dtype=int); y = np.array([])
            for m in cfg.models:
                if stage == "S2_regime":
                    q = np.full(P.n, np.nan); q[rows] = regime[f["label"]][m]
                elif stage == "S4_exit" and ("S3_sizing", m) in shared:
                    q = shared[("S3_sizing", m)]
                else:
                    if len(tr) < 200 or len(np.unique(y)) < 2:
                        q = np.full(P.n, np.nan) if stage != "S5_params" else np.full((P.n, 3), np.nan)
                    else:
                        (qp,) = ml.fit_predict(m, Xtr, y, [Xp], seed=cfg.seed)
                        q = np.full(P.n, np.nan) if qp.ndim == 1 else np.full((P.n, 3), np.nan)
                        q[rows] = qp
                    if stage in ("S3_sizing", "S4_exit") and cfg.pred_smooth_span > 1:
                        q = ml.smooth_predictions(q, P.inst_start, cfg.pred_smooth_span)
                    if stage == "S3_sizing":
                        shared[("S3_sizing", m)] = q
                if stage == "S2_regime" and cfg.pred_smooth_span > 1:
                    q = ml.smooth_predictions(q, P.inst_start, cfg.pred_smooth_span)
                preds[m] = q
            # ---- choose intensity per model on validation, then model on validation
            best = {}
            for m, q in preds.items():
                cands = []
                for inten in INTENSITY[stage]:
                    pos_val = _decide(stage, p, pv, starts, q, inten, val_m, P, rebalance, H)
                    cands.append((val_score(pos_val), inten))
                sc, inten = max(cands, key=lambda z: z[0])
                best[m] = (sc, inten)
            sel_model = max(best, key=lambda m: best[m][0])
            for m in cfg.models:
                pos_test = _decide(stage, p, pv, starts, preds[m], best[m][1], val_m, P, rebalance, H)
                res.positions[(stage, m)][test_m] = pos_test[test_m]
            res.positions[(stage, "selected")][test_m] = res.positions[(stage, sel_model)][test_m]
            for m in cfg.models:
                res.selection.append({"fold": f["label"], "stage": stage, "model": m,
                                      "intensity": best[m][1], "val_sharpe": best[m][0],
                                      "val_sharpe_base": base_val, "selected": m == sel_model,
                                      "n_train": int(len(tr)), "pos_rate_train": float(np.mean(y)) if len(y) and stage != "S5_params" else np.nan})
            res.timings[(f["label"], stage)] = time.time() - t0
        log(f"  {strategy.id} fold {f['label']} done ({time.time() - t_start:.0f}s)")
    return res


def _decide(stage, p, pv, starts, q, inten, val_m, P, rebalance, H):
    """Map model output q to modified target positions (for the full panel)."""
    if stage == "S1_filter":
        ref = q[starts & val_m]
        thr = np.nanquantile(ref, inten) if np.isfinite(ref).any() else -np.inf
        return ml.apply_filter(p, starts, q, thr)
    if stage == "S2_regime":
        ref = q[val_m]
        thr = np.nanquantile(ref, inten) if np.isfinite(ref).any() else -np.inf
        return ml.apply_gate(p, q, thr)
    if stage == "S4_exit":
        ref = q[val_m & (p != 0)]
        thr = np.nanquantile(ref, inten) if np.isfinite(ref).any() else -np.inf
        return ml.apply_exit(p, starts, q, thr)
    if stage == "S3_sizing":
        return ml.apply_sizing(p, q, q[val_m & (p != 0)])
    if stage == "S5_params":
        probs = np.where(rebalance[:, None], q, np.nan)
        return ml.apply_variant(pv, probs, P.inst_pos, H, 1)
    raise ValueError(stage)


from .features import FEATURE_NAMES as _FEATS  # noqa: E402
