"""Out-of-sample evaluation of a StrategyResult: metrics, ML-vs-baseline deltas and
significance tests at portfolio, market-group, instrument and sub-period level,
plus cost / execution / sizing sensitivity. Produces registry rows."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import registry as R
from . import stats as ST
from .features import FEATURE_NAMES
from .metrics import compute_metrics, trade_returns
from .panel import Panel, portfolio_returns, simulate_panel, vol_weights
from .pipeline import RunConfig, StrategyResult
from .strategies import Strategy

FEATURE_SET_ID = f"tfml_v1_{len(FEATURE_NAMES)}feat"


def _bt_id(execution, cost_mult, sizing, tv=0.10):
    return f"{execution}|cost{cost_mult:g}|{sizing}" + (f"{tv:g}" if sizing == "vol_target" else "")


class _SimCache:
    """Memoises panel simulations per (config key, execution, cost, sizing)."""

    def __init__(self, P: Panel, res: StrategyResult, w: np.ndarray):
        self.P, self.res, self.w, self.c = P, res, w, {}

    max_entries = 6   # LRU; baseline entries are pinned (memory: ~40 MB per entry)

    def get(self, k, execution, cost_mult, sized: bool):
        key = (k, execution, cost_mult, sized)
        if key in self.c:
            self.c[key] = self.c.pop(key)   # move to most-recent
            return self.c[key]
        pos = self.res.positions[k]
        net, held, cost = simulate_panel(self.P, pos * self.w if sized else pos, execution, cost_mult)
        turn = np.abs(held - self.P.lag(held, 1))
        self.c[key] = (net, held, cost, turn)
        unpinned = [x for x in self.c if x[0] != ("base", None)]
        while len(unpinned) > self.max_entries:
            self.c.pop(unpinned.pop(0))
        return self.c[key]


def _pf_metrics(P, cache: _SimCache, k, mask, execution, cost_mult, sized=True, inst_subset=None):
    net, held, cost, turn = cache.get(k, execution, cost_mult, sized)
    pr = portfolio_returns(P, net, mask, inst_subset)
    exp = portfolio_returns(P, (held != 0).astype(float), mask, inst_subset)
    cst = portfolio_returns(P, cost, mask, inst_subset)
    trn = portfolio_returns(P, turn, mask, inst_subset)
    m = compute_metrics(pr, None, cst, trn, ann=None)
    m["exposure"] = float(exp.mean())
    # trade statistics pooled over instruments (unit-size trades)
    rows = mask if inst_subset is None else mask & np.isin(P.inst, inst_subset)
    net_u, held_u, _, _ = cache.get(k, execution, cost_mult, False)
    trs = []
    for kk in np.unique(P.inst[rows]):
        s_, e_ = P.bounds[kk], P.bounds[kk + 1]
        sel = rows[s_:e_]
        trs.append(trade_returns(net_u[s_:e_][sel], held_u[s_:e_][sel]))
    trs = np.concatenate(trs) if trs else np.array([])
    m["n_trades"] = int(len(trs))
    m["win_rate_trades"] = float((trs > 0).mean()) if len(trs) else np.nan
    tp, tn = trs[trs > 0].sum(), -trs[trs < 0].sum()
    m["profit_factor_trades"] = float(tp / tn) if tn > 0 else np.nan
    act = held_u[rows] != 0
    m["win_rate_days"] = float((net_u[rows][act] > 0).mean()) if act.any() else np.nan
    m["avg_hold_bars"] = float(act.sum() / len(trs)) if len(trs) else np.nan
    return m, pr


def evaluate(res: StrategyResult, P: Panel, cfg: RunConfig, strategy: Strategy,
             dataset_hash: str, out_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    created = datetime.now(timezone.utc).isoformat()
    commit = R.git_commit()
    chash = R.code_hash()
    w = vol_weights(P, cfg.target_vol)
    cache = _SimCache(P, res, w)
    oos = res.oos_mask
    keys = list(res.positions)
    base_key = ("base", None)
    validation = cfg.validation_id()
    bt_main = _bt_id(cfg.execution, cfg.cost_mult, "vol_target", cfg.target_vol)
    sel_counts = {}
    for r in res.selection:
        if r["selected"]:
            sel_counts.setdefault(r["stage"], {}).setdefault(r["model"], 0)
            sel_counts[r["stage"]][r["model"]] += 1

    def row(level, market, instrument, stage, model, backtest, period, metrics, extra=None):
        d = {"run_id": cfg.run_id, "created_at": created, "git_commit": commit, "code_hash": chash,
             "strategy": strategy.id, "strategy_family": strategy.family,
             "strategy_params": json.dumps(strategy.params, default=str),
             "strategy_reference": strategy.reference,
             "indicators": json.dumps(list(strategy.indicators)), "feature_set": FEATURE_SET_ID,
             "market": market, "instrument": instrument,
             "dataset": json.dumps({"frequency": P.frequency, "n_instruments": int(len(P.meta)),
                                    "sha256_of_raw_hashes": dataset_hash}),
             "ml_stage": stage or "none", "ml_model": model or "none",
             "selected_models": json.dumps(sel_counts.get(stage, {})) if model == "selected" else None,
             "validation": validation, "backtest": backtest, "level": level, "period": period}
        d.update({k: (float(v) if v is not None and np.isfinite(v) else None) if isinstance(v, (int, float, np.floating, np.integer)) else v
                  for k, v in metrics.items()})
        if extra:
            d.update(extra)
        d["exp_id"] = R.exp_id(d)
        d["result_path"] = str((out_dir / f"{strategy.id}.parquet").relative_to(R.REPO))
        return d

    rows = []
    sel_keys = [base_key] + [(s, "selected") for s in cfg.stages]
    # ---------------- portfolio level, all instruments, every configuration
    pf = {}
    for k in keys:
        m, pr = _pf_metrics(P, cache, k, oos, cfg.execution, cfg.cost_mult)
        pf[k] = (m, pr)
    base_m, base_pr = pf[base_key]
    period = f"{pd.Timestamp(P.date[oos].min()).date()}..{pd.Timestamp(P.date[oos].max()).date()}"
    series = {"base": base_pr}
    for k in keys:
        m, pr = pf[k]
        extra = {}
        if k != base_key:
            series[f"{k[0]}|{k[1]}"] = pr
            al = pd.concat([pr, base_pr], axis=1).dropna()
            ann = len(al) / max((al.index[-1] - al.index[0]).days / 365.25, 1e-9)
            bt = ST.sharpe_diff_bootstrap(al.iloc[:, 0].values, al.iloc[:, 1].values, ann=ann, seed=cfg.seed)
            d, z, p_hac = ST.sharpe_diff_hac(al.iloc[:, 0].values, al.iloc[:, 1].values)
            extra = {"base_sharpe": base_m["sharpe"], "d_sharpe": m["sharpe"] - base_m["sharpe"],
                     "d_cagr": m["cagr"] - base_m["cagr"], "d_max_drawdown": m["max_drawdown"] - base_m["max_drawdown"],
                     "d_sortino": m["sortino"] - base_m["sortino"], "d_calmar": (m["calmar"] or np.nan) - (base_m["calmar"] or np.nan),
                     "d_turnover": m["turnover"] - base_m["turnover"],
                     "p_boot_two": bt["p_two"], "p_boot_one": bt["p_one"], "ci_lo": bt["ci_lo"], "ci_hi": bt["ci_hi"],
                     "p_hac_two": p_hac, "z_hac": z}
        rows.append(row("portfolio", "ALL", "PORTFOLIO", k[0] if k != base_key else None, k[1],
                        bt_main, period, m, extra))
    # SPA / Reality Check: does any ML configuration beat the baseline? (all stage x model)
    ml_keys = [k for k in keys if k != base_key and k[1] != "selected"]
    D = pd.concat([pf[k][1] - base_pr for k in ml_keys], axis=1).dropna().to_numpy()
    spa = ST.spa_test(D, seed=cfg.seed)
    rw = ST.romano_wolf(D, seed=cfg.seed)
    for r_, k in zip([r for r in rows if r["level"] == "portfolio" and r["ml_stage"] != "none" and r["ml_model"] != "selected"], ml_keys):
        r_["p_romano_wolf"] = float(rw[ml_keys.index(k)])
    rows.append(row("spa", "ALL", "PORTFOLIO", "ALL", "ALL", bt_main, period, {},
                    {"p_spa": spa["p_spa"], "p_rc": spa["p_rc"],
                     "best_config": "|".join(map(str, ml_keys[spa["best_k"]])) if spa["best_k"] is not None else None,
                     "n_configs": len(ml_keys)}))
    # ---------------- market groups (base + selected + every model)
    groups = {g: np.array(list(idx)) for g, idx in P.meta.groupby("group").groups.items()}
    gbase = {g: _pf_metrics(P, cache, base_key, oos, cfg.execution, cfg.cost_mult, True, sub)
             for g, sub in groups.items()}
    for k in keys:
        for g, sub in groups.items():
            m, pr = gbase[g] if k == base_key else _pf_metrics(P, cache, k, oos, cfg.execution,
                                                              cfg.cost_mult, True, sub)
            gb = gbase[g]
            extra = {}
            if k != base_key:
                al = pd.concat([pr, gb[1]], axis=1).dropna()
                if len(al) > 60:
                    ann = len(al) / max((al.index[-1] - al.index[0]).days / 365.25, 1e-9)
                    bt = ST.sharpe_diff_bootstrap(al.iloc[:, 0].values, al.iloc[:, 1].values, ann=ann,
                                                  seed=cfg.seed, n_boot=1000)
                    extra = {"base_sharpe": gb[0]["sharpe"], "d_sharpe": m["sharpe"] - gb[0]["sharpe"],
                             "d_cagr": m["cagr"] - gb[0]["cagr"],
                             "d_max_drawdown": m["max_drawdown"] - gb[0]["max_drawdown"],
                             "p_boot_two": bt["p_two"], "p_boot_one": bt["p_one"]}
            rows.append(row("market", g, f"GROUP:{g}", k[0] if k != base_key else None, k[1],
                            bt_main, period, m, extra))
            if k in sel_keys:
                series[f"{g}::{k[0]}|{k[1]}"] = pr

    # ---------------- instrument level (unit sizing, faithful rules), every configuration
    bt_unit = _bt_id(cfg.execution, cfg.cost_mult, "unit")
    inst_rows = {}
    for i in P.meta.index:
        s_, e_ = P.bounds[i], P.bounds[i + 1]
        sl = np.flatnonzero(oos[s_:e_]) + s_
        if len(sl) >= 250:
            inst_rows[i] = sl
    base_inst = {}
    for k in keys:
        net, held, cost, turn = cache.get(k, cfg.execution, cfg.cost_mult, False)
        for i, sl in inst_rows.items():
            meta = P.meta.loc[i]
            idx = pd.DatetimeIndex(P.date[sl])
            m = compute_metrics(pd.Series(net[sl], idx), pd.Series(held[sl], idx), pd.Series(cost[sl], idx),
                                pd.Series(turn[sl], idx))
            extra = {}
            if k == base_key:
                base_inst[i] = (m, net[sl].copy())
            else:
                bm, bnet = base_inst[i]
                d, z, p_hac = ST.sharpe_diff_hac(net[sl], bnet)
                extra = {"base_sharpe": bm["sharpe"], "d_sharpe": m["sharpe"] - bm["sharpe"],
                         "d_cagr": m["cagr"] - bm["cagr"], "d_max_drawdown": m["max_drawdown"] - bm["max_drawdown"],
                         "p_hac_two": p_hac, "z_hac": z}
            r_ = row("instrument", meta["group"], meta["id"], k[0] if k != base_key else None, k[1], bt_unit,
                     period, m, extra)
            r_["asset_class"] = meta["asset_class"]; r_["region"] = meta["region"]
            rows.append(r_)

    # ---------------- sub-periods (base + selected)
    for (a, b) in cfg.subperiods:
        pm = oos & (P.date >= np.datetime64(f"{a}-01-01")) & (P.date < np.datetime64(f"{b + 1}-01-01"))
        if pm.sum() < 1000:
            continue
        sb = None
        for k in sel_keys:
            m, pr = _pf_metrics(P, cache, k, pm, cfg.execution, cfg.cost_mult)
            extra = {}
            if k == base_key:
                sb = (m, pr)
            else:
                al = pd.concat([pr, sb[1]], axis=1).dropna()
                ann = len(al) / max((al.index[-1] - al.index[0]).days / 365.25, 1e-9)
                bt = ST.sharpe_diff_bootstrap(al.iloc[:, 0].values, al.iloc[:, 1].values, ann=ann, seed=cfg.seed, n_boot=1000)
                extra = {"base_sharpe": sb[0]["sharpe"], "d_sharpe": m["sharpe"] - sb[0]["sharpe"],
                         "p_boot_two": bt["p_two"], "p_boot_one": bt["p_one"]}
            rows.append(row("subperiod", "ALL", "PORTFOLIO", k[0] if k != base_key else None, k[1],
                            bt_main, f"{a}-{b}", m, extra))

    # ---------------- backtest-mechanism sensitivity (base + selected)
    variants = [("next_open", 0.0, "vol_target"), ("next_open", 3.0, "vol_target"),
                ("next_close", 1.0, "vol_target"), ("same_close", 1.0, "vol_target"),
                ("next_open", 1.0, "unit")]
    for ex, cm, sz in variants:
        sb = None
        for k in sel_keys:
            m, pr = _pf_metrics(P, cache, k, oos, ex, cm * cfg.cost_mult, sz == "vol_target")
            extra = {}
            if k == base_key:
                sb = (m, pr)
            else:
                al = pd.concat([pr, sb[1]], axis=1).dropna()
                ann = len(al) / max((al.index[-1] - al.index[0]).days / 365.25, 1e-9)
                bt = ST.sharpe_diff_bootstrap(al.iloc[:, 0].values, al.iloc[:, 1].values, ann=ann, seed=cfg.seed, n_boot=1000)
                extra = {"base_sharpe": sb[0]["sharpe"], "d_sharpe": m["sharpe"] - sb[0]["sharpe"],
                         "p_boot_two": bt["p_two"], "p_boot_one": bt["p_one"]}
            rows.append(row("sensitivity", "ALL", "PORTFOLIO", k[0] if k != base_key else None, k[1],
                            _bt_id(ex, cm * cfg.cost_mult, sz, cfg.target_vol), period, m, extra))

    sdf = pd.DataFrame(series)
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf.astype("float32").to_parquet(out_dir / f"{strategy.id}.parquet")
    return rows, sdf
