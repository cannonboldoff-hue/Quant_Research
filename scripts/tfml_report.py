"""Build every table, figure and headline number used in paper/ from the experiment
registry (results/tfml/registry.sqlite) and the stored OOS return series.

    python scripts/tfml_report.py            # all runs present in the registry
    python scripts/tfml_report.py --run benchmark

Outputs: paper/tables/*.md|csv, paper/figures/*.png, paper/results_summary.json.
Nothing here is hand-entered: if a number is in the paper it was computed below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qresearch.stats.risk import pbo
from qresearch.tfml import data as D
from qresearch.tfml import registry as R
from qresearch.tfml import stats as ST
from qresearch.tfml import strategies as S
from qresearch.tfml.features import FEATURE_SPECS
from qresearch.tfml.metrics import compute_metrics
from qresearch.tfml.ml import MODEL_NAMES
from qresearch.tfml.pipeline import STAGE_DESCRIPTION, STAGES

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
TAB = PAPER / "tables"
FIG = PAPER / "figures"
STAGE_LABEL = {"S1_filter": "S1 filter", "S2_regime": "S2 regime", "S3_sizing": "S3 sizing",
               "S4_exit": "S4 exit", "S5_params": "S5 params"}
ALPHA = 0.05
SUMMARY: dict = {}

# palette (dataviz reference instance, light mode)
BLUE, RED, GRAY_MID = "#2a78d6", "#e34948", "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def save(df: pd.DataFrame, name: str, floatfmt=".3f", index=False):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=index)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=index, floatfmt=floatfmt), encoding="utf-8")


def sig_counts(g: pd.DataFrame, pcol: str = "p_boot_two") -> pd.Series:
    p = pd.to_numeric(g[pcol], errors="coerce")
    d = pd.to_numeric(g["d_sharpe"], errors="coerce")
    return pd.Series({"sig_pos": int(((p < ALPHA) & (d > 0)).sum()), "sig_neg": int(((p < ALPHA) & (d < 0)).sum())})


# --------------------------------------------------------------------------- static tables
def data_tables():
    man = json.loads((D.PROCESSED / "manifest.json").read_text())
    rows = pd.DataFrame.from_dict(man["instruments"], orient="index")
    rows = rows[rows["status"].notna()]
    uni = D.universe_frame()
    ok = rows[rows["status"] == "ok"].copy()
    ok["freq"] = ok["frequency"]
    g = ok.groupby(["group", "freq"]).agg(
        instruments=("id", "nunique"), first_start=("start", "min"), last_start=("start", "max"),
        end=("end", "max"), total_bars=("rows", "sum"), spikes_removed=("removed_spikes", "sum"),
        hl_fixed=("fixed_hl_inconsistency", "sum"), rows_trimmed_gap=("dropped_before_gap", "sum"),
        nonpositive_dropped=("dropped_nonpositive", "sum")).reset_index()
    src = uni.groupby("group").agg(source=("source", "first"), asset_class=("asset_class", "first"),
                                   survivorship=("survivorship", "first"), benchmark=("benchmark", "first"),
                                   exchanges=("exchange", lambda x: len(set(x))),
                                   regions=("region", lambda x: ", ".join(sorted(set(x)))))
    g = g.merge(src, left_on="group", right_index=True, how="left")
    save(g, "T1_data_coverage")
    bad = rows[rows["status"] != "ok"][["status"]].reset_index().rename(columns={"index": "series"})
    save(bad, "T1b_data_exclusions")
    trimmed = rows[(rows.get("dropped_before_gap", 0) > 0)][["dropped_before_gap", "start", "notes"]].reset_index()
    save(trimmed.rename(columns={"index": "series"}), "T1c_gap_trims")
    SUMMARY["data"] = {"n_series_ok": int(len(ok)), "n_instruments_ok": int(ok["id"].nunique()),
                       "n_bars_total": int(ok["rows"].sum()),
                       "n_benchmark_instruments_daily": int(ok[(ok["freq"] == "daily") & ok["group"].isin(uni[uni.benchmark]["group"].unique())]["id"].nunique()),
                       "groups": sorted(ok["group"].unique().tolist()),
                       "n_exchanges": int(uni["exchange"].nunique()), "n_regions": int(uni["region"].nunique()),
                       "spikes_removed": int(ok["removed_spikes"].sum()),
                       "hl_fixed": int(ok["fixed_hl_inconsistency"].sum()),
                       "n_gap_trimmed_series": int(len(trimmed)),
                       "excluded": bad.to_dict("records")}
    val = TAB / "T0_data_validation.csv"
    if val.exists():
        SUMMARY["data"]["validation"] = pd.read_csv(val).to_dict("records")


def method_tables():
    st = pd.DataFrame([{"id": s.id, "name": s.name, "family": s.family, "reference": s.reference,
                        "params": ", ".join(f"{k}={v}" for k, v in s.params.items()),
                        "indicators": ", ".join(s.indicators), "direction": "long/cash" if s.long_only else "long/short"}
                       for s in S.STRATEGIES])
    save(st, "T2_strategies")
    ft = pd.DataFrame([{"feature": k, "family": v[0], "definition": v[1]} for k, v in FEATURE_SPECS.items()])
    save(ft, "T3_features")
    stg = pd.DataFrame([{"stage": k, "pipeline role": v} for k, v in STAGE_DESCRIPTION.items()])
    save(stg, "T4_ml_stages")
    SUMMARY["n_strategies"] = len(st)
    SUMMARY["n_features"] = len(ft)
    SUMMARY["n_indicator_families"] = int(ft["family"].nunique())
    SUMMARY["n_strategy_families"] = int(st["family"].nunique())
    SUMMARY["strategy_families"] = st["family"].value_counts().to_dict()


# --------------------------------------------------------------------------- run-level analysis
def load(run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = R.read("select * from experiments where run_id = ?", params=(run,))
    s = R.read("select * from ml_selection where run_id = ?", params=(run,))
    for c in ["d_sharpe", "p_boot_two", "p_boot_one", "p_hac_two", "sharpe", "base_sharpe", "d_cagr",
              "d_max_drawdown", "d_turnover", "cagr", "max_drawdown", "turnover"]:
        if c in e:
            e[c] = pd.to_numeric(e[c], errors="coerce")
    return e, s


def headline(e: pd.DataFrame, run: str) -> pd.DataFrame:
    pf = e[(e.level == "portfolio") & (e.ml_model == "selected")].copy()
    pf["p"] = pf["p_boot_two"]
    pf["p_holm"] = ST.holm(pf["p"].to_numpy())
    pf["p_bh"] = ST.bh(pf["p"].to_numpy())
    pf["p_by"] = ST.bh(pf["p"].to_numpy(), dependent=True)
    out = []
    for stg in STAGES:
        x = pf[pf.ml_stage == stg]
        if x.empty:
            continue
        d = x["d_sharpe"]
        out.append({"stage": stg, "n_strategies": len(x), "mean_dSharpe": d.mean(), "median_dSharpe": d.median(),
                    "share_improved": (d > 0).mean(),
                    "sig_pos_raw": int(((x.p < ALPHA) & (d > 0)).sum()), "sig_neg_raw": int(((x.p < ALPHA) & (d < 0)).sum()),
                    "sig_pos_holm": int(((x.p_holm < ALPHA) & (d > 0)).sum()), "sig_neg_holm": int(((x.p_holm < ALPHA) & (d < 0)).sum()),
                    "sig_pos_bh": int(((x.p_bh < ALPHA) & (d > 0)).sum()), "sig_neg_bh": int(((x.p_bh < ALPHA) & (d < 0)).sum()),
                    "mean_base_sharpe": x["base_sharpe"].mean(), "mean_ml_sharpe": x["sharpe"].mean(),
                    "mean_d_cagr": x["d_cagr"].mean(), "mean_d_maxdd": x["d_max_drawdown"].mean(),
                    "mean_d_turnover": x["d_turnover"].mean()})
    h = pd.DataFrame(out)
    from scipy.stats import binomtest
    h["sign_test_p"] = [binomtest(int(round(r.share_improved * r.n_strategies)), int(r.n_strategies)).pvalue for r in h.itertuples()]
    save(h, f"{run}_T5_headline_by_stage")
    pf_out = pf[["strategy", "strategy_family", "ml_stage", "base_sharpe", "sharpe", "d_sharpe", "p", "p_holm", "p_bh", "selected_models"]]
    save(pf_out, f"{run}_T6_strategy_stage_detail")
    SUMMARY.setdefault("runs", {}).setdefault(run, {})["headline"] = h.to_dict("records")
    SUMMARY["runs"][run]["n_tests_portfolio_selected"] = int(pf["p"].notna().sum())
    SUMMARY["runs"][run]["n_sig_pos_bh_total"] = int(((pf.p_bh < ALPHA) & (pf.d_sharpe > 0)).sum())
    SUMMARY["runs"][run]["n_sig_neg_bh_total"] = int(((pf.p_bh < ALPHA) & (pf.d_sharpe < 0)).sum())
    SUMMARY["runs"][run]["n_sig_pos_holm_total"] = int(((pf.p_holm < ALPHA) & (pf.d_sharpe > 0)).sum())
    meta = R.read("select * from runs where run_id = ?", params=(run,))
    if not meta.empty:
        cfg = json.loads(meta["config"].iloc[0])
        SUMMARY["runs"][run]["meta"] = {"n_instruments": int(meta["n_instruments"].iloc[0]), "n_rows": int(meta["n_rows"].iloc[0]),
                                        "code_hash": meta["code_hash"].iloc[0] if "code_hash" in meta else None,
                                        "dataset_hash": meta["dataset_hash"].iloc[0], "validation": cfg["validation"],
                                        "frequency": cfg["frequency"], "first_test_year": cfg["first_test_year"],
                                        "oos_period": pf["period"].iloc[0] if len(pf) else None,
                                        "n_strategies_done": int(pf["strategy"].nunique()),
                                        "n_registry_rows": int(len(e))}
    return pf


def base_table(e, run):
    b = e[(e.level == "portfolio") & (e.ml_stage == "none")]
    cols = ["strategy", "strategy_family", "cagr", "total_return", "ann_vol", "sharpe", "sortino", "calmar",
            "max_drawdown", "win_rate_trades", "profit_factor_trades", "turnover", "cost_drag", "exposure",
            "n_trades", "stability_r2", "pct_pos_years"]
    t = b[cols].sort_values("sharpe", ascending=False)
    save(t, f"{run}_T7_baseline_metrics")
    sel = e[(e.level == "portfolio") & (e.ml_model == "selected")]
    save(sel[["strategy", "ml_stage"] + cols[2:]], f"{run}_T7b_ml_selected_metrics")
    SUMMARY["runs"][run]["baseline"] = {"mean_sharpe": float(b.sharpe.mean()), "median_sharpe": float(b.sharpe.median()),
                                        "n_positive_sharpe": int((b.sharpe > 0).sum()), "n": int(len(b)),
                                        "best": b.loc[b.sharpe.idxmax(), "strategy"], "best_sharpe": float(b.sharpe.max()),
                                        "worst": b.loc[b.sharpe.idxmin(), "strategy"], "worst_sharpe": float(b.sharpe.min()),
                                        "mean_cagr": float(b.cagr.mean()), "mean_maxdd": float(b.max_drawdown.mean())}


def model_tables(e, s, run):
    pf = e[(e.level == "portfolio") & (e.ml_stage != "none") & (e.ml_model != "selected")]
    m = pf.groupby(["ml_stage", "ml_model"]).agg(mean_dSharpe=("d_sharpe", "mean"), median_dSharpe=("d_sharpe", "median"),
                                                 share_improved=("d_sharpe", lambda x: (x > 0).mean())).reset_index()
    m = m.merge(pf.groupby(["ml_stage", "ml_model"]).apply(sig_counts).reset_index(), on=["ml_stage", "ml_model"])
    if not s.empty:
        sel = s[s.selected == 1].groupby(["stage", "model"]).size().rename("times_selected").reset_index()
        tot = s[s.selected == 1].groupby("stage").size().rename("n_choices")
        sel = sel.merge(tot, left_on="stage", right_index=True)
        sel["selection_share"] = sel.times_selected / sel.n_choices
        m = m.merge(sel.rename(columns={"stage": "ml_stage", "model": "ml_model"}), how="left", on=["ml_stage", "ml_model"])
        m["times_selected"] = m["times_selected"].fillna(0).astype(int)
        m["selection_share"] = m["selection_share"].fillna(0.0)
        m["n_choices"] = m.groupby("ml_stage")["n_choices"].transform("max")
    save(m, f"{run}_T8_model_comparison")
    rows = []
    if not s.empty:
        vs = s.groupby(["strategy", "stage", "model"])["val_sharpe"].mean().reset_index()
        mm = vs.merge(pf[["strategy", "ml_stage", "ml_model", "d_sharpe"]],
                      left_on=["strategy", "stage", "model"], right_on=["strategy", "ml_stage", "ml_model"])
        for (st_, stg), g in mm.groupby(["strategy", "stage"]):
            if len(g) >= 3 and g.val_sharpe.nunique() > 1 and g.d_sharpe.nunique() > 1:
                rows.append({"strategy": st_, "stage": stg, "spearman": g[["val_sharpe", "d_sharpe"]].corr("spearman").iloc[0, 1]})
    rc = pd.DataFrame(rows)
    if not rc.empty:
        rc_s = rc.groupby("stage")["spearman"].agg(["mean", "median", "count"]).reset_index()
        save(rc_s, f"{run}_T8b_validation_test_rank_consistency")
        SUMMARY["runs"][run]["val_test_spearman"] = rc_s.to_dict("records")
    selrows = e[(e.level == "portfolio") & (e.ml_model == "selected")][["strategy", "ml_stage", "d_sharpe"]]
    avg = pf.groupby(["strategy", "ml_stage"])["d_sharpe"].agg(avg_model="mean", best_ex_post="max", worst_ex_post="min").reset_index()
    cmp_ = selrows.merge(avg, on=["strategy", "ml_stage"]).groupby("ml_stage")[["d_sharpe", "avg_model", "best_ex_post", "worst_ex_post"]].mean().reset_index()
    cmp_ = cmp_.rename(columns={"d_sharpe": "selected_model"})
    save(cmp_, f"{run}_T8c_selected_vs_alternatives")
    SUMMARY["runs"][run]["model_comparison"] = m.to_dict("records")
    SUMMARY["runs"][run]["selected_vs_alternatives"] = cmp_.to_dict("records")


def market_tables(e, run):
    mk = e[(e.level == "market") & (e.ml_model == "selected")].copy()
    mk["p_bh"] = np.nan
    for stg, g in mk.groupby("ml_stage"):
        mk.loc[g.index, "p_bh"] = ST.bh(g["p_boot_two"].to_numpy())
    t = mk.groupby(["market", "ml_stage"]).agg(mean_dSharpe=("d_sharpe", "mean"), share_improved=("d_sharpe", lambda x: (x > 0).mean()),
                                              n=("d_sharpe", "count")).reset_index()
    t = t.merge(mk.groupby(["market", "ml_stage"]).apply(sig_counts).add_suffix("_raw").reset_index(), on=["market", "ml_stage"])
    t = t.merge(mk.groupby(["market", "ml_stage"]).apply(lambda g: sig_counts(g, "p_bh")).add_suffix("_bh").reset_index(),
                on=["market", "ml_stage"])
    bm = e[(e.level == "market") & (e.ml_stage == "none")].groupby("market")["sharpe"].mean().rename("mean_base_sharpe")
    t = t.merge(bm, left_on="market", right_index=True)
    save(t, f"{run}_T9_market_groups")
    SUMMARY["runs"][run]["market"] = t.to_dict("records")
    return t


def instrument_tables(e, run):
    ins = e[(e.level == "instrument") & (e.ml_model == "selected")].copy()
    if ins.empty:
        return None
    ins["p_bh"] = np.nan
    for stg, g in ins.groupby("ml_stage"):
        ins.loc[g.index, "p_bh"] = ST.bh(g["p_hac_two"].to_numpy())
    t = ins.groupby(["asset_class", "ml_stage"]).agg(
        n_tests=("d_sharpe", "count"), mean_dSharpe=("d_sharpe", "mean"),
        share_improved=("d_sharpe", lambda x: (x > 0).mean())).reset_index()
    t = t.merge(ins.groupby(["asset_class", "ml_stage"]).apply(lambda g: sig_counts(g, "p_hac_two")).add_suffix("_raw").reset_index(),
                on=["asset_class", "ml_stage"])
    t = t.merge(ins.groupby(["asset_class", "ml_stage"]).apply(lambda g: sig_counts(g, "p_bh")).add_suffix("_bh").reset_index(),
                on=["asset_class", "ml_stage"])
    save(t, f"{run}_T10_instrument_level")
    tot = ins.groupby("ml_stage").agg(n_tests=("d_sharpe", "count"), share_improved=("d_sharpe", lambda x: (x > 0).mean()),
                                      mean_dSharpe=("d_sharpe", "mean")).reset_index()
    tot = tot.merge(ins.groupby("ml_stage").apply(lambda g: sig_counts(g, "p_bh")).add_suffix("_bh").reset_index(), on="ml_stage")
    save(tot, f"{run}_T10b_instrument_level_total")
    SUMMARY["runs"][run]["instrument"] = tot.to_dict("records")
    return ins


def period_sensitivity_tables(e, run):
    sp = e[(e.level == "subperiod") & (e.ml_stage != "none")]
    if not sp.empty:
        t = sp.groupby(["period", "ml_stage"]).agg(mean_dSharpe=("d_sharpe", "mean"), share_improved=("d_sharpe", lambda x: (x > 0).mean()),
                                                  n=("d_sharpe", "count")).reset_index()
        t = t.merge(sp.groupby(["period", "ml_stage"]).apply(sig_counts).reset_index(), on=["period", "ml_stage"])
        bs = e[(e.level == "subperiod") & (e.ml_stage == "none")].groupby("period")["sharpe"].mean().rename("mean_base_sharpe")
        t = t.merge(bs, left_on="period", right_index=True)
        save(t, f"{run}_T11_subperiods")
        SUMMARY["runs"][run]["subperiods"] = t.to_dict("records")
    se = e[e.level == "sensitivity"]
    if not se.empty:
        bs = se[se.ml_stage == "none"].groupby("backtest")["sharpe"].mean().rename("mean_base_sharpe")
        sml = se[se.ml_stage != "none"]
        t = sml.groupby(["backtest", "ml_stage"]).agg(
            mean_dSharpe=("d_sharpe", "mean"), share_improved=("d_sharpe", lambda x: (x > 0).mean()),
            mean_ml_sharpe=("sharpe", "mean")).reset_index()
        t = t.merge(sml.groupby(["backtest", "ml_stage"]).apply(sig_counts).reset_index(), on=["backtest", "ml_stage"])
        t = t.merge(bs, left_on="backtest", right_index=True)
        msel = e[(e.level == "portfolio") & (e.ml_model == "selected")]
        main = msel.groupby("ml_stage").agg(
            mean_dSharpe=("d_sharpe", "mean"), share_improved=("d_sharpe", lambda x: (x > 0).mean()),
            mean_ml_sharpe=("sharpe", "mean")).reset_index()
        main = main.merge(msel.groupby("ml_stage").apply(sig_counts).reset_index(), on="ml_stage")
        main["backtest"] = e[e.level == "portfolio"]["backtest"].iloc[0] + " (benchmark)"
        main["mean_base_sharpe"] = e[(e.level == "portfolio") & (e.ml_stage == "none")]["sharpe"].mean()
        t = pd.concat([main[t.columns], t])
        save(t, f"{run}_T12_backtest_sensitivity")
        SUMMARY["runs"][run]["sensitivity"] = t.to_dict("records")
    fam = e[(e.level == "portfolio") & (e.ml_model == "selected")].groupby(["strategy_family", "ml_stage"]).agg(
        n=("d_sharpe", "count"), mean_dSharpe=("d_sharpe", "mean"), share_improved=("d_sharpe", lambda x: (x > 0).mean())).reset_index()
    fb = e[(e.level == "portfolio") & (e.ml_stage == "none")].groupby("strategy_family")["sharpe"].mean().rename("mean_base_sharpe")
    fam = fam.merge(fb, left_on="strategy_family", right_index=True)
    save(fam, f"{run}_T13_strategy_family")
    SUMMARY["runs"][run]["family"] = fam.to_dict("records")


def spa_table(e, run):
    sp = e[e.level == "spa"][["strategy", "p_spa", "p_rc", "best_config", "n_configs"]].copy()
    sp["p_spa"] = pd.to_numeric(sp.p_spa); sp["p_rc"] = pd.to_numeric(sp.p_rc)
    sp["p_spa_holm"] = ST.holm(sp.p_spa.to_numpy())
    save(sp, f"{run}_T14_spa_reality_check")
    SUMMARY["runs"][run]["spa"] = {"n": int(len(sp)), "n_spa_sig": int((sp.p_spa < ALPHA).sum()),
                                   "n_rc_sig": int((sp.p_rc < ALPHA).sum()),
                                   "n_spa_sig_holm": int((sp.p_spa_holm < ALPHA).sum())}


def placebo_tables(e, run):
    """ML vs placebo (same decision rule + intensity, random predictions)."""
    try:
        pl = R.read("select * from placebo where run_id = ?", params=(run,))
    except Exception:
        return
    if pl.empty:
        return
    ml_ = e[(e.level == "portfolio") & (e.ml_model == "selected")][["strategy", "ml_stage", "d_sharpe"]]
    rows = []
    for (st_, stg), g in pl.groupby(["strategy", "stage"]):
        m = ml_[(ml_.strategy == st_) & (ml_.ml_stage == stg)]
        if m.empty:
            continue
        d_ml = float(m.d_sharpe.iloc[0])
        draws = g.d_sharpe_placebo.to_numpy()
        rows.append({"strategy": st_, "stage": stg, "d_sharpe_ml": d_ml, "d_sharpe_placebo_mean": draws.mean(),
                     "d_sharpe_placebo_p95": np.quantile(draws, 0.95), "ml_minus_placebo": d_ml - draws.mean(),
                     "p_empirical": (1 + (draws >= d_ml).sum()) / (1 + len(draws)), "n_draws": len(draws)})
    d = pd.DataFrame(rows)
    d["p_bh"] = ST.bh(d.p_empirical.to_numpy())
    save(d, f"{run}_T19_placebo_detail")
    t = d.groupby("stage").agg(n=("strategy", "count"), mean_d_ml=("d_sharpe_ml", "mean"),
                               mean_d_placebo=("d_sharpe_placebo_mean", "mean"),
                               mean_ml_minus_placebo=("ml_minus_placebo", "mean"),
                               share_ml_beats_placebo_mean=("ml_minus_placebo", lambda x: (x > 0).mean()),
                               n_ml_above_placebo_p95=("p_empirical", lambda x: int((x <= 0.05).sum())),
                               n_bh=("p_bh", lambda x: int((x < ALPHA).sum()))).reset_index()
    save(t, f"{run}_T19_placebo_by_stage")
    SUMMARY["runs"][run]["placebo"] = t.to_dict("records")


def conditions_regression(e: pd.DataFrame, run: str):
    """Which conditions predict the ML improvement? OLS of instrument-level dSharpe (all
    models, not only selected) on stage, model, strategy family, asset class and baseline
    Sharpe, with standard errors clustered by instrument."""
    import statsmodels.formula.api as smf
    d = e[(e.level == "instrument") & (e.ml_stage != "none") & (e.ml_model != "selected")].copy()
    if d.empty:
        return
    d = d.dropna(subset=["d_sharpe", "base_sharpe"])
    d["d_sharpe"] = d["d_sharpe"].clip(-3, 3)
    ref_ac = "equity" if "equity" in set(d.asset_class) else sorted(set(d.asset_class))[0]
    ref_fam = "ma_crossover" if "ma_crossover" in set(d.strategy_family) else sorted(set(d.strategy_family))[0]
    f = (f"d_sharpe ~ C(ml_stage, Treatment('S1_filter')) + C(ml_model, Treatment('logreg')) + "
         f"C(strategy_family, Treatment('{ref_fam}')) + C(asset_class, Treatment('{ref_ac}')) + base_sharpe")
    res = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["instrument"])[0]})
    t = pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues}).reset_index().rename(columns={"index": "term"})
    t["term"] = (t["term"].str.replace(r"C\((\w+), Treatment\('[\w_]+'\)\)\[T\.", r"\1=", regex=True).str.replace("]", ""))
    save(t, f"{run}_T15_conditions_regression")
    SUMMARY["runs"][run]["regression"] = {"n_obs": int(res.nobs), "r2": float(res.rsquared),
                                          "ref_asset_class": ref_ac, "ref_family": ref_fam,
                                          "terms": t.to_dict("records")}


def overfitting_tables(run: str):
    """PBO (CSCV) across each strategy's ML configurations + baseline, DSR of the best config."""
    out_dir = R.RESULTS / run
    rows, comp = [], {}
    for f in sorted(out_dir.glob("*.parquet")):
        sdf = pd.read_parquet(f)
        cols = ["base"] + [c for c in sdf.columns if "|" in c and "::" not in c and not c.endswith("|selected")]
        M = sdf[cols].dropna(how="all").fillna(0.0)
        if len(M) < 500:
            continue
        ann = len(M) / ((M.index[-1] - M.index[0]).days / 365.25)
        p = pbo(M.to_numpy().T, n_splits=16)
        sr = M.mean() / M.std() * np.sqrt(ann)
        best = sr.drop("base").idxmax()
        dsr = ST.deflated_sharpe(sr[best], len(M), n_trials=len(cols) - 1, sr_var_trials=float(sr.drop("base").var()),
                                 skew=float(M[best].skew()), kurt=float(M[best].kurt() + 3), ann=ann)
        rows.append({"strategy": f.stem, "pbo": p, "best_config_ex_post": best, "best_sharpe": sr[best],
                     "base_sharpe": sr["base"], "dsr_best": dsr})
        comp[f.stem] = sdf[["base"] + [c for c in sdf.columns if c.endswith("|selected") and "::" not in c]]
    t = pd.DataFrame(rows)
    if not t.empty:
        save(t, f"{run}_T16_pbo_dsr")
        SUMMARY["runs"][run]["pbo"] = {"median_pbo": float(t.pbo.median()), "mean_pbo": float(t.pbo.mean()),
                                       "n_pbo_above_half": int((t.pbo > 0.5).sum()),
                                       "n_dsr_above_095": int((t.dsr_best > 0.95).sum()), "n": int(len(t))}
    return comp


def composite(comp: dict, run: str):
    """Equal-weight multi-strategy composite: average daily return across all strategies,
    baseline vs each ML stage (selected model)."""
    if not comp:
        return None
    keys = ["base"] + [f"{s}|selected" for s in STAGES]
    cols = {}
    for k in keys:
        frames = [v[k] for v in comp.values() if k in v]
        if frames:
            cols[k] = pd.concat(frames, axis=1).mean(axis=1)
    cdf = pd.DataFrame(cols).dropna()
    ann = len(cdf) / ((cdf.index[-1] - cdf.index[0]).days / 365.25)
    base_sr = compute_metrics(cdf["base"], ann=ann)["sharpe"]
    rows = []
    for k in keys:
        if k not in cdf:
            continue
        m = compute_metrics(cdf[k], ann=ann)
        extra = {}
        if k != "base":
            bt = ST.sharpe_diff_bootstrap(cdf[k].values, cdf["base"].values, ann=ann, n_boot=5000)
            extra = {"d_sharpe": m["sharpe"] - base_sr, "p_boot_two": bt["p_two"], "ci_lo": bt["ci_lo"], "ci_hi": bt["ci_hi"]}
        rows.append({"config": k.replace("|selected", ""), **{x: m[x] for x in ["cagr", "total_return", "ann_vol", "sharpe", "sortino",
                                                                                 "calmar", "max_drawdown", "pct_pos_years", "stability_r2"]}, **extra})
    t = pd.DataFrame(rows)
    save(t, f"{run}_T17_composite_portfolio")
    SUMMARY["runs"][run]["composite"] = t.to_dict("records")
    cdf.to_csv(TAB / f"{run}_composite_daily_returns.csv")
    return cdf


# --------------------------------------------------------------------------- figures
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
                         "savefig.bbox": "tight", "axes.titlecolor": INK, "axes.titlesize": 10})
    return plt


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [RED, GRAY_MID, BLUE])


def fig_heatmap(pf: pd.DataFrame, run: str):
    plt = _mpl()
    from matplotlib.colors import TwoSlopeNorm
    piv = pf.pivot(index="strategy", columns="ml_stage", values="d_sharpe")[list(STAGES)]
    pv = pf.pivot(index="strategy", columns="ml_stage", values="p_bh")[list(STAGES)]
    order = pf[pf.ml_stage == "S1_filter"].set_index("strategy")["base_sharpe"].sort_values(ascending=False).index
    piv, pv = piv.loc[order], pv.loc[order]
    lim = float(np.nanmax(np.abs(piv.values))) or 1
    fig, ax = plt.subplots(figsize=(5.2, 0.19 * len(piv) + 1.2))
    im = ax.imshow(piv.values, cmap=_div_cmap(), norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(len(STAGES)), [STAGE_LABEL[s] for s in STAGES])
    ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                star = "*" if pv.values[i, j] < ALPHA else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=6, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("ΔSharpe (ML − rule-based)")
    ax.set_title("OOS ΔSharpe by strategy and ML stage\n(rows sorted by baseline Sharpe; * = BH-adjusted p<0.05)")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{run}_F1_heatmap_dsharpe.png")
    plt.close(fig)


def fig_distribution(pf, run):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 3))
    data = [pf[pf.ml_stage == s]["d_sharpe"].dropna().values for s in STAGES]
    ax.axhline(0, color=INK2, lw=1)
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=BLUE)
    for k in ("whiskers", "caps"):
        for l_ in bp[k]:
            l_.set(color=BLUE)
    for l_ in bp["medians"]:
        l_.set(color=INK, lw=1.5)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(d)), d, s=9, color=BLUE, alpha=0.6, lw=0)
    ax.set_xticks(range(1, len(STAGES) + 1), [STAGE_LABEL[s] for s in STAGES])
    ax.set_ylabel("ΔSharpe (selected model − baseline)")
    ax.set_title(f"Distribution of OOS ML improvement across strategies ({run})")
    fig.savefig(FIG / f"{run}_F2_dsharpe_distribution.png")
    plt.close(fig)


def fig_composite(cdf, run):
    if cdf is None:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    eq = np.log((1 + cdf).cumprod())
    ax.plot(eq.index, eq["base"], color=INK, lw=2, label="Rule-based (baseline)")
    for i, s in enumerate(STAGES):
        k = f"{s}|selected"
        if k in eq:
            ax.plot(eq.index, eq[k], color=SERIES[i], lw=1.4, label=STAGE_LABEL[s])
    ax.set_ylabel("log cumulative return")
    ax.set_title(f"Equal-weight composite of all strategies: baseline vs ML stages (OOS, {run})")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.savefig(FIG / f"{run}_F3_composite_equity.png")
    plt.close(fig)


def fig_models(e, run):
    plt = _mpl()
    pf = e[(e.level == "portfolio") & (e.ml_stage != "none")]
    g = pf.groupby(["ml_stage", "ml_model"])["d_sharpe"].mean().unstack()
    models = [m for m in (*MODEL_NAMES, "selected") if m in g.columns]
    stg = [s for s in STAGES if s in g.index]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.axhline(0, color=INK2, lw=1)
    w = 0.13
    for j, m in enumerate(models):
        col = INK if m == "selected" else SERIES[j]
        ax.bar(np.arange(len(stg)) + (j - len(models) / 2 + 0.5) * w, g.loc[stg, m].values, width=w * 0.85, color=col, label=m)
    ax.set_xticks(range(len(stg)), [STAGE_LABEL[s] for s in stg])
    ax.set_ylabel("mean ΔSharpe across strategies")
    ax.set_title("Model comparison per ML stage (OOS, portfolio level)")
    ax.legend(frameon=False, fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIG / f"{run}_F4_model_comparison.png")
    plt.close(fig)


def fig_markets(t, run):
    plt = _mpl()
    from matplotlib.colors import TwoSlopeNorm
    piv = t.pivot(index="market", columns="ml_stage", values="mean_dSharpe")[[s for s in STAGES if s in set(t.ml_stage)]]
    lim = float(np.nanmax(np.abs(piv.values))) or 1
    fig, ax = plt.subplots(figsize=(5.2, 0.35 * len(piv) + 1.2))
    im = ax.imshow(piv.values, cmap=_div_cmap(), norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(piv.shape[1]), [STAGE_LABEL[s] for s in piv.columns])
    ax.set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=7, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02).set_label("mean ΔSharpe")
    ax.set_title("Mean OOS ΔSharpe by market group (selected model)")
    fig.savefig(FIG / f"{run}_F5_market_heatmap.png")
    plt.close(fig)


def fig_sensitivity(e, run):
    se = e[(e.level.isin(["sensitivity", "portfolio"])) & (e.ml_model == "selected")]
    if se.empty:
        return
    plt = _mpl()
    g = se.groupby(["backtest", "ml_stage"])["d_sharpe"].mean().unstack()[[s for s in STAGES if s in set(se.ml_stage)]]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.axhline(0, color=INK2, lw=1)
    w = 0.8 / len(g)
    for j, (bt, row) in enumerate(g.iterrows()):
        ax.bar(np.arange(g.shape[1]) + (j - len(g) / 2 + 0.5) * w, row.values, width=w * 0.85, color=SERIES[j % 8], label=bt)
    ax.set_xticks(range(g.shape[1]), [STAGE_LABEL[s] for s in g.columns])
    ax.set_ylabel("mean ΔSharpe")
    ax.set_title("ML improvement under alternative backtest assumptions")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIG / f"{run}_F6_backtest_sensitivity.png")
    plt.close(fig)


# --------------------------------------------------------------------------- cross-run
def cross_run(runs: list[str]):
    rows = []
    for r in runs:
        e, _ = load(r)
        pf = e[(e.level == "portfolio") & (e.ml_model == "selected")]
        if pf.empty:
            continue
        pf = pf.assign(p_bh=ST.bh(pf.p_boot_two.to_numpy()))
        for stg, g in pf.groupby("ml_stage"):
            rows.append({"run": r, "stage": stg, "n_strategies": len(g), "mean_base_sharpe": g.base_sharpe.mean(),
                         "mean_dSharpe": g.d_sharpe.mean(), "median_dSharpe": g.d_sharpe.median(),
                         "share_improved": (g.d_sharpe > 0).mean(),
                         "sig_pos_bh": int(((g.p_bh < ALPHA) & (g.d_sharpe > 0)).sum()),
                         "sig_neg_bh": int(((g.p_bh < ALPHA) & (g.d_sharpe < 0)).sum())})
    t = pd.DataFrame(rows)
    if not t.empty:
        save(t, "T18_cross_run_robustness")
        SUMMARY["cross_run"] = t.to_dict("records")


def ablation_table(runs: list[str]):
    rows = []
    for r in [x for x in runs if x.startswith("ablation_")]:
        e, _ = load(r)
        pf = e[(e.level == "portfolio") & (e.ml_stage != "none")]
        for (stg, mdl), g in pf.groupby(["ml_stage", "ml_model"]):
            rows.append({"feature_block": r.replace("ablation_", ""), "stage": stg, "model": mdl, "n": len(g),
                         "mean_dSharpe": g.d_sharpe.mean(), "median_dSharpe": g.d_sharpe.median(),
                         "share_improved": (g.d_sharpe > 0).mean(),
                         "sig_pos": int(((g.p_boot_two < ALPHA) & (g.d_sharpe > 0)).sum()),
                         "sig_neg": int(((g.p_boot_two < ALPHA) & (g.d_sharpe < 0)).sum())})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    save(t, "T20_feature_family_ablation")
    SUMMARY["ablation"] = t.to_dict("records")
    sel = t[t.model == "selected"].pivot(index="feature_block", columns="stage", values="mean_dSharpe")
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 3))
    order = [b for b in ["all", "momentum", "volatility", "trend", "oscillator_channel", "other"] if b in sel.index]
    sel = sel.loc[order]
    w = 0.8 / max(len(sel.columns), 1)
    ax.axhline(0, color=INK2, lw=1)
    for j, stg in enumerate(sel.columns):
        ax.bar(np.arange(len(sel)) + (j - len(sel.columns) / 2 + 0.5) * w, sel[stg].values, width=w * 0.85,
               color=SERIES[STAGES.index(stg)], label=STAGE_LABEL[stg])
    ax.set_xticks(range(len(sel)), sel.index, rotation=15)
    ax.set_ylabel("mean ΔSharpe (selected model)")
    ax.set_title("Which indicator families carry the ML improvement? (ablation protocol)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "F7_feature_ablation.png")
    plt.close(fig)


def null_table(runs: list[str]):
    """Random-walk null test: ML 'improvements' on markets with no predictable structure."""
    rows = []
    for r in [x for x in runs if x.startswith("null_")]:
        e, _ = load(r)
        pf = e[(e.level == "portfolio") & (e.ml_stage != "none")]
        base = e[(e.level == "portfolio") & (e.ml_stage == "none")]
        ins = e[(e.level == "instrument") & (e.ml_model == "selected")]
        for stg, g in pf[pf.ml_model == "selected"].groupby("ml_stage"):
            allm = pf[pf.ml_stage == stg]
            gi = ins[ins.ml_stage == stg]
            rows.append({"run": r, "stage": stg, "n_strategies": len(g), "mean_base_sharpe": base.sharpe.mean(),
                         "mean_dSharpe_selected": g.d_sharpe.mean(), "mean_dSharpe_all_models": allm.d_sharpe.mean(),
                         "share_improved": (g.d_sharpe > 0).mean(),
                         "sig_pos_raw": int(((g.p_boot_two < ALPHA) & (g.d_sharpe > 0)).sum()),
                         "sig_neg_raw": int(((g.p_boot_two < ALPHA) & (g.d_sharpe < 0)).sum()),
                         "instrument_tests": len(gi),
                         "instrument_false_pos_rate": float(((gi.p_hac_two < ALPHA) & (gi.d_sharpe > 0)).mean()) if len(gi) else np.nan,
                         **{f"mean_dSharpe_{lab}": float(pd.to_numeric(e[(e.level == "sensitivity") & (e.ml_stage == stg)
                                                                            & (e.backtest == bt)].d_sharpe).mean())
                            for lab, bt in [("cost0", "next_open|cost0|vol_target0.1"),
                                            ("cost3", "next_open|cost3|vol_target0.1")]}})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    save(t, "T21_null_randomwalk")
    SUMMARY["null_test"] = t.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*")
    args = ap.parse_args()
    runs = args.run or R.read("select distinct run_id from experiments")["run_id"].tolist()
    data_tables()
    method_tables()
    for run in [r for r in runs if not r.startswith(("ablation_", "null_"))]:
        e, s = load(run)
        if e.empty:
            continue
        pf = headline(e, run)
        base_table(e, run)
        model_tables(e, s, run)
        t = market_tables(e, run)
        instrument_tables(e, run)
        period_sensitivity_tables(e, run)
        spa_table(e, run)
        placebo_tables(e, run)
        conditions_regression(e, run)
        comp = overfitting_tables(run)
        cdf = composite(comp, run)
        fig_heatmap(pf, run)
        fig_distribution(pf, run)
        fig_composite(cdf, run)
        fig_models(e, run)
        fig_markets(t, run)
        fig_sensitivity(e, run)
        print(f"report: {run} done ({len(e)} registry rows)")
    cross_run([r for r in runs if not r.startswith(("ablation_", "null_"))])
    ablation_table(runs)
    null_table(runs)
    (PAPER / "results_summary.json").write_text(
        json.dumps(SUMMARY, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    for t in ["experiments", "ml_selection", "runs", "placebo"]:
        try:
            R.read(f"select * from {t}").to_csv(R.RESULTS / f"registry_{t}.csv.gz", index=False, compression="gzip")
        except Exception:
            pass


if __name__ == "__main__":
    main()
