"""End-to-end test of the walk-forward ML pipeline on a small synthetic panel:
training labels are realised before validation, OOS positions live only in test
blocks, the baseline is reproduced exactly, and evaluation produces registry rows."""
import numpy as np
import pandas as pd
import pytest

from qresearch.tfml import pipeline as PL
from qresearch.tfml import strategies as S
from qresearch.tfml.features import compute_features
from qresearch.tfml.panel import Panel


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-01", periods=n)
    lr = rng.normal(0.0002, 0.01, n) + 0.003 * np.sin(np.arange(n) / 60 + seed)
    c = 100 * np.exp(np.cumsum(lr))
    o = c * np.exp(rng.normal(0, 0.002, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.003, "low": np.minimum(o, c) * 0.997,
                         "close": c, "volume": 1000.0}, index=idx)


@pytest.fixture(scope="module")
def panel():
    frames = [_frame(1800, s) for s in range(4)]
    X = np.concatenate([compute_features(f).to_numpy(np.float32) for f in frames])
    lens = np.array([len(f) for f in frames])
    big = pd.concat(frames)
    meta = pd.DataFrame({"id": [f"S{i}" for i in range(4)], "group": ["g1", "g1", "g2", "g2"],
                         "asset_class": ["equity", "equity", "fx", "fx"], "region": "x",
                         "first_date": [f.index[0] for f in frames], "last_date": [f.index[-1] for f in frames]})
    inst = np.repeat(np.arange(4), lens).astype(np.int32)
    pos = np.concatenate([np.arange(l) for l in lens])
    return Panel("daily", meta, inst, big.index.to_numpy(), pos, pos == 0, big.open.to_numpy(), big.high.to_numpy(),
                 big.low.to_numpy(), big.close.to_numpy(), big.volume.to_numpy(), np.full(len(big), 2e-4), X, 252.0,
                 np.r_[0, np.cumsum(lens)])


def test_pipeline_end_to_end(panel, tmp_path):
    cfg = PL.RunConfig(run_id="t", first_test_year=2009, last_test_year=2010, val_years=1,
                       models=("logreg", "lgbm"), max_train_rows=5000, subperiods=((2009, 2010),))
    regime = PL.fit_regime_models(panel, cfg, log=lambda *a: None)
    strat = S.get("ema_12_26")
    res = PL.run_strategy(strat, panel, cfg, regime, log=lambda *a: None)
    test = (panel.date >= np.datetime64("2009-01-01")) & (panel.date < np.datetime64("2011-01-01"))
    assert (res.oos_mask == test).all()
    base = np.concatenate([strat.positions(panel.frame(k)).to_numpy() for k in range(4)])
    assert np.allclose(res.positions[("base", None)][test], base[test])
    for k, p in res.positions.items():
        assert (p[~test] == 0).all(), k             # nothing leaks outside the OOS blocks
        assert np.isfinite(p).all() and np.abs(p).max() <= 2.0 + 1e-9
    sel = pd.DataFrame(res.selection)
    assert set(sel.stage) == set(PL.STAGES)
    assert (sel.groupby(["fold", "stage"]).selected.sum() == 1).all()   # exactly one model chosen
    # S1 filter can only remove trades: wherever it holds a position it equals the baseline
    s1 = res.positions[("S1_filter", "selected")]
    nz = s1 != 0
    assert np.allclose(s1[nz], base[nz])

    from qresearch.tfml.evaluate import evaluate
    import qresearch.tfml.registry as R
    import shutil
    out = R.RESULTS / "_pytest_tmp"            # result paths are stored repo-relative
    try:
        rows, series = evaluate(res, panel, cfg, strat, "hash", out)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    df = pd.DataFrame(rows)
    assert {"portfolio", "market", "instrument", "subperiod", "sensitivity", "spa"} <= set(df.level)
    pf = df[(df.level == "portfolio") & (df.ml_stage != "none")]
    assert pf["d_sharpe"].notna().all() and pf["p_boot_two"].between(0, 1).all()
    for col in ["strategy", "indicators", "market", "instrument", "dataset", "ml_stage", "ml_model",
                "validation", "backtest", "sharpe", "result_path", "exp_id"]:
        assert col in df.columns
    assert df.exp_id.is_unique


def test_training_labels_realised_before_validation(panel):
    cfg = PL.RunConfig(first_test_year=2009, last_test_year=2009, val_years=1)
    f = PL.folds(cfg, panel.date.max())[0]
    label_row = np.arange(panel.n) + cfg.h_bar + cfg.embargo_bars
    ld = PL._label_date(panel, label_row)
    tr = PL._train_rows(panel, cfg, ld, f, np.ones(panel.n, bool))
    assert tr.any()
    assert (ld[tr] < f["val_start"]).all()
    assert (panel.date[tr] < f["val_start"]).all()
    # rows whose label window crosses the instrument end have no label
    last = panel.bounds[1:] - 1
    assert np.isnat(ld[last]).all()
