"""Tests for qresearch.tfml: causality (no look-ahead) of every feature and
strategy, backtest accounting, ML decision rules, fold purging and statistics.
Synthetic data only -- runs without downloaded datasets."""
import numpy as np
import pandas as pd
import pytest

from qresearch.tfml import backtest as B
from qresearch.tfml import ml, stats as ST, strategies as S
from qresearch.tfml.features import FEATURE_NAMES, compute_features
from qresearch.tfml.metrics import compute_metrics, trade_returns
from qresearch.tfml.pipeline import RunConfig, folds


def _ohlcv(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    lr = rng.normal(0.0003, 0.012, n) + 0.002 * np.sin(np.arange(n) / 40)
    c = 100 * np.exp(np.cumsum(lr))
    o = c * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    v = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


@pytest.fixture(scope="module")
def df():
    return _ohlcv()


def test_features_are_causal(df):
    full = compute_features(df)
    assert list(full.columns) == FEATURE_NAMES and len(FEATURE_NAMES) >= 30
    for cut in (500, 700):
        part = compute_features(df.iloc[:cut])
        a, b = full.iloc[cut - 1], part.iloc[-1]
        both = a.notna() & b.notna()
        assert np.allclose(a[both], b[both], atol=1e-5), (a - b)[both].abs().sort_values().tail()
        assert (a.isna() == b.isna()).all()


@pytest.mark.parametrize("strat", S.STRATEGIES, ids=lambda s: s.id)
def test_strategy_positions_are_causal_and_bounded(df, strat):
    ctx = {"rf": pd.Series(0.02, index=df.index)}
    full = strat.positions(df, ctx)
    assert full.between(-1, 1).all()
    if strat.long_only:
        assert (full >= 0).all()
    for cut in (450, 650, 899):
        part = strat.positions(df.iloc[:cut], ctx)
        assert part.iloc[-1] == pytest.approx(full.iloc[cut - 1]), f"{strat.id} looks ahead at {cut}"


def test_at_least_30_strategies_with_references():
    assert len(S.STRATEGIES) >= 30
    assert len({s.id for s in S.STRATEGIES}) == len(S.STRATEGIES)
    assert all(s.reference for s in S.STRATEGIES)


def test_variant_params_scale_lookbacks():
    s = S.get("sma_50_200")
    assert s.variant_params(2.0)["slow"] == 400 and s.variant_params(0.5)["fast"] == 25
    ps = S.get("psar")
    assert ps.variant_params(2.0)["af"] == pytest.approx(0.01)  # slower = smaller acceleration


def test_next_open_accounting():
    idx = pd.bdate_range("2020-01-01", periods=5)
    d = pd.DataFrame({"open": [10, 11, 12, 13, 14.0], "close": [10.5, 11.5, 12.5, 13.5, 14.5],
                      "high": 15.0, "low": 9.0, "volume": 0.0}, index=idx)
    target = pd.Series([1, 1, 0, 0, 0.0], index=idx)        # long decided at close of bar 0
    sim = B.simulate(d, target, cost_per_side_bps=10)
    # bar 1: entered at open 11 -> earns 11.5/11-1, pays 10bps
    assert sim["net"].iloc[1] == pytest.approx(11.5 / 11 - 1 - 0.001)
    # bar 2: still long (target at bar 1 was 1): gap + intraday, no trade
    assert sim["net"].iloc[2] == pytest.approx((12 / 11.5 - 1) + (12.5 / 12 - 1))
    # bar 3: overnight gap earned by the old long, flat intraday, exit cost
    assert sim["net"].iloc[3] == pytest.approx(13 / 12.5 - 1 - 0.001)
    assert sim["net"].iloc[0] == 0 and sim["net"].iloc[4] == 0
    # same-close convention earns bar 1 close-to-close with the bar-0 signal
    sc = B.simulate(d, target, 0, execution="same_close")
    assert sc["net"].iloc[1] == pytest.approx(11.5 / 10.5 - 1)


def test_signal_cannot_earn_its_own_gap():
    idx = pd.bdate_range("2020-01-01", periods=3)
    d = pd.DataFrame({"open": [10, 20, 20.0], "close": [10, 20, 20.0], "high": 20.0, "low": 10.0,
                      "volume": 0.0}, index=idx)
    sim = B.simulate(d, pd.Series([1, 1, 1.0], index=idx), 0)
    assert sim["net"].sum() == pytest.approx(0.0)   # the jump happens overnight before entry


def test_trade_returns_and_metrics():
    net = np.array([0, 0.01, 0.02, 0, -0.01, -0.01, 0])
    held = np.array([0, 1, 1, 0, -1, -1, 0])
    tr = trade_returns(net, held)
    assert len(tr) == 2 and tr[0] == pytest.approx(1.01 * 1.02 - 1)
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 1000), index=pd.bdate_range("2010-01-01", periods=1000))
    m = compute_metrics(r)
    assert m["max_drawdown"] <= 0 and np.isfinite(m["sharpe"]) and m["years"] > 3


def test_decision_rules():
    p = np.array([0, 1, 1, 1, 0, -1, -1, 1, 1.0])
    inst_start = np.zeros(len(p), bool); inst_start[0] = True
    starts = ml.trade_starts(p, inst_start)
    assert starts.tolist() == [False, True, False, False, False, True, False, True, False]
    q = np.array([np.nan, 0.2, 0.9, 0.9, np.nan, 0.8, 0.1, 0.6, 0.6])
    f = ml.apply_filter(p, starts, q, 0.5)
    assert f.tolist() == [0, 0, 0, 0, 0, -1, -1, 1, 1]          # first trade skipped entirely
    e = ml.apply_exit(p, starts, q, 0.5)
    assert e.tolist() == [0, 0, 0, 0, 0, -1, 0, 1, 1]           # exits stay flat until next start
    g = ml.apply_gate(p, q, 0.5)
    assert g.tolist() == [0, 0, 1, 1, 0, -1, 0, 1, 1]
    s = ml.apply_sizing(p, q, np.array([0.1, 0.5, 0.9]))
    assert s.max() <= 2 and s[2] == pytest.approx(2.0)


def test_folds_are_ordered_and_disjoint():
    cfg = RunConfig(first_test_year=2010, last_test_year=2013)
    fs = folds(cfg, np.datetime64("2013-06-30", "ns"))
    assert [f["label"] for f in fs] == ["2010", "2011", "2012", "2013"]
    for f in fs:
        assert f["val_start"] < f["test_start"] < f["test_end"]
    cfg_r = RunConfig(validation="wf_rolling", rolling_train_years=5, first_test_year=2010, last_test_year=2010)
    f = folds(cfg_r, np.datetime64("2012-01-01", "ns"))[0]
    assert f["train_start"] == np.datetime64("2003-01-01", "ns")


def test_sharpe_difference_tests():
    rng = np.random.default_rng(0)
    a = rng.normal(0.001, 0.01, 3000)
    d, z, p = ST.sharpe_diff_hac(a, a.copy())
    assert d == pytest.approx(0) and p == pytest.approx(1)
    b = rng.normal(0.0, 0.01, 3000)
    d, z, p = ST.sharpe_diff_hac(a + 0.002, b)
    assert d > 0 and p < 0.01
    bt = ST.sharpe_diff_bootstrap(a + 0.002, b, n_boot=300)
    assert bt["d_sharpe"] > 0 and bt["p_one"] < 0.05


def test_multiple_testing_adjustments():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    from statsmodels.stats.multitest import multipletests
    assert np.allclose(ST.holm(p), multipletests(p, method="holm")[1])
    assert np.allclose(ST.bh(p), multipletests(p, method="fdr_bh")[1])
    assert np.allclose(ST.bh(p, dependent=True), multipletests(p, method="fdr_by")[1])


def test_spa_detects_superior_and_null_models():
    rng = np.random.default_rng(3)
    null = rng.normal(0, 0.01, (2000, 5))
    assert ST.spa_test(null, n_boot=300)["p_spa"] > 0.05
    good = null.copy(); good[:, 2] += 0.002
    assert ST.spa_test(good, n_boot=300)["p_spa"] < 0.05


def test_mcginley_is_stable_after_crash():
    """Regression: the uncapped McGinley recursion diverged (inf / ZeroDivisionError) after a
    large price drop relative to the average (seen on WIPRO.NS and ETH-USD with n=7)."""
    from qresearch.tfml import indicators as I
    p = pd.Series(np.r_[np.full(50, 100.0), np.full(50, 20.0), np.linspace(20, 60, 100)])
    md = I.mcginley(p, 7)
    assert np.isfinite(md).all()
    assert md.between(p.min() - 1e-9, p.max() + 1e-9).all()     # never overshoots the price range
