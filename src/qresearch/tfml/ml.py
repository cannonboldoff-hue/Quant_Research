"""ML model zoo and the decision rules that turn model outputs into positions.

Model hyper-parameters are fixed a priori (no tuning on test data, and no tuning
at all beyond the validation-based choice of model and intervention intensity),
which limits the researcher degrees of freedom the study has to account for.
"""
from __future__ import annotations

import warnings

import numba
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("logreg", "rf", "et", "lgbm", "mlp")


def make_model(name: str, seed: int = 0, n_jobs: int = 1, n_train: int = 100_000):
    # leaf sizes scale with the training set: ~0.5% of rows, within [20, 200]
    leaf = int(np.clip(n_train // 200, 20, 200))
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=500))
    if name == "rf":
        # Random forest (bagged, feature-subsampled, un-boosted trees) via LightGBM's "rf"
        # mode -- same estimator family as sklearn's RandomForest, ~5x faster at this size.
        from lightgbm import LGBMClassifier
        return LGBMClassifier(boosting_type="rf", n_estimators=100, num_leaves=64, max_depth=7,
                              min_child_samples=leaf, subsample=0.25, subsample_freq=1,
                              colsample_bytree=0.15, n_jobs=n_jobs, random_state=seed, verbose=-1)
    if name == "et":
        return make_pipeline(SimpleImputer(strategy="median"),
                             ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_leaf=leaf,
                                                  max_features="sqrt", bootstrap=True, max_samples=0.25,
                                                  n_jobs=n_jobs, random_state=seed))
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=leaf,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                              n_jobs=n_jobs, random_state=seed, verbose=-1)
    if name == "mlp":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, batch_size=512,
                                           learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                                           n_iter_no_change=5, random_state=seed))
    raise ValueError(name)


def fit_predict(name: str, X_tr, y_tr, X_pred_list, seed: int = 0):
    """Fit a model and return predicted class-probability arrays for each X in X_pred_list.
    Binary -> P(y=1) vector; multiclass -> (n, k) matrix ordered by class label."""
    m = make_model(name, seed, n_train=len(y_tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_tr, y_tr)
        classes = list(m.classes_)
        outs = []
        for X in X_pred_list:
            if len(X) == 0:
                outs.append(np.zeros((0,)) if len(classes) <= 2 else np.zeros((0, len(classes))))
                continue
            P = m.predict_proba(X)
            if len(classes) == 2:
                outs.append(P[:, 1])
            else:
                full = np.zeros((len(X), 3))
                for j, c in enumerate(classes):
                    full[:, int(c)] = P[:, j]
                outs.append(full)
    return outs


# --------------------------------------------------------------------------- decision rules (numba)
@numba.njit(cache=True)
def trade_starts(p, inst_start):
    n = p.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        s = np.sign(p[i])
        if s != 0 and (inst_start[i] or np.sign(p[i - 1]) != s):
            out[i] = True
    return out


@numba.njit(cache=True)
def apply_filter(p, starts, q, thr):
    """S1 meta-label filter: at each trade start, take the trade iff q >= thr
    (q NaN -> no model opinion -> take). Skipped trades stay flat until the next start."""
    n = p.shape[0]
    out = np.zeros(n)
    take = True
    for i in range(n):
        if starts[i]:
            take = np.isnan(q[i]) or q[i] >= thr
        if p[i] == 0:
            out[i] = 0.0
        else:
            out[i] = p[i] if take else 0.0
    return out


@numba.njit(cache=True)
def apply_exit(p, starts, q, thr):
    """S4 ML exit: inside a trade, exit when q < thr; stay flat until the next trade start."""
    n = p.shape[0]
    out = np.zeros(n)
    exited = False
    for i in range(n):
        if starts[i]:
            exited = False
        if p[i] == 0:
            exited = False
            continue
        if not exited and not np.isnan(q[i]) and q[i] < thr:
            exited = True
        out[i] = 0.0 if exited else p[i]
    return out


@numba.njit(cache=True)
def smooth_predictions(q, inst_start, span):
    """Causal EMA of model outputs within each contiguous run of predictions (reset at
    instrument starts and NaN gaps). Applied before every bar-level decision rule so that
    decisions do not churn on day-to-day prediction noise. span is fixed a priori (5)."""
    a = 2.0 / (span + 1.0)
    out = np.full(q.shape[0], np.nan)
    prev = np.nan
    for i in range(q.shape[0]):
        if inst_start[i]:
            prev = np.nan
        x = q[i]
        if np.isnan(x):
            prev = np.nan
            continue
        prev = x if np.isnan(prev) else a * x + (1 - a) * prev
        out[i] = prev
    return out


def apply_gate(p, q, thr):
    """S2 regime gate: hold the base position only when P(trending regime) >= thr."""
    keep = np.isnan(q) | (q >= thr)
    return np.where(keep, p, 0.0)


def apply_sizing(p, q, ref):
    """S3 sizing: scale by 2*F_ref(q), F_ref = empirical CDF of the validation-period
    predictions (mean multiplier ~1, range [0, 2])."""
    ref = np.sort(ref[~np.isnan(ref)])
    if len(ref) == 0:
        return p.copy()
    w = np.searchsorted(ref, np.nan_to_num(q, nan=np.median(ref)), side="right") / len(ref)
    w = np.where(np.isnan(q), 1.0, 2.0 * w)
    return p * w


@numba.njit(cache=True)
def apply_variant(pv, probs, inst_pos, rebalance_every, default_v):
    """S5 parameter adaptation: every ``rebalance_every`` bars choose the variant with
    the highest predicted probability of being best; hold its positions until the next
    rebalance. pv: (n, 3) variant positions; probs: (n, 3) (NaN rows -> default)."""
    n = pv.shape[0]
    out = np.zeros(n)
    v = default_v
    for i in range(n):
        if inst_pos[i] == 0:
            v = default_v
        if inst_pos[i] % rebalance_every == 0:
            if np.isnan(probs[i, 0]):
                v = default_v
            else:
                best = 0
                for j in range(1, 3):
                    if probs[i, j] > probs[i, best]:
                        best = j
                v = best
        out[i] = pv[i, v]
    return out
