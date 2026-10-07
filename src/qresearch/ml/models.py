"""ML trade-classification models — consolidates ``train_ml_model`` (2 near-
identical notebook copies), ``live_trade_decision``, ``get_score``,
``label_signals``, ``classify_trade``, ``predict_signals``,
``align_signals_to_ohlcv``, and a generalized Optuna ``objective``.

Both ``train_ml_model`` copies hardcoded the same 5-column feature list
(RSI/VWAP/%K/%D/volume/close) and differed only in an XGBoost compat kwarg —
collapsed into one function taking ``feature_cols`` explicitly. The optuna
``objective`` originally supported LightGBM and CatBoost with a SMOTE
resampling step; CatBoost and imbalanced-learn are not in requirements.txt,
so only the LightGBM path (already a dependency) is ported — pass
``class_weight`` if the data is imbalanced instead of resampling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["RSI", "VWAP", "%K", "%D", "volume", "close"]


def train_classifier(df: pd.DataFrame, tradebook: pd.DataFrame, feature_cols: list[str] | None = None,
                     date_col: str = "date", entry_col: str = "Entry Time", pnl_col: str = "PNL",
                     test_size: float = 0.2, random_state: int = 42):
    """Train an XGBoost classifier to predict trade profitability from
    pre-computed indicator columns — ``train_ml_model``. Returns (model,
    held-out accuracy)."""
    from xgboost import XGBClassifier  # local import: optional dependency

    feature_cols = feature_cols or DEFAULT_FEATURES
    merged = pd.merge(df, tradebook, left_on=date_col, right_on=entry_col, how="inner")
    merged = merged.dropna(subset=feature_cols + [pnl_col])

    X, y = merged[feature_cols], (merged[pnl_col] > 0).astype(int)
    # Chronological split: the original random train_test_split mixed future trades into
    # the training set (look-ahead leakage on time-ordered data); the held-out set is now
    # strictly the last ``test_size`` fraction of trades by entry time.
    order = merged[entry_col].argsort(kind="stable").to_numpy()
    X, y = X.iloc[order], y.iloc[order]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model, float(model.score(X_test, y_test))


def predict_signals(df: pd.DataFrame, model, feature_cols: list[str] | None = None,
                    threshold: float = 0.7) -> pd.DataFrame:
    """Attach predicted profitability probability and filter high-confidence
    trades — ``live_trade_decision``/``predict_signals``."""
    feature_cols = feature_cols or DEFAULT_FEATURES
    out = df.copy()
    out["predicted_probability"] = model.predict_proba(out[feature_cols].astype(np.float32))[:, 1]
    return out[out["predicted_probability"] > threshold]


def label_trade_outcome(signal: pd.Series, future_return: pd.Series) -> pd.Series:
    """1 if a long signal (1) was followed by a positive return, or a short
    signal (2) by a negative one; 0 otherwise; NaN where signal==0 —
    ``classify_trade``."""
    out = pd.Series(np.nan, index=signal.index)
    out[(signal == 1) & (future_return > 0)] = 1
    out[(signal == 1) & (future_return <= 0)] = 0
    out[(signal == 2) & (future_return < 0)] = 1
    out[(signal == 2) & (future_return >= 0)] = 0
    return out


def label_signals(df: pd.DataFrame, tradebook: pd.DataFrame, ticker_col: str = "Ticker",
                  date_col: str = "Date", entry_col: str = "Entry Time") -> pd.DataFrame:
    """Label each non-flat signal bar with whether the matching tradebook
    entry exited via take-profit — ``label_signals``."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    tb = tradebook.copy()
    tb[entry_col] = pd.to_datetime(tb[entry_col])
    tb["Signal"] = tb["Direction"].str.lower().map({"long": 1, "short": 2})
    tb["label"] = (tb["Exit Reason"].fillna("") == "Take Profit").astype(int)

    merged = pd.merge(df[df["Signal"] != 0], tb[[entry_col, ticker_col, "Signal", "label"]],
                      left_on=[date_col, ticker_col, "Signal"], right_on=[entry_col, ticker_col, "Signal"], how="left")
    df.loc[merged.index, "label"] = merged["label"]
    df["label"] = df["label"].fillna(0).astype(int)
    return df[df["Signal"] != 0].copy()


def align_signals_to_ohlcv(ohlcv: pd.DataFrame, signals: pd.DataFrame, signal_col: str = "signal",
                           ticker_col: str = "Ticker", date_col: str = "Date") -> pd.DataFrame:
    """Left-join a filtered-signal frame back onto the full OHLCV frame,
    filling non-signal bars with 0 — ``align_signals_to_ohlcv``."""
    ohlcv, signals = ohlcv.copy(), signals.copy()
    ohlcv[date_col] = pd.to_datetime(ohlcv[date_col])
    signals[date_col] = pd.to_datetime(signals[date_col])
    merged = ohlcv.merge(signals[[ticker_col, date_col, signal_col]], on=[ticker_col, date_col], how="left")
    merged[signal_col] = merged[signal_col].fillna(0).astype(int)
    return merged


def get_score(model, x_train, x_test, y_train, y_test) -> float:
    """Fit and score a model in one call — ``get_score``."""
    model.fit(x_train, y_train)
    return float(model.score(x_test, y_test))


def lgbm_objective(trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 3, random_state: int = 42) -> float:
    """Optuna objective: mean CV ROC-AUC for a LightGBM classifier over a
    standard hyperparameter search space — generalized ``objective``
    (LightGBM path only; see module docstring)."""
    import lightgbm as lgb

    params = {
        "objective": "binary", "metric": "auc", "verbosity": -1, "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": random_state, "n_jobs": -1,
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_val = scaler.transform(X.iloc[val_idx])
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y.iloc[train_idx])
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y.iloc[val_idx], preds))
    return float(np.mean(scores))


def walk_forward_topn(features: pd.DataFrame, target_col: str = "forward_return", group_col: str = "quarter",
                      ticker_col: str = "ticker", top_n: int = 10, min_train_periods: int = 3) -> pd.DataFrame:
    """Walk-forward regression: train on all periods before period *t*,
    predict period *t*, keep the top-N tickers by predicted return each
    period. Generalizes ``step_forward_prediction_with_top_10_tickers`` off
    its hardcoded quarterly grouping. Non-feature columns to exclude are
    inferred as ``[group_col, ticker_col, target_col]`` plus any datetime
    column named "Date"."""
    from xgboost import XGBRegressor  # local import: optional dependency

    periods = sorted(features[group_col].unique())
    drop_cols = [c for c in (group_col, ticker_col, target_col, "Date") if c in features.columns]
    results = []
    for i in range(min_train_periods, len(periods)):
        train = features[features[group_col].isin(periods[:i])]
        test = features[features[group_col] == periods[i]]
        X_train, y_train = train.drop(columns=drop_cols), train[target_col]
        X_test = test.drop(columns=drop_cols)

        scaler = StandardScaler()
        X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train_s, y_train)

        preds = test[[c for c in (group_col, ticker_col) if c in test.columns]].copy()
        preds["predicted_return"] = model.predict(X_test_s)
        preds["actual_return"] = test[target_col].values
        results.append(preds.sort_values("predicted_return", ascending=False).head(top_n))

    return pd.concat(results).reset_index(drop=True) if results else features.iloc[0:0]
