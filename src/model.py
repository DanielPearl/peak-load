"""Natural-gas price model — quantile-GBM ensemble + per-strike classifiers.

Same shape as the retail-gas-prices and unemployment-claims bots so all
three can be reasoned about identically:

  • Quantile GBMs at {0.05, 0.25, 0.50, 0.75, 0.95}, each side a small
    ensemble of seeded GBMs wrapped in `_EnsembleRegressor`. The median
    (q=0.50) gets an ElasticNet meta-voice for stability on small-N
    training sets.
  • Walk-forward permutation-importance feature selection + correlation
    pruning. Trained on the FULL feature set, model keeps only the
    features that look stable across folds and aren't redundant with a
    higher-importance neighbor.
  • Per-strike classifier ensemble: one HistGradientBoostingClassifier
    (with class_weight='balanced' + isotonic holdout calibration) per
    $/MMBTU strike on the training grid. ``prob_above(strike)`` looks
    up the closest trained classifier and isotonic-interpolates between
    adjacent strikes when needed.
  • Residual std (robust MAD-based) on the median's test-set residuals
    — used both as a Gaussian fallback when no classifiers train and
    to size the dynamic threshold grid in run_daily.py.

The natural-gas-price problem is a forecasting-the-LEVEL problem
(Kalshi resolves on `henry_hub_spot >= threshold` at 5pm EDT each day),
so the per-strike classifier target is the cleanest framing — each
classifier trains directly on the binary outcome it'll be priced
against.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


# --------------------------------------------------------------------------- #
# Feature selection — walk-forward permutation importance + correlation prune
# --------------------------------------------------------------------------- #

def _walk_forward_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Walk-forward permutation importance.

    Trains a quick HGB regressor on each TimeSeriesSplit fold, measures
    permutation importance on the held-out tail of that fold, and
    returns:
        mean_importance — dict[feature -> mean perm importance across folds]
        positive_folds  — dict[feature -> # of folds where importance > 0]

    The two-channel return lets the caller require BOTH a meaningful
    average AND stability across folds — which is more honest than
    importance alone (a feature can have a positive mean while being
    near-zero in 4/5 folds and a fluke giant in the 5th).
    """
    importances: Dict[str, List[float]] = {c: [] for c in X.columns}
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_i, (tr, te) in enumerate(splitter.split(X)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05,
                max_depth=4, l2_regularization=0.1,
                random_state=random_state + fold_i,
            )),
        ])
        model.fit(X_tr, y_tr)
        perm = permutation_importance(
            model, X_te, y_te, n_repeats=8,
            random_state=random_state + fold_i,
            scoring="neg_mean_absolute_error", n_jobs=1,
        )
        for col, imp in zip(X.columns, perm.importances_mean):
            importances[col].append(float(imp))
    mean_imp = {c: float(np.mean(v)) for c, v in importances.items()}
    pos_folds = {c: int(sum(1 for x in v if x > 0))
                 for c, v in importances.items()}
    return mean_imp, pos_folds


def _correlation_prune(
    X: pd.DataFrame,
    keep_order: List[str],
    correlation_max: float = 0.95,
) -> List[str]:
    """Drop features whose abs correlation > ``correlation_max`` with an
    already-kept feature. ``keep_order`` should be ranked by importance,
    so the more-important version of a redundant pair survives.
    """
    kept: List[str] = []
    if not keep_order:
        return kept
    corr = X[keep_order].corr().abs()
    for c in keep_order:
        if any(corr.loc[c, k] > correlation_max for k in kept):
            continue
        kept.append(c)
    return kept


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 30,
    min_positive_folds: int = 3,
    correlation_max: float = 0.95,
    n_splits: int = 5,
    random_state: int = 42,
    importance_csv_path: Optional[str] = None,
) -> List[str]:
    """Pipeline: walk-forward importance → stability filter → corr prune
    → top-N. Returns the surviving feature columns.

    Defaults are tuned for the NG-price problem: ~50 candidate features
    (weather + storage + production + lags + calendar), keep top 30 to
    leave headroom for the GBMs to find subtle interactions without
    overfitting noise.
    """
    mean_imp, pos_folds = _walk_forward_feature_importance(
        X, y, n_splits=n_splits, random_state=random_state)
    n_total = len(X.columns)
    # Stability: importance > 0 in at least N of K folds AND mean > 0.
    stable = [c for c in X.columns
              if pos_folds[c] >= min_positive_folds and mean_imp[c] > 0]
    log.info("feature selection: %d/%d features stable "
             "(positive in >= %d/%d folds AND mean > 0)",
             len(stable), n_total, min_positive_folds, n_splits)
    stable_sorted = sorted(stable, key=lambda c: mean_imp[c], reverse=True)
    after_corr = _correlation_prune(X, stable_sorted,
                                     correlation_max=correlation_max)
    log.info("feature selection: %d -> %d after correlation prune (>%.2f)",
             len(stable_sorted), len(after_corr), correlation_max)
    selected = after_corr[:max_features]
    log.info("feature selection: kept top %d (max_features=%d)",
             len(selected), max_features)
    if importance_csv_path:
        rows = pd.DataFrame({
            "feature": list(X.columns),
            "mean_importance": [mean_imp[c] for c in X.columns],
            "positive_folds": [pos_folds[c] for c in X.columns],
            "selected": [c in selected for c in X.columns],
        }).sort_values("mean_importance", ascending=False)
        try:
            Path(importance_csv_path).parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(importance_csv_path, index=False)
            log.info("feature importance audit written to %s",
                     importance_csv_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write importance audit: %s", exc)
    return selected


# --------------------------------------------------------------------------- #
# Ensemble wrappers
# --------------------------------------------------------------------------- #

@dataclass
class _EnsembleRegressor:
    """Mean of several seeded GBMs + an optional ElasticNet meta voice.
    Same shape as the gas-prices / claims version."""
    members: List[Pipeline]
    meta_model: Optional[Pipeline] = None
    meta_weight: float = 0.0

    def predict(self, X):
        gbm_pred = np.mean([m.predict(X) for m in self.members], axis=0)
        if self.meta_model is None or self.meta_weight <= 0:
            return gbm_pred
        meta_pred = self.meta_model.predict(X)
        return (1.0 - self.meta_weight) * gbm_pred + self.meta_weight * meta_pred


@dataclass
class _ClassifierEnsemble:
    """Mean of several seeded calibrated classifiers — binary direction."""
    members: List

    def predict_proba(self, X) -> np.ndarray:
        """Return P(class=1) for every row in X."""
        ps = np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)
        return np.asarray(ps, dtype=float)


def _interpolate_threshold_prob(
    threshold_probs: Dict[float, float],
    strike: float,
) -> float:
    """Look up P(price >= strike) from the trained-threshold dict.

    Exact match → use that classifier's output. Otherwise linear
    interpolation between the two nearest trained thresholds. Outside
    the trained range → use the closest-trained probability (clipped).
    Probabilities are guaranteed monotone non-increasing in strike since
    'price >= $3.00' must be at least as likely as 'price >= $3.50' —
    we isotonic-fix any small reversals from independent classifiers.
    """
    if not threshold_probs:
        return 0.5
    sorted_t = sorted(threshold_probs.keys())
    sorted_p = [threshold_probs[t] for t in sorted_t]
    fixed_p = list(sorted_p)
    for i in range(1, len(fixed_p)):
        fixed_p[i] = min(fixed_p[i], fixed_p[i - 1])
    if strike <= sorted_t[0]:
        return max(0.01, min(0.99, fixed_p[0]))
    if strike >= sorted_t[-1]:
        return max(0.01, min(0.99, fixed_p[-1]))
    for i in range(len(sorted_t) - 1):
        lo, hi = sorted_t[i], sorted_t[i + 1]
        if lo <= strike <= hi:
            w = (strike - lo) / (hi - lo) if hi > lo else 0.0
            p = (1.0 - w) * fixed_p[i] + w * fixed_p[i + 1]
            return max(0.01, min(0.99, p))
    return 0.5


# --------------------------------------------------------------------------- #
# Trained-model artifact
# --------------------------------------------------------------------------- #

@dataclass
class NatGasModel:
    """Trained model bundle for the Natural Gas Price bot.

    Public surface (used by run_daily.py / dashboard):
      • predict(X)                  → median forecast ($/MMBTU)
      • prob_above(X, threshold)    → P(price >= threshold) per row
      • predict_quantiles(X)        → all five quantile predictions
      • threshold_probabilities()   helper still works via prob_above
      • feature_columns / residual_std / metrics / model_name kept for
        cross-bot dashboard compatibility.
    """
    feature_columns: List[str]
    quantile_models: Dict[float, _EnsembleRegressor]
    threshold_classifiers: Dict[float, _ClassifierEnsemble]
    threshold_grid: List[float]
    residual_std: float
    train_end_date: pd.Timestamp
    metrics: Dict[str, float] = field(default_factory=dict)
    model_name: str = "quantile_gbm_ensemble"

    # ---- core inference ----------------------------------------------- #

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Median-quantile point forecast of NG spot ($/MMBTU)."""
        row = X[self.feature_columns]
        return self.quantile_models[0.50].predict(row)

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Return {q: forecast_array} for every trained quantile."""
        row = X[self.feature_columns]
        return {q: self.quantile_models[q].predict(row) for q in QUANTILES}

    def prob_above(self, X: pd.DataFrame, threshold_usd: float) -> np.ndarray:
        """P(price >= threshold_usd) for each row in X.

        Path A — per-strike classifiers (trained, the honest path): for
        each row, evaluate every trained threshold classifier, then
        isotonic-interpolate at ``threshold_mw``.

        Path B — fallback (no classifiers, e.g. tiny dataset): Gaussian
        residuals around the median forecast.
        """
        row = X[self.feature_columns]
        if self.threshold_classifiers:
            n_rows = len(row)
            out = np.empty(n_rows, dtype=float)
            # Pre-compute per-trained-threshold prob arrays once, then
            # interp per row. Cheaper than the obvious double-loop.
            per_thr = {
                thr: clf.predict_proba(row)
                for thr, clf in self.threshold_classifiers.items()
            }
            for i in range(n_rows):
                tp_row = {thr: float(p[i]) for thr, p in per_thr.items()}
                out[i] = _interpolate_threshold_prob(tp_row, threshold_usd)
            return out
        # Gaussian fallback. residual_std drives the spread.
        forecast = self.predict(X)
        sigma = max(self.residual_std, 1e-6)
        z = (threshold_usd - forecast) / sigma
        return 1.0 - norm.cdf(z)


# --------------------------------------------------------------------------- #
# Pipeline builders
# --------------------------------------------------------------------------- #

def _make_quantile_pipeline(alpha: float, random_state: int = 42,
                             n_estimators: int = 400) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=10,
            random_state=random_state,
        )),
    ])


def _make_classifier_pipeline(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])


# --------------------------------------------------------------------------- #
# Threshold training grid (data-derived for NG prices)
# --------------------------------------------------------------------------- #

def _default_training_grid(
    series: "pd.Series",
    step_usd: float = 0.10,
) -> List[float]:
    """Build a wide threshold grid for per-strike classifier training.

    Spans the 5th-95th percentile of historical prices, snapped to
    ``step_usd`` boundaries. For NG, that's typically $1.50-$8.00 in
    $0.10 ticks (~65 strikes) — wide enough to cover any plausible
    Kalshi strike, narrow enough to keep training time reasonable.
    """
    lo = float(np.floor(series.quantile(0.05) / step_usd) * step_usd)
    hi = float(np.ceil(series.quantile(0.95) / step_usd) * step_usd)
    return [round(lo + i * step_usd, 3)
            for i in range(int(round((hi - lo) / step_usd)) + 1)]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    *,
    test_size_days: int = 120,
    walk_forward_splits: int = 5,
    ensemble_seeds: int = 5,
    classifier_seeds: int = 3,
    calibration_holdout_days: int = 60,
    random_state: int = 42,
    importance_csv_path: Optional[str] = None,
    meta_model_weight: float = 0.25,
    threshold_training_grid: Optional[List[float]] = None,
    threshold_training_step_usd: float = 0.10,
    max_features: int = 30,
    feature_min_positive_folds: int = 3,
    feature_correlation_max: float = 0.95,
) -> NatGasModel:
    """Train the natural-gas-price model end-to-end.

    Steps:
      1. Hold out the last `test_size_days` for OOS metrics.
      2. Walk-forward feature selection + correlation pruning.
      3. ElasticNet meta-model on the median target.
      4. Quantile-GBM ensemble at five quantiles.
      5. Per-strike classifier ensemble across the $/MMBTU training
         grid (one classifier per strike, calibrated).
      6. Residual std on test-set residuals (Gaussian fallback +
         dynamic-grid sizing in run_daily.py).

    Returns a `NatGasModel` ready for save_model / inference.
    """
    df = df.dropna(subset=["target"]).copy()
    if len(df) <= test_size_days + 10:
        raise ValueError(
            f"Not enough rows: {len(df)} <= test_size_days={test_size_days} + 10")

    train = df.iloc[:-test_size_days].copy()
    test = df.iloc[-test_size_days:].copy()

    X_train_raw = train[feature_columns]
    X_test_raw = test[feature_columns]
    y_train = train["target"]
    y_test = test["target"]

    # ---- 1. Walk-forward feature selection -------------------------- #
    log.info("walk-forward feature selection over %d splits "
             "(start: %d candidate features)",
             walk_forward_splits, len(feature_columns))
    selected = select_features(
        X_train_raw, y_train,
        max_features=max_features,
        min_positive_folds=feature_min_positive_folds,
        correlation_max=feature_correlation_max,
        n_splits=walk_forward_splits,
        random_state=random_state,
        importance_csv_path=importance_csv_path,
    )
    if not selected:
        log.warning("feature selector kept 0 features — falling back to full set")
        selected = list(feature_columns)
    X_train = X_train_raw[selected]
    X_test = X_test_raw[selected]
    feature_columns = selected

    # ---- 2. ElasticNet meta-model on the median target -------------- #
    meta_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9], cv=5,
            random_state=random_state, max_iter=20000,
        )),
    ])
    meta_model.fit(X_train, y_train)
    log.info("trained ElasticNet meta-model (weight=%.2f in median ensemble)",
             meta_model_weight)

    # ---- 3. Quantile ensemble --------------------------------------- #
    quantile_models: Dict[float, _EnsembleRegressor] = {}
    for q in QUANTILES:
        members = []
        for seed_offset in range(ensemble_seeds):
            m = _make_quantile_pipeline(
                q, random_state=random_state + seed_offset * 7919)
            m.fit(X_train, y_train)
            members.append(m)
        meta_for_this_q = meta_model if abs(q - 0.5) < 1e-6 else None
        quantile_models[q] = _EnsembleRegressor(
            members, meta_model=meta_for_this_q,
            meta_weight=meta_model_weight if meta_for_this_q else 0.0,
        )
    log.info("trained %d quantile ensembles × %d members each",
             len(QUANTILES), ensemble_seeds)

    # ---- 4. Residual std on TEST set --------------------------------- #
    # Two uses:
    #   • Gaussian fallback for prob_above when zero per-strike
    #     classifiers were trained (very small dataset).
    #   • Sizing the dynamic threshold grid in run_daily.py — strikes
    #     get placed at forecast ± span_sigma·residual_std.
    #
    # Use MAD (median absolute deviation × 1.4826) instead of std for
    # robustness — NG's tail events (winter spikes, glut crashes) blow
    # up the plain std and produce a $3+ residual that makes the dynamic
    # grid unusable. MAD recovers a "typical day" residual that's an
    # order of magnitude smaller and properly sizes the grid against
    # everyday moves rather than once-a-decade dislocations.
    median_test_pred_for_resid = quantile_models[0.50].predict(X_test)
    residuals = y_test.values - median_test_pred_for_resid
    residual_std = float(np.median(np.abs(residuals - np.median(residuals)))
                          * 1.4826)
    # Floor at $0.05/MMBTU so the grid still has at least Kalshi tick-
    # spacing's worth of breadth even on a flat residual.
    residual_std = max(residual_std, 0.05)

    # ---- 5. Per-strike classifier ensemble -------------------------- #
    if threshold_training_grid is None:
        threshold_training_grid = _default_training_grid(
            y_train, step_usd=threshold_training_step_usd)

    threshold_classifiers: Dict[float, _ClassifierEnsemble] = {}
    threshold_metrics: Dict[float, Dict[str, float]] = {}
    have_holdout = len(X_train) > calibration_holdout_days + 60
    for thr in threshold_training_grid:
        y_thr_train = (y_train >= thr).astype(int)
        y_thr_test = (y_test >= thr).astype(int)
        # Skip thresholds where training data has only one class.
        if y_thr_train.nunique() < 2:
            continue
        members: List = []
        for seed_offset in range(classifier_seeds):
            seed = random_state + seed_offset * 7919
            base_clf = _make_classifier_pipeline(random_state=seed)
            if have_holdout:
                fit_end = len(X_train) - calibration_holdout_days
                X_fit_t, X_cal_t = X_train.iloc[:fit_end], X_train.iloc[fit_end:]
                y_fit_t = y_thr_train.iloc[:fit_end]
                y_cal_t = y_thr_train.iloc[fit_end:]
                if y_fit_t.nunique() < 2:
                    # Fit window only one class — fall back to full-train.
                    base_clf.fit(X_train, y_thr_train)
                    members.append(base_clf)
                    continue
                base_clf.fit(X_fit_t, y_fit_t)
                if y_cal_t.nunique() == 2 and len(X_cal_t) >= 20:
                    cal_clf = CalibratedClassifierCV(
                        base_clf, method="isotonic", cv="prefit")
                    cal_clf.fit(X_cal_t, y_cal_t)
                    members.append(cal_clf)
                else:
                    members.append(base_clf)
            else:
                base_clf.fit(X_train, y_thr_train)
                members.append(base_clf)
        threshold_classifiers[thr] = _ClassifierEnsemble(members=members)
        # Per-threshold test-set metrics.
        if y_thr_test.nunique() == 2:
            test_probs = np.mean(
                [m.predict_proba(X_test)[:, 1] for m in members], axis=0)
            test_pred = (test_probs >= 0.5).astype(int)
            threshold_metrics[thr] = {
                "n_train_pos": int(y_thr_train.sum()),
                "n_test_pos": int(y_thr_test.sum()),
                "test_accuracy": float(accuracy_score(y_thr_test, test_pred)),
                "test_precision": float(
                    precision_score(y_thr_test, test_pred, zero_division=0)),
                "test_recall": float(
                    recall_score(y_thr_test, test_pred, zero_division=0)),
                "test_f1": float(f1_score(y_thr_test, test_pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_thr_test, test_probs)),
            }
    if threshold_classifiers:
        thr_lo = min(threshold_classifiers)
        thr_hi = max(threshold_classifiers)
        log.info("trained %d per-strike classifiers ($%.2f..$%.2f / MMBTU)",
                 len(threshold_classifiers), thr_lo, thr_hi)
    else:
        log.warning("0 per-strike classifiers trained — model will fall "
                    "back to Gaussian residual probability")

    # ---- 6. Median forecast OOS metrics (point-forecast diagnostics) - #
    # Reuse the test-set median predictions we already computed for
    # residual_std — same numbers, no need to fit twice.
    median_test_pred = median_test_pred_for_resid
    pf_metrics = {
        "mae": float(mean_absolute_error(y_test, median_test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, median_test_pred))),
        "r2": float(r2_score(y_test, median_test_pred)),
    }
    log.info("median forecast OOS — MAE %.0f  RMSE %.0f  R2 %.3f",
             pf_metrics["mae"], pf_metrics["rmse"], pf_metrics["r2"])

    # Headline classifier numbers come from the AVERAGE across trained
    # per-strike classifiers — those are what actually matter for the
    # binary trading decision against Kalshi.
    if threshold_metrics:
        avg_acc = float(np.mean([m["test_accuracy"] for m in threshold_metrics.values()]))
        avg_prec = float(np.mean([m["test_precision"] for m in threshold_metrics.values()]))
        avg_rec = float(np.mean([m["test_recall"] for m in threshold_metrics.values()]))
        avg_f1 = float(np.mean([m["test_f1"] for m in threshold_metrics.values()]))
        avg_auc = float(np.mean([m["test_roc_auc"] for m in threshold_metrics.values()]))
    else:
        avg_acc = avg_prec = avg_rec = avg_f1 = avg_auc = 0.0

    metrics = {
        # Point-forecast diagnostics (the dashboard reads these).
        "mae": pf_metrics["mae"],
        "rmse": pf_metrics["rmse"],
        "r2": pf_metrics["r2"],
        "residual_std": residual_std,
        # Per-strike classifier headline numbers (the trading-decision
        # accuracy that actually matters).
        "per_strike_avg_accuracy": avg_acc,
        "per_strike_avg_precision": avg_prec,
        "per_strike_avg_recall": avg_rec,
        "per_strike_avg_f1": avg_f1,
        "per_strike_avg_roc_auc": avg_auc,
        "per_strike_count": len(threshold_classifiers),
        "n_features_selected": len(feature_columns),
    }
    log.info("test-set headline — MAE=$%.3f/MMBTU  R2=%.3f  | per-strike avg: "
             "acc=%.3f prec=%.3f rec=%.3f F1=%.3f AUC=%.3f over %d thresholds",
             pf_metrics["mae"], pf_metrics["r2"],
             avg_acc, avg_prec, avg_rec, avg_f1, avg_auc,
             len(threshold_classifiers))

    return NatGasModel(
        feature_columns=list(feature_columns),
        quantile_models=quantile_models,
        threshold_classifiers=threshold_classifiers,
        threshold_grid=sorted(threshold_classifiers.keys()),
        residual_std=residual_std,
        train_end_date=train.index[-1],
        metrics=metrics,
        model_name="quantile_gbm_ensemble",
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_model(model: NatGasModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("saved model to %s", path)


def load_model(path: str | Path) -> Optional[NatGasModel]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Probability across a threshold grid
# --------------------------------------------------------------------------- #

def threshold_probabilities(
    model: NatGasModel,
    feature_row: pd.DataFrame,
    thresholds_usd: List[float],
) -> Dict[float, float]:
    """Return P(price >= thr) for every threshold in the grid.

    Single-row input → dict[threshold -> probability]. Uses the
    per-strike classifiers via ``model.prob_above`` so probabilities
    are honest-priced against the same binary the market resolves on.
    """
    return {float(thr): float(model.prob_above(feature_row, thr)[0])
            for thr in thresholds_usd}
