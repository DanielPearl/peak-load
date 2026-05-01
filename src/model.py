"""Peak-load forecasting models + residual-based probability conversion.

Trains two models and keeps the better one:

  Baseline:   Ridge regression
  Stronger:   GradientBoostingRegressor (sklearn HGB, fast)

Both are wrapped in a sklearn Pipeline with median imputation and
standard scaling. The stronger model usually wins on a feature-rich
dataset; baseline is kept as a sanity check / ablation.

Residual-based probability conversion:

  After training, compute residuals on a holdout window. The
  empirical residual distribution gives us a calibrated way to
  convert a point forecast into a probability for any threshold:

      P(load > threshold) = P(forecast + residual > threshold)
                          = P(residual > threshold - forecast)
                          = 1 - F_resid(threshold - forecast)

  We use a Gaussian CDF with the empirical std as a closed-form
  approximation. Empirical CDF would be marginally more accurate
  but adds dependencies and is harder to reason about.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model artifact
# --------------------------------------------------------------------------- #

@dataclass
class PeakLoadModel:
    """Trained model + the metadata needed for probability conversion.

    `feature_columns` is locked at training time so inference can
    reorder / fill missing columns deterministically. `residual_std`
    is the empirical std of OOS residuals — drives prob_above().
    """
    model_name: str                 # "ridge" or "hgb_gbm"
    feature_columns: List[str]
    pipeline: Pipeline
    residual_std: float
    train_end_date: pd.Timestamp
    metrics: Dict[str, float] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[self.feature_columns])

    def prob_above(self, X: pd.DataFrame, threshold_mw: float) -> np.ndarray:
        """P(actual load > threshold_mw) for each input row.

        Uses Gaussian residual approximation:
            forecast = pipeline.predict(X)
            actual ≈ forecast + N(0, residual_std)
            P(actual > thr) = 1 - Φ((thr - forecast) / residual_std)
        """
        forecast = self.predict(X)
        sigma = max(self.residual_std, 1e-6)
        z = (threshold_mw - forecast) / sigma
        return 1.0 - norm.cdf(z)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    test_size_days: int = 90,
    random_state: int = 42,
) -> PeakLoadModel:
    """Train baseline + stronger model. Return the better one (by RMSE).

    Time-aware split: last ``test_size_days`` are held out for OOS
    metrics and residual-std estimation. Random splits are wrong here
    because they leak future weather/load patterns into training.
    """
    df = df.dropna(subset=["target"]).copy()
    if len(df) <= test_size_days + 10:
        raise ValueError(
            f"Not enough rows: {len(df)} <= test_size_days={test_size_days} + 10")

    train = df.iloc[:-test_size_days]
    test = df.iloc[-test_size_days:]
    X_train, y_train = train[feature_columns], train["target"]
    X_test, y_test = test[feature_columns], test["target"]

    # ── Baseline: RidgeCV ─────────────────────────────────────────
    # CV-tuned alpha avoids the ill-conditioned solve that fixed-α
    # Ridge fell into on near-collinear feature columns (lag features
    # + weather are highly correlated). Without this the linear
    # solver was producing matmul overflows and bogus near-perfect
    # OOS metrics — masking the actual error structure.
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))),
    ])
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    # Guard against numerical blowups: any non-finite prediction means
    # the linear solve is unstable on this dataset; reject and let HGB
    # win automatically. (RidgeCV usually fixes this but defense-in-
    # depth is cheap.)
    if not np.all(np.isfinite(ridge_pred)):
        log.warning("ridge produced non-finite predictions — rejecting")
        ridge_metrics = {"mae": float("inf"), "rmse": float("inf"), "r2": -1e9}
    else:
        ridge_metrics = _score(y_test, ridge_pred)
    log.info("ridge OOS — MAE %.0f  RMSE %.0f  R2 %.3f  (alpha=%g)",
             ridge_metrics["mae"], ridge_metrics["rmse"], ridge_metrics["r2"],
             ridge.named_steps["model"].alpha_)

    # ── Stronger: HGB regressor ────────────────────────────────────
    hgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, max_depth=5,
            l2_regularization=0.1, random_state=random_state,
        )),
    ])
    hgb.fit(X_train, y_train)
    hgb_pred = hgb.predict(X_test)
    hgb_metrics = _score(y_test, hgb_pred)
    log.info("HGB OOS  — MAE %.0f  RMSE %.0f  R2 %.3f",
             hgb_metrics["mae"], hgb_metrics["rmse"], hgb_metrics["r2"])

    # Pick the lower-RMSE model. Keep the loser's metrics in audit.
    if hgb_metrics["rmse"] <= ridge_metrics["rmse"]:
        chosen, name, pred, metrics = hgb, "hgb_gbm", hgb_pred, hgb_metrics
    else:
        chosen, name, pred, metrics = ridge, "ridge", ridge_pred, ridge_metrics

    # Residual std = empirical OOS noise. Drives prob_above().
    residuals = y_test.values - pred
    residual_std = float(np.std(residuals))
    metrics["residual_std"] = residual_std
    metrics["baseline_ridge_rmse"] = ridge_metrics["rmse"]
    metrics["stronger_hgb_rmse"] = hgb_metrics["rmse"]
    log.info("chose %s (residual_std=%.0f MW)", name, residual_std)

    return PeakLoadModel(
        model_name=name,
        feature_columns=list(feature_columns),
        pipeline=chosen,
        residual_std=residual_std,
        train_end_date=train.index[-1],
        metrics=metrics,
    )


def _score(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_model(model: PeakLoadModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("saved model to %s", path)


def load_model(path: str | Path) -> Optional[PeakLoadModel]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Probability across a threshold grid
# --------------------------------------------------------------------------- #

def threshold_probabilities(
    model: PeakLoadModel,
    feature_row: pd.DataFrame,
    thresholds_mw: List[int],
) -> Dict[int, float]:
    """Return P(load > thr) for every threshold in the grid.

    Single-row input -> dict[threshold -> probability]. For batched
    inference (many days at once) call model.prob_above directly.
    """
    return {int(thr): float(model.prob_above(feature_row, thr)[0])
            for thr in thresholds_mw}
