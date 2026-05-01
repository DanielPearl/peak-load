"""Feature engineering for the peak-load model.

Three layers:

  1. Weather-derived features:  CDD/HDD, heat index, lagged temps.
  2. Calendar features:         day-of-week, weekend, month, holiday,
                                season — all leakage-safe (known in advance).
  3. Lagged target features:    yesterday's peak, last week, rolling
                                averages — captures load persistence.

All features are leakage-safe: they only use information available
BEFORE today's peak hour. The forecast row built by `build_today_row`
is the same shape as a training row but with the target column
omitted (we're predicting it).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# US federal holidays for the next several years. Hardcoded so we
# don't take a holidays-package dependency for a small list. Update
# when this list expires.
US_FEDERAL_HOLIDAYS_2024_2027 = [
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
    "2024-11-11", "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13",
    "2025-11-11", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-10-12",
    "2026-11-11", "2026-11-26", "2026-12-25",
    # 2027 — extend before this expires
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-10-11",
    "2027-11-11", "2027-11-25", "2027-12-24",
]
HOLIDAY_SET = set(pd.to_datetime(US_FEDERAL_HOLIDAYS_2024_2027).date)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _heat_index_c(temp_c: float, humidity_pct: float) -> float:
    """Rothfusz-equivalent heat index in Celsius.

    Heat index ('feels like') is what actually drives AC demand —
    100°F at 30% humidity feels different than 100°F at 80%. The
    standard formula is computed in F so we convert in/out.
    """
    if pd.isna(temp_c) or pd.isna(humidity_pct):
        return np.nan
    t_f = temp_c * 9 / 5 + 32
    rh = humidity_pct
    if t_f < 80:    # formula only valid above 80°F; below, return temp itself
        return temp_c
    hi = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh
          - 0.22475541 * t_f * rh - 6.83783e-3 * t_f * t_f
          - 5.481717e-2 * rh * rh + 1.22874e-3 * t_f * t_f * rh
          + 8.5282e-4 * t_f * rh * rh - 1.99e-6 * t_f * t_f * rh * rh)
    return (hi - 32) * 5 / 9


def _season(month: int) -> str:
    """Meteorological season — drives the heat/cool relationship."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


# --------------------------------------------------------------------------- #
# Public: build training feature table
# --------------------------------------------------------------------------- #

def build_features(panel: pd.DataFrame, target: str = "daily_peak_load_mw"
                   ) -> Tuple[pd.DataFrame, List[str]]:
    """Build the (features, target) table for training.

    Parameters
    ----------
    panel : pd.DataFrame
        Daily-indexed panel from data_loaders.build_panel(). Must contain
        the target column; weather/renewables columns are optional but
        improve accuracy when present.
    target : str
        Either 'daily_peak_load_mw' or 'net_peak_load_mw'.

    Returns
    -------
    df, feature_cols
        df has all features + the target column. feature_cols is the
        list of column names the model should use. Rows where the target
        is NaN are dropped (training set requires a label).
    """
    out = panel.copy().sort_index()
    if target not in out.columns:
        raise KeyError(f"target column {target!r} not in panel")
    out = out.rename(columns={target: "target"})

    _add_weather_features(out)
    _add_calendar_features(out)
    _add_lag_features(out, "target")
    if "solar_generation_mw" in out.columns:
        _add_lag_features(out, "solar_generation_mw", prefix="solar")
    if "wind_generation_mw" in out.columns:
        _add_lag_features(out, "wind_generation_mw", prefix="wind")

    out = out.dropna(subset=["target"])
    out = out.replace([np.inf, -np.inf], np.nan)
    # Critical leakage guard: BOTH load columns are functions of each
    # other (`net_peak_load_mw = daily_peak_load_mw − solar − wind`),
    # so whichever isn't the target would leak the target into
    # features. Drop the non-target load column unconditionally.
    leakage_cols = {"target", "daily_peak_load_mw", "net_peak_load_mw"}
    feature_cols = [c for c in out.columns if c not in leakage_cols]
    return out, feature_cols


def build_today_row(panel: pd.DataFrame, forecast_row: pd.Series,
                    target: str = "daily_peak_load_mw") -> pd.DataFrame:
    """Build a single-row feature DataFrame for prediction.

    `panel` is the historical daily panel up to (but not including)
    today. `forecast_row` is a Series with weather forecast columns
    for today (max_temp_c, min_temp_c, avg_temp_c, humidity_pct,
    dew_point_c). Calendar features come from today's date.

    Returns a 1-row DataFrame with the same columns as build_features
    output, ready to feed model.predict().
    """
    panel = panel.copy().sort_index()
    if target in panel.columns:
        panel = panel.rename(columns={target: "target"})
    today = forecast_row.name if forecast_row.name is not None else pd.Timestamp.utcnow().normalize()
    today = pd.Timestamp(today).normalize()

    # Append a placeholder row at today's date with weather from the
    # forecast and target=NaN. Then run the same feature pipeline so
    # lags / rollings are computed identically to training.
    today_row = pd.Series(forecast_row).copy()
    today_row["target"] = np.nan
    panel.loc[today] = today_row

    _add_weather_features(panel)
    _add_calendar_features(panel)
    _add_lag_features(panel, "target")
    if "solar_generation_mw" in panel.columns:
        _add_lag_features(panel, "solar_generation_mw", prefix="solar")
    if "wind_generation_mw" in panel.columns:
        _add_lag_features(panel, "wind_generation_mw", prefix="wind")
    return panel.loc[[today]]


# --------------------------------------------------------------------------- #
# Feature builders (in-place on the panel)
# --------------------------------------------------------------------------- #

def _add_weather_features(out: pd.DataFrame) -> None:
    """Cooling/heating degree days + heat index. CDD/HDD are the
    classic load drivers — they capture the asymmetry that 80°F
    drives AC load whereas 60°F doesn't drive heating much."""
    if "avg_temp_c" not in out.columns:
        return
    avg_f = out["avg_temp_c"] * 9 / 5 + 32
    # CDD: degrees above 65°F. HDD: degrees below 65°F. Standard reference.
    out["cooling_degree_days"] = (avg_f - 65).clip(lower=0)
    out["heating_degree_days"] = (65 - avg_f).clip(lower=0)
    if "humidity_pct" in out.columns and "max_temp_c" in out.columns:
        out["heat_index_c"] = [
            _heat_index_c(t, h)
            for t, h in zip(out["max_temp_c"], out["humidity_pct"])
        ]
        out["heat_index_f"] = out["heat_index_c"] * 9 / 5 + 32
    # Temperature deltas — the diurnal range correlates with weather
    # regime stability (a wide range = clear sky = different load).
    if "max_temp_c" in out.columns and "min_temp_c" in out.columns:
        out["temp_range_c"] = out["max_temp_c"] - out["min_temp_c"]


def _add_calendar_features(out: pd.DataFrame) -> None:
    """Day-of-week / weekend / holiday / month / season indicators.

    These are leakage-safe by definition (known years in advance) and
    capture the strong weekly + seasonal pattern in electricity load.
    """
    idx = out.index
    out["day_of_week"] = idx.weekday
    out["is_weekend"] = (idx.weekday >= 5).astype(int)
    out["is_holiday"] = pd.Series(
        [d.date() in HOLIDAY_SET for d in idx], index=idx).astype(int)
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    # Season one-hot. Winter is the dropped reference category.
    season = pd.Series([_season(m) for m in idx.month], index=idx)
    for s in ("spring", "summer", "fall"):
        out[f"is_{s}"] = (season == s).astype(int)
    # Cyclical encodings so the model can pick up periodicity without
    # treating week / month as linear quantities.
    out["dow_sin"] = np.sin(2 * np.pi * idx.weekday / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * idx.weekday / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12.0)


def _add_lag_features(out: pd.DataFrame, col: str, prefix: str = None) -> None:
    """Lagged values + rolling stats. ``col`` is the column to lag;
    ``prefix`` overrides the feature-name prefix (defaults to the
    column name with `_load`/`_mw` stripped for tidiness).
    """
    p = prefix or col.replace("_load_mw", "").replace("_mw", "")
    for lag in (1, 7, 14):
        out[f"{p}_lag_{lag}"] = out[col].shift(lag)
    out[f"{p}_rolling_7"] = out[col].shift(1).rolling(7, min_periods=3).mean()
    out[f"{p}_rolling_30"] = out[col].shift(1).rolling(30, min_periods=10).mean()
    # Same weekday last week — captures weekly seasonality directly.
    out[f"{p}_same_weekday_lw"] = out[col].shift(7)
