"""Feature engineering for the Natural Gas Price prediction model.

Layers:
  1. Weather features:    HDD/CDD national + per-region, anomalies,
                          lags, rolling means, cold/heat-wave duration
                          counters, wind/humidity, LNG-terminal weather
                          + storm flags, Gulf hurricane disruption,
                          day-over-day forecast-revision deltas.
  2. Storage features:    levels, 5y deviation, change-week-over-week,
                          days-since-last-report (Thursday spike).
  3. Production features: lagged + monthly-change.
  4. Calendar features:   day-of-week, weekend, holiday, month, quarter.
  5. Lagged target:       yesterday's NG, last week, rolling 7/30,
                          plus log-returns (NG prices are volatility-
                          clustered so log-return lags help the GBMs
                          pick up momentum vs. mean-reversion regimes).
  6. Cross-Kalshi features: snapshot probabilities from related Kalshi
                          markets — only populated at inference time.
  7. Forecast revisions:  numeric deltas (today vs prior run) for the
                          weather forecast — only populated at
                          inference time. NaN at train time; the
                          median imputer fills, the walk-forward
                          selector prunes the channels that don't add
                          signal.

All features are leakage-safe: they use information available BEFORE
the prediction date's 5pm settle.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# US federal holidays — daily indicator for power-burn / commercial demand.
# Update this list before it expires.
US_FEDERAL_HOLIDAYS_2024_2027 = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
    "2024-11-11", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13",
    "2025-11-11", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-10-12",
    "2026-11-11", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-10-11",
    "2027-11-11", "2027-11-25", "2027-12-24",
]
HOLIDAY_SET = set(pd.to_datetime(US_FEDERAL_HOLIDAYS_2024_2027).date)


# --------------------------------------------------------------------------- #
# Public: build training feature table
# --------------------------------------------------------------------------- #

def build_features(panel: pd.DataFrame,
                   target: str = "natgas_henry_hub_usd_mmbtu"
                   ) -> Tuple[pd.DataFrame, List[str]]:
    """Build the (features, target) table for training.

    Parameters
    ----------
    panel : pd.DataFrame
        Daily-indexed panel from data_loaders.build_panel(). Must
        contain the target column; weather/storage/production are
        optional but improve accuracy when present.
    target : str
        Default 'natgas_henry_hub_usd_mmbtu'.

    Returns
    -------
    df, feature_cols
        df has all features + the renamed `target` column. feature_cols
        is the list of column names the model should use. Rows where
        target is NaN are dropped.
    """
    out = panel.copy().sort_index()
    if target not in out.columns:
        raise KeyError(f"target column {target!r} not in panel")
    out = out.rename(columns={target: "target"})

    _add_weather_features(out)
    _add_storage_features(out)
    _add_production_features(out)
    _add_calendar_features(out)
    _add_target_lag_features(out)

    out = out.dropna(subset=["target"])
    out = out.replace([np.inf, -np.inf], np.nan)

    # Leakage guard: drop the raw target columns + any other Henry Hub
    # alias that might've slipped in.
    leakage_cols = {"target", "natgas_henry_hub_usd_mmbtu"}
    feature_cols = [c for c in out.columns if c not in leakage_cols]
    return out, feature_cols


def build_today_row(panel: pd.DataFrame, forecast_row: pd.Series,
                    cross_kalshi_features: pd.Series = None,
                    forecast_revisions: pd.Series = None,
                    target: str = "natgas_henry_hub_usd_mmbtu"
                    ) -> pd.DataFrame:
    """Build a single-row feature DataFrame for prediction.

    `panel` is the historical daily panel up to (but not including)
    today. `forecast_row` is a Series with the columns produced by
    `fetch_weather_forecast` (national + regional + LNG-terminal
    weather, wind/humidity, storm flags). Optional
    `cross_kalshi_features` is the snapshot from
    `fetch_cross_kalshi_features` — these are appended as columns on
    the today-row. Optional `forecast_revisions` is the day-over-day
    weather-forecast delta from `compute_forecast_revisions`.

    Returns a 1-row DataFrame with the same columns as `build_features`
    output, ready for `model.predict()`.
    """
    panel = panel.copy().sort_index()
    if target in panel.columns:
        panel = panel.rename(columns={target: "target"})
    today = (forecast_row.name if forecast_row.name is not None
             else pd.Timestamp.utcnow().normalize())
    today = pd.Timestamp(today).normalize()

    today_row = pd.Series(forecast_row).copy()
    today_row["target"] = np.nan
    panel.loc[today] = today_row

    _add_weather_features(panel)
    _add_storage_features(panel)
    _add_production_features(panel)
    _add_calendar_features(panel)
    _add_target_lag_features(panel)

    today_features = panel.loc[[today]]

    # Cross-Kalshi snapshot — append as constant columns. The model was
    # trained without these (NaN at training, the imputer median-fills).
    # Once we have a historical archive of cross-Kalshi prices we can
    # backfill at training time and the model will actually USE them.
    if cross_kalshi_features is not None and not cross_kalshi_features.empty:
        for k, v in cross_kalshi_features.items():
            today_features[k] = v
    # Forecast-revision deltas — also snapshot-only. NaN at train time
    # (we don't have a historical record of yesterday's forecast), so
    # the imputer fills and the walk-forward selector prunes. As we
    # accumulate a forecast-history archive we can backfill these at
    # training time too.
    if forecast_revisions is not None and not forecast_revisions.empty:
        for k, v in forecast_revisions.items():
            today_features[k] = v
    return today_features


# --------------------------------------------------------------------------- #
# Feature builders (in-place on the panel)
# --------------------------------------------------------------------------- #

def _add_weather_features(out: pd.DataFrame) -> None:
    """National + regional HDD/CDD lags, rolling averages, wave-duration
    counters, wind/humidity, LNG-terminal weather, and Gulf storm flag
    — the full weather stack for NG-demand and supply-disruption signals.

    Anomalies (deviation from rolling normal) capture sudden cold snaps
    / heat waves which is what moves NG price. Wave-duration counters
    flag *persistent* events — a 3-day cold spell prices very differently
    from a 1-day blip because storage gets drawn measurably. LNG-terminal
    wind + storm flags capture export-demand disruption; Gulf storm flag
    captures production-side disruption. Wind in general matters for
    renewable/wind-power offset, which displaces NG burn for power.
    """
    # ── National HDD ────────────────────────────────────────────────
    if "national_hdd" in out.columns:
        for lag in (1, 2, 3, 7, 14):
            out[f"hdd_lag_{lag}"] = out["national_hdd"].shift(lag)
        out["hdd_rolling_7"] = (out["national_hdd"]
                                .shift(1).rolling(7, min_periods=3).mean())
        out["hdd_rolling_30"] = (out["national_hdd"]
                                  .shift(1).rolling(30, min_periods=10).mean())
        # Cold-wave duration: consecutive prior days with HDD > 25 (a
        # noticeably cold day). Streak resets on any below-threshold day.
        out["cold_wave_days"] = _streak_above(
            out["national_hdd"].shift(1), threshold=25.0)
        # Sum of HDD over the last 3 days — captures sustained cold.
        out["hdd_sum_3d"] = (out["national_hdd"]
                              .shift(1).rolling(3, min_periods=2).sum())

    # ── National CDD ────────────────────────────────────────────────
    if "national_cdd" in out.columns:
        for lag in (1, 2, 3, 7, 14):
            out[f"cdd_lag_{lag}"] = out["national_cdd"].shift(lag)
        out["cdd_rolling_7"] = (out["national_cdd"]
                                .shift(1).rolling(7, min_periods=3).mean())
        out["cdd_rolling_30"] = (out["national_cdd"]
                                  .shift(1).rolling(30, min_periods=10).mean())
        out["heat_wave_days"] = _streak_above(
            out["national_cdd"].shift(1), threshold=15.0)
        out["cdd_sum_3d"] = (out["national_cdd"]
                              .shift(1).rolling(3, min_periods=2).sum())

    if "national_avg_temp_f" in out.columns:
        for lag in (1, 7):
            out[f"temp_lag_{lag}"] = out["national_avg_temp_f"].shift(lag)

    # ── Wind + humidity (national, power-burn substitution signal) ──
    if "national_wind_mph" in out.columns:
        for lag in (1, 2, 7):
            out[f"wind_lag_{lag}"] = out["national_wind_mph"].shift(lag)
        out["wind_rolling_7"] = (out["national_wind_mph"]
                                  .shift(1).rolling(7, min_periods=3).mean())
    if "national_humidity_pct" in out.columns:
        for lag in (1, 7):
            out[f"humidity_lag_{lag}"] = (
                out["national_humidity_pct"].shift(lag))

    # ── Regional HDD/CDD — per-region demand decomposition.
    # NG demand is dominated by Northeast + Midwest in winter (heating)
    # and South in summer (power-burn for AC). Letting the model see
    # the regional breakdown rather than only the national aggregate
    # gives it a sharper read on where in the demand stack a given
    # weather print sits.
    region_cols = [c for c in out.columns if c.startswith("region_")]
    for col in region_cols:
        for lag in (1, 7):
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        if col.endswith("_hdd") or col.endswith("_cdd"):
            out[f"{col}_rolling_7"] = (out[col]
                                        .shift(1).rolling(7, min_periods=3).mean())

    # ── LNG-terminal weather (export-side disruption) ───────────────
    if "lng_terminal_wind_mph" in out.columns:
        out["lng_wind_lag_1"] = out["lng_terminal_wind_mph"].shift(1)
        out["lng_wind_rolling_3"] = (out["lng_terminal_wind_mph"]
                                      .shift(1).rolling(3, min_periods=1).mean())
    if "lng_terminal_storm_flag" in out.columns:
        out["lng_storm_lag_1"] = out["lng_terminal_storm_flag"].shift(1)
        # 7-day storm-disruption count — a single-day storm vs a multi-
        # day system disrupts very differently. Sums the prior 7 days
        # of binary storm flags.
        out["lng_storm_count_7d"] = (out["lng_terminal_storm_flag"]
                                      .shift(1)
                                      .rolling(7, min_periods=1).sum())
    if "lng_terminal_avg_temp_f" in out.columns:
        out["lng_temp_lag_1"] = out["lng_terminal_avg_temp_f"].shift(1)

    # ── Gulf hurricane / storm disruption ────────────────────────────
    # Tropical-storm-force winds in the Gulf hit ~17% of US dry-gas
    # production from offshore + onshore Louisiana / Texas. Plumb in
    # both the active flag (binary) and a rolling 7-day count (proxy
    # for "we're in storm season AND a system has been active recently").
    if "gulf_storm_active" in out.columns:
        out["gulf_storm_lag_1"] = out["gulf_storm_active"].shift(1)
        out["gulf_storm_count_7d"] = (out["gulf_storm_active"]
                                       .shift(1)
                                       .rolling(7, min_periods=1).sum())
    if "gulf_max_wind_mph" in out.columns:
        out["gulf_wind_lag_1"] = out["gulf_max_wind_mph"].shift(1)
        out["gulf_wind_rolling_3"] = (out["gulf_max_wind_mph"]
                                       .shift(1).rolling(3, min_periods=1).mean())


def _streak_above(series: pd.Series, threshold: float) -> pd.Series:
    """Count consecutive elements > threshold ending at each position.
    Resets to 0 on any below-threshold value. Pre-shift the series if
    you want a leakage-safe view (this function does no shifting).
    """
    flag = (series > threshold).astype(float)
    # Standard cumcount-within-groups trick: cumsum of inverse marks
    # the streak ID; counting within each ID gives the streak length.
    group = (flag != flag.shift(1)).cumsum()
    streak = flag.groupby(group).cumsum() * flag
    return streak.astype(float)


def _add_storage_features(out: pd.DataFrame) -> None:
    """Storage level + change-week-over-week + 5-y deviation.

    NG traders react to the weekly EIA storage report on Thursdays.
    The DELTA matters more than the absolute level — a draw bigger
    than the 5-year norm is bullish; a build bigger is bearish. We
    expose lagged levels (so the model only sees prior-week numbers)
    and compute the week-over-week change explicitly.
    """
    if "ng_storage_bcf" not in out.columns:
        return
    # Lag-7 = last week's published level (since reports are weekly).
    out["storage_lag_7"] = out["ng_storage_bcf"].shift(7)
    out["storage_lag_14"] = out["ng_storage_bcf"].shift(14)
    out["storage_change_wow"] = (out["ng_storage_bcf"].shift(7)
                                  - out["ng_storage_bcf"].shift(14))
    if "ng_storage_deviation_bcf" in out.columns:
        out["storage_deviation_lag_7"] = (
            out["ng_storage_deviation_bcf"].shift(7))


def _add_production_features(out: pd.DataFrame) -> None:
    """Production lag + month-over-month change.

    Production is monthly so daily resolution doesn't add much; what
    matters is the slow trend (year-over-year is meaningful).
    """
    if "ng_production_bcfd" not in out.columns:
        return
    out["production_lag_30"] = out["ng_production_bcfd"].shift(30)
    out["production_lag_365"] = out["ng_production_bcfd"].shift(365)
    out["production_yoy"] = (out["ng_production_bcfd"]
                              - out["ng_production_bcfd"].shift(365))


def _add_calendar_features(out: pd.DataFrame) -> None:
    """Day-of-week / weekend / holiday / month / season indicators.

    Weekly cycle matters for NG: power-burn drops on weekends, EIA
    storage reports release on Thursdays, NYMEX expiry mechanics fall
    on the same day each month.
    """
    idx = out.index
    out["day_of_week"] = idx.weekday
    out["is_weekend"] = (idx.weekday >= 5).astype(int)
    out["is_holiday"] = pd.Series(
        [d.date() in HOLIDAY_SET for d in idx], index=idx).astype(int)
    out["is_thursday"] = (idx.weekday == 3).astype(int)   # storage report day
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["is_winter"] = idx.month.isin([12, 1, 2]).astype(int)
    out["is_summer"] = idx.month.isin([6, 7, 8]).astype(int)
    out["is_shoulder"] = idx.month.isin([3, 4, 5, 9, 10, 11]).astype(int)
    out["dow_sin"] = np.sin(2 * np.pi * idx.weekday / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * idx.weekday / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12.0)


def _add_target_lag_features(out: pd.DataFrame) -> None:
    """Lagged price + rolling stats + log-returns + trend + vol regime.

    NG is volatility-clustered: a big move yesterday raises the
    probability of a big move today (regardless of direction). Log-
    return lags let the GBMs learn that dynamic. Plain price lags
    handle the level / mean-reversion forces. Trend features capture
    the bigger-picture regime (above-trend vs below-trend, short-term
    vs long-term momentum). Realized-vol features tell the model
    whether we're in a high-vol or low-vol regime, which determines
    how seriously to take any given lag signal.
    """
    p = out["target"]
    p_lag1 = p.shift(1)  # always shift before any rolling op (leakage guard)

    # ── Plain price lags. Short to long. ─────────────────────────────
    for lag in (1, 2, 3, 5, 7, 14, 30, 60, 90):
        out[f"target_lag_{lag}"] = p.shift(lag)

    # ── Rolling means (level reference). ────────────────────────────
    sma_7 = p_lag1.rolling(7, min_periods=3).mean()
    sma_30 = p_lag1.rolling(30, min_periods=10).mean()
    sma_90 = p_lag1.rolling(90, min_periods=30).mean()
    out["target_rolling_7"] = sma_7
    out["target_rolling_30"] = sma_30
    out["target_rolling_90"] = sma_90
    out["target_rolling_30_std"] = p_lag1.rolling(30, min_periods=10).std()
    out["target_rolling_90_std"] = p_lag1.rolling(90, min_periods=30).std()

    # ── Trend features: where the price sits relative to its trend.
    # The point isn't the level itself (lags already encode that) — it's
    # the *deviation* from the trend, which is what mean-reversion or
    # breakout strategies key on. SMA-crossover features encode the
    # short-vs-long-momentum regime.
    out["trend_dev_30"] = p_lag1 - sma_30        # above/below 30d trend
    out["trend_dev_90"] = p_lag1 - sma_90        # above/below 90d trend
    out["trend_sma7_minus_sma30"] = sma_7 - sma_30   # short vs medium MA
    out["trend_sma30_minus_sma90"] = sma_30 - sma_90  # medium vs long MA
    # Rate of change over various windows.
    out["roc_7"] = (p_lag1 / p.shift(8) - 1.0)   # 7d return shifted
    out["roc_30"] = (p_lag1 / p.shift(31) - 1.0)
    out["roc_90"] = (p_lag1 / p.shift(91) - 1.0)

    # ── Log-returns: ln(p_t / p_{t-1}). Stable across price regimes.
    log_p = np.log(p.replace(0, np.nan))
    log_ret_1 = (log_p - log_p.shift(1)).shift(1)
    out["log_return_lag_1"] = log_ret_1
    out["log_return_lag_2"] = log_ret_1.shift(1)
    out["log_return_lag_7"] = log_ret_1.shift(6)
    out["log_return_abs_rolling_7"] = (log_ret_1.abs()
                                        .rolling(7, min_periods=3).mean())
    # Realized-vol features: rolling std of log returns. NG vol is
    # bursty — high-vol periods last days. Knowing we're in one tells
    # the per-strike classifiers to widen their forecast distribution.
    out["log_return_std_14"] = log_ret_1.rolling(14, min_periods=5).std()
    out["log_return_std_30"] = log_ret_1.rolling(30, min_periods=10).std()
    out["log_return_std_90"] = log_ret_1.rolling(90, min_periods=30).std()
    # Vol-of-vol: is volatility itself trending? (Useful for regime
    # detection — vol-of-vol up = approaching a dislocation.)
    out["log_return_vol_30_minus_90"] = (
        out["log_return_std_30"] - out["log_return_std_90"]
    )

    # ── Momentum acceleration: is the trend speeding up or fading?
    out["log_return_accel"] = log_ret_1 - log_ret_1.shift(7)
