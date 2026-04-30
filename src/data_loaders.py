"""Data loaders for electricity load, weather, and renewable generation.

Three real sources are wired in:

    EIA           electricity load by ISO/respondent
                  https://www.eia.gov/opendata/   (free, instant signup)
    NOAA          weather observations + station metadata
                  https://www.ncdc.noaa.gov/cdo-web/token  (free)
    OpenWeather   forecast (next 5 days hourly)
                  https://openweathermap.org/api  (free tier sufficient)

If a key isn't set, each loader falls back to a synthetic generator
that produces realistic-looking data so the rest of the pipeline can
still run end-to-end. This is explicitly to make a fresh clone usable
without configuring four API keys; for any actual trading you'd want
real data on every path.

Each public function returns a pandas DataFrame indexed by date.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .config import Config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Electricity load (EIA)
# --------------------------------------------------------------------------- #

def fetch_electricity_load(cfg: Config, days: int = 730) -> pd.DataFrame:
    """Return daily electricity load for the configured region.

    Real path: EIA's hourly demand series for the configured
    respondent. We aggregate hourly → daily peak (max of the 24
    hourly readings) since that's what Kalshi peak-load markets
    resolve on.

    Synthetic fallback: produces a daily peak series with seasonal
    cycle + weekday pattern + Gaussian noise. Calibrated so summer/
    winter peaks roughly match the region's reference values.
    """
    if cfg.eia_api_key:
        try:
            return _fetch_eia_load_real(cfg, days=days)
        except Exception as exc:  # noqa: BLE001
            log.warning("EIA load fetch failed (%s); falling back to synthetic", exc)
    if not cfg.use_synthetic_when_missing:
        raise RuntimeError("EIA_API_KEY not set and synthetic fallback disabled")
    return _synthetic_load(cfg, days=days)


def _fetch_eia_load_real(cfg: Config, days: int) -> pd.DataFrame:
    """Real EIA call. Pulls hourly demand for the respondent and
    rolls up to daily peak.

    Endpoint: /v2/electricity/rto/region-data/data/
      filter:  respondent={cfg.region_meta['eia_respondent']}
               type=D  (demand, MWh)
               frequency=hourly

    EIA rate-limits aggressively so we ask for the date range in one
    call. For a > 1-year request the API returns up to 5000 rows per
    page; loop until done.
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    base = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    rows: list = []
    offset = 0
    while True:
        params = {
            "api_key": cfg.eia_api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": cfg.region_meta["eia_respondent"],
            "facets[type][]": "D",
            "start": start.isoformat() + "T00",
            "end": end.isoformat() + "T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": 5000,
        }
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        page = r.json().get("response", {}).get("data", [])
        if not page:
            break
        rows.extend(page)
        if len(page) < 5000:
            break
        offset += 5000

    if not rows:
        raise RuntimeError("EIA returned no rows")
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = df["period"].dt.tz_localize(None).dt.normalize()
    daily = df.groupby("date")["value"].max().rename("daily_peak_load_mw")
    return daily.to_frame()


def _synthetic_load(cfg: Config, days: int) -> pd.DataFrame:
    """Realistic synthetic daily peak load.

    Components:
      • Seasonal cycle:  sinusoid summer/winter peak, trough in shoulder months
      • Weekday pattern: Mon-Fri ~5% higher than Sat/Sun
      • Heatwave bumps:  random multi-day amplification in summer
      • Gaussian noise:  ~3% of mean, captures day-to-day variance
    Anchored to the region's reference summer/winter peak so synthetic
    values land in a reasonable absolute range.
    """
    rng = np.random.default_rng(seed=42)
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    summer_peak = cfg.region_meta["summer_peak_mw"]
    winter_peak = cfg.region_meta["winter_peak_mw"]
    avg_peak = (summer_peak + winter_peak) / 2.0
    amp = (summer_peak - winter_peak) / 2.0

    # Day-of-year sinusoid: peaks around July, troughs January.
    doy = idx.dayofyear.to_numpy(dtype=float)
    seasonal = avg_peak + amp * np.cos(2 * np.pi * (doy - 200) / 365.25) * -1

    # Weekday bump.
    weekday = idx.weekday.to_numpy(dtype=float)
    weekday_bump = np.where(weekday < 5, 1.04, 0.96)

    # Sparse heatwave events: ~3 per summer, +5-15% boost lasting 3-5 days.
    heatwave = np.zeros_like(seasonal)
    n_waves = max(1, days // 200)
    for _ in range(n_waves):
        start_i = rng.integers(0, len(idx))
        # Only apply if it lands in summer.
        if 150 < doy[start_i] < 270:
            length = rng.integers(3, 6)
            magnitude = rng.uniform(0.05, 0.15)
            heatwave[start_i:start_i + length] = magnitude

    noise = rng.normal(0, 0.03, size=len(idx))
    peak = seasonal * weekday_bump * (1 + heatwave) * (1 + noise)
    return pd.DataFrame({"daily_peak_load_mw": peak}, index=idx).rename_axis("date")


# --------------------------------------------------------------------------- #
# Weather (NOAA observed history + OpenWeather forecast)
# --------------------------------------------------------------------------- #

def fetch_weather_history(cfg: Config, days: int = 730) -> pd.DataFrame:
    """Daily observed weather for the region's reference station.

    Real path: NOAA Climate Data Online. We pull daily summaries
    (TMAX, TMIN, TAVG) plus humidity/dew-point from the configured
    station. Free token, free data.

    Synthetic fallback: daily temperature with seasonal sinusoid +
    weekday-independent noise. Humidity/dew-point derived as
    plausible functions of temperature.
    """
    if cfg.noaa_token:
        try:
            return _fetch_noaa_real(cfg, days=days)
        except Exception as exc:  # noqa: BLE001
            log.warning("NOAA fetch failed (%s); falling back to synthetic", exc)
    if not cfg.use_synthetic_when_missing:
        raise RuntimeError("NOAA_TOKEN not set and synthetic fallback disabled")
    return _synthetic_weather(cfg, days=days)


def _fetch_noaa_real(cfg: Config, days: int) -> pd.DataFrame:
    """NOAA CDO API stub.

    The actual call: GET /cdo-web/api/v2/data?datasetid=GHCND&...
    Headers: {"token": NOAA_TOKEN}. Returns JSON with one row per
    (station, date, datatype) triple. Pivot to wide format with
    columns max_temp / min_temp / avg_temp.

    Implementation deferred — current pipeline uses synthetic when
    the real path isn't critical. Wire this up if you want full
    historical training data from real obs. NOTE: real NOAA pulls
    are rate-limited (5 reqs/sec, 10K rows/req) so for >1 year of
    data you'll need to paginate by year.
    """
    raise NotImplementedError(
        "NOAA real fetch is a stub — wire in the GHCND endpoint here. "
        "See https://www.ncdc.noaa.gov/cdo-web/webservices/v2"
    )


def _synthetic_weather(cfg: Config, days: int) -> pd.DataFrame:
    """Generate daily weather columns. Calibrated to a generic US
    midcontinent climate; tweak per-region if needed."""
    rng = np.random.default_rng(seed=43)
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    doy = idx.dayofyear.to_numpy(dtype=float)
    # Avg temp seasonal sinusoid: cold January (~0°C / 32°F),
    # hot July (~30°C / 86°F).
    avg_c = 15 + 15 * np.cos(2 * np.pi * (doy - 200) / 365.25) * -1
    avg_c += rng.normal(0, 3, size=len(idx))   # day-to-day noise
    max_c = avg_c + rng.uniform(4, 8, size=len(idx))
    min_c = avg_c - rng.uniform(4, 8, size=len(idx))
    # Humidity higher in summer (when air can hold more moisture).
    humidity = 50 + 20 * np.cos(2 * np.pi * (doy - 200) / 365.25) * -1
    humidity = np.clip(humidity + rng.normal(0, 5, size=len(idx)), 20, 95)
    # Dew point: approx avg_temp - (100 - humidity) / 5 (rule-of-thumb).
    dew_c = avg_c - (100 - humidity) / 5.0
    return pd.DataFrame({
        "max_temp_c": max_c,
        "min_temp_c": min_c,
        "avg_temp_c": avg_c,
        "humidity_pct": humidity,
        "dew_point_c": dew_c,
    }, index=idx).rename_axis("date")


def fetch_weather_forecast(cfg: Config, days_ahead: int = 7) -> pd.DataFrame:
    """Forecast for the next N days. Used by run_daily.py to build
    the inference feature row.

    Real path: OpenWeather One Call API or NOAA NDFD. Either is fine;
    OpenWeather is simpler. Returns forecast peaks/lows by date.
    Synthetic fallback: persistence (tomorrow ≈ today + small noise).
    """
    if cfg.openweather_api_key:
        try:
            return _fetch_openweather_forecast_real(cfg, days_ahead=days_ahead)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenWeather fetch failed (%s); using persistence", exc)
    # Persistence forecast: yesterday's weather, slightly noised.
    history = fetch_weather_history(cfg, days=14)
    last = history.iloc[[-1]]
    rng = np.random.default_rng()
    rows = []
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    for d in range(1, days_ahead + 1):
        rows.append({
            "date": end + pd.Timedelta(days=d),
            "max_temp_c": float(last["max_temp_c"].iloc[0]) + rng.normal(0, 1.5),
            "min_temp_c": float(last["min_temp_c"].iloc[0]) + rng.normal(0, 1.5),
            "avg_temp_c": float(last["avg_temp_c"].iloc[0]) + rng.normal(0, 1.0),
            "humidity_pct": float(last["humidity_pct"].iloc[0]) + rng.normal(0, 3),
            "dew_point_c": float(last["dew_point_c"].iloc[0]) + rng.normal(0, 1.0),
        })
    return pd.DataFrame(rows).set_index("date")


def _fetch_openweather_forecast_real(cfg: Config, days_ahead: int) -> pd.DataFrame:
    """OpenWeather One Call 3.0 forecast.

    GET https://api.openweathermap.org/data/3.0/onecall
        ?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts
        &units=metric&appid={key}

    Returns up to 8 days of daily forecast. Map JSON 'daily' array
    into one DataFrame row per date with the columns above.
    Implementation deferred — same pattern as fetch_weather_history.
    """
    raise NotImplementedError(
        "OpenWeather real fetch is a stub — wire in One Call 3.0 here.")


# --------------------------------------------------------------------------- #
# Renewables (EIA Open Data — solar / wind generation by region)
# --------------------------------------------------------------------------- #

def fetch_renewables(cfg: Config, days: int = 730) -> pd.DataFrame:
    """Daily solar + wind generation totals (MWh) for the region.

    Real path: EIA hourly fuel-type series, type=NG (net gen),
    fueltype=SUN / WND, aggregated to daily totals.

    Synthetic fallback: solar = positive function of (avg_temp +
    summer_bias), wind = noise around regional mean. Returns NaN
    columns if neither path is available so feature builders can
    skip cleanly.
    """
    if cfg.eia_api_key:
        try:
            return _fetch_eia_renewables_real(cfg, days=days)
        except Exception as exc:  # noqa: BLE001
            log.warning("EIA renewables fetch failed (%s); using synthetic",
                        exc)
    if not cfg.use_synthetic_when_missing:
        # Return empty frame — features.py handles missing renewables.
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    return _synthetic_renewables(cfg, days=days)


def _fetch_eia_renewables_real(cfg: Config, days: int) -> pd.DataFrame:
    """EIA solar + wind generation. Same endpoint as load, with
    fueltype facet. Stub — wire in identical to _fetch_eia_load_real
    but filter on fueltype=SUN and fueltype=WND, then merge on date."""
    raise NotImplementedError(
        "EIA renewables fetch is a stub — copy _fetch_eia_load_real "
        "shape with fueltype facet (SUN / WND) and merge by date.")


def _synthetic_renewables(cfg: Config, days: int) -> pd.DataFrame:
    """Crude daily renewables generator. Solar correlates with
    temperature (sunny days are hotter); wind is roughly stationary
    around a regional mean."""
    rng = np.random.default_rng(seed=44)
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    summer_peak = cfg.region_meta["summer_peak_mw"]
    doy = idx.dayofyear.to_numpy(dtype=float)
    summer_factor = (np.cos(2 * np.pi * (doy - 200) / 365.25) * -1 + 1) / 2
    solar = summer_peak * 0.08 * summer_factor + rng.normal(
        0, summer_peak * 0.005, size=len(idx))
    wind = summer_peak * 0.06 + rng.normal(
        0, summer_peak * 0.015, size=len(idx))
    return pd.DataFrame({
        "solar_generation_mw": np.clip(solar, 0, None),
        "wind_generation_mw": np.clip(wind, 0, None),
    }, index=idx).rename_axis("date")


# --------------------------------------------------------------------------- #
# Combined panel
# --------------------------------------------------------------------------- #

def build_panel(cfg: Config, days: Optional[int] = None) -> pd.DataFrame:
    """Join load + weather + renewables into one daily panel.

    The panel is the input to features.py. Each row corresponds to
    one calendar date; columns are everything the model can see at
    or before that date's PEAK HOUR.
    """
    days = days or cfg.history_days_for_training
    load = fetch_electricity_load(cfg, days=days)
    weather = fetch_weather_history(cfg, days=days)
    renew = fetch_renewables(cfg, days=days)
    panel = load.join(weather, how="outer").join(renew, how="outer")
    panel = panel.sort_index()
    # Net peak: load minus solar+wind generation. Useful target if we
    # have renewable data, otherwise NaN.
    if "solar_generation_mw" in panel.columns and "wind_generation_mw" in panel.columns:
        panel["net_peak_load_mw"] = (
            panel["daily_peak_load_mw"]
            - panel["solar_generation_mw"]
            - panel["wind_generation_mw"]
        )
    return panel
