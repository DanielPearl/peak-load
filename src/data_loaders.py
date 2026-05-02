"""Data loaders for the Natural Gas Price prediction bot.

Real sources wired in:

  EIA           Henry Hub daily spot price + weekly storage +
                production + consumption + LNG exports
                https://www.eia.gov/opendata/   (free)
  NOAA          Daily weather observations for consumption-weighted
                US weather index (HDD / CDD national).
                https://www.ncdc.noaa.gov/cdo-web/token  (free)
  OpenWeather   Forecast for next-day weather (One Call 3.0)
  Kalshi        Cross-market implied probabilities for crude oil,
                war/conflict, hurricane, fed/policy events

If a key isn't set, each loader falls back to a synthetic generator
so the rest of the pipeline can still run end-to-end.

Each public function returns a pandas DataFrame indexed by date.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from .config import Config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# EIA Henry Hub daily spot price (the prediction target)
# --------------------------------------------------------------------------- #

EIA_BASE = "https://api.eia.gov/v2"


def fetch_natgas_henry_hub(cfg: Config, days: int = 1095) -> pd.DataFrame:
    """Henry Hub natural gas spot price, daily, $/MMBTU.

    Real path: EIA series ``NG.RNGWHHD.D`` via /seriesid endpoint.
    Synthetic fallback: AR(1) random walk anchored at $3/MMBTU with
    weather-correlated shocks (driven by the unified panel).

    Returns DataFrame indexed by date with column
    ``natgas_henry_hub_usd_mmbtu``.
    """
    if cfg.eia_api_key:
        try:
            return _fetch_eia_henry_hub_real(cfg, days=days)
        except Exception as exc:  # noqa: BLE001
            log.warning("EIA Henry Hub fetch failed (%s); using synthetic", exc)
    if not cfg.use_synthetic_when_missing:
        raise RuntimeError("EIA_API_KEY not set and synthetic fallback off")
    return _synthetic_henry_hub(cfg, days=days)


def _fetch_eia_henry_hub_real(cfg: Config, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url = f"{EIA_BASE}/seriesid/NG.RNGWHHD.D"
    params = {
        "api_key": cfg.eia_api_key,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "length": 5000,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json().get("response", {}).get("data", []) or []
    if not rows:
        raise RuntimeError("EIA Henry Hub returned no rows")
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (df.set_index("period")[["value"]]
              .rename(columns={"value": "natgas_henry_hub_usd_mmbtu"})
              .sort_index())


def _synthetic_henry_hub(cfg: Config, days: int) -> pd.DataFrame:
    return _synthetic_panel(cfg, days)[["natgas_henry_hub_usd_mmbtu"]]


# --------------------------------------------------------------------------- #
# EIA weekly storage report — strongest single feature for NG prices
# --------------------------------------------------------------------------- #

def fetch_natgas_storage(cfg: Config, days: int = 1095) -> pd.DataFrame:
    """Weekly NG underground storage, lower-48 working gas (Bcf).

    Real path: EIA series NG.NW2_EPG0_SWO_R48_BCF.W (released Thursdays
    ~10:30am ET, settlement-relevant for the trading days that follow).

    Forward-filled to daily index so it joins on the daily panel.
    """
    if cfg.eia_api_key:
        try:
            return _fetch_eia_storage_real(cfg, days=days)
        except Exception as exc:  # noqa: BLE001
            log.warning("EIA storage fetch failed (%s); using synthetic", exc)
    if not cfg.use_synthetic_when_missing:
        return pd.DataFrame()
    return _synthetic_storage(cfg, days=days)


def _fetch_eia_storage_real(cfg: Config, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url = f"{EIA_BASE}/seriesid/NG.NW2_EPG0_SWO_R48_BCF.W"
    params = {
        "api_key": cfg.eia_api_key,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "length": 5000,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json().get("response", {}).get("data", []) or []
    if not rows:
        raise RuntimeError("EIA storage returned no rows")
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    weekly = (df.set_index("period")[["value"]]
                .rename(columns={"value": "ng_storage_bcf"})
                .sort_index())
    daily_index = pd.date_range(weekly.index.min(),
                                 weekly.index.max() + pd.Timedelta(days=10),
                                 freq="D")
    daily = weekly.reindex(daily_index).ffill()
    daily.index.name = "date"
    return daily


def _synthetic_storage(cfg: Config, days: int) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    doy = idx.dayofyear.to_numpy(dtype=float)
    # NG storage seasonal: peaks ~3700 Bcf late Oct, bottoms ~1400 Bcf
    # late March (winter draw). Sinusoidal proxy.
    seasonal = 2550 + 1100 * np.cos(2 * np.pi * (doy - 105) / 365.25)
    rng = np.random.default_rng(seed=44)
    noise = rng.normal(0, 80, size=len(idx))
    return pd.DataFrame({"ng_storage_bcf": seasonal + noise},
                        index=idx).rename_axis("date")


# --------------------------------------------------------------------------- #
# EIA US dry-gas production
# --------------------------------------------------------------------------- #

def fetch_natgas_production(cfg: Config, days: int = 1095) -> pd.DataFrame:
    """US dry natural gas production, monthly Bcf/day, ffilled daily.

    Real EIA series: NG.N9070US2.M. Used to capture slow supply trend
    (shale productivity, rig activity, freeze-off events).
    """
    if cfg.eia_api_key:
        try:
            return _fetch_eia_production_real(cfg, days=days)
        except Exception as exc:  # noqa: BLE001
            log.warning("EIA production fetch failed (%s); synthetic", exc)
    if not cfg.use_synthetic_when_missing:
        return pd.DataFrame()
    return _synthetic_production(cfg, days=days)


def _fetch_eia_production_real(cfg: Config, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url = f"{EIA_BASE}/seriesid/NG.N9070US2.M"
    params = {
        "api_key": cfg.eia_api_key,
        "start": start.isoformat()[:7],
        "end": end.isoformat()[:7],
        "length": 5000,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json().get("response", {}).get("data", []) or []
    if not rows:
        raise RuntimeError("EIA production returned no rows")
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    monthly = (df.set_index("period")[["value"]]
                 .rename(columns={"value": "ng_production_bcfd"})
                 .sort_index())
    daily_index = pd.date_range(monthly.index.min(),
                                 monthly.index.max() + pd.Timedelta(days=40),
                                 freq="D")
    return monthly.reindex(daily_index).ffill()


def _synthetic_production(cfg: Config, days: int) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    n = len(idx)
    trend = np.linspace(95, 103, n)
    rng = np.random.default_rng(seed=45)
    noise = rng.normal(0, 1.0, size=n)
    return pd.DataFrame({"ng_production_bcfd": trend + noise},
                        index=idx).rename_axis("date")


# --------------------------------------------------------------------------- #
# Consumption-weighted national weather aggregate
# --------------------------------------------------------------------------- #

def fetch_national_weather(cfg: Config, days: int = 1095) -> pd.DataFrame:
    """Consumption-weighted national HDD / CDD aggregate.

    NG demand is heating-driven in winter, power-burn-driven in summer.
    Right weather feature is a population/consumption-weighted national
    average, not any one city's forecast.

    Returns columns:
      national_avg_temp_f      — weighted mean daily avg temp
      national_hdd             — heating degree days (°F below 65)
      national_cdd             — cooling degree days (°F above 65)
      hdd_anomaly_30d          — deviation from 30-day rolling normal
      cdd_anomaly_30d          — same for cooling
    """
    if cfg.noaa_token:
        try:
            return _fetch_noaa_national_real(cfg, days=days)
        except Exception as exc:  # noqa: BLE001
            log.warning("NOAA national weather fetch failed (%s); synthetic",
                        exc)
    if not cfg.use_synthetic_when_missing:
        raise RuntimeError("NOAA_TOKEN not set and synthetic fallback off")
    return _synthetic_national_weather(cfg, days=days)


def _fetch_noaa_national_real(cfg: Config, days: int) -> pd.DataFrame:
    """NOAA real fetch — currently a stub.

    Production wiring: GHCND daily endpoint per station, fetch TAVG (or
    compute from TMAX/TMIN if TAVG missing), pivot, weight-average.
    Rate-limited to 5 reqs/sec, 10K rows/req — paginate by year.
    """
    raise NotImplementedError(
        "NOAA national weather fetch is a stub — wire when needed.")


def _synthetic_national_weather(cfg: Config, days: int) -> pd.DataFrame:
    panel = _synthetic_panel(cfg, days)
    cols = ["national_avg_temp_f", "national_hdd", "national_cdd",
            "hdd_anomaly_30d", "cdd_anomaly_30d"]
    return panel[cols]


# --------------------------------------------------------------------------- #
# Weather forecast (next-day) — for inference row
# --------------------------------------------------------------------------- #

def fetch_weather_forecast(cfg: Config, days_ahead: int = 7) -> pd.DataFrame:
    """National-weighted weather forecast for the next N days."""
    if cfg.openweather_api_key:
        try:
            return _fetch_openweather_national_forecast(cfg,
                                                         days_ahead=days_ahead)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenWeather forecast failed (%s); persistence", exc)
    history = fetch_national_weather(cfg, days=14)
    if history.empty:
        return pd.DataFrame()
    last = history.iloc[-1]
    rng = np.random.default_rng()
    rows = []
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    for d in range(1, days_ahead + 1):
        new_temp = float(last["national_avg_temp_f"]) + rng.normal(0, 2.0)
        rows.append({
            "date": end + pd.Timedelta(days=d),
            "national_avg_temp_f": new_temp,
            "national_hdd": max(0, 65 - new_temp),
            "national_cdd": max(0, new_temp - 65),
            "hdd_anomaly_30d": float(last.get("hdd_anomaly_30d", 0.0))
                                + rng.normal(0, 1.0),
            "cdd_anomaly_30d": float(last.get("cdd_anomaly_30d", 0.0))
                                + rng.normal(0, 1.0),
        })
    return pd.DataFrame(rows).set_index("date")


def _fetch_openweather_national_forecast(cfg: Config, days_ahead: int
                                          ) -> pd.DataFrame:
    """OpenWeather One Call 3.0 — fetch each reference station, then
    weight-average into the national aggregate columns.
    """
    rows_by_date: Dict[pd.Timestamp, dict] = {}
    total_weight = sum(s.get("weight", 0.0)
                       for s in cfg.weather_reference_stations) or 1.0
    for st in cfg.weather_reference_stations:
        params = {
            "lat": st["lat"], "lon": st["lon"],
            "exclude": "current,minutely,hourly,alerts",
            "units": "imperial",
            "appid": cfg.openweather_api_key,
        }
        r = requests.get("https://api.openweathermap.org/data/3.0/onecall",
                         params=params, timeout=15)
        r.raise_for_status()
        for d in (r.json().get("daily") or [])[:days_ahead]:
            ts = pd.to_datetime(d["dt"], unit="s", utc=True
                                 ).tz_localize(None).normalize()
            temp = (d.get("temp") or {}).get("day", float("nan"))
            w = st.get("weight", 0.0) / total_weight
            row = rows_by_date.setdefault(ts,
                                            {"temp_w": 0.0, "weight_used": 0.0})
            row["temp_w"] += float(temp) * w
            row["weight_used"] += w
    out = []
    for ts in sorted(rows_by_date):
        r = rows_by_date[ts]
        denom = r["weight_used"] or 1.0
        avg = r["temp_w"] / denom
        out.append({
            "date": ts,
            "national_avg_temp_f": avg,
            "national_hdd": max(0.0, 65 - avg),
            "national_cdd": max(0.0, avg - 65),
            "hdd_anomaly_30d": float("nan"),
            "cdd_anomaly_30d": float("nan"),
        })
    return pd.DataFrame(out).set_index("date")


# --------------------------------------------------------------------------- #
# Cross-Kalshi feature loader — implied probabilities from related markets
# --------------------------------------------------------------------------- #

def fetch_cross_kalshi_features(cfg: Config) -> pd.Series:
    """Pull current implied probabilities from cross-Kalshi feature
    series defined in cfg.cross_kalshi_series.

    For each series: average yes_ask across all currently-open markets
    to get an aggregate "consensus probability". This is a SNAPSHOT
    feature (same value for all rows on the inference date). Walk-
    forward selection prunes channels that don't add signal.

    Returns a pandas Series with one entry per (label, derived stat)
    pair. Dormant series produce NaN so the median imputer fills.
    """
    out: Dict[str, float] = {}
    if not (cfg.kalshi_api_key_id and cfg.kalshi_private_key_path):
        return pd.Series(dtype=float)
    from .kalshi import _SignedClient
    try:
        client = _SignedClient(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Kalshi cross-feature client init failed: %s", exc)
        return pd.Series(dtype=float)
    for series_ticker, label, _why in cfg.cross_kalshi_series:
        try:
            resp = client.get("/markets",
                               params={"series_ticker": series_ticker,
                                        "status": "open", "limit": 200})
            ms = resp.get("markets", []) or []
            asks = []
            vols = []
            for m in ms:
                # Read price using both legacy int field + newer
                # ``_dollars`` string field. Many low-liquidity markets
                # only have the dollar form populated.
                price = _market_price_prob(m)
                if price is not None:
                    asks.append(price)
                v = m.get("volume") or m.get("volume_fp") or 0
                try:
                    vols.append(int(float(v)) if v else 0)
                except (TypeError, ValueError):
                    vols.append(0)
            # n_open populates even when prices are missing — count of
            # open markets is its own signal channel.
            out[f"xk_{label}_n_open"] = float(len(ms))
            out[f"xk_{label}_vol_sum"] = float(sum(vols))
            if asks:
                out[f"xk_{label}_avg_prob"] = float(np.mean(asks))
                out[f"xk_{label}_max_prob"] = float(np.max(asks))
            else:
                out[f"xk_{label}_avg_prob"] = float("nan")
                out[f"xk_{label}_max_prob"] = float("nan")
        except Exception as exc:  # noqa: BLE001
            log.debug("cross-Kalshi fetch failed for %s: %s",
                      series_ticker, exc)
    return pd.Series(out, dtype=float)


def _market_price_prob(m: dict) -> Optional[float]:
    """Extract a probability (0..1) from a Kalshi market dict, trying
    yes_ask → yes_ask_dollars → last_price_dollars → midpoint(bid, ask)
    in that order. Returns None when no price is available.
    """
    ya = m.get("yes_ask")
    if ya not in (None, ""):
        try:
            return float(ya) / 100.0
        except (TypeError, ValueError):
            pass
    for k in ("yes_ask_dollars", "last_price_dollars"):
        v = m.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    bid = m.get("yes_bid_dollars")
    ask = m.get("yes_ask_dollars")
    if bid not in (None, "") and ask not in (None, ""):
        try:
            return (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            pass
    return None


# --------------------------------------------------------------------------- #
# Unified synthetic panel — NG price DRIVEN BY weather + storage + noise
# --------------------------------------------------------------------------- #

_SYNTH_CACHE: dict = {}


def _synthetic_panel(cfg: Config, days: int) -> pd.DataFrame:
    """Generate a unified synthetic panel where NG price is a function
    of weather (HDD/CDD), storage deviation, and AR shocks.

    The model needs weather + storage features to fit well; calendar
    features alone won't because the AR + irreducible noise stops a
    free lunch. Cached per-`days`.
    """
    key = days
    if key in _SYNTH_CACHE:
        return _SYNTH_CACHE[key]

    rng = np.random.default_rng(seed=42)
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    n = len(idx)
    doy = idx.dayofyear.to_numpy(dtype=float)

    # ── Weather: nationally-weighted average temp °F ────────────────
    seasonal_temp = 55 + 25 * np.cos(2 * np.pi * (doy - 200) / 365.25) * -1
    ar_phi = 0.85
    shocks = rng.normal(0, 5, size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = ar_phi * ar[i - 1] + shocks[i] * np.sqrt(1 - ar_phi ** 2)
    avg_f = seasonal_temp + ar
    hdd = np.maximum(65 - avg_f, 0)
    cdd = np.maximum(avg_f - 65, 0)
    hdd_norm = (pd.Series(hdd, index=idx)
                .rolling(30, min_periods=10).mean().values)
    cdd_norm = (pd.Series(cdd, index=idx)
                .rolling(30, min_periods=10).mean().values)
    hdd_anom = hdd - np.nan_to_num(hdd_norm, nan=hdd.mean())
    cdd_anom = cdd - np.nan_to_num(cdd_norm, nan=cdd.mean())

    # ── Storage: seasonal + responsive to weather shocks ───────────
    seasonal_storage = 2550 + 1100 * np.cos(2 * np.pi * (doy - 105) / 365.25)
    storage_noise = rng.normal(0, 60, size=n)
    storage = seasonal_storage + storage_noise

    # ── NG production: slow upward trend ───────────────────────────
    production = np.linspace(95, 103, n) + rng.normal(0, 1.0, size=n)

    # ── NG price: weather + storage deficit + AR(1) shocks ────────
    base_price = 3.00
    hdd_premium = hdd_anom * 0.015
    cdd_premium = cdd_anom * 0.012
    storage_5y_avg = seasonal_storage
    storage_deficit = (storage_5y_avg - storage) / 100.0
    storage_premium = storage_deficit * 0.03
    price_shocks = rng.normal(0, 0.10, size=n)
    price_ar = np.zeros(n)
    for i in range(1, n):
        price_ar[i] = (0.78 * price_ar[i - 1]
                        + price_shocks[i] * np.sqrt(1 - 0.78 ** 2))
    price = (base_price + hdd_premium + cdd_premium
             + storage_premium + price_ar)
    price = np.clip(price, 1.50, 12.00)

    panel = pd.DataFrame({
        "natgas_henry_hub_usd_mmbtu": price,
        "national_avg_temp_f": avg_f,
        "national_hdd": hdd,
        "national_cdd": cdd,
        "hdd_anomaly_30d": hdd_anom,
        "cdd_anomaly_30d": cdd_anom,
        "ng_storage_bcf": storage,
        "ng_storage_5y_avg_bcf": storage_5y_avg,
        "ng_production_bcfd": production,
    }, index=idx).rename_axis("date")

    _SYNTH_CACHE[key] = panel
    return panel


# --------------------------------------------------------------------------- #
# Combined panel
# --------------------------------------------------------------------------- #

def build_panel(cfg: Config, days: Optional[int] = None) -> pd.DataFrame:
    """Join NG price + storage + production + weather into a daily panel.

    Cross-Kalshi features are NOT added here — they're a snapshot at
    inference time, plumbed through `build_today_row`. (At training
    time we don't have a historical Kalshi market-price archive yet;
    that's a future enhancement.)
    """
    days = days or cfg.history_days_for_training
    price = fetch_natgas_henry_hub(cfg, days=days)
    storage = fetch_natgas_storage(cfg, days=days)
    production = fetch_natgas_production(cfg, days=days)
    weather = fetch_national_weather(cfg, days=days)
    panel = (price
             .join(storage, how="outer")
             .join(production, how="outer")
             .join(weather, how="outer"))
    panel = panel.sort_index()
    # Trim to the requested window. EIA's /seriesid endpoint returns
    # the full historical series regardless of start/end params (~5000
    # rows for daily NG since 2006), so the join produces a 20-year
    # panel even when we only want 3 years. Trim explicitly so weather
    # / storage NaN rows from far-history don't pollute training (the
    # median imputer would otherwise cancel out the weather signal).
    cutoff = pd.Timestamp.utcnow().normalize().tz_localize(None) \
              - pd.Timedelta(days=days)
    panel = panel.loc[panel.index >= cutoff]
    if ("ng_storage_bcf" in panel.columns
            and "ng_storage_5y_avg_bcf" in panel.columns):
        panel["ng_storage_deviation_bcf"] = (
            panel["ng_storage_bcf"] - panel["ng_storage_5y_avg_bcf"])
    return panel
