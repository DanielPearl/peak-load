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

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    rows = _fetch_eia_seriesid_paged(
        cfg, "NG.RNGWHHD.D", start.isoformat(), end.isoformat())
    if not rows:
        raise RuntimeError("EIA Henry Hub returned no rows")
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (df.set_index("period")[["value"]]
              .rename(columns={"value": "natgas_henry_hub_usd_mmbtu"})
              .sort_index())


def _fetch_eia_seriesid_paged(cfg: Config, series_id: str,
                                start: str, end: str,
                                page_size: int = 5000) -> list:
    """Paginate EIA's /seriesid endpoint past its per-request row cap.

    EIA caps `length` at 5000 rows. 27 years of daily Henry Hub is
    ~10500 rows, so a single call truncates the historical tail. Loop
    with `offset` until a short page (< page_size) comes back.
    """
    url = f"{EIA_BASE}/seriesid/{series_id}"
    offset = 0
    out: list = []
    while True:
        params = {
            "api_key": cfg.eia_api_key,
            "start": start,
            "end": end,
            "length": page_size,
            "offset": offset,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        page = r.json().get("response", {}).get("data", []) or []
        if not page:
            break
        out.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        # Hard cap to avoid runaway loops if EIA misbehaves.
        if offset > 100000:
            log.warning("EIA pagination hit safety cap at %d rows", offset)
            break
    return out


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
    base_cols = ["national_avg_temp_f", "national_hdd", "national_cdd",
                 "hdd_anomaly_30d", "cdd_anomaly_30d",
                 "national_wind_mph", "national_humidity_pct",
                 "lng_terminal_avg_temp_f", "lng_terminal_wind_mph",
                 "lng_terminal_storm_flag",
                 "gulf_max_wind_mph", "gulf_storm_active"]
    region_cols = [c for c in panel.columns if c.startswith("region_")]
    return panel[base_cols + region_cols]


# --------------------------------------------------------------------------- #
# Weather forecast (next-day) — for inference row
# --------------------------------------------------------------------------- #

def fetch_weather_forecast(cfg: Config, days_ahead: int = 7) -> pd.DataFrame:
    """National + regional + LNG-terminal weather forecast for the next
    N days, including wind / humidity / storm-flag derivations.
    """
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
        row = {
            "date": end + pd.Timedelta(days=d),
            "national_avg_temp_f": new_temp,
            "national_hdd": max(0, 65 - new_temp),
            "national_cdd": max(0, new_temp - 65),
            "hdd_anomaly_30d": float(last.get("hdd_anomaly_30d", 0.0))
                                + rng.normal(0, 1.0),
            "cdd_anomaly_30d": float(last.get("cdd_anomaly_30d", 0.0))
                                + rng.normal(0, 1.0),
        }
        # Carry forward every other weather channel produced by the
        # synthetic panel so build_today_row sees the same column set
        # at inference as at training. Add tiny noise on wind so the
        # forecast row isn't a perfect persistence — otherwise wind/
        # storm flags would be deterministic and the model would treat
        # them as constants.
        for col in last.index:
            if col in row:
                continue
            val = last[col]
            if col.endswith("_storm_flag") or col == "gulf_storm_active":
                row[col] = int(val) if pd.notna(val) else 0
            elif col.endswith("_mph"):
                row[col] = float(val) + rng.normal(0, 1.5) if pd.notna(val) else 0.0
            else:
                row[col] = float(val) if pd.notna(val) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def _fetch_openweather_national_forecast(cfg: Config, days_ahead: int
                                          ) -> pd.DataFrame:
    """OpenWeather One Call 3.0 — fetch each reference station + each
    LNG-terminal station, then derive the national/regional/LNG
    aggregates plus wind / humidity / storm-flag channels.

    Per station per day we collect (temp, wind_speed, humidity, alerts).
    National aggregates are weight-averaged across reference stations;
    regional aggregates split by ``region`` tag; LNG-terminal aggregates
    use the lng_terminal_stations list with their own weights. Storm
    flags fire when any station's wind > 35mph.
    """
    rows_by_date: Dict[pd.Timestamp, dict] = {}
    nat_total_w = sum(s.get("weight", 0.0)
                      for s in cfg.weather_reference_stations) or 1.0
    lng_total_w = sum(s.get("weight", 0.0)
                      for s in cfg.lng_terminal_stations) or 1.0
    regions = {s.get("region", "unassigned")
               for s in cfg.weather_reference_stations}
    region_total_w = {
        reg: sum(s.get("weight", 0.0)
                 for s in cfg.weather_reference_stations
                 if s.get("region") == reg) or 1.0
        for reg in regions
    }

    def _accumulate(stations: list, kind: str, weight_total: float,
                    region_filter: Optional[str] = None) -> None:
        """Call OpenWeather for each station and accumulate weighted
        contributions onto the matching row. `kind` is 'national',
        'lng', or 'region:<name>' — controls which sub-dict to update.
        """
        for st in stations:
            if region_filter and st.get("region") != region_filter:
                continue
            params = {
                "lat": st["lat"], "lon": st["lon"],
                "exclude": "current,minutely,hourly",
                "units": "imperial",
                "appid": cfg.openweather_api_key,
            }
            try:
                r = requests.get(
                    "https://api.openweathermap.org/data/3.0/onecall",
                    params=params, timeout=15)
                r.raise_for_status()
                payload = r.json()
            except Exception as exc:  # noqa: BLE001
                log.warning("OpenWeather %s station %s failed: %s",
                            kind, st.get("name"), exc)
                continue
            alerts_active = bool(payload.get("alerts"))
            for d in (payload.get("daily") or [])[:days_ahead]:
                ts = pd.to_datetime(d["dt"], unit="s", utc=True
                                     ).tz_localize(None).normalize()
                temp = (d.get("temp") or {}).get("day", float("nan"))
                wind = d.get("wind_speed", float("nan"))
                hum = d.get("humidity", float("nan"))
                w = st.get("weight", 0.0) / weight_total
                row = rows_by_date.setdefault(ts, {})
                bucket = row.setdefault(kind, {
                    "temp_w": 0.0, "wind_w": 0.0, "hum_w": 0.0,
                    "max_wind": 0.0, "weight_used": 0.0,
                    "alerts": 0, "storm_flag": 0,
                })
                if not np.isnan(temp):
                    bucket["temp_w"] += float(temp) * w
                if not np.isnan(wind):
                    bucket["wind_w"] += float(wind) * w
                    bucket["max_wind"] = max(bucket["max_wind"], float(wind))
                    if float(wind) > 35.0:
                        bucket["storm_flag"] = 1
                if not np.isnan(hum):
                    bucket["hum_w"] += float(hum) * w
                bucket["weight_used"] += w
                if alerts_active:
                    bucket["alerts"] = 1

    _accumulate(cfg.weather_reference_stations, "national", nat_total_w)
    _accumulate(cfg.lng_terminal_stations, "lng", lng_total_w)
    for reg in regions:
        _accumulate(cfg.weather_reference_stations,
                    f"region:{reg}", region_total_w[reg], region_filter=reg)

    out = []
    for ts in sorted(rows_by_date):
        buckets = rows_by_date[ts]
        nat = buckets.get("national", {})
        denom = nat.get("weight_used", 0.0) or 1.0
        avg_temp = nat.get("temp_w", 0.0) / denom
        avg_wind = nat.get("wind_w", 0.0) / denom
        avg_hum = nat.get("hum_w", 0.0) / denom
        row = {
            "date": ts,
            "national_avg_temp_f": avg_temp,
            "national_hdd": max(0.0, 65 - avg_temp),
            "national_cdd": max(0.0, avg_temp - 65),
            "hdd_anomaly_30d": float("nan"),
            "cdd_anomaly_30d": float("nan"),
            "national_wind_mph": avg_wind,
            "national_humidity_pct": avg_hum,
        }
        # LNG terminal aggregate.
        lng = buckets.get("lng")
        if lng and lng.get("weight_used"):
            denom_l = lng["weight_used"] or 1.0
            row["lng_terminal_avg_temp_f"] = lng["temp_w"] / denom_l
            row["lng_terminal_wind_mph"] = lng["wind_w"] / denom_l
            # Storm flag: terminal wind > 35mph OR any active alert.
            row["lng_terminal_storm_flag"] = int(
                lng["storm_flag"] or lng["alerts"])
        # Gulf channel: max wind across Gulf-region reference stations.
        gulf = buckets.get("region:gulf")
        if gulf and gulf.get("weight_used"):
            row["gulf_max_wind_mph"] = float(gulf["max_wind"])
            # Storm flag: any wind > 40mph or active alert in Gulf.
            row["gulf_storm_active"] = int(
                gulf["max_wind"] > 40.0 or gulf["alerts"])
        # Per-region temp / HDD / CDD.
        for reg in regions:
            bucket = buckets.get(f"region:{reg}")
            if not (bucket and bucket.get("weight_used")):
                continue
            denom_r = bucket["weight_used"] or 1.0
            reg_temp = bucket["temp_w"] / denom_r
            row[f"region_{reg}_temp_f"] = reg_temp
            row[f"region_{reg}_hdd"] = max(0.0, 65 - reg_temp)
            row[f"region_{reg}_cdd"] = max(0.0, reg_temp - 65)
        out.append(row)
    return pd.DataFrame(out).set_index("date")


# --------------------------------------------------------------------------- #
# Forecast-revision tracking
# --------------------------------------------------------------------------- #
#
# A "forecast revision" is the change between today's forecast and the
# forecast we made for the same horizon during a prior run. Big upward
# revisions in winter HDD often precede price rallies as the market
# re-prices demand. We persist the most recent forecast snapshot to
# disk so the next run can compute deltas — no historical store needed.

_FORECAST_HISTORY_VERSION = 1


def load_previous_forecast(path: Path) -> Optional[pd.Series]:
    """Read the prior run's saved forecast row, if any. Returns a
    pandas Series of the day-ahead forecast values keyed by column."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not load previous forecast at %s: %s", path, exc)
        return None
    if payload.get("version") != _FORECAST_HISTORY_VERSION:
        return None
    data = payload.get("forecast") or {}
    return pd.Series({k: float(v) for k, v in data.items()
                       if isinstance(v, (int, float))}, dtype=float)


def save_current_forecast(path: Path, forecast_row: pd.Series) -> None:
    """Persist today's day-ahead forecast row so the next run can
    compute revisions. Strings/timestamps stripped — numeric only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric = {k: float(v) for k, v in forecast_row.items()
                if isinstance(v, (int, float)) and not pd.isna(v)}
    payload = {
        "version": _FORECAST_HISTORY_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "forecast_date": str(forecast_row.name) if forecast_row.name is not None
                         else None,
        "forecast": numeric,
    }
    path.write_text(json.dumps(payload, indent=2))


def compute_forecast_revisions(current: pd.Series,
                                 previous: Optional[pd.Series]
                                 ) -> pd.Series:
    """Compute revision deltas for the columns we want to track. Only
    weather channels — price/cross-Kalshi revisions are different
    objects. Missing previous → NaN deltas (imputer fills median).
    """
    tracked = [
        "national_avg_temp_f", "national_hdd", "national_cdd",
        "national_wind_mph", "national_humidity_pct",
        "lng_terminal_wind_mph", "gulf_max_wind_mph",
    ]
    region_cols = [c for c in current.index
                   if c.startswith("region_") and (c.endswith("_hdd")
                                                    or c.endswith("_cdd")
                                                    or c.endswith("_temp_f"))]
    out: Dict[str, float] = {}
    for col in tracked + region_cols:
        if col not in current.index:
            continue
        cur = current.get(col, float("nan"))
        if previous is None or col not in previous.index:
            out[f"{col}_revision_1d"] = float("nan")
            continue
        prev = previous.get(col, float("nan"))
        if pd.isna(cur) or pd.isna(prev):
            out[f"{col}_revision_1d"] = float("nan")
        else:
            out[f"{col}_revision_1d"] = float(cur) - float(prev)
    return pd.Series(out, dtype=float)


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

    Each series fetch is independent, so we run them in parallel via a
    threadpool — turns ~14 sequential GETs (~30s) into ~3s wall time.

    Returns a pandas Series with one entry per (label, derived stat)
    pair. Dormant series produce NaN so the median imputer fills.
    """
    out: Dict[str, float] = {}
    if not (cfg.kalshi_api_key_id and cfg.kalshi_private_key_path):
        return pd.Series(dtype=float)

    from kalshi_sdk import batch_fetch
    from .kalshi import _SignedClient

    try:
        client = _SignedClient(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Kalshi cross-feature client init failed: %s", exc)
        return pd.Series(dtype=float)

    def _fetch_series(series_ticker: str) -> list:
        resp = client.get("/markets",
                          params={"series_ticker": series_ticker,
                                  "status": "open", "limit": 200})
        return resp.get("markets", []) or []

    series_keys = [series for series, _label, _why in cfg.cross_kalshi_series]
    label_for = {series: label for series, label, _why in cfg.cross_kalshi_series}
    fetched = batch_fetch(series_keys, _fetch_series, max_workers=8)

    for series_ticker, label, _why in cfg.cross_kalshi_series:
        ms = fetched.get(series_ticker, [])
        asks: list[float] = []
        vols: list[int] = []
        for m in ms:
            price = _market_price_prob(m)
            if price is not None:
                asks.append(price)
            v = m.get("volume") or m.get("volume_fp") or 0
            try:
                vols.append(int(float(v)) if v else 0)
            except (TypeError, ValueError):
                vols.append(0)
        out[f"xk_{label}_n_open"] = float(len(ms))
        out[f"xk_{label}_vol_sum"] = float(sum(vols))
        if asks:
            out[f"xk_{label}_avg_prob"] = float(np.mean(asks))
            out[f"xk_{label}_max_prob"] = float(np.max(asks))
        else:
            out[f"xk_{label}_avg_prob"] = float("nan")
            out[f"xk_{label}_max_prob"] = float("nan")
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

    # ── Wind + humidity (national-weighted averages) ────────────────
    # Wind: log-normal with a seasonal lift in winter (storm season).
    wind = np.clip(
        7.0 + 3.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)
        + rng.normal(0, 2.5, size=n), 0, 60)
    # Humidity: % — anti-correlated with cold snaps in winter, peaks
    # mid-summer.
    humidity = np.clip(
        62 + 12 * np.cos(2 * np.pi * (doy - 200) / 365.25) * -1
        + rng.normal(0, 6, size=n), 10, 100)

    # ── Regional temps: each region tracks national with its own
    # seasonal offset + local noise, then we derive HDD/CDD per region.
    # Synthetic but realistic enough that the model can learn that
    # e.g. an extreme northeast cold snap matters more than a mild
    # west one for total NG demand. Anomaly = region HDD/CDD minus its
    # own 30-day rolling normal.
    region_offsets = {
        "northeast": -8.0,
        "midwest":   -6.0,
        "south":     +6.0,
        "west":      +2.0,
        "gulf":      +9.0,
    }
    region_cols: Dict[str, np.ndarray] = {}
    for reg, offset in region_offsets.items():
        local_noise = rng.normal(0, 3, size=n)
        reg_temp = avg_f + offset + local_noise
        reg_hdd = np.maximum(65 - reg_temp, 0)
        reg_cdd = np.maximum(reg_temp - 65, 0)
        reg_hdd_norm = (pd.Series(reg_hdd, index=idx)
                          .rolling(30, min_periods=10).mean().values)
        reg_cdd_norm = (pd.Series(reg_cdd, index=idx)
                          .rolling(30, min_periods=10).mean().values)
        region_cols[f"region_{reg}_temp_f"] = reg_temp
        region_cols[f"region_{reg}_hdd"] = reg_hdd
        region_cols[f"region_{reg}_cdd"] = reg_cdd
        region_cols[f"region_{reg}_hdd_anomaly_30d"] = (
            reg_hdd - np.nan_to_num(reg_hdd_norm, nan=reg_hdd.mean()))
        region_cols[f"region_{reg}_cdd_anomaly_30d"] = (
            reg_cdd - np.nan_to_num(reg_cdd_norm, nan=reg_cdd.mean()))

    # ── LNG-terminal-weighted weather. Gulf-dominated, so it tracks
    # the Gulf region temp closely with extra wind variance (most
    # disruption is wind/storm-driven, not temp). storm_flag fires
    # when terminal wind > 35mph — proxy for tropical-storm-level
    # loadings disruption.
    gulf_temp = region_cols["region_gulf_temp_f"]
    lng_terminal_temp_f = gulf_temp + rng.normal(0, 1.5, size=n)
    lng_terminal_wind = np.clip(
        wind + 4.0 + rng.normal(0, 4.0, size=n), 0, 100)
    lng_terminal_storm_flag = (lng_terminal_wind > 35).astype(int)

    # ── Gulf storm flag — fires more readily than LNG-terminal one;
    # tracks tropical-cyclone activity in hurricane season.
    hurricane_season = np.asarray((idx.month >= 6) & (idx.month <= 11),
                                    dtype=int)
    gulf_max_wind = np.clip(
        wind + 6.0 * hurricane_season + rng.normal(0, 5.0, size=n), 0, 120)
    gulf_storm_active = (gulf_max_wind > 40).astype(int)

    # ── Storage: seasonal + responsive to weather shocks ───────────
    seasonal_storage = 2550 + 1100 * np.cos(2 * np.pi * (doy - 105) / 365.25)
    storage_noise = rng.normal(0, 60, size=n)
    storage = seasonal_storage + storage_noise

    # ── NG production: slow upward trend, hit by Gulf storms ───────
    production = (np.linspace(95, 103, n)
                  - gulf_storm_active * rng.uniform(2, 5, size=n)
                  + rng.normal(0, 1.0, size=n))

    # ── NG price: weather + storage deficit + AR(1) shocks + Gulf
    # storm premium (production-loss / LNG-disruption tug-of-war).
    base_price = 3.00
    hdd_premium = hdd_anom * 0.015
    cdd_premium = cdd_anom * 0.012
    storage_5y_avg = seasonal_storage
    storage_deficit = (storage_5y_avg - storage) / 100.0
    storage_premium = storage_deficit * 0.03
    storm_premium = gulf_storm_active * 0.08
    price_shocks = rng.normal(0, 0.10, size=n)
    price_ar = np.zeros(n)
    for i in range(1, n):
        price_ar[i] = (0.78 * price_ar[i - 1]
                        + price_shocks[i] * np.sqrt(1 - 0.78 ** 2))
    price = (base_price + hdd_premium + cdd_premium
             + storage_premium + storm_premium + price_ar)
    price = np.clip(price, 1.50, 12.00)

    panel_cols = {
        "natgas_henry_hub_usd_mmbtu": price,
        "national_avg_temp_f": avg_f,
        "national_hdd": hdd,
        "national_cdd": cdd,
        "hdd_anomaly_30d": hdd_anom,
        "cdd_anomaly_30d": cdd_anom,
        "national_wind_mph": wind,
        "national_humidity_pct": humidity,
        "lng_terminal_avg_temp_f": lng_terminal_temp_f,
        "lng_terminal_wind_mph": lng_terminal_wind,
        "lng_terminal_storm_flag": lng_terminal_storm_flag,
        "gulf_max_wind_mph": gulf_max_wind,
        "gulf_storm_active": gulf_storm_active,
        "ng_storage_bcf": storage,
        "ng_storage_5y_avg_bcf": storage_5y_avg,
        "ng_production_bcfd": production,
    }
    panel_cols.update(region_cols)
    panel = pd.DataFrame(panel_cols, index=idx).rename_axis("date")

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
