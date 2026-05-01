"""Configuration for the peak-load forecasting bot.

All knobs live here, loaded from .env at startup. Anything that
varies between dev / prod / per-region tuning belongs in .env, not
hardcoded in the source. Sensible defaults so a fresh clone runs
without configuring anything.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env vars work either way


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"


# --------------------------------------------------------------------------- #
# Region presets
# --------------------------------------------------------------------------- #

# Each ISO/region has its own load profile, weather pattern, and
# Kalshi market series. Defaults are tuned for ERCOT (Texas) since
# it has the most weather-driven volatility and active peak-load
# Kalshi markets. Override via ENERGY_REGION env var.
REGION_PRESETS = {
    "ercot": {
        "name": "ERCOT (Texas)",
        "eia_respondent": "ERCO",
        "noaa_station": "KAUS",       # Austin-Bergstrom — central proxy
        "default_lat": 30.27, "default_lon": -97.74,
        "summer_peak_mw": 78000,      # historical reference — for sanity-check
        "winter_peak_mw": 65000,
        "kalshi_series_prefix": "KXERCOTPL",
    },
    "nyiso": {
        "name": "NYISO (New York)",
        "eia_respondent": "NYIS",
        "noaa_station": "KNYC",
        "default_lat": 40.78, "default_lon": -73.97,
        "summer_peak_mw": 33000,
        "winter_peak_mw": 25000,
        "kalshi_series_prefix": "KXNYISOPL",
    },
    "pjm": {
        "name": "PJM",
        "eia_respondent": "PJM",
        "noaa_station": "KPHL",
        "default_lat": 39.95, "default_lon": -75.17,
        "summer_peak_mw": 165000,
        "winter_peak_mw": 140000,
        "kalshi_series_prefix": "KXPJMPL",
    },
    "caiso": {
        "name": "CAISO (California)",
        "eia_respondent": "CISO",
        "noaa_station": "KSFO",
        "default_lat": 37.62, "default_lon": -122.37,
        "summer_peak_mw": 50000,
        "winter_peak_mw": 35000,
        "kalshi_series_prefix": "KXCAISOPL",
    },
}


@dataclass
class Config:
    # ── Region ────────────────────────────────────────────────────────
    region: str
    region_meta: dict

    # ── API keys ──────────────────────────────────────────────────────
    eia_api_key: str
    noaa_token: str
    openweather_api_key: str
    kalshi_api_key_id: str
    kalshi_private_key_path: str
    kalshi_env: str           # "demo" | "prod"

    # ── Modeling ──────────────────────────────────────────────────────
    forecast_horizon_days: int = 1     # typically 1 — predict tomorrow's peak
    history_days_for_training: int = 730   # ~2 years of daily data
    test_size_days: int = 90
    random_state: int = 42
    target_column: str = "daily_peak_load_mw"

    # ── Threshold grid (in MW) ────────────────────────────────────────
    # Probabilities are computed at each threshold for Kalshi-comparison.
    # Centered around the region's typical seasonal peak ± a wide window.
    threshold_grid_mw: List[int] = field(default_factory=list)

    # ── Signal gates ──────────────────────────────────────────────────
    min_edge: float = 0.10            # |model_p − kalshi_p| ≥ 10pt to fire
    min_volume: int = 50              # market liquidity floor
    min_open_interest: int = 50
    max_spread_cents: int = 8

    # ── Sim risk caps ─────────────────────────────────────────────────
    # Mirror gas-prices / unemployment-claims defaults: 1 contract per
    # bet, 1 position open at a time, $1 cap. Daily cadence + hold-to-
    # resolution means a single position is the natural unit of risk.
    bet_size_cents: int = 100         # $1 cap per bet
    max_open_positions: int = 1       # one bet at a time, like the other bots
    max_total_exposure_cents: int = 200  # $2 ceiling
    max_bets_per_day: int = 5         # cap on signal-storm days

    # ── Hedge thresholds ──────────────────────────────────────────────
    # If a position's current price has moved enough from entry, open
    # an offsetting contract on the OTHER side to lock in P&L. Same
    # pattern as gas-prices/unemployment-claims hedge logic.
    hedge_enabled: bool = True
    hedge_profit_lock_cents: int = 20  # +20c in our favor → hedge
    hedge_stop_loss_cents: int = 15    # -15c against → hedge
    hedge_size_fraction: float = 1.0   # full hedge size

    # ── Validator thresholds (pre-trade gates beyond signals.py) ─────
    val_max_spread_cents: int = 8
    val_prob_bounds_cents_low: int = 5
    val_prob_bounds_cents_high: int = 95
    val_min_minutes_to_close: int = 30
    val_max_minutes_to_close: int = 60 * 24 * 7
    val_basis_risk_strike_window_mw: float = 1500
    val_basis_risk_max_hours_to_close: float = 4

    # ── Synthetic data fallback ───────────────────────────────────────
    # When no real APIs are configured, the loaders generate realistic
    # synthetic data so the pipeline runs end-to-end. Set to False to
    # require real data and fail loudly if any source is unreachable.
    use_synthetic_when_missing: bool = True

    # Demo mode for Kalshi markets. Default: off — when Kalshi has no
    # peak-load series listed for the region (the current state of the
    # exchange), the watchlist stays honestly empty rather than
    # surfacing fake tickers a user can't look up.
    #
    # Setting KALSHI_DEMO_MODE=true generates synthetic peak-load
    # markets with realistic-looking date+threshold tickers so the
    # full sim pipeline (open position, hold, mark-to-market, close on
    # resolution) can be exercised end-to-end against the dashboard.
    # Demo positions are clearly tagged with DEMO in the decision
    # metadata so they can be filtered out later.
    kalshi_demo_mode: bool = False

    # ── Output paths (filled in __post_init__) ────────────────────────
    model_path: Path = field(default_factory=lambda: MODELS_DIR / "peak_load.pkl")
    daily_csv_path: Path = field(default_factory=lambda: OUTPUTS_DIR / "daily_signals.csv")
    daily_json_path: Path = field(default_factory=lambda: OUTPUTS_DIR / "daily_signals.json")


def load_config() -> Config:
    """Read .env / OS environment and return a populated Config."""
    region = os.environ.get("ENERGY_REGION", "ercot").lower()
    if region not in REGION_PRESETS:
        raise ValueError(
            f"ENERGY_REGION={region!r} not recognized. "
            f"Pick one of: {sorted(REGION_PRESETS)}"
        )
    meta = REGION_PRESETS[region]

    # Threshold grid: anchor on summer peak, span ±15% in 1.5K steps so
    # we cover a full distribution of plausible Kalshi strikes for the
    # region. ERCOT default works out to ~17 thresholds spanning
    # 66K-90K MW. Override via THRESHOLD_GRID_MW="60000,65000,..." if
    # you want a custom set.
    grid_env = os.environ.get("THRESHOLD_GRID_MW", "").strip()
    if grid_env:
        thresholds = [int(x) for x in grid_env.split(",") if x.strip()]
    else:
        peak = meta["summer_peak_mw"]
        lo = int(peak * 0.85 // 1500) * 1500
        hi = int(peak * 1.15 // 1500) * 1500
        thresholds = list(range(lo, hi + 1, 1500))

    return Config(
        region=region,
        region_meta=meta,
        eia_api_key=os.environ.get("EIA_API_KEY", ""),
        noaa_token=os.environ.get("NOAA_TOKEN", ""),
        openweather_api_key=os.environ.get("OPENWEATHER_API_KEY", ""),
        kalshi_api_key_id=os.environ.get("KALSHI_API_KEY_ID", ""),
        kalshi_private_key_path=os.environ.get("KALSHI_PRIVATE_KEY_PATH", ""),
        kalshi_env=os.environ.get("KALSHI_ENV", "prod"),
        forecast_horizon_days=int(os.environ.get("FORECAST_HORIZON_DAYS", "1")),
        history_days_for_training=int(os.environ.get("HISTORY_DAYS", "730")),
        test_size_days=int(os.environ.get("TEST_SIZE_DAYS", "90")),
        target_column=os.environ.get("TARGET_COLUMN", "daily_peak_load_mw"),
        threshold_grid_mw=thresholds,
        min_edge=float(os.environ.get("MIN_EDGE", "0.10")),
        min_volume=int(os.environ.get("MIN_VOLUME", "50")),
        min_open_interest=int(os.environ.get("MIN_OPEN_INTEREST", "50")),
        max_spread_cents=int(os.environ.get("MAX_SPREAD_CENTS", "8")),
        bet_size_cents=int(os.environ.get("BET_SIZE_CENTS", "100")),
        max_open_positions=int(os.environ.get("MAX_OPEN_POSITIONS", "1")),
        max_total_exposure_cents=int(os.environ.get(
            "MAX_TOTAL_EXPOSURE_CENTS", "200")),
        max_bets_per_day=int(os.environ.get("MAX_BETS_PER_DAY", "5")),
        hedge_enabled=(
            os.environ.get("HEDGE_ENABLED", "true").lower()
            in ("true", "1", "yes")),
        hedge_profit_lock_cents=int(os.environ.get(
            "HEDGE_PROFIT_LOCK_CENTS", "20")),
        hedge_stop_loss_cents=int(os.environ.get(
            "HEDGE_STOP_LOSS_CENTS", "15")),
        hedge_size_fraction=float(os.environ.get(
            "HEDGE_SIZE_FRACTION", "1.0")),
        use_synthetic_when_missing=(
            os.environ.get("USE_SYNTHETIC_WHEN_MISSING", "true").lower()
            in ("true", "1", "yes")),
        kalshi_demo_mode=(
            os.environ.get("KALSHI_DEMO_MODE", "false").lower()
            in ("true", "1", "yes")),
    )
