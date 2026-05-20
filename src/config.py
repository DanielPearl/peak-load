"""Configuration for the Natural Gas Price prediction bot.

Targets Kalshi's KXNATGASD daily series — Pyth-settled Henry Hub
natural gas spot price, daily 5pm EDT settlement, $/MMBTU thresholds
at $0.005 spacing.

All knobs live here. Sensible defaults so a fresh clone with valid
EIA + Kalshi credentials runs end-to-end.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"


# --------------------------------------------------------------------------- #
# Cross-Kalshi feature markets
# --------------------------------------------------------------------------- #

# Kalshi series whose implied probabilities feed in as daily features.
# Each entry: (series_ticker, label, why-it-matters).
#
# Walk-forward feature selection prunes whichever channels don't
# survive the stability filter — so it's safe to be inclusive here
# and let the data decide.
CROSS_KALSHI_FEATURE_SERIES = [
    # ── Crude oil — heat-content competitor + macro energy proxy ─────
    ("KXBRENTD",  "brent_daily",  "Brent crude daily — global oil + Russia premium"),
    ("KXWTI",     "wti_daily",    "WTI crude daily — US oil reference"),
    ("KXHOILW",   "hoil_weekly",  "Heating oil weekly — winter heating proxy"),
    # ── Retail gasoline — reflects refining margins / driving demand ──
    ("KXAAAGASD", "aaa_gas_daily", "AAA retail gas daily"),
    ("KXAAAGASW", "aaa_gas_weekly", "AAA retail gas weekly"),
    # ── Geopolitics / war — shock features ────────────────────────────
    ("KXRUSSIAUKR",   "russia_ukraine",  "Russia/Ukraine ceasefire / escalation"),
    ("KXIRANISRAEL",  "iran_israel",     "Iran/Israel conflict markets"),
    ("KXISRAELHAMAS", "israel_hamas",    "Israel/Hamas ceasefire markets"),
    ("KXVENZ",        "venezuela",       "Venezuela / Maduro markets — oil sanctions"),
    # ── Hurricane / storm — Gulf production + LNG terminal disruption ─
    # Gulf-coast strikes hit (a) ~17% of US NG production from offshore
    # platforms, (b) every major LNG export terminal — Sabine Pass,
    # Corpus Christi, Cameron, Freeport, Calcasieu Pass — so any active
    # Gulf threat is BOTH a supply hit AND an export-demand hit. Pull
    # FL paths, the LA/TX paths where the terminals sit, and the
    # season-total markets that proxy overall Atlantic hurricane risk.
    ("KXHURPATHFLA",  "hurr_florida",    "Hurricane hits FL"),
    ("KXHURPATHLA",   "hurr_louisiana",  "Hurricane hits LA — Sabine/Cameron/CCP corridor"),
    ("KXHURPATHTX",   "hurr_texas",      "Hurricane hits TX — Freeport/CCorpus corridor"),
    ("KXHURCATFL",    "hurr_warning_fl", "Hurricane warning FL"),
    ("KXHURCTOTMAJ",  "hurr_total_major","Major hurricanes total this season"),
    ("KXHURNAMED",    "hurr_named_total","Named storms total — Atlantic season activity"),
    ("KXHURCAT5",     "hurr_cat5",       "Any Cat-5 hurricane this season"),
    # ── Macro / Fed / policy — discount-rate effect on commodities ────
    ("KXFEDDECISION", "fed_decision",    "Fed rate decision next meeting"),
    ("KXRECESSION",   "recession",       "US recession this year"),
]


# --------------------------------------------------------------------------- #
# Config dataclass
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # ── Target market ─────────────────────────────────────────────────
    # KXNATGASD = Henry Hub daily NG spot, Pyth-settled at 5pm EDT,
    # thresholds in $/MMBTU at $0.005 spacing.
    kalshi_series_prefix: str = "KXNATGASD"
    target_column: str = "natgas_henry_hub_usd_mmbtu"

    # ── API keys ──────────────────────────────────────────────────────
    eia_api_key: str = ""
    noaa_token: str = ""
    openweather_api_key: str = ""
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""

    # ── Modeling ──────────────────────────────────────────────────────
    forecast_horizon_days: int = 1
    # Full EIA Henry Hub history goes back to 1997-01-07 (~10500 daily
    # obs). 10000 captures roughly 27 years; the loader paginates past
    # EIA's 5000-row per-request cap. Weather/storage/cross-Kalshi
    # companion data thins out before ~2010, so the median imputer +
    # walk-forward selector handle the partial-coverage early period.
    history_days_for_training: int = 10000
    test_size_days: int = 120
    random_state: int = 42

    # ── Threshold grid ($/MMBTU) ─────────────────────────────────────
    # Static fallback covers historical NG range; run_daily.py builds
    # a dynamic grid centered ±$1 around today's spot at training-time
    # residual_std spacing.
    threshold_grid_usd: List[float] = field(default_factory=list)

    # ── Signal gates ──────────────────────────────────────────────────
    min_edge: float = 0.05            # |model_p − kalshi_p| ≥ 5pt
    min_volume: int = 25              # NG strikes thinner than electricity
    min_open_interest: int = 25
    max_spread_cents: int = 10        # NG spreads can be wider

    # ── Sim risk caps ─────────────────────────────────────────────────
    bet_size_cents: int = 100
    max_open_positions: int = 1
    max_total_exposure_cents: int = 200
    max_bets_per_day: int = 5

    # ── Hedge thresholds ──────────────────────────────────────────────
    hedge_enabled: bool = True
    hedge_profit_lock_cents: int = 20
    hedge_stop_loss_cents: int = 15
    hedge_size_fraction: float = 1.0

    # ── Validator thresholds ──────────────────────────────────────────
    val_max_spread_cents: int = 10
    # Skip contracts at the extremes. Tightened from [15, 85] to [20, 80]:
    # at 85c the half-spread + Kalshi fee eat ~30% of the 15c upside, and
    # the model is least informative near the tails. [20, 80] preserves
    # the actionable middle band without asking the model to confidently
    # call near-certainty events.
    val_prob_bounds_cents_low: int = 20
    val_prob_bounds_cents_high: int = 80
    # Hard cap on what we'll pay for either side, regardless of edge. At
    # 75c+ the loss-vs-gain ratio is 3:1+ and a single missed call dwarfs
    # many wins. Variance protection independent of edge math. Set 100
    # to disable.
    val_max_entry_price_cents: int = 75
    val_min_minutes_to_close: int = 30
    val_max_minutes_to_close: int = 60 * 24 * 7
    # Strike-window in $/MMBTU. 5¢ above/below spot is a "close to the
    # money" zone where settle-print noise dominates → don't trade
    # near the strike right before close.
    val_basis_risk_strike_window_usd: float = 0.05
    val_basis_risk_max_hours_to_close: float = 4

    # ── Synthetic data fallback ───────────────────────────────────────
    # EIA / weather can be synthetic; Kalshi is REAL ONLY (no demo mode).
    use_synthetic_when_missing: bool = True

    # ── Reference data ────────────────────────────────────────────────
    # Henry Hub coordinates (Erath, LA — physical hub in southern LA):
    henry_hub_lat: float = 29.97
    henry_hub_lon: float = -91.50
    # Population/consumption-weighted weather stations for national HDD
    # /CDD aggregation. Weights sum to ~0.72 — the rest absorbs into
    # the synthetic baseline. Real production should refresh these
    # against EIA's gas-consumption-by-state data.
    #
    # The `region` tag groups stations into NG-demand regions so the
    # feature builder can compute regional HDD/CDD aggregates. Regions
    # match EIA's natural-gas demand reporting groups (Northeast,
    # Midwest, South, West) plus a separate "Gulf" channel for the
    # production/LNG-export footprint that hurricanes hit.
    weather_reference_stations: List[dict] = field(default_factory=lambda: [
        {"name": "NYC",     "lat": 40.78, "lon": -73.97, "weight": 0.18, "region": "northeast"},
        {"name": "Boston",  "lat": 42.36, "lon": -71.06, "weight": 0.07, "region": "northeast"},
        {"name": "Chicago", "lat": 41.88, "lon": -87.63, "weight": 0.14, "region": "midwest"},
        {"name": "Atlanta", "lat": 33.75, "lon": -84.39, "weight": 0.08, "region": "south"},
        {"name": "Houston", "lat": 29.76, "lon": -95.36, "weight": 0.10, "region": "gulf"},
        {"name": "Phoenix", "lat": 33.45, "lon": -112.07,"weight": 0.06, "region": "west"},
        {"name": "Denver",  "lat": 39.74, "lon": -104.99,"weight": 0.05, "region": "west"},
        {"name": "Seattle", "lat": 47.61, "lon": -122.33,"weight": 0.04, "region": "west"},
    ])

    # LNG export-terminal weather stations. Tropical storms, fog, and
    # high winds at these terminals disrupt loadings → bullish for
    # domestic NG (less gas leaves the country) on the short horizon
    # but bearish for the next monthly print (cargoes that couldn't
    # load build up stranded gas). Weights ≈ each terminal's share of
    # peak US LNG export capacity (~13 Bcf/day total in 2025-26).
    lng_terminal_stations: List[dict] = field(default_factory=lambda: [
        {"name": "Sabine_Pass",    "lat": 29.73, "lon": -93.87, "weight": 0.30},  # Cheniere LA
        {"name": "Corpus_Christi", "lat": 27.83, "lon": -97.39, "weight": 0.18},  # Cheniere TX
        {"name": "Cameron",        "lat": 29.78, "lon": -93.33, "weight": 0.15},  # Sempra LA
        {"name": "Freeport",       "lat": 28.95, "lon": -95.36, "weight": 0.16},  # Freeport TX
        {"name": "Calcasieu_Pass", "lat": 29.79, "lon": -93.34, "weight": 0.12},  # Venture Global LA
        {"name": "Cove_Point",     "lat": 38.39, "lon": -76.40, "weight": 0.05},  # Dominion MD
        {"name": "Elba_Island",    "lat": 32.01, "lon": -80.95, "weight": 0.04},  # Kinder Morgan GA
    ])

    # ── Cross-Kalshi feature markets ──────────────────────────────────
    cross_kalshi_series: list = field(
        default_factory=lambda: list(CROSS_KALSHI_FEATURE_SERIES))

    # ── Output paths ──────────────────────────────────────────────────
    model_path: Path = field(
        default_factory=lambda: MODELS_DIR / "natgas_price.pkl")
    daily_csv_path: Path = field(
        default_factory=lambda: OUTPUTS_DIR / "daily_signals.csv")
    daily_json_path: Path = field(
        default_factory=lambda: OUTPUTS_DIR / "daily_signals.json")


def load_config() -> Config:
    """Read .env / OS environment and return a populated Config."""
    grid_env = os.environ.get("THRESHOLD_GRID_USD", "").strip()
    if grid_env:
        thresholds = [float(x) for x in grid_env.split(",") if x.strip()]
    else:
        # $1.50 to $8.00 every $0.05 = 131 strikes. Covers historical
        # NG spot range from glut bottoms ($1.50 in 2020) to winter
        # spikes ($8+ in 2022). Dynamic grid in run_daily narrows.
        thresholds = [round(1.50 + 0.05 * i, 3) for i in range(131)]

    return Config(
        eia_api_key=os.environ.get("EIA_API_KEY", ""),
        noaa_token=os.environ.get("NOAA_TOKEN", ""),
        openweather_api_key=os.environ.get("OPENWEATHER_API_KEY", ""),
        kalshi_api_key_id=os.environ.get("KALSHI_API_KEY_ID", ""),
        kalshi_private_key_path=os.environ.get("KALSHI_PRIVATE_KEY_PATH", ""),
        forecast_horizon_days=int(os.environ.get("FORECAST_HORIZON_DAYS", "1")),
        history_days_for_training=int(os.environ.get("HISTORY_DAYS", "10000")),
        test_size_days=int(os.environ.get("TEST_SIZE_DAYS", "120")),
        target_column=os.environ.get("TARGET_COLUMN",
                                       "natgas_henry_hub_usd_mmbtu"),
        threshold_grid_usd=thresholds,
        min_edge=float(os.environ.get("MIN_EDGE", "0.05")),
        min_volume=int(os.environ.get("MIN_VOLUME", "25")),
        min_open_interest=int(os.environ.get("MIN_OPEN_INTEREST", "25")),
        max_spread_cents=int(os.environ.get("MAX_SPREAD_CENTS", "10")),
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
    )
