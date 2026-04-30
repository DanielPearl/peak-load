# peak-load — Energy / Weather Kalshi Forecasting Bot

Daily-cadence forecasting model for **electricity peak load**, designed
to compare model probabilities against Kalshi peak-load market prices
and surface mispricings.

The model predicts `daily_peak_load_mw` (or, when renewables data is
available, `net_peak_load_mw = load − solar − wind`) for a configurable
ISO region — ERCOT, NYISO, PJM, or CAISO — converts the point forecast
into per-threshold probabilities via empirical residual std, then
compares against Kalshi market prices.

## Quick start

```bash
git clone git@github.com:DanielPearl/peak-load.git
cd peak-load

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # populate optional API keys

python scripts/train.py      # train the model (uses synthetic data
                             # if EIA / NOAA / OpenWeather keys are
                             # not configured — fully runnable
                             # without any real credentials)

python scripts/run_daily.py  # build today's signals
```

Outputs land in:

```
outputs/daily_signals.csv    one row per Kalshi market scored today
outputs/daily_signals.json   same data plus model metadata
data/sim.db                  SQLite mirror for the unified dashboard
```

## Architecture

```
src/
  config.py         all knobs, .env loader, region presets
  data_loaders.py   EIA load + NOAA weather + OpenWeather forecast
                    (real-API stubs + synthetic fallback)
  features.py       CDD/HDD, heat index, lags, rolling stats, calendar
  model.py          Ridge baseline + HGB stronger model, residual-std
                    probability conversion
  kalshi.py         signed REST client + synthetic market generator
  signals.py        edge computation, liquidity filters, BUY/NO_TRADE
scripts/
  train.py          one-shot training entry point
  run_daily.py      daily signals pipeline (cron-ready)
```

### Data flow

```
EIA load            ┐
NOAA weather       ─┼─►  build_panel ──►  build_features ──►  train_model
EIA renewables      ┘                                                │
                                                                     ▼
OpenWeather forecast ─►  build_today_row ──►  threshold_probabilities
                                                                     │
                                                                     ▼
Kalshi markets ─────────────────────────────────────────►  compute_signals
                                                                     │
                                                                     ▼
                                              outputs/daily_signals.{csv,json}
                                              data/sim.db (dashboard)
```

### Synthetic data fallback

If `EIA_API_KEY` / `NOAA_TOKEN` / `OPENWEATHER_API_KEY` /
`KALSHI_API_KEY_ID` aren't set, each loader falls back to a synthetic
generator that produces realistic-looking data. This lets a clone run
end-to-end before any keys are configured. For real trading you want
all four populated; for development this fallback is convenient.

## Region presets

`ENERGY_REGION=ercot` (default) | `nyiso` | `pjm` | `caiso`. Each
preset wires up the right EIA respondent code, NOAA station, lat/lon
for OpenWeather, reference summer/winter peak (used to anchor the
threshold grid), and Kalshi series prefix.

## Threshold probabilities

The model outputs a point forecast. We convert that to a per-threshold
probability using empirical OOS residual std:

```
P(load > thr) = 1 − Φ((thr − forecast) / residual_std)
```

The threshold grid auto-populates (every 1.5 K MW around the region's
typical peak ± 15%) but can be overridden via `THRESHOLD_GRID_MW`.

## Signal logic

For each Kalshi market:

```
edge = model_prob − (yes_ask / 100)
if edge >=  MIN_EDGE  → BUY_YES
if edge <= -MIN_EDGE  → BUY_NO
otherwise            → NO_TRADE
```

Plus liquidity filters: `MIN_VOLUME`, `MIN_OPEN_INTEREST`,
`MAX_SPREAD_CENTS`.

## Deploying on a DigitalOcean droplet

```bash
ssh root@your-droplet
cd /root
git clone git@github.com:DanielPearl/peak-load.git
cd peak-load

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # populate API keys

# One-time training
python scripts/train.py

# Daily cron — run before Kalshi peak-load markets close.
# ERCOT/CAISO: ~6pm local, NYISO/PJM: ~5pm local. UTC offsets vary.
# Example: ERCOT (UTC-5) → run at 21:00 UTC = 16:00 CDT.
crontab -e
# Add:
#   0 21 * * * cd /root/peak-load && /root/peak-load/.venv/bin/python scripts/run_daily.py >> /var/log/peak-load.log 2>&1
```

The bot writes both human-readable outputs (`outputs/`) and a SQLite
`data/sim.db` that the unified Kalshi dashboard reads alongside the
gas-prices and unemployment-claims bots.

## What's next

- Wire real NOAA + OpenWeather paths (currently `NotImplementedError` stubs)
- Real EIA renewables fetch (same TODO)
- Track signal performance over time → calibration curve, P&L histogram
- Explore quantile-regression instead of point-forecast + Gaussian std
- Add holidays for years past 2027 (HOLIDAY_SET in features.py)

## License

MIT — see LICENSE.
