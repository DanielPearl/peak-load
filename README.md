# Natural Gas — Henry Hub Daily Price Bot

Daily-cadence forecasting model for **Henry Hub natural gas spot price**
(`KXNATGASD` on Kalshi, Pyth-settled at 5pm EDT each day, $/MMBTU
thresholds at $0.005 ticks). Compares model probabilities against
Kalshi market prices and surfaces mispricings.

NG is the most weather-driven commodity in the US energy complex:
winter heating-degree-days drive residential demand, summer
cooling-degree-days drive power-burn demand from gas-fired plants
dispatching against AC load. The model captures that mechanism plus
the slower fundamentals (storage, production, LNG exports) that move
prices over weeks.

## Quick start

```bash
git clone git@github.com:DanielPearl/natural-gas.git
cd natural-gas

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # populate Kalshi + EIA credentials

python scripts/train.py      # train the model (uses synthetic data
                             # if EIA / NOAA keys aren't configured —
                             # the pipeline runs end-to-end without
                             # real credentials but real data is
                             # required for trading signals)

python scripts/run_daily.py  # build today's signals against real Kalshi
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
  config.py         all knobs, .env loader, cross-Kalshi feature
                    series, weather-station weighting
  data_loaders.py   EIA Henry Hub spot + storage + production +
                    weather (national HDD/CDD aggregate) + cross-
                    Kalshi event features (war, hurricane, oil)
  features.py       weather lags + storage delta + production yoy +
                    target lags / log-returns + calendar (Thursday
                    storage-report indicator), all leakage-safe
  model.py          quantile-GBM ensemble + ElasticNet meta-voice +
                    per-strike calibrated classifier ensemble +
                    walk-forward feature selection + correlation prune
  kalshi.py         signed REST client. Real Kalshi only — no demo mode
  signals.py        edge computation (model_p − kalshi_implied), gates
                    on liquidity / spread / edge thresholds
  simulator.py      paper-trading simulator with hedge logic
  validators.py     pre-trade gates: liquidity, spread, time-to-close,
                    basis-risk zone

scripts/
  train.py          full training pipeline → models/natgas_price.pkl
  run_daily.py      daily inference + position lifecycle
  smoke_test.py     CI sanity check
```

## Data sources

| Source | What | Real path |
|---|---|---|
| EIA | Henry Hub daily spot | `NG.RNGWHHD.D` series |
| EIA | Weekly storage | `NG.NW2_EPG0_SWO_R48_BCF.W` |
| EIA | Monthly production | `NG.N9070US2.M` (forward-filled to daily) |
| NOAA | Daily weather observations | GHCND (stub — wire when needed) |
| OpenWeather | Next-day forecast | One Call 3.0 (real) |
| Kalshi | Cross-market features | 14 series: crude oil, war/conflict, hurricane, Fed, retail gas |

## Cross-Kalshi event features

The model exposes a `xk_*` family of features pulled from related
Kalshi markets at inference time:

- **Crude oil**: `KXBRENTD`, `KXWTI`, `KXBRENTW` — global oil markets,
  partial substitute for NG in industrial heat
- **War / geopolitics**: `KXRUSSIAUKR`, `KXIRANISRAEL`, `KXISRAELHAMAS`,
  `KXVENZ` — risk premium drivers (Russia is the world's #1 NG
  exporter, so any escalation moves global LNG and indirectly Henry Hub)
- **Hurricane**: `KXHURPATHFLA`, `KXHURCATFL`, `KXHURCTOTMAJ` — Gulf
  of Mexico hurricane impact on production + LNG terminal disruption
- **Macro / Fed**: `KXFEDDECISION`, `KXRECESSION` — discount-rate
  effect + demand outlook
- **Retail gas**: `KXAAAGASD`, `KXAAAGASW` — refining-margin signal
  (oil products move with NG via crack spreads)

For each series we expose `_avg_prob`, `_max_prob`, `_n_open`,
`_vol_sum` channels. Walk-forward selection prunes whichever don't
add signal. At training time these features are NaN historically (no
backfill of Kalshi prices yet); at inference they get today's snapshot.

## Risk caps

Defaults — override via `.env`:

| Setting | Default |
|---|---|
| Bet size | $1 (100 cents) |
| Max open positions | 1 |
| Max total exposure | $2 |
| Max bets per day | 5 |
| Min edge to fire | 5 percentage points |
| Hedge profit lock | +20¢ in our favor |
| Hedge stop-loss | −15¢ against |

## Cron deployment

```cron
# Daily at 4pm EDT — one hour before KXNATGASD 5pm settlement.
0 20 * * * cd /root/natural-gas && /root/natural-gas/.venv/bin/python scripts/run_daily.py >> /var/log/natural-gas.log 2>&1
```

The signed-REST client uses the same RSA private key as the other
Kalshi bots in this monorepo (gas-prices, unemployment-claims,
whale-watcher) — they share `Secret Keys/kalshi_api_key.txt`.

## Sister bot

The **retail gas price bot** (`/Kalshi/Gas Prices/`) targets
`KXAAAGASW` — AAA weekly retail gasoline. Different commodity,
different mechanism (downstream of crude, refining margins, hurricane
refinery disruption). The two bots are complementary, not duplicative.
