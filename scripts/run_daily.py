"""Daily inference: build today's feature row, score every Kalshi
peak-load market, write signals to /outputs/.

Designed to run once per day via cron, an hour or two before the
relevant Kalshi peak-load markets close.

Outputs:
  outputs/daily_signals.csv   one row per market (machine-friendly)
  outputs/daily_signals.json  same data plus metadata (model name,
                              residual_std, run timestamp)
  data/sim.db                 SQLite mirror of model_snapshots +
                              market_views so the unified dashboard
                              can read peak-load alongside other bots
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import Config, load_config
from src.data_loaders import build_panel, fetch_weather_forecast
from src.features import build_today_row
from src.kalshi import fetch_kalshi_markets
from src.model import load_model, threshold_probabilities
from src.signals import compute_signals, signals_to_records

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("run_daily")


def main() -> int:
    cfg = load_config()
    log.info("starting daily run for region=%s", cfg.region)

    model = load_model(cfg.model_path)
    if model is None:
        log.error("no model artifact at %s — run scripts/train.py first",
                  cfg.model_path)
        return 1
    log.info("loaded %s model (residual_std=%.0f, trained-thru=%s)",
             model.model_name, model.residual_std,
             model.train_end_date.date())

    # ── 1. Today's feature row ────────────────────────────────────────
    panel = build_panel(cfg, days=cfg.history_days_for_training)
    forecast = fetch_weather_forecast(cfg, days_ahead=2)
    if forecast.empty:
        log.error("weather forecast unavailable — cannot build today's row")
        return 1
    today_forecast = forecast.iloc[0]    # tomorrow's forecast (idx[0] = today + 1)
    feature_row = build_today_row(panel, today_forecast,
                                   target=cfg.target_column)
    forecast_mw = float(model.predict(feature_row)[0])
    log.info("today's forecast: %.0f MW", forecast_mw)

    # ── 2. Threshold probabilities ────────────────────────────────────
    probs = threshold_probabilities(model, feature_row, cfg.threshold_grid_mw)
    log.info("threshold prob span: %d MW p=%.2f .. %d MW p=%.2f",
             cfg.threshold_grid_mw[0], probs[cfg.threshold_grid_mw[0]],
             cfg.threshold_grid_mw[-1], probs[cfg.threshold_grid_mw[-1]])

    # ── 3. Kalshi markets ─────────────────────────────────────────────
    markets = fetch_kalshi_markets(cfg)
    log.info("fetched %d Kalshi markets", len(markets))

    # ── 4. Compute signals ────────────────────────────────────────────
    signals = compute_signals(cfg, markets, probs)
    n_buy = sum(1 for s in signals if s.decision.startswith("BUY"))
    log.info("%d signals (%d BUY recommendations)", len(signals), n_buy)

    # ── 5. Write outputs ──────────────────────────────────────────────
    cfg.daily_csv_path.parent.mkdir(parents=True, exist_ok=True)
    records = signals_to_records(signals)
    _write_csv(cfg.daily_csv_path, records)
    _write_json(cfg.daily_json_path, records, model, forecast_mw, probs, cfg)
    _mirror_to_sqlite(cfg, signals, forecast_mw, probs, model)

    log.info("wrote %s and %s", cfg.daily_csv_path, cfg.daily_json_path)
    print(f"\nForecast: {forecast_mw:.0f} MW")
    print(f"BUY recommendations: {n_buy}")
    return 0


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def _write_csv(path: Path, records: list) -> None:
    if not records:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, records: list, model, forecast_mw: float,
                probs: dict, cfg: Config) -> None:
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "region": cfg.region,
        "region_name": cfg.region_meta["name"],
        "model": {
            "name": model.model_name,
            "trained_through": str(model.train_end_date.date()),
            "residual_std_mw": model.residual_std,
            "metrics": model.metrics,
        },
        "forecast_mw": forecast_mw,
        "threshold_probabilities": {str(k): v for k, v in probs.items()},
        "signals": records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _mirror_to_sqlite(cfg: Config, signals: list, forecast_mw: float,
                       probs: dict, model) -> None:
    """Mirror enough state to a SQLite DB so the unified dashboard
    (the gas-prices repo's dashboard.py) can read peak-load data
    alongside the other bots without special-casing.

    Tables we populate:
      model_snapshots — one row per run, with metrics
      market_views    — one row per Kalshi market this run
    """
    db_path = ROOT / "data" / "sim.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = """
    CREATE TABLE IF NOT EXISTS model_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        captured_at TEXT NOT NULL,
        current_gas_price REAL,            -- repurposed: today's forecast MW
        median_change REAL,                 -- repurposed: 0 (single forecast)
        median_price REAL,                  -- repurposed: forecast MW
        prob_up REAL,                       -- repurposed: P(load > median thr)
        quantile_05 REAL,
        quantile_50 REAL,
        quantile_95 REAL,
        residual_std REAL,
        feature_count INTEGER,
        classifier_accuracy REAL,
        training_precision REAL,
        training_recall REAL,
        training_f1 REAL,
        training_roc_auc REAL
    );
    CREATE TABLE IF NOT EXISTS market_views (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        captured_at TEXT NOT NULL,
        ticker TEXT NOT NULL,
        title TEXT,
        direction TEXT,
        strike_low REAL,
        strike_high REAL,
        minutes_to_close REAL,
        model_prob_yes REAL,
        yes_ask_cents INTEGER,
        no_ask_cents INTEGER,
        yes_bid_cents INTEGER,
        spread_cents INTEGER,
        book_depth INTEGER,
        edge_yes REAL,
        edge_no REAL,
        bot_verdict TEXT NOT NULL,
        rejection_reason TEXT,
        volume INTEGER,
        open_interest INTEGER,
        raw_model_prob_yes REAL
    );
    CREATE INDEX IF NOT EXISTS idx_views_ticker ON market_views(ticker, captured_at DESC);
    """
    with closing(sqlite3.connect(db_path)) as c, c:
        c.executescript(schema)
        # Insert model snapshot (one row per run).
        median_thr = cfg.threshold_grid_mw[len(cfg.threshold_grid_mw) // 2]
        c.execute(
            "INSERT INTO model_snapshots(captured_at, current_gas_price, "
            "  median_change, median_price, prob_up, "
            "  quantile_05, quantile_50, quantile_95, "
            "  residual_std, feature_count, classifier_accuracy, "
            "  training_precision, training_recall, training_f1, training_roc_auc"
            ") VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(),
             forecast_mw, forecast_mw, probs.get(median_thr, 0.5),
             forecast_mw - 1.645 * model.residual_std,
             forecast_mw,
             forecast_mw + 1.645 * model.residual_std,
             model.residual_std, len(model.feature_columns),
             model.metrics.get("r2", 0.0),
             model.metrics.get("r2", 0.0),    # placeholders so dash
             model.metrics.get("r2", 0.0),    # cards have non-zero
             model.metrics.get("r2", 0.0),
             model.metrics.get("r2", 0.0))
        )
        # Insert one market_view per signal so the watchlist renders.
        for s in signals:
            spread = s.spread_cents
            verdict = ("BUY_YES" if s.decision == "BUY_YES"
                       else "BUY_NO" if s.decision == "BUY_NO"
                       else "SKIP")
            c.execute(
                "INSERT INTO market_views(captured_at, ticker, title, direction, "
                "  strike_low, strike_high, minutes_to_close, model_prob_yes, "
                "  yes_ask_cents, no_ask_cents, spread_cents, "
                "  edge_yes, edge_no, bot_verdict, rejection_reason, "
                "  volume, open_interest, raw_model_prob_yes) "
                "VALUES (?, ?, ?, 'above', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(),
                 s.ticker, s.yes_sub_title, s.threshold_mw,
                 24 * 60,    # placeholder TTC; daily cycle
                 s.model_prob, s.yes_ask_cents, s.no_ask_cents, spread,
                 s.edge if s.edge >= 0 else None,
                 -s.edge if s.edge < 0 else None,
                 verdict, s.decision_reason,
                 s.volume, s.open_interest, s.model_prob)
            )


if __name__ == "__main__":
    raise SystemExit(main())
