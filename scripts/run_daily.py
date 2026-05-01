"""Daily inference: build today's feature row, score every Kalshi
peak-load market, write signals to /outputs/, AND open / close
simulated positions when an edge signal clears all gates.

Designed to run once per day via cron, an hour or two before the
relevant Kalshi peak-load markets close.

Per-tick flow:
  1. Load today's panel + weather forecast → build feature row
  2. Score → forecast MW + per-threshold probabilities
  3. Fetch open Kalshi peak-load markets
  4. Compute signals (edge, liquidity filters, BUY/SKIP)
  5. RESOLVE: any existing open position whose market has resolved
              gets closed at the settlement price
  6. OPEN:    each new BUY signal that clears risk caps opens a
              1-contract simulated position
  7. Write outputs: outputs/daily_signals.{csv,json} + data/sim.db
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import Config, load_config
from src.data_loaders import build_panel, fetch_weather_forecast
from src.features import build_today_row
from src.kalshi import fetch_kalshi_markets, fetch_market_status
from src.model import load_model, threshold_probabilities
from src.signals import compute_signals, signals_to_records
from src.simulator import PeakLoadSimulator

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

    sim = PeakLoadSimulator(cfg.daily_csv_path.parent.parent / "data" / "sim.db",
                            cfg=cfg)

    # ── 1. Today's feature row ────────────────────────────────────────
    panel = build_panel(cfg, days=cfg.history_days_for_training)
    forecast = fetch_weather_forecast(cfg, days_ahead=2)
    if forecast.empty:
        log.error("weather forecast unavailable — cannot build today's row")
        return 1
    today_forecast = forecast.iloc[0]
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
    log.info("fetched %d open Kalshi markets", len(markets))

    # ── 4. Compute signals ────────────────────────────────────────────
    signals = compute_signals(cfg, markets, probs)
    n_buy = sum(1 for s in signals if s.decision.startswith("BUY"))
    log.info("%d signals (%d BUY recommendations)", len(signals), n_buy)

    # ── 5. Resolve any open positions whose markets have closed ──────
    _resolve_open_positions(cfg, sim)

    # ── 6. Open new positions for BUY signals that clear risk caps ──
    n_opened = _open_positions_for_signals(sim, signals, forecast_mw)
    log.info("opened %d new position(s); %d position(s) currently open",
             n_opened, len(sim.open_positions()))

    # ── 7. Record snapshots for the dashboard ────────────────────────
    median_thr = cfg.threshold_grid_mw[len(cfg.threshold_grid_mw) // 2]
    sim.record_model_snapshot(
        forecast_mw=forecast_mw,
        residual_std=model.residual_std,
        median_threshold_prob=probs.get(median_thr, 0.5),
        n_features=len(model.feature_columns),
        r2=model.metrics.get("r2", 0.0),
        mae_mw=model.metrics.get("mae", 0.0),
        rmse_mw=model.metrics.get("rmse", 0.0),
    )
    for s in signals:
        sim.record_market_view(
            ticker=s.ticker, title=s.yes_sub_title,
            threshold_mw=s.threshold_mw,
            minutes_to_close=24 * 60,
            model_prob_yes=s.model_prob,
            yes_ask_cents=s.yes_ask_cents,
            no_ask_cents=s.no_ask_cents,
            spread_cents=s.spread_cents,
            edge=s.edge,
            verdict=("BUY_YES" if s.decision == "BUY_YES"
                     else "BUY_NO" if s.decision == "BUY_NO" else "SKIP"),
            reason=s.decision_reason,
            volume=s.volume, open_interest=s.open_interest,
        )

    # ── 8. Output files ──────────────────────────────────────────────
    cfg.daily_csv_path.parent.mkdir(parents=True, exist_ok=True)
    records = signals_to_records(signals)
    _write_csv(cfg.daily_csv_path, records)
    _write_json(cfg.daily_json_path, records, model, forecast_mw, probs, cfg)
    log.info("wrote %s and %s", cfg.daily_csv_path, cfg.daily_json_path)
    print(f"\nForecast: {forecast_mw:.0f} MW")
    print(f"BUY recommendations: {n_buy}")
    print(f"New positions opened: {n_opened}")
    print(f"Total open positions: {len(sim.open_positions())}")
    return 0


# --------------------------------------------------------------------------- #
# Position lifecycle
# --------------------------------------------------------------------------- #

def _open_positions_for_signals(
    sim: PeakLoadSimulator,
    signals: list,
    forecast_mw: float,
) -> int:
    """For each BUY signal, try to open a 1-contract position. The
    simulator's risk gates handle dedup (already-have-open-position)
    and global caps (max_open / max_exposure / max_bets_per_day).
    Returns the count of newly-opened positions.
    """
    opened = 0
    for s in signals:
        if s.decision == "BUY_YES":
            ask = s.yes_ask_cents
            side = "YES"
        elif s.decision == "BUY_NO":
            ask = s.no_ask_cents
            side = "NO"
        else:
            continue
        if ask is None:
            continue
        decision_md = {
            "ticker": s.ticker, "decision": s.decision,
            "decision_reason": s.decision_reason,
            "model_prob": s.model_prob,
            "kalshi_implied_prob": s.kalshi_implied_prob,
            "edge": s.edge, "spread_cents": s.spread_cents,
            "volume": s.volume, "open_interest": s.open_interest,
            "yes_ask_cents": s.yes_ask_cents,
            "no_ask_cents": s.no_ask_cents,
        }
        pid = sim.open_position(
            ticker=s.ticker, side=side, ask_cents=ask,
            decision_metadata=decision_md,
            threshold_mw=float(s.threshold_mw) if s.threshold_mw else None,
            forecast_mw=forecast_mw, signal_edge=s.edge,
        )
        if pid is not None:
            opened += 1
    return opened


def _resolve_open_positions(cfg: Config, sim: PeakLoadSimulator) -> None:
    """Check each open position's market on Kalshi. If the market has
    resolved (status != open), close the position at the settle price.

    Skipped silently if Kalshi creds aren't set or the per-market
    fetch fails — open positions just stay open until the next run
    that can reach Kalshi. Better to under-resolve than to mis-close.
    """
    for pos in sim.open_positions():
        m = fetch_market_status(cfg, pos["ticker"])
        if not m:
            continue
        status = m.get("status", "")
        if status in ("open", "active"):
            continue
        # Resolved. Read the settle value Kalshi exposes; fall back to
        # 100/0 based on the binary outcome if a settle isn't reported.
        settle = m.get("yes_settled_price") or m.get("settlement_value")
        if settle is None:
            # Try result field: "yes" or "no"
            result = (m.get("result") or "").lower()
            if result == "yes":
                settle_cents = 100
            elif result == "no":
                settle_cents = 0
            else:
                log.info("pid=%d market %s resolved but no settle/result "
                         "field — leaving open for next run",
                         pos["id"], pos["ticker"])
                continue
        else:
            try:
                settle_cents = int(round(float(settle)))
            except (TypeError, ValueError):
                continue
        # Translate the YES-side settle into the side we hold's payoff.
        # For a YES position: payoff = settle_cents (YES at 100 means win)
        # For a NO position:  payoff = 100 - settle_cents
        side = pos["side"]
        if side == "YES":
            exit_cents = settle_cents
        else:
            exit_cents = 100 - settle_cents
        sim.close_position(pos["id"], exit_cents)


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


if __name__ == "__main__":
    raise SystemExit(main())
