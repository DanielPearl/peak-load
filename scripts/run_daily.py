"""Daily inference: build today's feature row, score every Kalshi
KXNATGASD market, write signals to /outputs/, AND open / close
simulated positions when an edge signal clears all gates.

Designed to run once per day via cron, an hour before the 5pm EDT
KXNATGASD market settlement.

Per-tick flow:
  1. Load today's panel + weather forecast → build feature row
     (with cross-Kalshi feature snapshot from related markets)
  2. Score → forecast $/MMBTU + per-threshold probabilities
  3. Fetch open Kalshi KXNATGASD markets
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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import Config, load_config
from src.data_loaders import (
    build_panel,
    compute_forecast_revisions,
    fetch_cross_kalshi_features,
    fetch_weather_forecast,
    load_previous_forecast,
    save_current_forecast,
)
from src.features import build_today_row
from src.kalshi import fetch_kalshi_markets, fetch_market_status
from src.model import load_model, threshold_probabilities
from src.signals import compute_signals, signals_to_records
from src.simulator import NatGasSimulator
from src.validators import ValidatorCfg, validate_market

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("run_daily")


def main() -> int:
    cfg = load_config()
    log.info("starting daily run for natural-gas-price bot "
             "(target series=%s)", cfg.kalshi_series_prefix)

    model = load_model(cfg.model_path)
    if model is None:
        log.error("no model artifact at %s — run scripts/train.py first",
                  cfg.model_path)
        return 1
    log.info("loaded %s model (residual_std=$%.3f, trained-thru=%s)",
             model.model_name, model.residual_std,
             model.train_end_date.date())

    sim = NatGasSimulator(
        cfg.daily_csv_path.parent.parent / "data" / "sim.db", cfg=cfg)

    # ── 1. Today's feature row ────────────────────────────────────────
    panel = build_panel(cfg, days=cfg.history_days_for_training)
    forecast = fetch_weather_forecast(cfg, days_ahead=2)
    if forecast.empty:
        log.error("weather forecast unavailable — cannot build today's row")
        return 1
    today_forecast = forecast.iloc[0]

    # Cross-Kalshi feature snapshot: pull current implied probabilities
    # from related markets (crude oil, war/conflict, hurricane, fed
    # policy). These are appended to today's feature row so the model
    # sees them at inference. NaN for series with no open markets.
    cross_kalshi = fetch_cross_kalshi_features(cfg)
    if not cross_kalshi.empty:
        log.info("cross-Kalshi features: %d channels populated",
                 cross_kalshi.notna().sum())

    # Forecast-revision snapshot: load yesterday's saved forecast (if
    # any) and compute day-over-day weather deltas. First run after a
    # deploy produces NaN deltas (no prior); the median imputer fills.
    forecast_history_path = (cfg.daily_csv_path.parent.parent
                              / "data" / "last_forecast.json")
    previous_forecast = load_previous_forecast(forecast_history_path)
    revisions = compute_forecast_revisions(today_forecast, previous_forecast)
    if previous_forecast is not None:
        n_rev = int(revisions.notna().sum())
        log.info("forecast revisions: %d channels with day-over-day delta",
                 n_rev)
    else:
        log.info("forecast revisions: no prior snapshot — first run")
    # Persist today's forecast so tomorrow's run has a baseline.
    save_current_forecast(forecast_history_path, today_forecast)

    feature_row = build_today_row(panel, today_forecast,
                                   cross_kalshi_features=cross_kalshi,
                                   forecast_revisions=revisions,
                                   target=cfg.target_column)
    forecast_usd = float(model.predict(feature_row)[0])
    log.info("today's NG forecast: $%.3f / MMBTU", forecast_usd)

    # ── 2. Threshold grid centered on TODAY'S forecast ───────────────
    cfg.threshold_grid_usd = _dynamic_threshold_grid(
        forecast_usd, model.residual_std)
    log.info("threshold grid (dynamic): $%.3f .. $%.3f around forecast",
             cfg.threshold_grid_usd[0], cfg.threshold_grid_usd[-1])

    probs = threshold_probabilities(model, feature_row, cfg.threshold_grid_usd)
    log.info("threshold prob span: $%.3f p=%.2f .. $%.3f p=%.2f",
             cfg.threshold_grid_usd[0], probs[cfg.threshold_grid_usd[0]],
             cfg.threshold_grid_usd[-1], probs[cfg.threshold_grid_usd[-1]])

    # ── 3. Kalshi markets ─────────────────────────────────────────────
    markets = fetch_kalshi_markets(cfg, forecast_usd=forecast_usd)
    log.info("fetched %d open Kalshi markets", len(markets))

    # ── 4. Compute signals ────────────────────────────────────────────
    # Kalshi markets list strikes at $0.005 ticks, so we have ~80
    # different threshold values to score. Computing them one-by-one
    # via prob_above() runs ALL trained classifiers per call, which
    # is O(markets × strikes × seeds) calls. Instead, evaluate every
    # trained classifier ONCE on today's row, then isotonic-interp
    # per Kalshi strike — O(strikes × seeds) total.
    from src.model import _interpolate_threshold_prob
    per_thr_probs = {
        thr: float(clf.predict_proba(feature_row[model.feature_columns])[0])
        for thr, clf in model.threshold_classifiers.items()
    }
    market_probs = {
        m.threshold_value: _interpolate_threshold_prob(
            per_thr_probs, m.threshold_value)
        for m in markets if m.threshold_value is not None
    }
    signals = compute_signals(cfg, markets, market_probs)
    n_buy = sum(1 for s in signals if s.decision.startswith("BUY"))
    log.info("%d signals (%d BUY recommendations)", len(signals), n_buy)

    # ── 5a. Resolve any open positions whose markets have closed ────
    _resolve_open_positions(cfg, sim)

    # ── 5b. Hedge any open positions whose price has moved enough ───
    n_hedged = _maybe_hedge_open_positions(cfg, sim, markets)
    if n_hedged:
        log.info("opened %d hedge position(s)", n_hedged)

    # ── 6. Validate + open new positions for BUY signals ────────────
    val_cfg = _validator_cfg(cfg)
    validated_signals = []
    for s in signals:
        if s.decision in ("BUY_YES", "BUY_NO"):
            mkt = next((m for m in markets if m.ticker == s.ticker), None)
            if mkt is None:
                continue
            # Single-day cadence — minutes_to_close approximated until we
            # parse the per-market close_time. Daily NG markets close at
            # 5pm EDT so the worst-case is ~24h from any midday run.
            ok, reason = validate_market(
                mkt, val_cfg, forecast_value=forecast_usd,
                minutes_to_close=24 * 60.0)
            if not ok:
                log.info("validator skip %s: %s", s.ticker, reason)
                continue
            validated_signals.append(s)
        else:
            validated_signals.append(s)
    n_opened = _open_positions_for_signals(sim, validated_signals, forecast_usd)
    log.info("opened %d new position(s); %d position(s) currently open",
             n_opened, len(sim.open_positions()))

    # ── 7. Record snapshots for the dashboard ────────────────────────
    median_thr = cfg.threshold_grid_usd[len(cfg.threshold_grid_usd) // 2]
    sim.record_model_snapshot(
        forecast_value=forecast_usd,
        residual_std=model.residual_std,
        median_threshold_prob=probs.get(median_thr, 0.5),
        n_features=len(model.feature_columns),
        r2=model.metrics.get("r2", 0.0),
        mae=model.metrics.get("mae", 0.0),
        rmse=model.metrics.get("rmse", 0.0),
        # Per-strike binary-classification headline metrics — what the
        # bot actually scores for each "above strike Y/N" decision.
        # The dashboard's Model section reads these.
        classifier_accuracy=model.metrics.get("per_strike_avg_accuracy", 0.0),
        precision=model.metrics.get("per_strike_avg_precision", 0.0),
        recall=model.metrics.get("per_strike_avg_recall", 0.0),
        f1=model.metrics.get("per_strike_avg_f1", 0.0),
        roc_auc=model.metrics.get("per_strike_avg_roc_auc", 0.0),
    )
    for s in signals:
        sim.record_market_view(
            ticker=s.ticker, title=s.yes_sub_title,
            threshold_value=s.threshold_value,
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
    _write_json(cfg.daily_json_path, records, model, forecast_usd, probs, cfg)
    log.info("wrote %s and %s", cfg.daily_csv_path, cfg.daily_json_path)
    print(f"\nForecast: ${forecast_usd:.3f} / MMBTU")
    print(f"BUY recommendations: {n_buy}")
    print(f"New positions opened: {n_opened}")
    print(f"Total open positions: {len(sim.open_positions())}")
    return 0


# --------------------------------------------------------------------------- #
# Position lifecycle
# --------------------------------------------------------------------------- #

def _open_positions_for_signals(
    sim: NatGasSimulator,
    signals: list,
    forecast_usd: float,
) -> int:
    """For each BUY signal, try to open a 1-contract position. The
    simulator's risk gates handle dedup + global caps. Returns the
    count of newly-opened positions.
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
            threshold_value=float(s.threshold_value)
                if s.threshold_value is not None else None,
            forecast_value=forecast_usd, signal_edge=s.edge,
        )
        if pid is not None:
            opened += 1
    return opened


def _dynamic_threshold_grid(
    forecast_usd: float,
    residual_std: float,
    span_sigma: float = 2.5,
    step_usd: float = 0.05,
    n_strikes: int = 21,
) -> list:
    """Build a $/MMBTU threshold grid centered on today's forecast.

    Spans forecast ± span_sigma·residual_std, snapped to step_usd
    boundaries and symmetric around the forecast. Defaults
    (2.5σ, 21 strikes, $0.05 step) get strikes spanning roughly ±$0.50
    where probabilities transition smoothly from ~0.1% to ~99%.
    """
    center = round(forecast_usd / step_usd) * step_usd
    width_from_sigma = max(int(round(span_sigma * residual_std / step_usd)),
                            (n_strikes - 1) // 2)
    half = width_from_sigma
    return [round(center + (i - half) * step_usd, 4)
            for i in range(2 * half + 1)]


def _validator_cfg(cfg: Config) -> ValidatorCfg:
    """Translate the flat Config dataclass into a ValidatorCfg."""
    return ValidatorCfg(
        min_volume=cfg.min_volume,
        min_open_interest=cfg.min_open_interest,
        max_spread_cents=cfg.val_max_spread_cents,
        prob_bounds_cents=(cfg.val_prob_bounds_cents_low,
                            cfg.val_prob_bounds_cents_high),
        max_entry_price_cents=cfg.val_max_entry_price_cents,
        min_minutes_to_close=cfg.val_min_minutes_to_close,
        max_minutes_to_close=cfg.val_max_minutes_to_close,
        basis_risk_strike_window=cfg.val_basis_risk_strike_window_usd,
        basis_risk_max_hours_to_close=cfg.val_basis_risk_max_hours_to_close,
    )


def _maybe_hedge_open_positions(
    cfg: Config, sim: NatGasSimulator, markets: list,
) -> int:
    """Run hedge check across all open un-hedged positions."""
    if not cfg.hedge_enabled:
        return 0
    by_ticker = {m.ticker: m for m in markets}
    n = 0
    for pos in sim.open_positions():
        if pos["hedge_id"] is not None:
            continue
        decision = pos["decision_json"] or ""
        if "hedge" in decision:
            continue
        m = by_ticker.get(pos["ticker"])
        if m is None:
            continue
        hedge_pid = sim.maybe_hedge(pos, m.yes_ask_cents, m.no_ask_cents)
        if hedge_pid is not None:
            n += 1
    return n


def _resolve_open_positions(cfg: Config, sim: NatGasSimulator) -> None:
    """Check each open position's market on Kalshi. If resolved, close
    at settle price. Skipped silently on API failure (better to
    under-resolve than to mis-close).
    """
    for pos in sim.open_positions():
        m = fetch_market_status(cfg, pos["ticker"])
        if not m:
            continue
        status = m.get("status", "")
        if status in ("open", "active"):
            continue
        settle = m.get("yes_settled_price") or m.get("settlement_value")
        if settle is None:
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
        side = pos["side"]
        exit_cents = settle_cents if side == "YES" else 100 - settle_cents
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


def _write_json(path: Path, records: list, model, forecast_usd: float,
                probs: dict, cfg: Config) -> None:
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "target": cfg.target_column,
        "kalshi_series": cfg.kalshi_series_prefix,
        "model": {
            "name": model.model_name,
            "trained_through": str(model.train_end_date.date()),
            "residual_std_usd": model.residual_std,
            "metrics": model.metrics,
        },
        "forecast_usd_mmbtu": forecast_usd,
        "threshold_probabilities": {f"{k:.3f}": v for k, v in probs.items()},
        "signals": records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
