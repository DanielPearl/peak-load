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
from src.validators import ValidatorCfg, validate_market

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

    # ── 2. Threshold grid centered on TODAY'S forecast ───────────────
    # The static grid in cfg was anchored on the region's SUMMER peak,
    # which made every probability collapse to 0% / 100% on shoulder-
    # season days when actual forecast lands well below summer peak.
    # Real Kalshi markets list strikes near the expected peak, so we
    # mirror that: span forecast ± span_sigma × residual_std.
    cfg.threshold_grid_mw = _dynamic_threshold_grid(
        forecast_mw, model.residual_std)
    log.info("threshold grid (dynamic): %d MW .. %d MW around forecast",
             cfg.threshold_grid_mw[0], cfg.threshold_grid_mw[-1])

    probs = threshold_probabilities(model, feature_row, cfg.threshold_grid_mw)
    log.info("threshold prob span: %d MW p=%.2f .. %d MW p=%.2f",
             cfg.threshold_grid_mw[0], probs[cfg.threshold_grid_mw[0]],
             cfg.threshold_grid_mw[-1], probs[cfg.threshold_grid_mw[-1]])

    # ── 3. Kalshi markets ─────────────────────────────────────────────
    # Pass forecast so demo-mode markets anchor on it (not the static
    # summer-peak reference). Real-Kalshi path ignores forecast_mw.
    markets = fetch_kalshi_markets(cfg, forecast_mw=forecast_mw)
    log.info("fetched %d open Kalshi markets", len(markets))

    # ── 4. Compute signals ────────────────────────────────────────────
    signals = compute_signals(cfg, markets, probs)
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
            # Find matching market for the validator (we already have
            # the signal's metadata but validate_market wants the full
            # KalshiMarket object).
            mkt = next((m for m in markets if m.ticker == s.ticker), None)
            if mkt is None:
                continue
            # 24h * 60 = 1440 min; daily-cadence approx until we wire a
            # real per-market mtc on the synthetic path.
            ok, reason = validate_market(
                mkt, val_cfg, forecast_mw=forecast_mw,
                minutes_to_close=24 * 60.0)
            if not ok:
                log.info("validator skip %s: %s", s.ticker, reason)
                continue
            validated_signals.append(s)
        else:
            validated_signals.append(s)
    n_opened = _open_positions_for_signals(sim, validated_signals, forecast_mw)
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


def _dynamic_threshold_grid(
    forecast_mw: float,
    residual_std: float,
    span_sigma: float = 2.5,
    step_mw: int = 1500,
    n_strikes: int = 11,
) -> list[int]:
    """Build a threshold grid centered on today's forecast.

    Spans forecast ± span_sigma·residual_std, snapped to step_mw
    boundaries and symmetric around the forecast. With defaults
    (2.5σ, 11 strikes, 1500 MW step) we get strikes spanning roughly
    ±2σ where probabilities transition smoothly from ~0.1% to ~99%.

    Static-grid alternative (cfg.threshold_grid_mw, anchored on the
    region's summer peak) was useful for sanity-checking model output
    against a fixed reference but produced all-0% or all-100% rows
    on shoulder-season days when forecast and summer peak diverge.
    """
    center = int(round(forecast_mw / step_mw)) * step_mw
    # Width controlled by both σ and step — pick whichever is wider so
    # we always span at least n_strikes steps even when σ is small.
    width_from_sigma = int(round(span_sigma * residual_std / step_mw))
    half = max((n_strikes - 1) // 2, width_from_sigma)
    return [center + (i - half) * step_mw
            for i in range(2 * half + 1)]


def _validator_cfg(cfg: Config) -> ValidatorCfg:
    """Translate the flat Config dataclass into a ValidatorCfg."""
    return ValidatorCfg(
        min_volume=cfg.min_volume,
        min_open_interest=cfg.min_open_interest,
        max_spread_cents=cfg.val_max_spread_cents,
        prob_bounds_cents=(cfg.val_prob_bounds_cents_low,
                            cfg.val_prob_bounds_cents_high),
        min_minutes_to_close=cfg.val_min_minutes_to_close,
        max_minutes_to_close=cfg.val_max_minutes_to_close,
        basis_risk_strike_window_mw=cfg.val_basis_risk_strike_window_mw,
        basis_risk_max_hours_to_close=cfg.val_basis_risk_max_hours_to_close,
    )


def _maybe_hedge_open_positions(
    cfg: Config, sim: PeakLoadSimulator, markets: list,
) -> int:
    """Run the hedge check across all open un-hedged positions. Returns
    the count of hedges fired this tick.

    For each open position: find its current market (if still listed),
    pull the live yes/no asks, and ask the simulator if the hedge
    triggers fire. Hedges are EV-locking offsets — opening the OTHER
    side of the contract to compensate if our position reverses.
    """
    if not cfg.hedge_enabled:
        return 0
    by_ticker = {m.ticker: m for m in markets}
    n = 0
    for pos in sim.open_positions():
        if pos["hedge_id"] is not None:
            continue
        # Ignore hedge positions themselves (their decision_json marks
        # them with kind=hedge).
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
