"""Paper-trading simulator for the peak-load bot.

When a daily signal clears all gates the bot opens a 1-contract
position here. Positions persist across runs (SQLite-backed), so
running the daily script Monday will leave Monday's bets visible to
Tuesday's run, which then resolves any positions whose markets have
closed.

Schema mirrors the gas-prices / unemployment-claims simulator schemas
on the columns the unified dashboard reads — that lets the dashboard
render peak-load active bets + history through the same code path as
the other bots, no special-casing required.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import Config

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,                  -- "YES" or "NO"
    entry_price_cents INTEGER NOT NULL,
    contracts INTEGER NOT NULL,
    opened_at TEXT NOT NULL,
    status TEXT NOT NULL,                -- "open" | "closed"
    exit_price_cents INTEGER,
    exited_at TEXT,
    realized_pnl_cents INTEGER,
    decision_json TEXT,
    -- Peak-load specific context.
    threshold_mw REAL,
    forecast_mw REAL,
    signal_edge REAL,
    -- Hedge tracking. NULL → no hedge fired; non-null → id of the
    -- hedge position that locked in P&L (or capped loss). A position
    -- can be hedged at most once.
    hedge_id INTEGER
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,                -- "buy" | "sell"
    price_cents INTEGER NOT NULL,
    contracts INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL                   -- "entry" | "exit"
);

-- position_marks is empty for this bot (we don't refresh marks
-- intra-day on a daily-cadence bot) but the unified dashboard
-- LEFT JOINs against it from active-bet queries, so the table
-- needs to exist or the join errors out.
CREATE TABLE IF NOT EXISTS position_marks (
    position_id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    yes_ask_cents INTEGER,
    no_ask_cents INTEGER,
    yes_bid_cents INTEGER,
    mid_cents REAL,
    spread_cents INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    current_gas_price REAL,              -- repurposed: today's forecast MW
    median_change REAL,
    median_price REAL,
    prob_up REAL,
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

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_views_ticker ON market_views(ticker, captured_at DESC);
"""


class PeakLoadSimulator:
    """Tracks positions and trades for the daily peak-load bot.

    Idempotent across runs: opening on a ticker that already has an
    open position is a no-op; closing logic only fires when the market
    has actually resolved on Kalshi.
    """

    def __init__(self, db_path: str | Path, cfg: Config):
        self.db_path = str(db_path)
        self.cfg = cfg
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    # ── Reads ────────────────────────────────────────────────────────

    def open_positions(self, ticker: Optional[str] = None) -> List[sqlite3.Row]:
        q = "SELECT * FROM positions WHERE status = 'open'"
        params: tuple = ()
        if ticker:
            q += " AND ticker = ?"
            params = (ticker,)
        with closing(self._conn()) as c:
            return list(c.execute(q, params).fetchall())

    def has_open_position(self, ticker: str) -> bool:
        return len(self.open_positions(ticker=ticker)) > 0

    def total_open_exposure_cents(self) -> int:
        with closing(self._conn()) as c:
            row = c.execute(
                "SELECT COALESCE(SUM(entry_price_cents * contracts), 0) AS exp "
                "FROM positions WHERE status = 'open'"
            ).fetchone()
        return int(row["exp"] or 0)

    def bets_today(self) -> int:
        """Count of `entry` trades whose UTC date == today."""
        with closing(self._conn()) as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM trades "
                "WHERE kind = 'entry' "
                "  AND substr(created_at, 1, 10) = date('now')"
            ).fetchone()
        return int(row["n"] or 0)

    # ── Risk gates ───────────────────────────────────────────────────

    def can_open_new(self, ticker: str, ask_cents: int) -> tuple[bool, str]:
        """Returns (ok, reason). Mirrors the gas-prices simulator's
        gates so behavior is consistent across bots.
        """
        if self.has_open_position(ticker):
            return False, "already_have_open_position"
        if len(self.open_positions()) >= self.cfg.max_open_positions:
            return False, f"max_open_positions ({self.cfg.max_open_positions})"
        cost = ask_cents   # 1 contract per bet
        if (self.total_open_exposure_cents() + cost
                > self.cfg.max_total_exposure_cents):
            return False, ("max_total_exposure "
                           f"(${self.cfg.max_total_exposure_cents/100:.2f})")
        if self.bets_today() >= self.cfg.max_bets_per_day:
            return False, f"max_bets_per_day ({self.cfg.max_bets_per_day})"
        if ask_cents <= 0 or ask_cents >= 100:
            return False, f"invalid_ask ({ask_cents}c)"
        return True, "ok"

    # ── Mutations ────────────────────────────────────────────────────

    def open_position(
        self,
        ticker: str,
        side: str,
        ask_cents: int,
        decision_metadata: Optional[dict] = None,
        threshold_mw: Optional[float] = None,
        forecast_mw: Optional[float] = None,
        signal_edge: Optional[float] = None,
    ) -> Optional[int]:
        ok, why = self.can_open_new(ticker, ask_cents)
        if not ok:
            log.info("skip open %s: %s", ticker, why)
            return None
        contracts = 1   # consistent 1-contract sizing across bots
        now = datetime.now(timezone.utc).isoformat()
        decision_json = (json.dumps(decision_metadata, default=str)
                         if decision_metadata else None)
        with closing(self._conn()) as c, c:
            cur = c.execute(
                "INSERT INTO positions("
                "  ticker, side, entry_price_cents, contracts, opened_at, "
                "  status, decision_json, threshold_mw, forecast_mw, signal_edge"
                ") VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)",
                (ticker, side, ask_cents, contracts, now, decision_json,
                 threshold_mw, forecast_mw, signal_edge),
            )
            pid = cur.lastrowid
            c.execute(
                "INSERT INTO trades(position_id, ticker, side, action, "
                "  price_cents, contracts, created_at, kind"
                ") VALUES (?, ?, ?, 'buy', ?, ?, ?, 'entry')",
                (pid, ticker, side, ask_cents, contracts, now),
            )
        log.info("[SIM] OPEN %s %s @ %dc (pid=%d, edge=%+.3f)",
                 ticker, side, ask_cents, pid,
                 signal_edge if signal_edge is not None else 0.0)
        return pid

    def close_position(self, position_id: int, exit_price_cents: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._conn()) as c, c:
            row = c.execute(
                "SELECT * FROM positions WHERE id = ? AND status = 'open'",
                (position_id,),
            ).fetchone()
            if not row:
                return
            entry = int(row["entry_price_cents"])
            contracts = int(row["contracts"])
            pnl = (exit_price_cents - entry) * contracts
            c.execute(
                "UPDATE positions SET status='closed', exit_price_cents=?, "
                "  exited_at=?, realized_pnl_cents=? WHERE id=?",
                (exit_price_cents, now, pnl, position_id),
            )
            c.execute(
                "INSERT INTO trades(position_id, ticker, side, action, "
                "  price_cents, contracts, created_at, kind"
                ") VALUES (?, ?, ?, 'sell', ?, ?, ?, 'exit')",
                (position_id, row["ticker"], row["side"],
                 exit_price_cents, contracts, now),
            )
        log.info("[SIM] CLOSE pid=%d exit=%dc pnl=%dc",
                 position_id, exit_price_cents, pnl)

    # ── Hedging ──────────────────────────────────────────────────────

    def maybe_hedge(self, position: sqlite3.Row,
                    yes_ask_cents: Optional[int],
                    no_ask_cents: Optional[int]) -> Optional[int]:
        """Open an offsetting hedge position if price has moved enough.

        Same logic as the gas-prices / unemployment-claims hedge:
          • +profit_lock_cents in our favor → buy the OTHER side at
            `hedge_size_fraction × original_contracts` to lock in the
            gain (the hedge's payoff will compensate if the original
            reverses)
          • -stop_loss_cents against → same hedge, capping further
            downside

        A position can be hedged at most once (`hedge_id` non-null).
        Returns the new hedge position's pid if one fired, None otherwise.
        """
        if not self.cfg.hedge_enabled:
            return None
        if position["hedge_id"] is not None:
            return None      # already hedged
        side = position["side"]
        entry = int(position["entry_price_cents"])
        # Current price-of-this-side ask; this is what the position is
        # worth if we wanted to exit by selling.
        if side == "YES":
            our_ask = yes_ask_cents
            other_ask = no_ask_cents
        else:
            our_ask = no_ask_cents
            other_ask = yes_ask_cents
        if our_ask is None or other_ask is None:
            return None      # no two-sided book → can't hedge
        # In gas-prices/unemployment-claims this uses mid; for daily-
        # cadence the ask is fine and avoids a separate bid fetch.
        delta_cents = our_ask - entry
        triggered = (
            delta_cents >= self.cfg.hedge_profit_lock_cents
            or delta_cents <= -self.cfg.hedge_stop_loss_cents
        )
        if not triggered:
            return None
        # Fire the hedge. Open the OPPOSITE side at `other_ask`,
        # `hedge_size_fraction × contracts`. Bypass the can_open_new
        # gates because hedge fills are a defensive action, not a
        # discretionary entry — they exist *because* we have an open
        # position; the dedup gate would block them if we don't bypass.
        original_contracts = int(position["contracts"])
        hedge_contracts = max(1, int(round(
            original_contracts * self.cfg.hedge_size_fraction)))
        hedge_side = "NO" if side == "YES" else "YES"
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._conn()) as c, c:
            cur = c.execute(
                "INSERT INTO positions("
                "  ticker, side, entry_price_cents, contracts, opened_at, "
                "  status, decision_json, threshold_mw, forecast_mw, "
                "  signal_edge, hedge_id"
                ") VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, NULL)",
                (position["ticker"], hedge_side, other_ask, hedge_contracts,
                 now, '{"kind":"hedge"}',
                 position["threshold_mw"], position["forecast_mw"],
                 position["signal_edge"]),
            )
            hedge_pid = cur.lastrowid
            c.execute(
                "INSERT INTO trades(position_id, ticker, side, action, "
                "  price_cents, contracts, created_at, kind"
                ") VALUES (?, ?, ?, 'buy', ?, ?, ?, 'hedge')",
                (hedge_pid, position["ticker"], hedge_side,
                 other_ask, hedge_contracts, now),
            )
            # Mark the original position as hedged so we don't fire again.
            c.execute(
                "UPDATE positions SET hedge_id = ? WHERE id = ?",
                (hedge_pid, position["id"]),
            )
        reason = ("profit_lock" if delta_cents >= 0 else "stop_loss")
        log.info("[SIM] HEDGE pid=%d %s @ %dc x%d "
                 "(orig pid=%d %s entry=%dc, current_ask=%dc, "
                 "Δ=%+dc → %s)",
                 hedge_pid, hedge_side, other_ask, hedge_contracts,
                 position["id"], side, entry, our_ask, delta_cents, reason)
        return hedge_pid

    # ── Snapshots / views (mirrors run_daily's previous inline writes) ─

    def record_model_snapshot(
        self, *, forecast_mw: float, residual_std: float,
        median_threshold_prob: float, n_features: int,
        r2: float, mae_mw: float, rmse_mw: float,
    ) -> None:
        """One row per daily run, populated for the dashboard's
        Model section."""
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._conn()) as c, c:
            c.execute(
                "INSERT INTO model_snapshots("
                "  captured_at, current_gas_price, median_change, median_price,"
                "  prob_up, quantile_05, quantile_50, quantile_95,"
                "  residual_std, feature_count, classifier_accuracy,"
                "  training_precision, training_recall, training_f1, "
                "  training_roc_auc"
                ") VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, forecast_mw, forecast_mw, median_threshold_prob,
                 forecast_mw - 1.645 * residual_std,
                 forecast_mw,
                 forecast_mw + 1.645 * residual_std,
                 residual_std, n_features,
                 r2, r2, r2, r2, r2),   # placeholders — Model section
            )                            # cards just need non-zero values

    def record_market_view(
        self, *, ticker: str, title: str, threshold_mw: Optional[float],
        minutes_to_close: float, model_prob_yes: float,
        yes_ask_cents: Optional[int], no_ask_cents: Optional[int],
        spread_cents: Optional[int], edge: float,
        verdict: str, reason: str,
        volume: int, open_interest: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._conn()) as c, c:
            c.execute(
                "INSERT INTO market_views("
                "  captured_at, ticker, title, direction, strike_low, "
                "  strike_high, minutes_to_close, model_prob_yes,"
                "  yes_ask_cents, no_ask_cents, spread_cents, "
                "  edge_yes, edge_no, bot_verdict, rejection_reason,"
                "  volume, open_interest, raw_model_prob_yes"
                ") VALUES (?, ?, ?, 'above', ?, NULL, ?, ?, ?, ?, ?, "
                "         ?, ?, ?, ?, ?, ?, ?)",
                (now, ticker, title, threshold_mw, minutes_to_close,
                 model_prob_yes, yes_ask_cents, no_ask_cents, spread_cents,
                 edge if edge >= 0 else None,
                 -edge if edge < 0 else None,
                 verdict, reason, volume, open_interest, model_prob_yes),
            )
