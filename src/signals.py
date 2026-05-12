"""Edge computation + signal generation.

The bot's job here is straightforward: for each Kalshi market, compare
my model's probability to the market-implied probability, decide if
there's a tradeable edge, and emit a structured signal.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .config import Config
from .kalshi import KalshiMarket

log = logging.getLogger(__name__)


@dataclass
class Signal:
    ticker: str
    threshold_value: Optional[float]
    yes_sub_title: str
    model_prob: float
    kalshi_implied_prob: float
    edge: float                        # model_prob − kalshi_implied
    yes_ask_cents: Optional[int]
    no_ask_cents: Optional[int]
    volume: int
    open_interest: int
    spread_cents: Optional[int]
    decision: str                      # BUY_YES / BUY_NO / NO_TRADE
    decision_reason: str               # why we made that call


def implied_prob(yes_price_cents: Optional[int]) -> Optional[float]:
    """Kalshi YES ask in cents → implied probability (0–1)."""
    if yes_price_cents is None:
        return None
    return yes_price_cents / 100.0


def compute_signal(cfg: Config, market: KalshiMarket,
                   threshold_probs: Dict[float, float]) -> Signal:
    """Compute the trade signal for one Kalshi market.

    Inputs:
      market           Kalshi market (from kalshi.py)
      threshold_probs  dict[threshold_value -> model_prob], from
                       model.threshold_probabilities()

    Returns a Signal with decision in {BUY_YES, BUY_NO, NO_TRADE}.
    Filters:
      • min_volume         — ignore thinly-traded markets
      • min_open_interest  — same, but for skin-in-the-game
      • max_spread_cents   — wide spreads eat the edge
      • min_edge           — only act when edge ≥ ±10pt by default

    The decision_reason names the FIRST gate that failed (or
    "edge_yes" / "edge_no" if we trade). Helpful for diagnostics
    when scanning the daily output for "why didn't we bet?".
    """
    thr = market.threshold_value
    if thr is None or thr not in threshold_probs:
        return _no_trade(market, 0.0, 0.0,
                         "no_threshold_match")

    model_p = threshold_probs[thr]
    kalshi_p = implied_prob(market.yes_ask_cents) or 0.0
    edge = model_p - kalshi_p
    spread = (market.no_ask_cents - (100 - market.yes_ask_cents)
              if (market.yes_ask_cents is not None
                  and market.no_ask_cents is not None) else None)

    # Liquidity filter.
    if market.volume < cfg.min_volume:
        return _no_trade(market, model_p, kalshi_p,
                         f"volume_too_low ({market.volume}<{cfg.min_volume})",
                         edge=edge, spread=spread)
    if market.open_interest < cfg.min_open_interest:
        return _no_trade(market, model_p, kalshi_p,
                         (f"open_interest_too_low "
                          f"({market.open_interest}<{cfg.min_open_interest})"),
                         edge=edge, spread=spread)
    if spread is not None and spread > cfg.max_spread_cents:
        return _no_trade(market, model_p, kalshi_p,
                         f"spread_too_wide ({spread}c>{cfg.max_spread_cents}c)",
                         edge=edge, spread=spread)

    # Edge gate. Symmetric: positive edge (model thinks more likely
    # than market) → BUY YES; negative edge → BUY NO.
    max_entry = getattr(cfg, "val_max_entry_price_cents", 100)
    if edge >= cfg.min_edge:
        # Variance gate: refuse to pay more than max_entry for YES.
        if (market.yes_ask_cents is not None
                and market.yes_ask_cents > max_entry):
            return _no_trade(
                market, model_p, kalshi_p,
                (f"entry_too_expensive_yes "
                 f"({market.yes_ask_cents}c>{max_entry}c)"),
                edge=edge, spread=spread,
            )
        return _trade(market, model_p, kalshi_p, edge, spread,
                      decision="BUY_YES", reason=f"edge_yes={edge:+.3f}")
    if edge <= -cfg.min_edge:
        if (market.no_ask_cents is not None
                and market.no_ask_cents > max_entry):
            return _no_trade(
                market, model_p, kalshi_p,
                (f"entry_too_expensive_no "
                 f"({market.no_ask_cents}c>{max_entry}c)"),
                edge=edge, spread=spread,
            )
        return _trade(market, model_p, kalshi_p, edge, spread,
                      decision="BUY_NO", reason=f"edge_no={edge:+.3f}")
    return _no_trade(market, model_p, kalshi_p,
                     f"insufficient_edge ({edge:+.3f})",
                     edge=edge, spread=spread)


def compute_signals(cfg: Config, markets: List[KalshiMarket],
                    threshold_probs: Dict[float, float]) -> List[Signal]:
    """Run compute_signal across a market list, return all signals."""
    return [compute_signal(cfg, m, threshold_probs) for m in markets]


# --------------------------------------------------------------------------- #
# Convenience constructors
# --------------------------------------------------------------------------- #

def _trade(market: KalshiMarket, model_p: float, kalshi_p: float,
           edge: float, spread: Optional[int],
           decision: str, reason: str) -> Signal:
    return Signal(
        ticker=market.ticker, threshold_value=market.threshold_value,
        yes_sub_title=market.yes_sub_title,
        model_prob=model_p, kalshi_implied_prob=kalshi_p, edge=edge,
        yes_ask_cents=market.yes_ask_cents, no_ask_cents=market.no_ask_cents,
        volume=market.volume, open_interest=market.open_interest,
        spread_cents=spread, decision=decision, decision_reason=reason,
    )


def _no_trade(market: KalshiMarket, model_p: float, kalshi_p: float,
              reason: str, edge: float = 0.0,
              spread: Optional[int] = None) -> Signal:
    return Signal(
        ticker=market.ticker, threshold_value=market.threshold_value,
        yes_sub_title=market.yes_sub_title,
        model_prob=model_p, kalshi_implied_prob=kalshi_p, edge=edge,
        yes_ask_cents=market.yes_ask_cents, no_ask_cents=market.no_ask_cents,
        volume=market.volume, open_interest=market.open_interest,
        spread_cents=spread, decision="NO_TRADE", decision_reason=reason,
    )


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def signals_to_records(signals: List[Signal]) -> List[Dict]:
    """List[Signal] → list[dict] for CSV / JSON serialization."""
    return [asdict(s) for s in signals]
