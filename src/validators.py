"""Pre-trade validators for the peak-load bot.

Same shape as the gas-prices / unemployment-claims validator: each
gate returns ``(ok: bool, reason: str)``; the first failure short-
circuits with a reason that's logged so we can audit why a candidate
was skipped.

Unlike the other bots' validators (which run on a polling loop), this
runs once per daily tick. A few thresholds are tuned looser than the
continuous bots because the daily cadence already adds breathing room
(no spread-eaten-by-spam-trading risk).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .kalshi import KalshiMarket


@dataclass
class ValidatorCfg:
    # Liquidity gates — same intent as the other bots.
    min_volume: int = 50               # lifetime contracts traded
    min_open_interest: int = 50        # currently-held contracts
    max_spread_cents: int = 8          # round-trip spread cost
    # Price-band gate. Skip deep-in / deep-out tail markets where
    # asymmetric payoffs make the edge fragile.
    prob_bounds_cents: Tuple[int, int] = (5, 95)
    # Time gate. Peak-load markets typically resolve at end of day
    # so there's no "too far out" issue, but a minimum gives us
    # breathing room against minute-to-resolution noise.
    min_minutes_to_close: int = 30
    max_minutes_to_close: int = 60 * 24 * 7   # 7d ceiling for safety
    # Forecast-anchor basis-risk gate. Skip strikes within ±X MW of
    # the model's forecast when very close to close — the difference
    # between the model's forecast (off our weather forecast) and
    # actual realized peak load is dominated by forecast error there,
    # not real edge.
    basis_risk_strike_window_mw: float = 1500
    basis_risk_max_hours_to_close: float = 4


def validate_market(
    market: KalshiMarket,
    cfg: ValidatorCfg,
    forecast_mw: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
) -> Tuple[bool, str]:
    """Returns (ok, reason). First-fail-wins so logs are audit-friendly."""
    # Threshold parsed?
    if market.threshold_mw is None:
        return False, "no_threshold_match"

    # Liquidity.
    if market.volume < cfg.min_volume:
        return False, f"volume_too_low ({market.volume}<{cfg.min_volume})"
    if market.open_interest < cfg.min_open_interest:
        return False, (f"open_interest_too_low "
                       f"({market.open_interest}<{cfg.min_open_interest})")

    # Two-sided book.
    if market.yes_ask_cents is None or market.no_ask_cents is None:
        return False, "no_two_sided_book"
    spread = (market.no_ask_cents - (100 - market.yes_ask_cents))
    if spread > cfg.max_spread_cents:
        return False, f"spread_too_wide ({spread}c>{cfg.max_spread_cents}c)"

    # Price band (deep-tail filter).
    lo, hi = cfg.prob_bounds_cents
    if not (lo <= market.yes_ask_cents <= hi):
        return False, (f"yes_ask_outside_bounds "
                       f"({market.yes_ask_cents}c not in {lo}-{hi})")

    # Time-to-close gate.
    if minutes_to_close is not None:
        if minutes_to_close < cfg.min_minutes_to_close:
            return False, f"too_close_to_close ({minutes_to_close:.0f}min)"
        if minutes_to_close > cfg.max_minutes_to_close:
            return False, f"too_far_from_close ({minutes_to_close:.0f}min)"

    # Basis-risk gate. Strikes near the forecast value within the
    # last few hours have edges dominated by forecast error rather
    # than genuine model conviction.
    if (cfg.basis_risk_strike_window_mw > 0
            and cfg.basis_risk_max_hours_to_close > 0
            and forecast_mw is not None
            and minutes_to_close is not None
            and minutes_to_close < cfg.basis_risk_max_hours_to_close * 60):
        gap = abs(market.threshold_mw - forecast_mw)
        if gap <= cfg.basis_risk_strike_window_mw:
            return False, (f"basis_risk_zone (strike {market.threshold_mw} "
                           f"within {cfg.basis_risk_strike_window_mw:.0f} MW "
                           f"of forecast {forecast_mw:.0f}, "
                           f"TTC {minutes_to_close:.0f}min)")

    return True, "ok"
