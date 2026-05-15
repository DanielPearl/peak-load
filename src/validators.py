"""Pre-trade validators for the natural-gas-price bot.

Same shape and threshold floor as every other Kalshi bot's validator:
each gate returns ``(ok: bool, reason: str)``; the first failure
short-circuits with a reason that's logged so we can audit why a
candidate was skipped. Threshold defaults are inherited from the
shared ``kalshi_sdk.validators.UNIFIED_VALIDATOR_DEFAULTS`` floor so
no bot drifts looser than the cautious-side baseline.

NG runs once per daily tick (not a polling loop) and reads quotes off
the ``KalshiMarket`` directly — there is no Orderbook object on this
path. The function body is intentionally local rather than a wrapper
over ``shared.validate_market`` because the market shape (no Orderbook,
no `direction` enum, no `strike_low`) doesn't match. The THRESHOLDS,
however, are the shared cautious floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from kalshi_sdk.validators import UNIFIED_VALIDATOR_DEFAULTS

from .kalshi import KalshiMarket


@dataclass
class ValidatorCfg:
    # Liquidity gates — unified floor across every bot.
    min_volume: int = UNIFIED_VALIDATOR_DEFAULTS["min_volume"]
    min_open_interest: int = UNIFIED_VALIDATOR_DEFAULTS["min_open_interest"]
    max_spread_cents: int = UNIFIED_VALIDATOR_DEFAULTS["max_spread_cents"]
    # Price-band gate. Unified cautious bounds = (25, 75) — same as every
    # other bot. Tail-probability markets are skipped.
    prob_bounds_cents: Tuple[int, int] = UNIFIED_VALIDATOR_DEFAULTS["prob_bounds_cents"]
    # Hard cap on what we'll pay for the SIDE we'd actually buy. Variance
    # protection independent of edge math.
    max_entry_price_cents: int = UNIFIED_VALIDATOR_DEFAULTS["max_entry_price_cents"]
    # Time gate. Same min as every bot; max is left wide-open so the
    # daily NG tick can scan markets up to 2 weeks out.
    min_minutes_to_close: int = UNIFIED_VALIDATOR_DEFAULTS["min_minutes_to_close"]
    max_minutes_to_close: int = UNIFIED_VALIDATOR_DEFAULTS["max_minutes_to_close"]
    # Forecast-anchor basis-risk gate. Skip strikes within ±X $/MMBTU
    # of the model's forecast when very close to close — settlement-print
    # noise dominates the edge there.
    basis_risk_strike_window: float = 0.05
    basis_risk_max_hours_to_close: float = 4


def validate_market(
    market: KalshiMarket,
    cfg: ValidatorCfg,
    forecast_value: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
) -> Tuple[bool, str]:
    """Returns (ok, reason). First-fail-wins so logs are audit-friendly."""
    if market.threshold_value is None:
        return False, "no_threshold_match"

    if market.volume < cfg.min_volume:
        return False, f"volume_too_low ({market.volume}<{cfg.min_volume})"
    if market.open_interest < cfg.min_open_interest:
        return False, (f"open_interest_too_low "
                       f"({market.open_interest}<{cfg.min_open_interest})")

    if market.yes_ask_cents is None or market.no_ask_cents is None:
        return False, "no_two_sided_book"
    spread = (market.no_ask_cents - (100 - market.yes_ask_cents))
    if spread > cfg.max_spread_cents:
        return False, f"spread_too_wide ({spread}c>{cfg.max_spread_cents}c)"

    lo, hi = cfg.prob_bounds_cents
    if not (lo <= market.yes_ask_cents <= hi):
        return False, (f"yes_ask_outside_bounds "
                       f"({market.yes_ask_cents}c not in {lo}-{hi})")

    # Hard cap on entry price (same gate every other bot runs). The yes_ask
    # is the side we'd buy here (NG markets are always traded long-YES on
    # the bot's standard "above forecast" stance).
    if market.yes_ask_cents > cfg.max_entry_price_cents:
        return False, (f"entry_too_expensive ({market.yes_ask_cents}c>"
                       f"{cfg.max_entry_price_cents}c)")

    if minutes_to_close is not None:
        if minutes_to_close < cfg.min_minutes_to_close:
            return False, f"too_close_to_close ({minutes_to_close:.0f}min)"
        if minutes_to_close > cfg.max_minutes_to_close:
            return False, f"too_far_from_close ({minutes_to_close:.0f}min)"

    if (cfg.basis_risk_strike_window > 0
            and cfg.basis_risk_max_hours_to_close > 0
            and forecast_value is not None
            and minutes_to_close is not None
            and minutes_to_close < cfg.basis_risk_max_hours_to_close * 60):
        gap = abs(market.threshold_value - forecast_value)
        if gap <= cfg.basis_risk_strike_window:
            return False, (f"basis_risk_zone (strike ${market.threshold_value:.3f} "
                           f"within ${cfg.basis_risk_strike_window:.3f}/MMBTU "
                           f"of forecast ${forecast_value:.3f}, "
                           f"TTC {minutes_to_close:.0f}min)")

    return True, "ok"
