"""Kalshi data access for the Natural Gas Price bot.

Thin wrapper over the shared ``kalshi_sdk`` package. Public API is the
same as the pre-SDK implementation:

    fetch_kalshi_markets(cfg, forecast_usd=None) -> list[KalshiMarket]
    fetch_market_status(cfg, ticker)             -> dict | None
    KalshiMarket                                 — re-export from SDK
    _SignedClient                                — back-compat shim

Target series: KXNATGASD (Henry Hub daily, Pyth-settled at 5pm EDT).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from kalshi_sdk import KalshiClient, KalshiError, KalshiMarket, parse_market
from kalshi_sdk.cache import TTLCache

from .config import Config

log = logging.getLogger(__name__)

# Re-exported for back-compat with sites that imported these symbols.
__all__ = ["KalshiMarket", "fetch_kalshi_markets", "fetch_market_status", "_SignedClient"]

# Single-market lookups (e.g. position-status checks every daily run)
# don't change within a run; cache for 60s to suppress duplicates when
# the same ticker is checked twice in one invocation.
_market_cache = TTLCache(ttl_seconds=60.0)


def _client(cfg: Config) -> Optional[KalshiClient]:
    """Build a SDK client from a Natural Gas Config, or None if creds absent."""
    if not (cfg.kalshi_api_key_id and cfg.kalshi_private_key_path):
        return None
    try:
        return KalshiClient(
            api_key_id=cfg.kalshi_api_key_id,
            private_key_path=cfg.kalshi_private_key_path,
            cache_ttl=60.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Kalshi client init failed: %s", exc)
        return None


def fetch_kalshi_markets(cfg: Config,
                         forecast_usd: Optional[float] = None,
                         ) -> List[KalshiMarket]:
    """Return open KXNATGASD markets paginated through the series.

    Real Kalshi only. If creds missing or the call fails, returns [].
    `forecast_usd` is reserved for future filtering; currently unused.
    """
    client = _client(cfg)
    if client is None:
        log.info("Kalshi creds missing — empty watchlist")
        return []
    try:
        prefix = cfg.kalshi_series_prefix
        raws = client.iter_open_markets(series_ticker=prefix, prefix=prefix)
        return [parse_market(m) for m in raws]
    except KalshiError as exc:
        log.warning("Kalshi fetch failed (%s); empty watchlist", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("Kalshi fetch failed (%s); empty watchlist", exc)
        return []


def fetch_market_status(cfg: Config, ticker: str) -> Optional[dict]:
    """Fetch a single market's current state (used to detect resolution)."""
    cached = _market_cache.get(ticker)
    if cached is not None:
        return cached
    client = _client(cfg)
    if client is None:
        return None
    try:
        out = client.get_market(ticker).get("market", {})
        _market_cache.put(ticker, out)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("market-status fetch failed for %s: %s", ticker, exc)
        return None


class _SignedClient:
    """Back-compat shim. Old call sites used ``_SignedClient(cfg).get(path, params)``.

    Internally delegates to ``kalshi_sdk.KalshiClient``. New code should
    construct ``KalshiClient`` directly from the SDK.
    """

    def __init__(self, cfg: Config):
        c = _client(cfg)
        if c is None:
            raise RuntimeError(
                "KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH unset")
        self._client = c

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        # Mirrors KalshiClient._request signature for the GET path.
        return self._client._request("GET", path, params=params or {})
