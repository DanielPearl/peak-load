"""Kalshi REST client for the Natural Gas Price bot.

Real Kalshi only — no demo / synthetic markets. If credentials aren't
set or no markets are listed for the configured series, the watchlist
is honestly empty.

Endpoint: signed REST against api.elections.kalshi.com.
Target series: KXNATGASD (Henry Hub daily, Pyth-settled at 5pm EDT).
Ticker format: KXNATGASD-{YYMMM}DD17-T{price}, e.g.
  KXNATGASD-26MAY0417-T2.750  → "above $2.750" on May 4, 2026
"""
from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

from .config import Config

log = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class KalshiMarket:
    ticker: str
    yes_sub_title: str
    threshold_value: Optional[float]    # parsed from yes_sub_title or ticker
                                        # (in $/MMBTU for NG markets)
    yes_ask_cents: Optional[int]
    no_ask_cents: Optional[int]
    volume: int
    open_interest: int
    raw: dict


# --------------------------------------------------------------------------- #
# Signed REST client
# --------------------------------------------------------------------------- #

class _SignedClient:
    def __init__(self, cfg: Config):
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography lib not installed")
        if not cfg.kalshi_api_key_id or not cfg.kalshi_private_key_path:
            raise RuntimeError(
                "KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH unset")
        self.api_key = cfg.kalshi_api_key_id
        self.base = KALSHI_BASE
        key_bytes = Path(cfg.kalshi_private_key_path).expanduser().read_bytes()
        self.priv: RSAPrivateKey = serialization.load_pem_private_key(
            key_bytes, password=None)

    def _headers(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method}{path}".encode()
        sig = base64.b64encode(self.priv.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                         salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())).decode()
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Accept": "application/json",
        }

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        r = requests.get(self.base + path, headers=self._headers("GET", path),
                         params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def fetch_kalshi_markets(cfg: Config,
                         forecast_usd: Optional[float] = None,
                         ) -> List[KalshiMarket]:
    """Return open KXNATGASD markets nearest the upcoming close.

    Real Kalshi only. If no creds or no open markets, returns [].

    `forecast_usd` is reserved for future use (could filter to strikes
    near the forecast for a tighter watchlist).
    """
    if not (cfg.kalshi_api_key_id and cfg.kalshi_private_key_path
            and HAS_CRYPTO):
        log.info("Kalshi creds missing — empty watchlist")
        return []
    try:
        return _fetch_kalshi_real(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Kalshi fetch failed (%s); empty watchlist", exc)
        return []


def fetch_market_status(cfg: Config, ticker: str) -> Optional[dict]:
    """Fetch a single market's current state — used by the daily runner
    to detect when an open position's market has resolved.

    Returns the raw market dict or None on failure.
    """
    if not (cfg.kalshi_api_key_id and cfg.kalshi_private_key_path
            and HAS_CRYPTO):
        return None
    try:
        client = _SignedClient(cfg)
        return client.get(f"/markets/{ticker}").get("market", {})
    except Exception as exc:  # noqa: BLE001
        log.warning("market-status fetch failed for %s: %s", ticker, exc)
        return None


def _fetch_kalshi_real(cfg: Config) -> List[KalshiMarket]:
    """Real Kalshi /markets call paginating through all open KXNATGASD."""
    client = _SignedClient(cfg)
    out: List[KalshiMarket] = []
    cursor = None
    prefix = cfg.kalshi_series_prefix
    while True:
        params = {"limit": 200, "status": "open",
                  "series_ticker": prefix}
        if cursor:
            params["cursor"] = cursor
        resp = client.get("/markets", params=params)
        for m in resp.get("markets", []) or []:
            ticker = m.get("ticker", "")
            if not ticker.startswith(prefix):
                continue
            out.append(_parse_market(m))
        cursor = resp.get("cursor")
        if not cursor:
            break
    return out


def _parse_market(raw: dict) -> KalshiMarket:
    ticker = raw.get("ticker", "")
    sub = raw.get("yes_sub_title", "") or ""
    threshold = _parse_threshold(ticker, sub)
    return KalshiMarket(
        ticker=ticker,
        yes_sub_title=sub,
        threshold_value=threshold,
        # Kalshi has two price formats: legacy int cents (`yes_ask`) and
        # newer `yes_ask_dollars` string. Many low-liquidity markets only
        # populate the dollar version, so fall through both.
        yes_ask_cents=_price_cents(raw, side="yes_ask"),
        no_ask_cents=_price_cents(raw, side="no_ask"),
        volume=_fp_int(raw.get("volume_fp"), raw.get("volume")),
        open_interest=_fp_int(raw.get("open_interest_fp"),
                              raw.get("open_interest")),
        raw=raw,
    )


def _price_cents(raw: dict, side: str) -> Optional[int]:
    """Read a price (yes_ask / no_ask / yes_bid) handling both Kalshi
    formats — legacy ``{side}`` integer cents and newer
    ``{side}_dollars`` decimal strings. Falls back to last-trade price
    when ask is missing entirely.
    """
    legacy = raw.get(side)
    if legacy not in (None, ""):
        try:
            return int(legacy)
        except (TypeError, ValueError):
            pass
    dollars = raw.get(f"{side}_dollars")
    if dollars not in (None, ""):
        try:
            return int(round(float(dollars) * 100))
        except (TypeError, ValueError):
            pass
    return None


# Threshold parser — KXNATGASD tickers look like:
#   KXNATGASD-26MAY0417-T2.750   → 2.750
#   KXNATGASD-26MAY0417-B2.745   → 2.745  (some markets use B/T prefixes)
# Subtitles look like "above $2.750" or "$2.750 or above".
_THRESHOLD_RE_PRICE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")
_THRESHOLD_RE_TICKER = re.compile(r"-[BT](\d+(?:\.\d+)?)$")


def _parse_threshold(ticker: str, sub_title: str) -> Optional[float]:
    """Pull the $/MMBTU threshold from the ticker tail or subtitle.

    Tries the ticker tail first (`-T2.750` or `-B2.745`) since it's
    structurally cleaner; falls back to the subtitle text.
    """
    if ticker:
        m = _THRESHOLD_RE_TICKER.search(ticker)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    for src in (sub_title, ticker):
        if not src:
            continue
        m = _THRESHOLD_RE_PRICE.search(src)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _to_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fp_int(fp_value, legacy_value) -> int:
    """Kalshi's _fp fields are decimal strings; legacy fields are ints."""
    if fp_value not in (None, ""):
        try:
            return int(round(float(fp_value)))
        except (TypeError, ValueError):
            pass
    if legacy_value not in (None, ""):
        try:
            return int(legacy_value)
        except (TypeError, ValueError):
            pass
    return 0
