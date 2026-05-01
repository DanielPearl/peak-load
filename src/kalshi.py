"""Minimal Kalshi REST client + a synthetic-market stub.

Just enough to:
  • Fetch open peak-load markets for the configured region series
  • Read each market's yes_ask / no_ask / volume / open_interest

Heavy-lifting (orderbook depth, real-time WS, order placement) lives
in the gas-prices KalshiClient and isn't reproduced here — the
peak-load bot is a daily decision-support tool, not a live executor.

If KALSHI_* env vars aren't set, a synthetic market generator
produces realistic-looking peak-load contracts (one strike per
threshold in the grid) so the daily script runs end-to-end without
real credentials.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
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

KALSHI_BASE = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
}


@dataclass
class KalshiMarket:
    ticker: str
    yes_sub_title: str
    threshold_mw: Optional[int]      # parsed from yes_sub_title or ticker
    yes_ask_cents: Optional[int]
    no_ask_cents: Optional[int]
    volume: int
    open_interest: int
    raw: dict


# --------------------------------------------------------------------------- #
# Real client (signed REST)
# --------------------------------------------------------------------------- #

class _SignedClient:
    def __init__(self, cfg: Config):
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography lib not installed")
        if not cfg.kalshi_api_key_id or not cfg.kalshi_private_key_path:
            raise RuntimeError("KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH unset")
        self.api_key = cfg.kalshi_api_key_id
        self.base = KALSHI_BASE[cfg.kalshi_env]
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

def fetch_kalshi_markets(cfg: Config) -> List[KalshiMarket]:
    """Return the open peak-load markets for the configured region.

    Real path: Kalshi GET /markets?series_ticker={prefix}&status=open.
    Falls back to synthetic markets when credentials aren't set so a
    fresh clone runs end-to-end. Synthetic markets are anchored to
    the threshold grid in config.
    """
    prefix = cfg.region_meta["kalshi_series_prefix"]
    if cfg.kalshi_api_key_id and cfg.kalshi_private_key_path and HAS_CRYPTO:
        try:
            real = _fetch_kalshi_real(cfg, prefix)
            if real:
                return real
            log.info("Kalshi returned 0 markets for series %r — exchange "
                     "may not yet list a peak-load series for this region.",
                     prefix)
        except Exception as exc:  # noqa: BLE001
            log.warning("Kalshi real fetch failed (%s)", exc)
    # If KALSHI_DEMO_MODE=true, generate plausible markets so the
    # simulation pipeline (open position, mark, close on resolution)
    # can be exercised end-to-end against the dashboard. Tickers are
    # tagged DEMO so they're distinguishable from real ones.
    if cfg.kalshi_demo_mode:
        log.info("KALSHI_DEMO_MODE=true → generating demo markets")
        return _demo_markets(cfg)
    return []


def _demo_markets(cfg: Config) -> List[KalshiMarket]:
    """Synthetic peak-load markets for demo / pipeline-validation use.

    Prices are anchored on a Gaussian centered on the region's typical
    seasonal peak so high thresholds price low and vice versa, with
    ±3pt noise added so the model can find disagreements. Volumes /
    OIs are randomized but stay above the bot's liquidity floors.

    Tickers carry a DEMO suffix so they can't be confused with real
    Kalshi markets when reading the dashboard or output JSON.
    """
    import time as _time
    rng = np.random.default_rng(seed=int(_time.time()))
    avg_peak = (cfg.region_meta["summer_peak_mw"]
                + cfg.region_meta["winter_peak_mw"]) / 2
    market_sigma = avg_peak * 0.05
    out: List[KalshiMarket] = []
    from scipy.stats import norm
    for thr in cfg.threshold_grid_mw:
        p = float(1 - norm.cdf((thr - avg_peak) / market_sigma))
        p_noise = max(0.01, min(0.99, p + rng.normal(0, 0.03)))
        yes_cents = int(round(p_noise * 100))
        out.append(KalshiMarket(
            ticker=f"{cfg.region_meta['kalshi_series_prefix']}-DEMO-{thr}",
            yes_sub_title=f"Above {thr} MW (DEMO)",
            threshold_mw=thr,
            yes_ask_cents=yes_cents,
            no_ask_cents=100 - yes_cents,
            volume=int(rng.integers(100, 5000)),
            open_interest=int(rng.integers(80, 4000)),
            raw={"demo": True},
        ))
    return out


def fetch_market_status(cfg: Config, ticker: str) -> Optional[dict]:
    """Fetch a single market's current state — used by the daily
    runner to detect when an open position's market has resolved.

    Returns the raw market dict (status, settle value, prices) or None
    if the API call fails. Intentionally non-fatal: if Kalshi is
    unreachable we just leave the position open and try again
    tomorrow.
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


def _fetch_kalshi_real(cfg: Config, series_prefix: str) -> List[KalshiMarket]:
    """Real Kalshi call — same pattern as the gas-prices bot."""
    client = _SignedClient(cfg)
    out: List[KalshiMarket] = []
    cursor = None
    while True:
        params = {"limit": 200, "status": "open",
                  "series_ticker": series_prefix}
        if cursor:
            params["cursor"] = cursor
        resp = client.get("/markets", params=params)
        for m in resp.get("markets", []) or []:
            ticker = m.get("ticker", "")
            if not ticker.startswith(series_prefix):
                continue
            out.append(_parse_market(m))
        cursor = resp.get("cursor")
        if not cursor:
            break
    return out


def _parse_market(raw: dict) -> KalshiMarket:
    ticker = raw.get("ticker", "")
    sub = raw.get("yes_sub_title", "") or ""
    # Try to extract MW threshold from "Above 75000 MW" or "75000 MW or higher".
    threshold = _parse_threshold(ticker, sub)
    return KalshiMarket(
        ticker=ticker,
        yes_sub_title=sub,
        threshold_mw=threshold,
        yes_ask_cents=_to_int(raw.get("yes_ask")),
        no_ask_cents=_to_int(raw.get("no_ask")),
        volume=_fp_int(raw.get("volume_fp"), raw.get("volume")),
        open_interest=_fp_int(raw.get("open_interest_fp"), raw.get("open_interest")),
        raw=raw,
    )


def _parse_threshold(ticker: str, sub_title: str) -> Optional[int]:
    """Pull the MW threshold from the question text or ticker tail.

    Common patterns:
      'Above 75000 MW'      ← yes_sub_title
      'KXERCOTPL-26MAY15-75000'  ← ticker tail
    """
    import re
    for src in (sub_title, ticker):
        if not src:
            continue
        m = re.search(r"(\d{4,6})\s*(?:MW|mw)?\b", src)
        if m:
            return int(m.group(1))
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


# Synthetic-market fallback was removed — the dashboard was surfacing
# made-up tickers like KXERCOTPL-SIM-66000 that didn't exist on the
# real exchange. If Kalshi has no live peak-load series for the
# region, the watchlist is now correctly empty until real markets
# list. The forecast + threshold probabilities still produce.
