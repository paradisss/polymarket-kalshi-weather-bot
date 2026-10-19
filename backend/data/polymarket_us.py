"""Polymarket US (CFTC-regulated, open to US users): BTC Up/Down markets, live BRTI price, orders.

Public data comes from https://gateway.polymarket.us (no key). Everything here was checked
against live markets on 2026-10-12:
  * BTC series: btc-updown-15m, btc-updown-1h, btc-updown-1d. Market slug cpc-btc-updown-<tf>-<start>z.
  * Settlement index: CF Benchmarks BRTI. The gateway serves the same index as 1-second candles at
    /v1/asset-prices/history. priceToBeat == close of the 1s candle starting 1 s before windowStart,
    and settlementPrice == close of the candle 1 s before windowEnd (matched to the cent).
  * Quotes: /v1/markets/<slug>/bbo. Yes (long) = Up. Down ask = 1 - best bid.
  * Taker fee per share = feeCoefficient x p x (1 - p) (feeCoefficient comes with each market).
Orders (needs a Polymarket US API key): Ed25519 signature over timestamp+METHOD+path, sent as
X-PM-Access-Key / X-PM-Timestamp / X-PM-Signature to https://api.polymarket.us. Order prices are
always the LONG (Yes/Up) price, even for Down orders.
"""
import base64
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("trading_bot")

GATEWAY = "https://gateway.polymarket.us"
API = "https://api.polymarket.us"
UA = {"User-Agent": "Mozilla/5.0 (polymarket-btc-bot)"}
US_SERIES = {"15m": "btc-updown-15m", "1h": "btc-updown-1h", "1d": "btc-updown-1d"}
DEFAULT_FEE_COEF = 0.0695


def _ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _num(x) -> Optional[float]:
    if isinstance(x, dict):
        x = x.get("value")
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@dataclass
class UsMarket:
    slug: str
    timeframe: str
    window_start: datetime
    window_end: datetime
    price_to_beat: Optional[float]
    fee_coef: float
    tick: float
    up_bid: Optional[float] = None
    up_ask: Optional[float] = None
    bid_shares: Optional[float] = None
    ask_shares: Optional[float] = None
    state: str = ""

    @property
    def market_id(self) -> str:
        return self.slug

    @property
    def is_active(self) -> bool:
        now = datetime.now(timezone.utc)
        return self.window_start <= now < self.window_end

    @property
    def elapsed_frac(self) -> float:
        T = (self.window_end - self.window_start).total_seconds()
        el = (datetime.now(timezone.utc) - self.window_start).total_seconds()
        return max(0.0, min(1.0, el / T)) if T > 0 else 0.0

    @property
    def minutes_left(self) -> float:
        return max(0.0, (self.window_end - datetime.now(timezone.utc)).total_seconds() / 60)


def _parse_market(m: dict, tf: str) -> Optional[UsMarket]:
    a = m.get("assetPriceTerms") or {}
    ws, we = _ts(a.get("windowStart")), _ts(a.get("windowEnd"))
    if not (ws and we):
        return None
    return UsMarket(
        slug=m["slug"], timeframe=tf, window_start=ws, window_end=we,
        price_to_beat=_num(a.get("priceToBeat")),
        fee_coef=float(m.get("feeCoefficient") or DEFAULT_FEE_COEF),
        tick=float(m.get("orderPriceMinTickSize") or 0.01),
        up_bid=_num(m.get("bestBidQuote")), up_ask=_num(m.get("bestAskQuote")),
        state=m.get("status", ""),
    )


def _candidate_slugs(now: datetime) -> List[Tuple[str, str]]:
    """Slugs of the windows live right now (slug time = window start, UTC)."""
    from datetime import timedelta
    out = []
    q = now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)
    out.append(("15m", f"cpc-btc-updown-15m-{q:%Y-%m-%d-%H%M}z"))
    h = now.replace(minute=0, second=0, microsecond=0)
    out.append(("1h", f"cpc-btc-updown-1h-{h:%Y-%m-%d-%H%M}z"))
    for hour in (21, 22):  # daily window starts 5pm New York (21:00Z summer, 22:00Z winter)
        d = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if d > now:
            d -= timedelta(days=1)
        out.append(("1d", f"cpc-btc-updown-1d-{d:%Y-%m-%d-%H%M}z"))
    return out


async def fetch_us_btc_markets() -> List[UsMarket]:
    """Current (live) BTC Up/Down market of each series, with a fresh best bid/offer."""
    now = datetime.now(timezone.utc)
    found: Dict[str, UsMarket] = {}
    tf_by_series = {v: k for k, v in US_SERIES.items()}
    async with httpx.AsyncClient(timeout=15, headers=UA) as c:
        try:
            r = await c.get(f"{GATEWAY}/v1/search", params={"query": "BTC Up or Down", "limit": 50})
            r.raise_for_status()
            for ev in r.json().get("events", []):
                tf = tf_by_series.get(ev.get("seriesSlug"))
                if not tf:
                    continue
                for m in ev.get("markets", []):
                    um = _parse_market(m, tf)
                    if um and not m.get("closed"):
                        found[um.slug] = um
        except Exception as e:
            logger.debug(f"Polymarket US search failed: {e}")
        have_tf = {m.timeframe for m in found.values() if m.is_active}
        for tf, slug in _candidate_slugs(now):
            if tf in have_tf or slug in found:
                continue
            try:
                r = await c.get(f"{GATEWAY}/v1/market/slug/{slug}")
                if r.status_code != 200:
                    continue
                m = r.json().get("market") or {}
                um = _parse_market(m, tf) if m else None
                if um and not m.get("closed") and um.is_active:
                    found[um.slug] = um
                    have_tf.add(tf)
            except Exception as e:
                logger.debug(f"Polymarket US market lookup failed for {slug}: {e}")
        out = [m for m in found.values() if m.window_end > now]
        for um in out:
            try:
                r = await c.get(f"{GATEWAY}/v1/markets/{um.slug}/bbo")
                r.raise_for_status()
                d = r.json().get("marketData", {})
                um.up_bid = _num(d.get("bestBid"))
                um.up_ask = _num(d.get("bestAsk"))
                um.bid_shares = _num(d.get("bidShares"))
                um.ask_shares = _num(d.get("askShares"))
                um.state = d.get("state", um.state)
            except Exception as e:
                logger.debug(f"Polymarket US bbo failed for {um.slug}: {e}")
                um.up_bid = um.up_ask = None
    return out


# ---------------------------------------------------------------- BRTI (settlement index)
_brti_cache: Tuple[float, Optional[float]] = (0.0, None)


async def fetch_brti(seconds: int = 120) -> Tuple[Optional[float], List[float]]:
    """Latest BRTI (the exact index Polymarket US settles on) plus recent 1s closes."""
    now = int(time.time())
    try:
        async with httpx.AsyncClient(timeout=10, headers=UA) as c:
            r = await c.get(f"{GATEWAY}/v1/asset-prices/history", params={
                "assets": "crypto:btc", "from": now - seconds, "bucket": "ASSET_PRICE_BUCKET_1S"})
            r.raise_for_status()
            series = r.json().get("series", [])
        candles = series[0].get("candles", []) if series else []
        closes = [float(x["close"]) for x in candles if x.get("close")]
        return (closes[-1] if closes else None), closes
    except Exception as e:
        logger.debug(f"BRTI fetch failed: {e}")
        return None, []


async def fetch_brti_minute_vol(minutes: int = 60) -> Optional[float]:
    """Realized BRTI volatility per minute, in percent, from 1-minute candles."""
    import math
    now = int(time.time())
    try:
        async with httpx.AsyncClient(timeout=10, headers=UA) as c:
            r = await c.get(f"{GATEWAY}/v1/asset-prices/history", params={
                "assets": "crypto:btc", "from": now - minutes * 60, "bucket": "ASSET_PRICE_BUCKET_1M"})
            r.raise_for_status()
            series = r.json().get("series", [])
        closes = [float(x["close"]) for x in (series[0].get("candles", []) if series else [])]
        rets = [math.log(b / a) * 100 for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
        if len(rets) < 10:
            return None
        m = sum(rets) / len(rets)
        return math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
    except Exception as e:
        logger.debug(f"BRTI vol fetch failed: {e}")
        return None


async def fetch_us_resolution(slug: str) -> Tuple[bool, Optional[float]]:
    """(resolved, 1.0 if Up won / 0.0 if Down won) for a Polymarket US market."""
    try:
        async with httpx.AsyncClient(timeout=15, headers=UA) as c:
            r = await c.get(f"{GATEWAY}/v1/market/slug/{slug}")
            r.raise_for_status()
            m = r.json().get("market", {})
    except Exception as e:
        logger.debug(f"Polymarket US resolution failed for {slug}: {e}")
        return False, None
    if m.get("status") != "MARKET_STATUS_RESOLVED":
        return False, None
    a = m.get("assetPriceTerms") or {}
    sp, beat = _num(a.get("settlementPrice")), _num(a.get("priceToBeat"))
    if sp is not None and beat is not None:
        return True, 1.0 if sp >= beat else 0.0  # rule: Up if end >= start
    try:
        import json
        prices = m.get("outcomePrices")
        prices = json.loads(prices) if isinstance(prices, str) else prices
        first = float(prices[0])
        if first > 0.99:
            return True, 1.0
        if first < 0.01:
            return True, 0.0
    except Exception:
        pass
    return False, None


# ---------------------------------------------------------------- authenticated client
class PolymarketUSClient:
    """Minimal signed client for the Polymarket US trading API (needs an API key)."""

    def __init__(self, key_id: str, secret_b64: str):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        raw = base64.b64decode(secret_b64)
        self.key_id = key_id
        self._key = Ed25519PrivateKey.from_private_bytes(raw[:32])

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        sig = self._key.sign(f"{ts}{method}{path}".encode())
        return {"X-PM-Access-Key": self.key_id, "X-PM-Timestamp": ts,
                "X-PM-Signature": base64.b64encode(sig).decode(), "Content-Type": "application/json",
                **UA}

    async def _req(self, method: str, path: str, body: Optional[dict] = None, query: Optional[dict] = None):
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.request(method, API + path, headers=self._headers(method, path),
                                json=body, params=query)
        if r.status_code >= 400:
            raise RuntimeError(f"Polymarket US {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else {}

    async def balances(self):
        return await self._req("GET", "/v1/account/balances")

    async def positions(self, market: Optional[str] = None):
        return await self._req("GET", "/v1/portfolio/positions", query={"market": market} if market else None)

    async def order(self, order_id: str):
        return await self._req("GET", f"/v1/order/{order_id}")

    async def create_order(self, slug: str, side: str, action: str, side_price: float,
                           quantity: int, tick: float = 0.01) -> dict:
        """Limit IOC order. side: "up"/"down"; action: "buy"/"sell"; side_price in that side's terms."""
        intent = {("up", "buy"): "ORDER_INTENT_BUY_LONG", ("up", "sell"): "ORDER_INTENT_SELL_LONG",
                  ("down", "buy"): "ORDER_INTENT_BUY_SHORT", ("down", "sell"): "ORDER_INTENT_SELL_SHORT"}[(side, action)]
        long_px = side_price if side == "up" else 1 - side_price
        long_px = round(round(long_px / tick) * tick, 3)
        if not 0.01 <= long_px <= 0.99:
            raise ValueError(f"price {long_px} out of range")
        return await self._req("POST", "/v1/orders", body={
            "marketSlug": slug, "intent": intent, "type": "ORDER_TYPE_LIMIT",
            "price": {"value": f"{long_px:.3f}", "currency": "USD"},
            "quantity": int(quantity), "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
            "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
            "synchronousExecution": True,
        })
