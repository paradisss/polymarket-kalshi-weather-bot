"""BTC Up/Down market fetcher for Polymarket — 5m, 15m, hourly and daily windows."""
import asyncio
import httpx
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger("trading_bot")

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Timeframe -> Polymarket series, slug pattern, window length (s), how many open windows to track
TIMEFRAMES: Dict[str, dict] = {
    "5m": {
        "series": "btc-up-or-down-5m",
        "slug_re": re.compile(r"^btc-updown-5m-\d{10}$"),
        "seconds": 300,
        "count": 6,
    },
    "15m": {
        "series": "btc-up-or-down-15m",
        "slug_re": re.compile(r"^btc-updown-15m-\d{10}$"),
        "seconds": 900,
        "count": 4,
    },
    "1h": {
        "series": "btc-up-or-down-hourly",
        "slug_re": re.compile(r"^bitcoin-up-or-down-[a-z]+-\d{1,2}-\d{4}-\d{1,2}(am|pm)-et$"),
        "seconds": 3600,
        "count": 4,
    },
    "1d": {
        "series": "btc-up-or-down-daily",
        "slug_re": re.compile(r"^bitcoin-up-or-down-on-[a-z]+-\d{1,2}-\d{4}$"),
        "seconds": 86400,
        "count": 2,
    },
}

# Kept for backwards compatibility
SERIES_SLUG = TIMEFRAMES["5m"]["series"]


def enabled_timeframes() -> List[str]:
    """Timeframes switched on via BTC_TIMEFRAMES (default: all four)."""
    try:
        from backend.config import settings
        raw = getattr(settings, "BTC_TIMEFRAMES", "5m,15m,1h,1d") or "5m"
    except Exception:
        raw = "5m,15m,1h,1d"
    tfs = [t.strip() for t in raw.split(",") if t.strip() in TIMEFRAMES]
    return tfs or ["5m"]


def timeframe_from_slug(slug: str) -> Optional[str]:
    for tf, cfg in TIMEFRAMES.items():
        if cfg["slug_re"].match(slug or ""):
            return tf
    return None


def is_valid_btc_slug(slug: str) -> bool:
    """True only for real BTC Up/Down slugs of a known timeframe."""
    return timeframe_from_slug(slug) is not None


@dataclass
class BtcMarket:
    """A single BTC Up/Down market window."""
    slug: str
    market_id: str
    up_price: float
    down_price: float
    window_start: datetime
    window_end: datetime
    volume: float
    closed: bool
    timeframe: str = "5m"
    up_token_id: Optional[str] = None
    # Live Up order book (CLOB). Gamma's outcomePrices/bestBid lag the book by minutes,
    # so up_price/down_price are replaced by the CLOB mid when a book is available.
    up_bid: Optional[float] = None
    up_ask: Optional[float] = None
    quote_source: str = "gamma"

    @property
    def event_slug(self) -> str:
        return self.slug

    @property
    def spread(self) -> float:
        return abs(1.0 - self.up_price - self.down_price)

    @property
    def window_seconds(self) -> int:
        return TIMEFRAMES.get(self.timeframe, TIMEFRAMES["5m"])["seconds"]

    @property
    def time_until_end(self) -> float:
        """Seconds until this window ends."""
        return (self.window_end - datetime.now(timezone.utc)).total_seconds()

    @property
    def is_active(self) -> bool:
        """Window is currently in progress."""
        now = datetime.now(timezone.utc)
        return self.window_start <= now <= self.window_end and not self.closed

    @property
    def is_upcoming(self) -> bool:
        """Window hasn't started yet."""
        return datetime.now(timezone.utc) < self.window_start and not self.closed


def _round_to_5min(ts: float) -> int:
    """Round a unix timestamp down to the nearest 5-minute boundary."""
    return int(ts) // 300 * 300


def _compute_window_slugs(count: int = 5) -> List[str]:
    """Event slugs for the current and upcoming 5-min windows (btc-updown-5m-{start_ts})."""
    current_boundary = _round_to_5min(time.time())
    return [f"btc-updown-5m-{current_boundary + i * 300}" for i in range(count)]


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _parse_event_to_btc_market(event: dict, timeframe: Optional[str] = None) -> Optional[BtcMarket]:
    """Parse a Polymarket event into a BtcMarket."""
    markets = event.get("markets", [])
    if not markets:
        return None
    market = markets[0]
    slug = event.get("slug", "")
    tf = timeframe or timeframe_from_slug(slug) or "5m"

    # Outcome prices, matched to the Up/Down labels when present
    up_price, down_price = 0.5, 0.5
    up_token_id = None
    try:
        toks = market.get("clobTokenIds", "")
        toks = json.loads(toks) if isinstance(toks, str) and toks else toks
        outs = market.get("outcomes", "")
        outs = json.loads(outs) if isinstance(outs, str) and outs else outs
        if isinstance(toks, list) and toks:
            up_token_id = str(toks[outs.index("Up") if isinstance(outs, list) and "Up" in outs else 0])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        prices = market.get("outcomePrices", "")
        prices = json.loads(prices) if isinstance(prices, str) and prices else prices
        outcomes = market.get("outcomes", "")
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) and outcomes else outcomes
        if isinstance(prices, list) and len(prices) >= 2:
            iu = 0
            if isinstance(outcomes, list) and "Up" in outcomes:
                iu = outcomes.index("Up")
            up_price = float(prices[iu])
            down_price = float(prices[1 - iu])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Window timing: eventStartTime is the real window open. startDate is when the
    # market was *listed* (often a day earlier), so only fall back to end - length.
    window_end = _parse_dt(event.get("endDate") or market.get("endDate")) or datetime.now(timezone.utc)
    window_start = _parse_dt(market.get("eventStartTime") or event.get("startTime"))
    if window_start is None:
        from datetime import timedelta
        window_start = window_end - timedelta(seconds=TIMEFRAMES[tf]["seconds"])

    return BtcMarket(
        slug=slug,
        market_id=str(market.get("id", "")),
        up_price=up_price,
        down_price=down_price,
        window_start=window_start,
        window_end=window_end,
        volume=float(market.get("volumeNum") or market.get("volume") or 0),
        closed=bool(market.get("closed", False) or event.get("closed", False)),
        timeframe=tf,
        up_token_id=up_token_id,
    )


async def attach_clob_quotes(client: httpx.AsyncClient, markets: List[BtcMarket]) -> None:
    """Replace Gamma's lagging prices with the live CLOB Up book (one batched request)."""
    ids = [m.up_token_id for m in markets if m.up_token_id]
    if not ids:
        return
    try:
        resp = await client.post(f"{CLOB_API}/books", json=[{"token_id": t} for t in ids])
        resp.raise_for_status()
        books = {str(b.get("asset_id")): b for b in resp.json()}
    except Exception as e:
        logger.warning(f"CLOB books fetch failed (using Gamma prices): {e}")
        return
    for m in markets:
        b = books.get(m.up_token_id or "")
        if not b:
            continue
        bids = [float(x["price"]) for x in b.get("bids", [])]
        asks = [float(x["price"]) for x in b.get("asks", [])]
        if not bids or not asks:
            continue
        m.up_bid, m.up_ask = max(bids), min(asks)
        if m.up_ask >= m.up_bid:
            m.up_price = round((m.up_bid + m.up_ask) / 2, 4)
            m.down_price = round(1 - m.up_price, 4)
            m.quote_source = "clob"


async def fetch_btc_market_by_slug(slug: str) -> Optional[BtcMarket]:
    """Fetch a single BTC Up/Down market by its event slug."""
    if not is_valid_btc_slug(slug):
        logger.debug(f"Rejected invalid BTC slug: {slug}")
        return None
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{GAMMA_API}/events", params={"slug": slug})
            response.raise_for_status()
            events = response.json()
            if not events:
                return None
            event = events[0] if isinstance(events, list) else events
            return _parse_event_to_btc_market(event)
        except Exception as e:
            logger.debug(f"Failed to fetch BTC market {slug}: {e}")
            return None


async def _fetch_series(client: httpx.AsyncClient, tf: str) -> List[BtcMarket]:
    """Open windows for one timeframe, soonest first."""
    cfg = TIMEFRAMES[tf]
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        response = await client.get(
            f"{GAMMA_API}/events",
            params={
                "series_slug": cfg["series"],
                "closed": "false",
                "end_date_min": now_iso,
                "order": "endDate",
                "ascending": "true",
                "limit": cfg["count"] + 4,
            },
        )
        response.raise_for_status()
        out = []
        for event in response.json():
            if not cfg["slug_re"].match(event.get("slug", "")):
                continue
            m = _parse_event_to_btc_market(event, tf)
            if m and not m.closed and m.time_until_end > 0:
                out.append(m)
        out.sort(key=lambda m: m.window_end)
        return out[: cfg["count"]]
    except Exception as e:
        logger.warning(f"BTC {tf} series fetch failed: {e}")
        return []


async def fetch_active_btc_markets(timeframes: Optional[List[str]] = None) -> List[BtcMarket]:
    """Fetch current and upcoming BTC Up/Down windows for every enabled timeframe."""
    tfs = timeframes or enabled_timeframes()
    async with httpx.AsyncClient(timeout=15.0) as client:
        results = await asyncio.gather(*[_fetch_series(client, tf) for tf in tfs])
        await attach_clob_quotes(client, [m for found in results for m in found])

    markets: List[BtcMarket] = []
    seen = set()
    for tf, found in zip(tfs, results):
        # 5m fallback: compute expected slugs if the series query came back empty
        if tf == "5m" and not found:
            for slug in _compute_window_slugs(count=6):
                m = await fetch_btc_market_by_slug(slug)
                if m and not m.closed:
                    found.append(m)
        for m in found:
            if m.slug not in seen:
                seen.add(m.slug)
                markets.append(m)

    markets.sort(key=lambda m: m.window_end)
    counts = {tf: sum(1 for m in markets if m.timeframe == tf) for tf in tfs}
    logger.info("Fetched BTC markets: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return markets


async def fetch_btc_market_for_settlement(slug: str) -> Optional[BtcMarket]:
    """Fetch a BTC market for settlement purposes (includes closed markets)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{GAMMA_API}/events", params={"slug": slug})
            response.raise_for_status()
            events = response.json()
            if not events:
                return None
            event = events[0] if isinstance(events, list) else events
            return _parse_event_to_btc_market(event)
        except Exception as e:
            logger.warning(f"Failed to fetch BTC market for settlement {slug}: {e}")
            return None


if __name__ == "__main__":
    async def test():
        markets = await fetch_active_btc_markets()
        print(f"Found {len(markets)} markets")
        for m in markets:
            state = "LIVE" if m.is_active else "NEXT" if m.is_upcoming else "?"
            print(f"  [{m.timeframe:>3}] {state} {m.slug}  Up {m.up_price:.2f}/Down {m.down_price:.2f}  "
                  f"{m.window_start:%m-%d %H:%M}->{m.window_end:%m-%d %H:%M}Z  vol ${m.volume:,.0f}")
    asyncio.run(test())
