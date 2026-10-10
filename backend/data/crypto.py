"""Crypto price data fetcher using CoinGecko + Binance APIs."""
import httpx
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binance 1-min kline fetcher + technical indicators for BTC 5-min trading
# ---------------------------------------------------------------------------

BINANCE_API = "https://api.binance.com/api/v3"
BYBIT_API = "https://api.bybit.com/v5/market"
COINBASE_API = "https://api.exchange.coinbase.com"
KRAKEN_API = "https://api.kraken.com/0/public"

# 30-second cache to avoid hammering Binance during a single scan cycle
_kline_cache: Dict[str, Any] = {"data": None, "ts": 0.0}
_CACHE_TTL = 30.0


@dataclass
class BtcMicrostructure:
    """Real-time BTC technical indicators computed from 1-min candles."""
    # RSI (14-period Wilder smoothing)
    rsi: float = 50.0
    # Momentum: % change over various lookbacks
    momentum_1m: float = 0.0
    momentum_5m: float = 0.0
    momentum_15m: float = 0.0
    # VWAP deviation (positive = price above VWAP)
    vwap: float = 0.0
    vwap_deviation: float = 0.0
    # SMA crossover: sma5 - sma15 as fraction of price
    sma_crossover: float = 0.0
    # Volatility: stdev of 1-min returns
    volatility: float = 0.0
    # Current price
    price: float = 0.0
    # Source exchange
    source: str = "binance"


async def fetch_binance_klines(limit: int = 60) -> Optional[List[list]]:
    """
    Fetch recent 1-minute BTCUSDT candles from Binance.
    Falls back to Bybit if Binance fails (US geo-blocking).

    Returns list of [open_time, open, high, low, close, volume, ...] or None.
    """
    now = time.time()
    if _kline_cache["data"] is not None and (now - _kline_cache["ts"]) < _CACHE_TTL:
        return _kline_cache["data"]

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Try Coinbase first (US-accessible, reliable)
        try:
            import datetime as _dt
            end = _dt.datetime.now(_dt.timezone.utc)
            start = end - _dt.timedelta(minutes=limit)
            resp = await client.get(
                f"{COINBASE_API}/products/BTC-USD/candles",
                params={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "granularity": 60,
                },
            )
            resp.raise_for_status()
            rows = resp.json()
            # Coinbase returns [time, low, high, open, close, volume] newest-first
            rows = list(reversed(rows))
            candles = [
                [int(r[0]) * 1000, str(r[3]), str(r[2]), str(r[1]), str(r[4]), str(r[5])]
                for r in rows
            ]
            _kline_cache["data"] = candles
            _kline_cache["ts"] = now
            _kline_cache["_source"] = "coinbase"
            return candles
        except Exception as e:
            logger.warning(f"Coinbase kline fetch failed, trying Kraken: {e}")

        # Fallback 1: Kraken (US-accessible, free)
        try:
            resp = await client.get(
                f"{KRAKEN_API}/OHLC",
                params={"pair": "XBTUSD", "interval": 1},
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", {})
            ohlc_key = [k for k in result if k != "last"]
            if ohlc_key:
                rows = result[ohlc_key[0]]
                rows = rows[-limit:]
                candles = [
                    [int(r[0]) * 1000, str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[6])]
                    for r in rows
                ]
                _kline_cache["data"] = candles
                _kline_cache["ts"] = now
                _kline_cache["_source"] = "kraken"
                return candles
        except Exception as e:
            logger.warning(f"Kraken kline fetch failed, trying Binance: {e}")

        # Fallback 2: Binance (geo-blocked in US)
        try:
            resp = await client.get(
                f"{BINANCE_API}/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
            )
            resp.raise_for_status()
            candles = resp.json()
            _kline_cache["data"] = candles
            _kline_cache["ts"] = now
            _kline_cache["_source"] = "binance"
            return candles
        except Exception as e:
            logger.warning(f"Binance kline fetch failed, trying Bybit: {e}")

        # Fallback 3: Bybit
        try:
            resp = await client.get(
                f"{BYBIT_API}/kline",
                params={
                    "category": "spot",
                    "symbol": "BTCUSDT",
                    "interval": "1",
                    "limit": limit,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("result", {}).get("list", [])
            rows = list(reversed(rows))
            candles = [
                [int(r[0]), r[1], r[2], r[3], r[4], r[5]]
                for r in rows
            ]
            _kline_cache["data"] = candles
            _kline_cache["ts"] = now
            _kline_cache["_source"] = "bybit"
            return candles
        except Exception as e:
            logger.error(f"All kline sources failed: {e}")

        return None


def _compute_rsi(closes: List[float], period: int = 14) -> float:
    """Compute RSI using Wilder smoothing."""
    if len(closes) < period + 1:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    gains = [d if d > 0 else 0.0 for d in deltas[:period]]
    losses = [-d if d < 0 else 0.0 for d in deltas[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for d in deltas[period:]:
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


async def compute_btc_microstructure() -> Optional[BtcMicrostructure]:
    """
    Fetch 60 one-minute candles and compute all technical indicators.
    Returns BtcMicrostructure or None on failure.
    """
    candles = await fetch_binance_klines(limit=60)
    if not candles or len(candles) < 20:
        logger.warning("Not enough candle data for microstructure")
        return None

    closes = [float(c[4]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]

    current_price = closes[-1]

    # RSI (14-period)
    rsi = _compute_rsi(closes, 14)

    # Momentum: % change over lookback periods
    def pct_change(lookback: int) -> float:
        if len(closes) > lookback and closes[-1 - lookback] > 0:
            return (closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback] * 100
        return 0.0

    momentum_1m = pct_change(1)
    momentum_5m = pct_change(5)
    momentum_15m = pct_change(15)

    # VWAP (30-candle window)
    vwap_window = min(30, len(closes))
    typical_prices = [(highs[-i] + lows[-i] + closes[-i]) / 3 for i in range(1, vwap_window + 1)]
    vwap_volumes = [volumes[-i] for i in range(1, vwap_window + 1)]
    total_vol = sum(vwap_volumes)
    if total_vol > 0:
        vwap = sum(tp * v for tp, v in zip(typical_prices, vwap_volumes)) / total_vol
    else:
        vwap = current_price
    vwap_deviation = (current_price - vwap) / vwap * 100 if vwap > 0 else 0.0

    # SMA crossover: 5-period vs 15-period
    sma5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else current_price
    sma15 = sum(closes[-15:]) / 15 if len(closes) >= 15 else current_price
    sma_crossover = (sma5 - sma15) / current_price * 100 if current_price > 0 else 0.0

    # Volatility: stdev of 1-min returns (last 30 candles)
    vol_window = min(30, len(closes) - 1)
    returns = [
        (closes[-i] - closes[-i - 1]) / closes[-i - 1]
        for i in range(1, vol_window + 1)
        if closes[-i - 1] > 0
    ]
    if returns:
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        volatility = math.sqrt(variance) * 100  # as percentage
    else:
        volatility = 0.0

    source = _kline_cache.get("_source", "unknown")

    return BtcMicrostructure(
        rsi=rsi,
        momentum_1m=momentum_1m,
        momentum_5m=momentum_5m,
        momentum_15m=momentum_15m,
        vwap=vwap,
        vwap_deviation=vwap_deviation,
        sma_crossover=sma_crossover,
        volatility=volatility,
        price=current_price,
        source=source,
    )

# ---------------------------------------------------------------------------
# Helpers for longer BTC Up/Down windows (15m / 1h / 1d)
# ---------------------------------------------------------------------------

_open_price_cache: Dict[int, float] = {}
_hourly_vol_cache: Dict[str, Any] = {"value": None, "ts": 0.0}


async def fetch_btc_price_at(ts: datetime) -> Optional[float]:
    """BTC/USD open price of the 1-minute candle starting at `ts` (UTC). Cached."""
    import datetime as _dt
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.timezone.utc)
    key = int(ts.timestamp()) // 60 * 60
    if key in _open_price_cache:
        return _open_price_cache[key]
    start = _dt.datetime.fromtimestamp(key, _dt.timezone.utc)
    price = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                f"{COINBASE_API}/products/BTC-USD/candles",
                params={"start": start.isoformat(), "end": (start + _dt.timedelta(minutes=2)).isoformat(), "granularity": 60},
            )
            resp.raise_for_status()
            rows = [r for r in resp.json() if int(r[0]) == key]
            if rows:
                price = float(rows[0][3])  # [time, low, high, open, close, volume]
        except Exception as e:
            logger.warning(f"Coinbase open-price fetch failed: {e}")
        if price is None:
            try:
                resp = await client.get(f"{KRAKEN_API}/OHLC", params={"pair": "XBTUSD", "interval": 1, "since": key - 60})
                resp.raise_for_status()
                result = resp.json().get("result", {})
                for k, rows in result.items():
                    if k == "last":
                        continue
                    for r in rows:
                        if int(r[0]) == key:
                            price = float(r[1])
            except Exception as e:
                logger.warning(f"Kraken open-price fetch failed: {e}")
    if price:
        if len(_open_price_cache) > 500:
            _open_price_cache.clear()
        _open_price_cache[key] = price
    return price


async def fetch_btc_minute_vol_from_hourly() -> Optional[float]:
    """Per-minute BTC volatility (%), derived from the last 24 hourly candles. Cached 10 min."""
    now = time.time()
    if _hourly_vol_cache["value"] is not None and now - _hourly_vol_cache["ts"] < 600:
        return _hourly_vol_cache["value"]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{COINBASE_API}/products/BTC-USD/candles", params={"granularity": 3600})
            resp.raise_for_status()
            closes = [float(r[4]) for r in reversed(resp.json()[:25])]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
        if len(rets) < 6:
            return None
        mean = sum(rets) / len(rets)
        sd_hour = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
        value = sd_hour / math.sqrt(60) * 100
        _hourly_vol_cache.update(value=value, ts=now)
        return value
    except Exception as e:
        logger.warning(f"Hourly volatility fetch failed: {e}")
        return None


# Public Binance market-data mirror: api.binance.com refuses US IPs (HTTP 451), this one does not.
BINANCE_DATA_API = "https://data-api.binance.vision/api/v3"


async def fetch_window_path(start: datetime, window_min: float) -> Tuple[Optional[float], float]:
    """(open price at `start`, mean log-distance from open over the elapsed part of the window, %).
    Coinbase candles: 1m for windows up to 4h, 5m above. If the first candle isn't published yet
    (window just opened) the previous close stands in for the open."""
    import datetime as _dt
    if start.tzinfo is None:
        start = start.replace(tzinfo=_dt.timezone.utc)
    gran = 60 if window_min <= 240 else 300
    k = int(start.timestamp()) // gran * gran
    now_ts = int(time.time())
    iso = lambda t: _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{COINBASE_API}/products/BTC-USD/candles",
                                    params={"granularity": gran, "start": iso(k - 2 * gran), "end": iso(now_ts)})
            resp.raise_for_status()
            rows = sorted(resp.json(), key=lambda r: r[0])
    except Exception as e:
        logger.warning(f"Window path fetch failed: {e}")
        return None, 0.0
    first = [r for r in rows if int(r[0]) == k]
    prev = [r for r in rows if int(r[0]) < k]
    if first:
        op = float(first[0][3])
    elif prev and now_ts - start.timestamp() < 180:
        op = float(prev[-1][4])
    else:
        return None, 0.0
    el = [math.log(float(r[4]) / op) * 100 for r in rows if k <= int(r[0]) and int(r[0]) + gran <= now_ts]
    return op, (sum(el) / len(el) if el else 0.0)


async def fetch_binance_ref(timeframe: str, start: datetime) -> Tuple[Optional[float], Optional[float]]:
    """(reference price, last price) on Binance BTC/USDT for the Binance-resolved series:
    1h -> open of the 1h candle starting at `start`; 1d -> close of the 1m candle starting at `start`."""
    import datetime as _dt
    if start.tzinfo is None:
        start = start.replace(tzinfo=_dt.timezone.utc)
    ms = int(start.timestamp()) * 1000
    interval, col = ("1h", 1) if timeframe == "1h" else ("1m", 4)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            k = await client.get(f"{BINANCE_DATA_API}/klines",
                                 params={"symbol": "BTCUSDT", "interval": interval, "startTime": ms, "limit": 1})
            k.raise_for_status()
            rows = k.json()
            ref = float(rows[0][col]) if rows and int(rows[0][0]) == ms else None
            t = await client.get(f"{BINANCE_DATA_API}/ticker/price", params={"symbol": "BTCUSDT"})
            t.raise_for_status()
            return ref, float(t.json()["price"])
    except Exception as e:
        logger.warning(f"Binance reference fetch failed: {e}")
        return None, None


POLYMARKET_PRICE_API = "https://polymarket.com/api/crypto/crypto-price"
_pm_open_cache: Dict[str, float] = {}


async def fetch_polymarket_open(timeframe: str, start: datetime, end: datetime) -> Optional[float]:
    """Polymarket's official opening price ("price to beat") for a 5m/15m BTC window. Cached."""
    variant = {"5m": "fiveminute", "15m": "fifteen"}.get(timeframe)
    if not variant:
        return None
    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
    key = f"{variant}:{iso(start)}"
    if key in _pm_open_cache:
        return _pm_open_cache[key]
    try:
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(POLYMARKET_PRICE_API, params={"symbol": "BTC", "eventStartTime": iso(start),
                                                               "variant": variant, "endDate": iso(end)})
            r.raise_for_status()
            op = r.json().get("openPrice")
    except Exception as e:
        logger.warning(f"Polymarket open-price fetch failed: {e}")
        return None
    if op:
        if len(_pm_open_cache) > 500:
            _pm_open_cache.clear()
        _pm_open_cache[key] = float(op)
        return float(op)
    return None


# --- Live Chainlink BTC/USD from Polymarket's real-time data socket (stdlib websocket client) ---
import base64, os, socket, ssl, struct
import json as _json
from urllib.parse import urlparse

RTDS_HOST = "ws-live-data.polymarket.com"


def _ws_frame(payload, opcode=1):
    mask = os.urandom(4)
    n = len(payload)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 65536:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack(">Q", n)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def _ws_read(s):
    data = b""
    while True:
        b1, b2 = _recv_exact(s, 2)
        op, n = b1 & 0x0F, b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", _recv_exact(s, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", _recv_exact(s, 8))[0]
        if b2 & 0x80:
            mk = _recv_exact(s, 4)
            body = bytes(c ^ mk[i % 4] for i, c in enumerate(_recv_exact(s, n)))
        else:
            body = _recv_exact(s, n)
        if op == 9:
            s.sendall(_ws_frame(body, 10)); continue
        if op == 8:
            raise ConnectionError("ws closed")
        data += body
        if b1 & 0x80 and op in (0, 1, 2):
            return data.decode("utf-8", "replace")


def chainlink_btc(timeout=12):
    """Latest Chainlink BTC/USD (the 5m/15m settlement feed) and ~60s of 1-second history."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    ctx = ssl.create_default_context()
    for ca in (os.environ.get("SSL_CERT_FILE"), os.environ.get("REQUESTS_CA_BUNDLE"), "/root/.ccr/ca-bundle.crt"):
        if ca and os.path.exists(ca):
            try:
                ctx.load_verify_locations(ca)
            except Exception:  # noqa: BLE001
                pass
    if proxy:
        p = urlparse(proxy)
        s = socket.create_connection((p.hostname, p.port or 80), timeout=timeout)
        auth = ""
        if p.username:
            auth = "Proxy-Authorization: Basic " + base64.b64encode(f"{p.username}:{p.password or ''}".encode()).decode() + "\r\n"
        s.sendall(f"CONNECT {RTDS_HOST}:443 HTTP/1.1\r\nHost: {RTDS_HOST}:443\r\n{auth}\r\n".encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += s.recv(4096)
        if b" 200" not in resp.split(b"\r\n")[0]:
            raise ConnectionError(resp.split(b"\r\n")[0].decode())
    else:
        s = socket.create_connection((RTDS_HOST, 443), timeout=timeout)
    s = ctx.wrap_socket(s, server_hostname=RTDS_HOST)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET / HTTP/1.1\r\nHost: {RTDS_HOST}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nOrigin: https://polymarket.com\r\n"
               "User-Agent: Mozilla/5.0\r\n\r\n").encode())
    hdr = b""
    while b"\r\n\r\n" not in hdr:
        hdr += s.recv(1)
    if b" 101" not in hdr.split(b"\r\n")[0]:
        raise ConnectionError(hdr.split(b"\r\n")[0].decode())
    sub = {"action": "subscribe", "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*",
                                                      "filters": "{\"symbol\":\"btc/usd\"}"}]}
    s.sendall(_ws_frame(_json.dumps(sub).encode()))
    hist, latest, t_end = [], None, time.time() + timeout
    try:
        while time.time() < t_end and (latest is None or not hist):
            txt = _ws_read(s).strip()
            if not txt.startswith("{"):
                continue
            pl = _json.loads(txt).get("payload") or {}
            if isinstance(pl.get("data"), list):
                hist += [(int(d["timestamp"]), float(d["value"])) for d in pl["data"]]
            elif pl.get("value") is not None:
                latest = float(pl["value"])
                hist.append((int(pl["timestamp"]), latest))
    finally:
        try:
            s.close()
        except Exception:  # noqa: BLE001
            pass
    hist.sort()
    if latest is None and hist:
        latest = hist[-1][1]
    return latest, hist



_chainlink_cache: Dict[str, Any] = {"value": None, "ts": 0.0}


async def fetch_chainlink_btc() -> Optional[float]:
    """Latest Chainlink BTC/USD (5m/15m settlement feed) via Polymarket's socket. Cached 5 s."""
    now = time.time()
    if _chainlink_cache["value"] is not None and now - _chainlink_cache["ts"] < 5:
        return _chainlink_cache["value"]
    try:
        import asyncio as _asyncio
        value, _hist = await _asyncio.to_thread(chainlink_btc, 10)
    except Exception as e:
        logger.warning(f"Chainlink socket failed: {e}")
        return None
    if value:
        _chainlink_cache.update(value=value, ts=now)
    return value


# CoinGecko API (free tier, no key needed)
COINGECKO_API = "https://api.coingecko.com/api/v3"


@dataclass
class CryptoPrice:
    """Current crypto price data."""
    symbol: str  # BTC, ETH, etc.
    name: str
    current_price: float
    price_24h_ago: float
    change_24h: float  # Percentage
    change_7d: float
    market_cap: float
    volume_24h: float
    last_updated: datetime


# Map common symbols to CoinGecko IDs
SYMBOL_TO_ID = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
}


async def fetch_crypto_price(symbol: str) -> Optional[CryptoPrice]:
    """
    Fetch current price data for a cryptocurrency.

    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)

    Returns:
        CryptoPrice or None if not found
    """
    symbol_upper = symbol.upper()
    coin_id = SYMBOL_TO_ID.get(symbol_upper, symbol.lower())

    url = f"{COINGECKO_API}/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            market_data = data.get("market_data", {})
            current_price = market_data.get("current_price", {}).get("usd", 0)
            change_24h = market_data.get("price_change_percentage_24h", 0)
            change_7d = market_data.get("price_change_percentage_7d", 0)

            # Calculate price 24h ago
            price_24h_ago = current_price / (1 + change_24h / 100) if change_24h else current_price

            return CryptoPrice(
                symbol=symbol_upper,
                name=data.get("name", symbol_upper),
                current_price=current_price,
                price_24h_ago=price_24h_ago,
                change_24h=change_24h or 0,
                change_7d=change_7d or 0,
                market_cap=market_data.get("market_cap", {}).get("usd", 0),
                volume_24h=market_data.get("total_volume", {}).get("usd", 0),
                last_updated=datetime.utcnow()
            )

        except httpx.HTTPStatusError as e:
            logger.warning(f"CoinGecko API error for {symbol}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching crypto price for {symbol}: {e}")
            return None


async def fetch_multiple_prices(symbols: List[str]) -> Dict[str, CryptoPrice]:
    """
    Fetch prices for multiple cryptocurrencies efficiently.

    Uses CoinGecko's markets endpoint for batch fetching.
    """
    # Map symbols to CoinGecko IDs
    coin_ids = [SYMBOL_TO_ID.get(s.upper(), s.lower()) for s in symbols]

    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "sparkline": "false",
        "price_change_percentage": "24h,7d"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=15.0)
            response.raise_for_status()
            data = response.json()

            results = {}
            for coin in data:
                symbol = coin.get("symbol", "").upper()
                current_price = coin.get("current_price", 0)
                change_24h = coin.get("price_change_percentage_24h", 0) or 0
                change_7d = coin.get("price_change_percentage_7d_in_currency", 0) or 0

                price_24h_ago = current_price / (1 + change_24h / 100) if change_24h else current_price

                results[symbol] = CryptoPrice(
                    symbol=symbol,
                    name=coin.get("name", symbol),
                    current_price=current_price,
                    price_24h_ago=price_24h_ago,
                    change_24h=change_24h,
                    change_7d=change_7d,
                    market_cap=coin.get("market_cap", 0) or 0,
                    volume_24h=coin.get("total_volume", 0) or 0,
                    last_updated=datetime.utcnow()
                )

            return results

        except Exception as e:
            logger.error(f"Error fetching multiple crypto prices: {e}")
            return {}


def estimate_price_probability(
    current_price: float,
    threshold: float,
    direction: str,
    volatility_24h: float = 0.05
) -> float:
    """
    Estimate probability of price hitting threshold.

    Simple model based on current distance and volatility.
    In production, you'd use options pricing or ML models.

    Args:
        current_price: Current asset price
        threshold: Target price threshold
        direction: "above" or "below"
        volatility_24h: Estimated daily volatility (default 5%)

    Returns:
        Probability estimate 0-1
    """
    if current_price <= 0:
        return 0.5

    # Calculate distance as percentage
    distance = (threshold - current_price) / current_price

    # Simple probability based on normal distribution
    # This is a rough approximation - real models are more complex
    import math

    # Standard deviations away
    std_devs = abs(distance) / volatility_24h

    if direction == "above":
        if current_price >= threshold:
            return 0.95  # Already above
        # Probability of going up by distance
        prob = 0.5 * (1 - math.erf(std_devs / math.sqrt(2)))
    else:  # below
        if current_price <= threshold:
            return 0.95  # Already below
        # Probability of going down by distance
        prob = 0.5 * (1 - math.erf(std_devs / math.sqrt(2)))

    return max(0.05, min(0.95, prob))


# Quick test
if __name__ == "__main__":
    import asyncio

    async def test():
        print("Fetching BTC price...")
        btc = await fetch_crypto_price("BTC")
        if btc:
            print(f"  {btc.name}: ${btc.current_price:,.2f}")
            print(f"  24h change: {btc.change_24h:+.2f}%")
            print(f"  Market cap: ${btc.market_cap:,.0f}")

        print("\nFetching multiple prices...")
        prices = await fetch_multiple_prices(["BTC", "ETH", "SOL"])
        for symbol, price in prices.items():
            print(f"  {symbol}: ${price.current_price:,.2f} ({price.change_24h:+.2f}%)")

    asyncio.run(test())
