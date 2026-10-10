"""Signal generator for BTC Up/Down markets (5m, 15m, 1h, 1d)."""
import logging
import math
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field
import asyncio

from backend.config import settings, tf_setting
from backend.data.btc_markets import BtcMarket, fetch_active_btc_markets
from backend.data.crypto import (
    fetch_crypto_price, compute_btc_microstructure,
    fetch_btc_minute_vol_from_hourly, fetch_window_path, fetch_binance_ref,
    fetch_polymarket_open, fetch_chainlink_btc,
)
from backend.models.database import SessionLocal, Signal

logger = logging.getLogger("trading_bot")


@dataclass
class TradingSignal:
    """A trading signal for a BTC 5-min market."""
    market: BtcMarket

    # Core signal data
    model_probability: float = 0.5  # Our estimated probability of UP
    market_probability: float = 0.5  # Market's implied UP probability
    edge: float = 0.0
    direction: str = "up"  # "up" or "down"

    # Confidence and sizing
    confidence: float = 0.5
    kelly_fraction: float = 0.0
    suggested_size: float = 0.0

    # Metadata
    sources: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # BTC price context
    btc_price: float = 0.0
    btc_change_1h: float = 0.0
    btc_change_24h: float = 0.0

    # Price actually paid (Up ask / 1 - Up bid); None -> scheduler falls back to the mid
    entry_price: Optional[float] = None

    # Predict-only mode: market order-book probability of Up and how much of the window is done
    prediction_up: Optional[float] = None
    elapsed_frac: float = 0.0

    @property
    def passes_threshold(self) -> bool:
        """Check if signal passes the minimum edge threshold for its timeframe."""
        tf = getattr(self.market, "timeframe", "5m")
        return abs(self.edge) >= tf_setting("BTC_TF_MIN_EDGE", tf, settings.MIN_EDGE_THRESHOLD)


def calculate_edge(
    model_prob: float,
    market_price: float
) -> tuple[float, str]:
    """
    Calculate edge and determine direction.

    For BTC 5-min markets:
    - "up" is equivalent to "yes" (outcomePrices[0])
    - "down" is equivalent to "no" (outcomePrices[1])

    Returns:
        (edge, direction) where direction is "up" or "down"
    """
    # Edge for UP bet
    up_edge = model_prob - market_price

    # Edge for DOWN bet
    down_edge = (1 - model_prob) - (1 - market_price)

    if up_edge >= down_edge:
        return up_edge, "up"
    else:
        return down_edge, "down"


def calculate_kelly_size(
    edge: float,
    probability: float,
    market_price: float,
    direction: str,
    bankroll: float
) -> float:
    """
    Calculate position size using fractional Kelly criterion.

    Kelly formula: f = (p * b - q) / b
    where:
        f = fraction of bankroll to bet
        p = probability of winning
        q = probability of losing (1 - p)
        b = odds (payout ratio)
    """
    if direction == "up":
        win_prob = probability
        price = market_price
    else:
        win_prob = 1 - probability
        price = 1 - market_price

    if price <= 0 or price >= 1:
        return 0

    odds = (1 - price) / price

    lose_prob = 1 - win_prob
    kelly = (win_prob * odds - lose_prob) / odds

    # Apply fractional Kelly
    kelly *= settings.KELLY_FRACTION

    # Cap at maximum per-trade limit
    max_fraction = 0.05  # 5% max per trade
    kelly = min(kelly, max_fraction)

    kelly = max(kelly, 0)

    size = kelly * bankroll

    # Hard cap from config
    size = min(size, settings.MAX_TRADE_SIZE)

    return size


async def generate_btc_signal(market: BtcMarket) -> Optional[TradingSignal]:
    """
    Generate a trading signal for a BTC 5-min Up/Down market.

    Uses real 1-minute candle data from Binance to compute:
    - RSI (mean reversion), Momentum (trend), VWAP deviation,
      SMA crossover, and market skew as a weighted composite.
    - Convergence filter: requires 3/4 indicators to agree.
    - Entry price filter: only enter when price ≤ MAX_ENTRY_PRICE.
    """
    try:
        micro = await compute_btc_microstructure()
    except Exception as e:
        logger.warning(f"Failed to compute microstructure: {e}")
        return None

    if not micro:
        return None

    # Resolution-rules model for every timeframe; 5m can opt back into the old indicator model.
    if getattr(market, "timeframe", "5m") != "5m" or (settings.BTC_5M_MODEL or "rules").lower() != "micro":
        return await generate_horizon_signal(market, micro)

    market_up_prob = market.up_price

    # Skip resolved markets
    if market_up_prob < 0.02 or market_up_prob > 0.98:
        return None

    # --- Entry price filter: only trade when price ≤ 50c ---
    entry_price = market_up_prob  # will be overridden per-direction below
    # We check after direction is determined

    # --- Individual indicator signals (each returns a bias from -1 to +1) ---

    # 1) RSI: mean reversion — oversold (< 30) = UP, overbought (> 70) = DOWN
    if micro.rsi < 30:
        rsi_signal = 0.5 + (30 - micro.rsi) / 30  # 0.5 to 1.0
    elif micro.rsi > 70:
        rsi_signal = -0.5 - (micro.rsi - 70) / 30  # -0.5 to -1.0
    elif micro.rsi < 45:
        rsi_signal = (45 - micro.rsi) / 30  # slight UP lean
    elif micro.rsi > 55:
        rsi_signal = -(micro.rsi - 55) / 30  # slight DOWN lean
    else:
        rsi_signal = 0.0
    rsi_signal = max(-1.0, min(1.0, rsi_signal))

    # 2) Momentum: weighted blend of 1m, 5m, 15m changes
    #    Positive momentum = UP bias
    mom_blend = micro.momentum_1m * 0.5 + micro.momentum_5m * 0.35 + micro.momentum_15m * 0.15
    # Normalise: ±0.1% is a strong 5-min signal for BTC
    momentum_signal = max(-1.0, min(1.0, mom_blend / 0.10))

    # 3) VWAP deviation: price above VWAP = UP momentum, below = DOWN
    vwap_signal = max(-1.0, min(1.0, micro.vwap_deviation / 0.05))

    # 4) SMA crossover: sma5 > sma15 = bullish
    sma_signal = max(-1.0, min(1.0, micro.sma_crossover / 0.03))

    # 5) Market skew: contrarian — if market says UP strongly, fade it
    market_skew = market_up_prob - 0.50
    skew_signal = max(-1.0, min(1.0, -market_skew * 4))

    # --- Convergence filter: count how many indicators agree on direction ---
    indicator_signs = [
        rsi_signal,
        momentum_signal,
        vwap_signal,
        sma_signal,
    ]
    up_votes = sum(1 for s in indicator_signs if s > 0.05)
    down_votes = sum(1 for s in indicator_signs if s < -0.05)

    # Convergence: require 2/4 indicators to agree — these are noisy 50/50 markets
    has_convergence = up_votes >= 2 or down_votes >= 2

    # --- Weighted composite ---
    w = settings
    composite = (
        rsi_signal * w.WEIGHT_RSI
        + momentum_signal * w.WEIGHT_MOMENTUM
        + vwap_signal * w.WEIGHT_VWAP
        + sma_signal * w.WEIGHT_SMA
        + skew_signal * w.WEIGHT_MARKET_SKEW
    )

    # Convert composite (-1..+1) to probability (0.35..0.65)
    # Wider range lets us express real edge when indicators converge
    model_up_prob = 0.50 + composite * 0.15
    model_up_prob = max(0.35, min(0.65, model_up_prob))

    # Calculate edge and direction
    edge, direction = calculate_edge(model_up_prob, market_up_prob)

    # --- Entry price filter: only buy the cheap side (≤ MAX_ENTRY_PRICE) ---
    if direction == "up":
        entry_price = market_up_prob
    else:
        entry_price = market.down_price

    # Time-remaining filter: only trade windows in the sweet spot
    now = datetime.utcnow()
    # Handle timezone-aware window_end
    window_end = market.window_end
    if window_end.tzinfo is not None:
        window_end = window_end.replace(tzinfo=None)
    time_remaining = (window_end - now).total_seconds()
    time_ok = settings.MIN_TIME_REMAINING <= time_remaining <= settings.MAX_TIME_REMAINING

    passes_filters = has_convergence and entry_price <= settings.MAX_ENTRY_PRICE and time_ok

    # Zero out edge if filters fail (signal still returned for UI visibility)
    if not passes_filters:
        edge = 0.0

    # Confidence: based on convergence strength + volatility
    #   Low volatility = lower confidence (less movement expected)
    vol_factor = min(1.0, micro.volatility / 0.05) if micro.volatility > 0 else 0.5
    convergence_strength = max(up_votes, down_votes) / 4.0
    confidence = min(0.8, 0.3 + convergence_strength * 0.3 + abs(composite) * 0.2) * vol_factor

    # Kelly sizing
    bankroll = settings.INITIAL_BANKROLL
    suggested_size = calculate_kelly_size(
        edge=abs(edge),
        probability=model_up_prob,
        market_price=market_up_prob,
        direction=direction,
        bankroll=bankroll,
    )

    # Build reasoning
    filter_status = "ACTIONABLE" if passes_filters else "FILTERED"
    filter_reasons = []
    if not has_convergence:
        filter_reasons.append(f"convergence {max(up_votes, down_votes)}/4 < 2")
    if not time_ok:
        filter_reasons.append(f"time {time_remaining:.0f}s not in [{settings.MIN_TIME_REMAINING},{settings.MAX_TIME_REMAINING}]")
    if entry_price > settings.MAX_ENTRY_PRICE:
        filter_reasons.append(f"entry {entry_price:.0%} > {settings.MAX_ENTRY_PRICE:.0%}")
    filter_note = f" [{', '.join(filter_reasons)}]" if filter_reasons else ""

    reasoning = (
        f"[{filter_status}]{filter_note} "
        f"BTC ${micro.price:,.0f} | RSI:{micro.rsi:.0f} Mom1m:{micro.momentum_1m:+.3f}% "
        f"Mom5m:{micro.momentum_5m:+.3f}% VWAP:{micro.vwap_deviation:+.3f}% "
        f"SMA:{micro.sma_crossover:+.4f}% Vol:{micro.volatility:.4f}% | "
        f"Composite:{composite:+.3f} -> Model UP:{model_up_prob:.0%} vs Mkt:{market_up_prob:.0%} | "
        f"Edge:{edge:+.1%} -> {direction.upper()} @ {entry_price:.0%} | "
        f"Convergence:{max(up_votes, down_votes)}/4 | "
        f"Window ends: {market.window_end.strftime('%H:%M UTC')}"
    )

    return TradingSignal(
        market=market,
        model_probability=model_up_prob,
        market_probability=market_up_prob,
        edge=edge,
        direction=direction,
        confidence=confidence,
        kelly_fraction=suggested_size / bankroll if bankroll > 0 else 0,
        suggested_size=suggested_size,
        sources=[f"binance_microstructure_{micro.source}"],
        reasoning=reasoning,
        btc_price=micro.price,
        btc_change_1h=micro.momentum_5m * 12,  # rough annualisation for display
        btc_change_24h=micro.momentum_15m * 96,  # rough extrapolation for display
    )


def _indicator_signals(micro) -> dict:
    """Directional bias (-1..+1) from each short-term indicator."""
    if micro.rsi < 30:
        rsi = 0.5 + (30 - micro.rsi) / 30
    elif micro.rsi > 70:
        rsi = -0.5 - (micro.rsi - 70) / 30
    elif micro.rsi < 45:
        rsi = (45 - micro.rsi) / 30
    elif micro.rsi > 55:
        rsi = -(micro.rsi - 55) / 30
    else:
        rsi = 0.0
    clamp = lambda v: max(-1.0, min(1.0, v))
    mom = micro.momentum_1m * 0.5 + micro.momentum_5m * 0.35 + micro.momentum_15m * 0.15
    return {
        "rsi": clamp(rsi),
        "momentum": clamp(mom / 0.10),
        "vwap": clamp(micro.vwap_deviation / 0.05),
        "sma": clamp(micro.sma_crossover / 0.03),
    }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Every series settles on price at the END of the window vs the start (checked against settled
# markets), each on its own feed:
#   5m / 15m: Chainlink BTC/USD (Polymarket's live feed + its own "price to beat")
#   1h:       Binance BTC/USDT 1h candle close vs open
#   1d:       Binance BTC/USDT 1m close at 12:00 ET today vs 12:00 ET yesterday
SETTLEMENT_SOURCE = {"5m": "chainlink", "15m": "chainlink", "1h": "binance", "1d": "binance"}
BASIS_PCT = {"chainlink": 0.002, "binance": 0.003, "coinbase": 0.01}


async def generate_horizon_signal(market: BtcMarket, micro) -> Optional[TradingSignal]:
    """
    Rules model ("rules-v4") for BTC Up/Down windows:
      P(Up) = Phi( ln(S / open) / sqrt(sigma^2 * minutes_left + basis^2) )
    with S and open read from the feed the series settles on (SETTLEMENT_SOURCE); Coinbase is the
    fallback. sigma = max(1m realized vol, hourly-derived per-minute vol). No momentum tilt.
    Edge is measured against what we'd pay (Up ask / 1 - Up bid). Trades only late in the window
    (BTC_TF_MIN_ELAPSED) and on a clean CLOB quote (BTC_MAX_SPREAD).
    """
    from datetime import timezone as _tz
    tf = market.timeframe
    source = SETTLEMENT_SOURCE.get(tf, "chainlink")
    market_up_prob = market.up_price
    if market_up_prob < 0.02 or market_up_prob > 0.98:
        return None
    bid = market.up_bid if market.up_bid is not None else market_up_prob
    ask = market.up_ask if market.up_ask is not None else market_up_prob

    now = datetime.now(_tz.utc)
    T = market.window_seconds / 60
    t_rem = max(0.0, min(T, (market.window_end - max(now, market.window_start)).total_seconds() / 60))
    t_el = T - t_rem
    time_remaining = (market.window_end - now).total_seconds()

    sigma = max(micro.volatility or 0.0, (await fetch_btc_minute_vol_from_hourly()) or 0.0, 0.005)

    open_price, dist_pct, ref_src = None, 0.0, None
    if market.is_active:
        if source == "chainlink":
            open_price = await fetch_polymarket_open(tf, market.window_start, market.window_end)
            spot = await fetch_chainlink_btc()
            ref_src = "chainlink"
            if not (open_price and spot):
                open_price, _ = await fetch_window_path(market.window_start, T)
                spot, ref_src = micro.price, "coinbase"
            if open_price and spot:
                dist_pct = math.log(spot / open_price) * 100
        else:
            open_price, last = await fetch_binance_ref(tf, market.window_start)
            ref_src = "binance"
            if open_price and last:
                dist_pct = math.log(last / open_price) * 100
            else:
                open_price = None
        if not open_price:
            logger.debug(f"No open price for {market.slug}; skipping")
            return None

    sd = math.sqrt(sigma ** 2 * max(t_rem, 0.25) + BASIS_PCT[ref_src or source] ** 2)
    p_base = _norm_cdf(dist_pct / sd)
    model_up_prob = max(0.01, min(0.99, p_base))
    rule = ref_src or source

    # Edge vs the price we'd actually pay
    e_up, e_dn = model_up_prob - ask, bid - model_up_prob
    direction = "up" if e_up >= e_dn else "down"
    entry_price = ask if direction == "up" else 1 - bid
    edge = max(e_up, e_dn, 0.0)

    min_t = tf_setting("BTC_TF_MIN_TIME", tf, settings.MIN_TIME_REMAINING)
    max_t = tf_setting("BTC_TF_MAX_TIME", tf, settings.MAX_TIME_REMAINING)
    max_entry = tf_setting("BTC_TF_MAX_ENTRY", tf, settings.MAX_ENTRY_PRICE)
    min_el = tf_setting("BTC_TF_MIN_ELAPSED", tf, 0.5)
    reasons = []
    if not market.is_active:
        reasons.append("not started")
    elif t_el / T < min_el:
        reasons.append(f"too early ({t_el / T:.0%} < {min_el:.0%} elapsed)")
    if not (min_t <= time_remaining <= max_t):
        reasons.append(f"time {time_remaining:.0f}s not in [{min_t:.0f},{max_t:.0f}]")
    if entry_price > max_entry:
        reasons.append(f"entry {entry_price:.0%} > {max_entry:.0%}")
    if market.quote_source != "clob" or ask - bid > settings.BTC_MAX_SPREAD:
        reasons.append("no clean CLOB quote")
    passes_filters = not reasons
    if not passes_filters:
        edge = 0.0

    confidence = min(0.85, 0.3 + abs(model_up_prob - 0.5) * 1.1)
    bankroll = settings.INITIAL_BANKROLL
    suggested_size = calculate_kelly_size(
        edge=abs(edge), probability=model_up_prob,
        market_price=entry_price if direction == "up" else 1 - entry_price,
        direction=direction, bankroll=bankroll,
    )

    min_edge = tf_setting("BTC_TF_MIN_EDGE", tf, settings.MIN_EDGE_THRESHOLD)
    status = "ACTIONABLE" if passes_filters and edge >= min_edge else ("FILTERED" if reasons else "NO EDGE")
    beat = (f"Open ${open_price:,.0f} -> now {dist_pct:+.3f}% ({rule}) | "
            if open_price else "Not started (no price to beat yet) | ")
    reasoning = (
        f"[{status}]{' [' + ', '.join(reasons) + ']' if reasons else ''} BTC {tf} ({rule}) | "
        f"BTC ${micro.price:,.0f} | {beat}"
        f"{t_rem:.0f}m left ({t_el / T:.0%} done), vol {sigma:.3f}%/min -> sd {sd:.3f}% | "
        f"Model UP:{model_up_prob:.0%} vs book {bid:.2f}/{ask:.2f} | "
        f"Edge:{edge:+.1%} -> {direction.upper()} @ {entry_price:.0%} | "
        f"Window ends: {market.window_end.strftime('%m-%d %H:%M UTC')}"
    )

    if not settings.BTC_PAPER_TRADING:
        # Predict-only: the prediction IS the market's order-book probability (it beat the price
        # model in a backtest on settled windows); the price model stays in the reasoning text.
        fav = "up" if market_up_prob >= 0.5 else "down"
        agree = (model_up_prob >= 0.5) == (market_up_prob >= 0.5)
        return TradingSignal(
            market=market,
            model_probability=market_up_prob,
            market_probability=market_up_prob,
            edge=0.0,
            direction=fav,
            confidence=max(market_up_prob, 1 - market_up_prob),
            kelly_fraction=0.0,
            suggested_size=0.0,
            sources=["prediction_market_mid", f"check_rules_v4_{rule}"],
            reasoning=(f"[PREDICTION] BTC {tf}: {fav.upper()} {max(market_up_prob, 1 - market_up_prob):.0%} "
                       f"(order book {bid:.2f}/{ask:.2f}) | "
                       + beat + f"{t_rem:.0f}m left ({t_el / T:.0%} done) | "
                       f"price model {model_up_prob:.0%} Up ({'agrees' if agree else 'disagrees'}) | "
                       f"Window ends: {market.window_end.strftime('%m-%d %H:%M UTC')}"),
            btc_price=micro.price,
            btc_change_1h=micro.momentum_5m * 12,
            btc_change_24h=micro.momentum_15m * 96,
            prediction_up=market_up_prob,
            elapsed_frac=t_el / T,
        )

    return TradingSignal(
        market=market,
        model_probability=model_up_prob,
        market_probability=market_up_prob,
        edge=edge,
        direction=direction,
        confidence=confidence,
        kelly_fraction=suggested_size / bankroll if bankroll > 0 else 0,
        suggested_size=suggested_size,
        sources=[f"rules_v4_{rule}"],
        reasoning=reasoning,
        btc_price=micro.price,
        btc_change_1h=micro.momentum_5m * 12,
        btc_change_24h=micro.momentum_15m * 96,
        entry_price=entry_price,
    )


async def scan_for_signals() -> List[TradingSignal]:
    """
    Scan BTC Up/Down markets (all enabled timeframes) and generate signals.
    """
    signals = []

    logger.info("=" * 50)
    logger.info("BTC SCAN: Fetching markets from Polymarket...")

    try:
        markets = await fetch_active_btc_markets()
    except Exception as e:
        logger.error(f"Failed to fetch BTC markets: {e}")
        markets = []

    logger.info(f"Found {len(markets)} active BTC markets")

    for market in markets:
        try:
            signal = await generate_btc_signal(market)
            if signal:
                signals.append(signal)
        except Exception as e:
            logger.debug(f"Signal generation failed for {market.slug}: {e}")

        # Small delay to avoid CoinGecko rate limits
        # (only needed if we're making multiple calls - reuse first result)
        await asyncio.sleep(0.1)

    # Sort by absolute edge (best opportunities first)
    signals.sort(key=lambda s: abs(s.edge), reverse=True)

    actionable = [s for s in signals if s.passes_threshold]
    logger.info(f"=" * 50)
    logger.info(f"SCAN COMPLETE: {len(signals)} signals, {len(actionable)} actionable")

    for signal in actionable[:5]:
        logger.info(f"  {signal.market.slug}")
        logger.info(f"    Edge: {signal.edge:+.1%} -> {signal.direction.upper()} @ ${signal.suggested_size:.2f}")

    # Persist signals with non-zero edge to DB for calibration tracking
    _persist_signals(signals)

    return signals


def _persist_predictions(signals: list):
    """Predict-only mode: store ONE prediction per window, taken once the window is late enough
    (BTC_TF_MIN_ELAPSED) and quoted on the CLOB, so it can be scored after settlement."""
    late = []
    for s in signals:
        tf = getattr(s.market, "timeframe", "5m")
        if (s.prediction_up is not None and s.market.is_active and s.market.quote_source == "clob"
                and s.elapsed_frac >= tf_setting("BTC_TF_MIN_ELAPSED", tf, 0.5)):
            late.append(s)
    if not late:
        return
    db = SessionLocal()
    try:
        for s in late:
            if db.query(Signal).filter(Signal.market_ticker == s.market.market_id).first():
                continue
            db.add(Signal(
                market_ticker=s.market.market_id,
                platform="polymarket",
                market_type="btc",
                timestamp=s.timestamp,
                direction=s.direction,
                model_probability=s.prediction_up,
                market_price=s.prediction_up,
                edge=0.0,
                confidence=s.confidence,
                kelly_fraction=0.0,
                suggested_size=0.0,
                sources=s.sources + [f"slug:{s.market.slug}"],
                reasoning=s.reasoning,
                executed=False,
            ))
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist predictions: {e}")
        db.rollback()
    finally:
        db.close()


def _persist_signals(signals: list):
    """Save signals with non-zero edge to DB, deduplicating on (market_ticker, timestamp)."""
    if not settings.BTC_PAPER_TRADING:
        _persist_predictions(signals)
        return
    to_save = [s for s in signals if abs(s.edge) > 0]
    if not to_save:
        return

    db = SessionLocal()
    try:
        for signal in to_save:
            # Dedup: skip if we already logged this signal for this market window
            existing = db.query(Signal).filter(
                Signal.market_ticker == signal.market.market_id,
                Signal.timestamp >= signal.timestamp.replace(second=0, microsecond=0),
            ).first()
            if existing:
                continue

            db_signal = Signal(
                market_ticker=signal.market.market_id,
                platform="polymarket",
                timestamp=signal.timestamp,
                direction=signal.direction,
                model_probability=signal.model_probability,
                market_price=signal.market_probability,
                edge=signal.edge,
                confidence=signal.confidence,
                kelly_fraction=signal.kelly_fraction,
                suggested_size=signal.suggested_size,
                sources=signal.sources,
                reasoning=signal.reasoning,
                executed=False,
            )
            db.add(db_signal)

        db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist signals: {e}")
        db.rollback()
    finally:
        db.close()


async def get_actionable_signals() -> List[TradingSignal]:
    """Get only signals that pass the edge threshold."""
    all_signals = await scan_for_signals()
    return [s for s in all_signals if s.passes_threshold]


if __name__ == "__main__":
    async def test():
        print("Scanning BTC 5-min markets for signals...")
        signals = await scan_for_signals()
        print(f"\nFound {len(signals)} total signals")

        actionable = [s for s in signals if s.passes_threshold]
        print(f"Actionable signals (>{settings.MIN_EDGE_THRESHOLD:.0%} edge): {len(actionable)}")

        for signal in actionable[:5]:
            print(f"\n{signal.market.slug}")
            print(f"  BTC: ${signal.btc_price:,.0f} ({signal.btc_change_24h:+.2f}%)")
            print(f"  Model UP: {signal.model_probability:.1%} vs Market UP: {signal.market_probability:.1%}")
            print(f"  Edge: {signal.edge:+.1%} -> {signal.direction.upper()}")
            print(f"  Size: ${signal.suggested_size:.2f}")

    asyncio.run(test())
