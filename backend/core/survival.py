"""Survival mode: paper-trade candidate strategies on Polymarket US BTC Up/Down windows.

Runs on Polymarket US (the venue a US user can legally trade): its order books, its per-market
taker fee (feeCoefficient x p x (1-p) per share, on buys and on early sells) and its settlement
(CF Benchmarks BRTI, read from the same feed Polymarket US publishes, so the bot's price and the
price to beat match the app exactly).

Every trade buys the chosen side at its ask. Each strategy has its own virtual bankroll and rules:
  * stake = 1% of the strategy's current bankroll
  * daily stop: no new trades once today's realized P&L <= -3% of the bankroll at day start
  * drawdown pause: the strategy stops for good once its bankroll falls 10% below its peak
    (strategies listed in SURVIVAL_NO_PAUSE skip this so they can reach the 200-trade proof)
Strategies listed in SURVIVAL_RETIRED take no new trades but stay on the scoreboard as RETIRED.
A strategy is PROVEN only after >= 200 settled trades with the 95% lower bound of mean P&L per
trade above zero. Real orders are only ever placed by backend/core/us_live.py, and only for
PROVEN strategies when the owner has switched live trading on.
"""
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.config import settings
from backend.models.database import SessionLocal, SurvivalTrade

logger = logging.getLogger("trading_bot")

VENUE = "polymarket_us"
START_BANKROLL = 1000.0
STAKE_FRAC = 0.01
DAILY_STOP_FRAC = 0.03
DRAWDOWN_PAUSE_FRAC = 0.10
MIN_PROOF_TRADES = 200
DEFAULT_FEE_COEF = 0.0695
MAX_SPREAD = 0.05
BASIS_PCT = 0.003  # same index as settlement; covers a few seconds of feed lag

STRATEGIES: Dict[str, str] = {
    "favorite_late": "Buy the favorite (mid >= 85%) in the last 20% of the window, ask <= 97c",
    "favorite_mid": "Buy the favorite (mid 65-85%) after 60% of the window has passed",
    "model_gap": "Buy the side the BRTI price model likes when it beats the ask by >= 8 pts (after 60%)",
    "underdog_late": "Control: buy the underdog (mid <= 15%) in the last 20% of the window",
    "dip_tp80": "Buy low (ask 20-40c, 15-60% into the window), sell when the bid reaches 80c, else hold",
    "dip_tp80_sl": "Same as dip_tp80 but also cut the loss if the bid falls to 10c",
    "late_gap": "Last 20% of the window: buy the side the BRTI model prices >= 10 pts above its ask (ask <= 70c)",
    "late_gap_big": "Last 20% of the window: buy the side the BRTI model prices >= 20 pts above its ask (ask <= 70c)",
}

# Exit rules for strategies that sell before settlement: (take-profit bid, stop-loss bid or None)
EXITS = {"dip_tp80": (0.80, None), "dip_tp80_sl": (0.80, 0.10)}


def _csv(value: Optional[str]) -> set:
    return {s.strip() for s in (value or "").split(",") if s.strip()}


def retired() -> set:
    """Strategies that take no new trades (history kept, shown as RETIRED)."""
    return _csv(getattr(settings, "SURVIVAL_RETIRED", ""))


def pause_exempt() -> set:
    """Strategies that keep trading past the 10% drawdown pause (the daily stop still applies)."""
    return _csv(getattr(settings, "SURVIVAL_NO_PAUSE", ""))


def timeframe_limits() -> Dict[str, set]:
    """SURVIVAL_TIMEFRAMES "model_gap:1h,favorite_late:15m|1h" -> {strategy: {window lengths it may trade}}."""
    out: Dict[str, set] = {}
    for part in _csv(getattr(settings, "SURVIVAL_TIMEFRAMES", "")):
        name, _, tfs = part.partition(":")
        allowed = {t.strip() for t in tfs.split("|") if t.strip()}
        if name.strip() and allowed:
            out[name.strip()] = allowed
    return out


def allowed_on(strategy: str, timeframe: str, limits: Optional[Dict[str, set]] = None) -> bool:
    lim = timeframe_limits() if limits is None else limits
    return strategy not in lim or timeframe in lim[strategy]


def taker_fee(shares: float, price: float, coef: float = DEFAULT_FEE_COEF) -> float:
    return shares * coef * price * (1 - price)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass
class Quote:
    """One live window, normalized: book, BRTI model and timing."""
    market_id: str
    slug: str
    timeframe: str
    window_end: datetime
    is_active: bool
    elapsed_frac: float
    up_bid: Optional[float]
    up_ask: Optional[float]
    fee_coef: float
    tick: float
    price_to_beat: Optional[float] = None
    brti: Optional[float] = None
    model_up: Optional[float] = None

    @property
    def clean(self) -> bool:
        b, a = self.up_bid, self.up_ask
        return b is not None and a is not None and 0 < b < a < 1 and a - b <= MAX_SPREAD

    @property
    def mid(self) -> Optional[float]:
        return (self.up_bid + self.up_ask) / 2 if self.clean else None

    def side(self, side: str):
        """(ask, bid, mid, model) for buying/selling that side."""
        b, a, m, mod = self.up_bid, self.up_ask, self.mid, self.model_up
        if side == "up":
            return a, b, m, mod
        return 1 - b, 1 - a, 1 - m, (None if mod is None else 1 - mod)


def model_up_prob(brti: float, beat: float, minutes_left: float, vol_pct_min: float) -> float:
    """P(Up) = Phi( ln(S/open) / sqrt(sigma^2 * t_left + basis^2) ), all in percent."""
    dist = math.log(brti / beat) * 100
    sd = math.sqrt(max(vol_pct_min, 0.005) ** 2 * max(minutes_left, 0.05) + BASIS_PCT ** 2)
    return max(0.01, min(0.99, _norm_cdf(dist / sd)))


async def fetch_us_quotes() -> List[Quote]:
    from backend.data.polymarket_us import fetch_us_btc_markets, fetch_brti, fetch_brti_minute_vol
    markets = await fetch_us_btc_markets()
    brti, _ = await fetch_brti(30)
    vol = await fetch_brti_minute_vol(60) or 0.03
    out = []
    for m in markets:
        q = Quote(market_id=m.slug, slug=m.slug, timeframe=m.timeframe,
                  window_end=m.window_end.replace(tzinfo=None), is_active=m.is_active,
                  elapsed_frac=m.elapsed_frac, up_bid=m.up_bid, up_ask=m.up_ask,
                  fee_coef=m.fee_coef, tick=m.tick, price_to_beat=m.price_to_beat, brti=brti)
        if m.is_active and brti and m.price_to_beat:
            q.model_up = model_up_prob(brti, m.price_to_beat, m.minutes_left, vol)
        out.append(q)
    return out


def choose(strategy: str, q: Quote) -> Optional[str]:
    """Return the side ("up"/"down") the strategy buys on this quote, or None."""
    if not q.is_active or not q.clean:
        return None
    el = q.elapsed_frac
    fav = "up" if q.mid >= 0.5 else "down"
    dog = "down" if fav == "up" else "up"
    f_ask, _, f_mid, _ = q.side(fav)
    d_ask, _, d_mid, _ = q.side(dog)
    if strategy == "favorite_late":
        if el >= 0.8 and f_mid >= 0.85 and f_ask <= 0.97:
            return fav
    elif strategy == "favorite_mid":
        if el >= 0.6 and 0.65 <= f_mid < 0.85:
            return fav
    elif strategy == "model_gap":
        if el >= 0.6 and q.model_up is not None:
            for s in ("up", "down"):
                ask, _, _, mod = q.side(s)
                if mod - ask >= 0.08 and ask <= 0.95:
                    return s
    elif strategy in ("late_gap", "late_gap_big"):
        need = 0.10 if strategy == "late_gap" else 0.20
        if el >= 0.8 and q.model_up is not None:
            best = None
            for s in ("up", "down"):
                ask, _, _, mod = q.side(s)
                if ask <= 0.70 and mod - ask >= need and (best is None or mod - ask > best[1]):
                    best = (s, mod - ask)
            if best:
                return best[0]
    elif strategy in ("dip_tp80", "dip_tp80_sl"):
        if 0.15 <= el <= 0.6:
            for s in ("up", "down"):
                if 0.20 <= q.side(s)[0] <= 0.40:
                    return s
    elif strategy == "underdog_late":
        if el >= 0.8 and d_mid <= 0.15 and d_ask >= 0.03:
            return dog
    return None


def _bankroll_state(db, strategy: str) -> dict:
    """Realized bankroll, peak, today's P&L and day-start bankroll from settled trades."""
    rows = (db.query(SurvivalTrade.pnl, SurvivalTrade.settled_at)
            .filter(SurvivalTrade.strategy == strategy, SurvivalTrade.settled == True,  # noqa: E712
                    SurvivalTrade.venue == VENUE)
            .order_by(SurvivalTrade.settled_at.asc()).all())
    bank, peak = START_BANKROLL, START_BANKROLL
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    day_start, today_pnl = None, 0.0
    for pnl, at in rows:
        if at and at >= today and day_start is None:
            day_start = bank
        if at and at >= today:
            today_pnl += pnl or 0.0
        bank += pnl or 0.0
        peak = max(peak, bank)
    if day_start is None:
        day_start = bank
    return {"bankroll": bank, "peak": peak, "today_pnl": today_pnl, "day_start": day_start}


def _blocked(st: dict, strategy: Optional[str] = None) -> Optional[str]:
    if strategy in retired():
        return "retired"
    if st["bankroll"] <= st["peak"] * (1 - DRAWDOWN_PAUSE_FRAC) and strategy not in pause_exempt():
        return "paused: 10% drawdown"
    if st["today_pnl"] <= -DAILY_STOP_FRAC * st["day_start"]:
        return "daily stop: -3% today"
    return None


def _apply_exits(db, quotes: List[Quote]) -> List[SurvivalTrade]:
    """Sell open take-profit/stop-loss positions at the current bid (taker fee on the sale too)."""
    by_market = {q.market_id: q for q in quotes}
    rows = (db.query(SurvivalTrade)
            .filter(SurvivalTrade.settled == False,  # noqa: E712
                    SurvivalTrade.venue == VENUE,
                    SurvivalTrade.strategy.in_(list(EXITS)),
                    SurvivalTrade.market_id.in_(list(by_market))).all())
    closed = []
    for t in rows:
        q = by_market[t.market_id]
        if not q.clean:
            continue
        bid = q.side(t.direction)[1]
        tp, sl = EXITS[t.strategy]
        reason = "take_profit" if bid >= tp else ("stop_loss" if sl is not None and bid <= sl else None)
        if not reason:
            continue
        sell_fee = taker_fee(t.shares, bid, t.fee_coef or DEFAULT_FEE_COEF)
        t.pnl = t.shares * bid - sell_fee - t.stake
        t.fee = (t.fee or 0.0) + sell_fee
        t.exit_price, t.exit_reason = bid, reason
        t.result = "win" if t.pnl > 0 else "loss"
        t.settled, t.settled_at = True, datetime.utcnow()
        closed.append(t)
    return closed


def record_survival_trades(quotes: List[Quote]) -> dict:
    """Apply exits, then open at most one paper trade per (strategy, market)."""
    db = SessionLocal()
    opened, exited = [], []
    try:
        exited = _apply_exits(db, quotes)
        limits = timeframe_limits()
        for strat in STRATEGIES:
            st = _bankroll_state(db, strat)
            if _blocked(st, strat):
                continue
            for q in quotes:
                if not allowed_on(strat, q.timeframe, limits):
                    continue
                side = choose(strat, q)
                if not side:
                    continue
                if db.query(SurvivalTrade.id).filter(SurvivalTrade.strategy == strat,
                                                    SurvivalTrade.venue == VENUE,
                                                    SurvivalTrade.market_id == q.market_id).first():
                    continue
                ask, _, mid, mod = q.side(side)
                stake = round(max(1.0, st["bankroll"] * STAKE_FRAC), 2)
                shares = stake / (ask + q.fee_coef * ask * (1 - ask))
                t = SurvivalTrade(
                    strategy=strat, venue=VENUE, market_id=q.market_id, slug=q.slug,
                    timeframe=q.timeframe, window_end=q.window_end, direction=side,
                    entry_price=ask, mid_at_entry=mid, model_at_entry=mod,
                    elapsed_frac=q.elapsed_frac, shares=shares, stake=stake,
                    fee=taker_fee(shares, ask, q.fee_coef), fee_coef=q.fee_coef,
                    price_to_beat=q.price_to_beat, index_at_entry=q.brti,
                )
                db.add(t)
                opened.append(t)
        db.commit()
        summary = {
            "opened": [(t.strategy, t.slug, t.direction, t.entry_price) for t in opened],
            "exited": [(t.strategy, t.slug, t.direction, t.exit_reason, t.exit_price) for t in exited],
        }
    except Exception as e:
        logger.warning(f"Survival: failed to record trades: {e}")
        db.rollback()
        summary = {"opened": [], "exited": [], "error": str(e)}
    finally:
        db.close()
    return summary


async def run_survival_cycle() -> dict:
    """One survival pass on Polymarket US: quotes -> paper exits/entries -> live mirror (if on)."""
    quotes = await fetch_us_quotes()
    summary = record_survival_trades(quotes)
    summary["windows"] = len(quotes)
    try:
        from backend.core.us_live import mirror_live
        summary["live"] = await mirror_live(summary, quotes)
    except Exception as e:
        logger.warning(f"Live mirror failed: {e}")
        summary["live"] = {"error": str(e)}
    return summary


async def settle_survival_trades(db, limit: int = 80) -> int:
    """Settle open survival trades whose market has resolved."""
    from backend.data.polymarket_us import fetch_us_resolution
    from backend.core.settlement import fetch_polymarket_resolution
    now = datetime.utcnow()
    open_rows = (db.query(SurvivalTrade)
                 .filter(SurvivalTrade.settled == False)  # noqa: E712
                 .order_by(SurvivalTrade.opened_at.asc()).limit(limit).all())
    cache: Dict[str, Optional[float]] = {}
    done = 0
    for t in open_rows:
        if t.window_end and t.window_end > now:
            continue
        key = f"{t.venue}:{t.market_id}"
        if key not in cache:
            try:
                if t.venue == VENUE:
                    ok, val = await fetch_us_resolution(t.slug or t.market_id)
                else:
                    ok, val = await fetch_polymarket_resolution(t.market_id, t.slug)
            except Exception as e:
                logger.debug(f"Survival resolution failed for {t.market_id}: {e}")
                ok, val = False, None
            cache[key] = val if ok else None
        val = cache[key]
        if val is None:
            continue
        t.outcome = "up" if val == 1.0 else "down"
        won = t.outcome == t.direction
        t.result = "win" if won else "loss"
        t.pnl = (t.shares - t.stake) if won else -t.stake
        t.settled = True
        t.settled_at = now
        t.exit_reason = "settled"
        done += 1
    if done:
        try:
            db.commit()
        except Exception as e:
            logger.error(f"Survival: failed to commit settlements: {e}")
            db.rollback()
            return 0
    return done


def strategy_stats(db, strat: str) -> dict:
    rows = (db.query(SurvivalTrade)
            .filter(SurvivalTrade.strategy == strat, SurvivalTrade.venue == VENUE).all())
    settled = [r for r in rows if r.settled]
    pnls = [r.pnl or 0.0 for r in settled]
    n = len(pnls)
    wins = sum(1 for r in settled if r.result == "win")
    mean = sum(pnls) / n if n else 0.0
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1)) if n > 1 else 0.0
    lower95 = mean - 1.96 * sd / math.sqrt(n) if n > 1 else None
    upper95 = mean + 1.96 * sd / math.sqrt(n) if n > 1 else None
    staked = sum(r.stake for r in settled)
    st = _bankroll_state(db, strat)
    blocked = _blocked(st, strat)
    if blocked == "retired":
        status = "RETIRED"
    elif blocked and blocked.startswith("paused"):
        status = "PAUSED"
    elif n >= MIN_PROOF_TRADES and lower95 is not None and lower95 > 0:
        status = "PROVEN"
    elif n >= 30 and upper95 is not None and upper95 < 0:
        status = "FAILING"
    else:
        status = "TESTING"
    by_tf: Dict[str, dict] = {}
    for r in settled:
        d = by_tf.setdefault(r.timeframe or "?", {"n": 0, "wins": 0, "pnl": 0.0})
        d["n"] += 1
        d["wins"] += r.result == "win"
        d["pnl"] = round(d["pnl"] + (r.pnl or 0.0), 2)
    return {
        "strategy": strat, "description": STRATEGIES[strat], "status": status, "blocked": blocked,
        "trades": n, "open": len(rows) - n, "wins": wins,
        "win_rate": wins / n if n else None,
        "avg_entry": (sum(r.entry_price for r in settled) / n) if n else None,
        "pnl": round(sum(pnls), 2), "fees": round(sum(r.fee or 0.0 for r in settled), 2),
        "roi": (sum(pnls) / staked) if staked else None,
        "mean_pnl": mean, "lower95": lower95, "upper95": upper95,
        "bankroll": round(st["bankroll"], 2), "peak": round(st["peak"], 2),
        "drawdown": (1 - st["bankroll"] / st["peak"]) if st["peak"] else 0.0,
        "today_pnl": round(st["today_pnl"], 2), "by_timeframe": by_tf,
        "needs": max(0, MIN_PROOF_TRADES - n),
        "pause_exempt": strat in pause_exempt(),
        "exits": {k: sum(1 for r in settled if r.exit_reason == k)
                  for k in ("take_profit", "stop_loss", "settled")},
    }


def scoreboard(db) -> dict:
    """Per-strategy survival stats and status."""
    out: List[dict] = [strategy_stats(db, s) for s in STRATEGIES]
    recent = (db.query(SurvivalTrade).filter(SurvivalTrade.venue == VENUE)
              .order_by(SurvivalTrade.opened_at.desc()).limit(25).all())
    try:
        from backend.core.us_live import live_status
        live = live_status(db)
    except Exception as e:
        live = {"enabled": False, "error": str(e)}
    return {
        "venue": VENUE,
        "rules": {
            "start_bankroll": START_BANKROLL, "stake_frac": STAKE_FRAC,
            "daily_stop_frac": DAILY_STOP_FRAC, "drawdown_pause_frac": DRAWDOWN_PAUSE_FRAC,
            "min_proof_trades": MIN_PROOF_TRADES,
            "pause_exempt": sorted(pause_exempt() & set(STRATEGIES)),
            "retired": sorted(retired() & set(STRATEGIES)),
            "timeframes": {k: sorted(v) for k, v in timeframe_limits().items() if k in STRATEGIES},
            "fee": "shares x feeCoefficient (0.0695) x p x (1-p), on buys and early sells",
            "entry": "ask of the chosen side on the Polymarket US book",
            "settlement": "CF Benchmarks BRTI (same feed as Polymarket US)",
            "real_money": bool(live.get("enabled")),
        },
        "strategies": out,
        "live": live,
        "recent": [{
            "strategy": r.strategy, "slug": r.slug, "timeframe": r.timeframe,
            "direction": r.direction, "entry_price": r.entry_price, "stake": r.stake,
            "fee": round(r.fee or 0.0, 4), "result": r.result,
            "exit_price": r.exit_price, "exit_reason": r.exit_reason,
            "price_to_beat": r.price_to_beat, "index_at_entry": r.index_at_entry,
            "pnl": None if r.pnl is None else round(r.pnl, 2),
            "opened_at": r.opened_at.isoformat() + "Z" if r.opened_at else None,
        } for r in recent],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
