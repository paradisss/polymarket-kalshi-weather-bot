"""Survival mode: paper-trade candidate strategies on BTC Up/Down windows with real costs.

Every trade buys the chosen side at its ask and pays Polymarket's crypto taker fee
(shares x 0.07 x p x (1-p)). Each strategy has its own virtual bankroll and risk rules:
  * stake = 1% of the strategy's current bankroll
  * daily stop: no new trades once today's realized P&L <= -3% of the bankroll at day start
  * drawdown pause: the strategy stops for good once its bankroll falls 10% below its peak
A strategy is PROVEN only after >= 200 settled trades with the 95% lower bound of mean P&L
per trade above zero. Nothing here places real orders.
"""
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

from backend.models.database import SessionLocal, SurvivalTrade

logger = logging.getLogger("trading_bot")

START_BANKROLL = 1000.0
STAKE_FRAC = 0.01
DAILY_STOP_FRAC = 0.03
DRAWDOWN_PAUSE_FRAC = 0.10
MIN_PROOF_TRADES = 200
FEE_RATE = 0.07
MAX_SPREAD = 0.05

STRATEGIES: Dict[str, str] = {
    "favorite_late": "Buy the favorite (mid >= 85%) in the last 20% of the window, ask <= 97c",
    "favorite_mid": "Buy the favorite (mid 65-85%) after 60% of the window has passed",
    "model_gap": "Buy the side the price model likes when it beats the ask by >= 8 pts (after 60%)",
    "underdog_late": "Control: buy the underdog (mid <= 15%) in the last 20% of the window",
    "dip_tp80": "Buy low (ask 20-40c, 15-60% into the window), sell when the bid reaches 80c, else hold",
    "dip_tp80_sl": "Same as dip_tp80 but also cut the loss if the bid falls to 10c",
}

# Exit rules for strategies that sell before settlement: (take-profit bid, stop-loss bid or None)
EXITS = {"dip_tp80": (0.80, None), "dip_tp80_sl": (0.80, 0.10)}


def taker_fee(shares: float, price: float) -> float:
    return shares * FEE_RATE * price * (1 - price)


def _side_quotes(sig):
    """Return {side: (ask, mid, model)} for Up and Down, or None without a clean CLOB quote."""
    m = sig.market
    if m.quote_source != "clob" or m.up_bid is None or m.up_ask is None:
        return None
    bid, ask = float(m.up_bid), float(m.up_ask)
    if ask - bid > MAX_SPREAD or not (0 < bid < ask < 1):
        return None
    mid = sig.prediction_up if sig.prediction_up is not None else (bid + ask) / 2
    mod = sig.price_model_up
    return {
        "up": (ask, mid, mod),
        "down": (1 - bid, 1 - mid, None if mod is None else 1 - mod),
    }


def choose(strategy: str, sig) -> Optional[str]:
    """Return the side ("up"/"down") the strategy buys on this signal, or None."""
    if sig.prediction_up is None or not sig.market.is_active:
        return None
    q = _side_quotes(sig)
    if not q:
        return None
    el = sig.elapsed_frac or 0.0
    fav = "up" if q["up"][1] >= 0.5 else "down"
    dog = "down" if fav == "up" else "up"
    if strategy == "favorite_late":
        if el >= 0.8 and q[fav][1] >= 0.85 and q[fav][0] <= 0.97:
            return fav
    elif strategy == "favorite_mid":
        if el >= 0.6 and 0.65 <= q[fav][1] < 0.85:
            return fav
    elif strategy == "model_gap":
        if el >= 0.6 and q["up"][2] is not None:
            for side in ("up", "down"):
                ask, _, mod = q[side]
                if mod - ask >= 0.08 and ask <= 0.95:
                    return side
    elif strategy in ("dip_tp80", "dip_tp80_sl"):
        if 0.15 <= el <= 0.6:
            for side in ("up", "down"):
                if 0.20 <= q[side][0] <= 0.40:
                    return side
    elif strategy == "underdog_late":
        if el >= 0.8 and q[dog][1] <= 0.15 and q[dog][0] >= 0.03:
            return dog
    return None


def _bankroll_state(db, strategy: str) -> dict:
    """Realized bankroll, peak, today's P&L and day-start bankroll from settled trades."""
    rows = (db.query(SurvivalTrade.pnl, SurvivalTrade.settled_at)
            .filter(SurvivalTrade.strategy == strategy, SurvivalTrade.settled == True)  # noqa: E712
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


def _blocked(st: dict) -> Optional[str]:
    if st["bankroll"] <= st["peak"] * (1 - DRAWDOWN_PAUSE_FRAC):
        return "paused: 10% drawdown"
    if st["today_pnl"] <= -DAILY_STOP_FRAC * st["day_start"]:
        return "daily stop: -3% today"
    return None


def _side_bid(sig, side: str) -> Optional[float]:
    m = sig.market
    if m.quote_source != "clob" or m.up_bid is None or m.up_ask is None:
        return None
    if float(m.up_ask) - float(m.up_bid) > MAX_SPREAD:
        return None
    return float(m.up_bid) if side == "up" else 1 - float(m.up_ask)


def _apply_exits(db, signals: list) -> int:
    """Sell open take-profit/stop-loss positions at the current bid (taker fee on the sale too)."""
    by_market = {s.market.market_id: s for s in signals}
    rows = (db.query(SurvivalTrade)
            .filter(SurvivalTrade.settled == False,  # noqa: E712
                    SurvivalTrade.strategy.in_(list(EXITS)),
                    SurvivalTrade.market_id.in_(list(by_market))).all())
    closed = 0
    for t in rows:
        bid = _side_bid(by_market[t.market_id], t.direction)
        if bid is None:
            continue
        tp, sl = EXITS[t.strategy]
        reason = "take_profit" if bid >= tp else ("stop_loss" if sl is not None and bid <= sl else None)
        if not reason:
            continue
        sell_fee = taker_fee(t.shares, bid)
        t.pnl = t.shares * bid - sell_fee - t.stake
        t.fee = (t.fee or 0.0) + sell_fee
        t.exit_price, t.exit_reason = bid, reason
        t.result = "win" if t.pnl > 0 else "loss"
        t.settled, t.settled_at = True, datetime.utcnow()
        closed += 1
    return closed


def record_survival_trades(signals: list) -> int:
    """Open at most one paper trade per (strategy, market) for each signal a strategy accepts."""
    cands = [s for s in signals if s.prediction_up is not None]
    if not cands:
        return 0
    db = SessionLocal()
    opened = 0
    try:
        _apply_exits(db, cands)
        for strat in STRATEGIES:
            st = _bankroll_state(db, strat)
            if _blocked(st):
                continue
            for s in cands:
                side = choose(strat, s)
                if not side:
                    continue
                mid_id = s.market.market_id
                if db.query(SurvivalTrade.id).filter(SurvivalTrade.strategy == strat,
                                                    SurvivalTrade.market_id == mid_id).first():
                    continue
                q = _side_quotes(s)[side]
                price = q[0]
                stake = round(max(1.0, st["bankroll"] * STAKE_FRAC), 2)
                shares = stake / (price + FEE_RATE * price * (1 - price))
                db.add(SurvivalTrade(
                    strategy=strat, market_id=mid_id, slug=s.market.slug,
                    timeframe=getattr(s.market, "timeframe", "5m"),
                    window_end=s.market.window_end.replace(tzinfo=None) if s.market.window_end else None,
                    direction=side, entry_price=price, mid_at_entry=q[1], model_at_entry=q[2],
                    elapsed_frac=s.elapsed_frac, shares=shares, stake=stake,
                    fee=taker_fee(shares, price),
                ))
                opened += 1
        db.commit()
    except Exception as e:
        logger.warning(f"Survival: failed to record trades: {e}")
        db.rollback()
        opened = 0
    finally:
        db.close()
    return opened


async def settle_survival_trades(db, limit: int = 80) -> int:
    """Settle open survival trades whose market has resolved."""
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
        if t.market_id not in cache:
            try:
                ok, val = await fetch_polymarket_resolution(t.market_id, t.slug)
            except Exception as e:
                logger.debug(f"Survival resolution failed for {t.market_id}: {e}")
                ok, val = False, None
            cache[t.market_id] = val if ok else None
        val = cache[t.market_id]
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


def scoreboard(db) -> dict:
    """Per-strategy survival stats and status."""
    out: List[dict] = []
    for strat, desc in STRATEGIES.items():
        rows = db.query(SurvivalTrade).filter(SurvivalTrade.strategy == strat).all()
        settled = [r for r in rows if r.settled]
        pnls = [r.pnl or 0.0 for r in settled]
        n = len(pnls)
        wins = sum(1 for r in settled if r.result == "win")
        mean = sum(pnls) / n if n else 0.0
        sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1)) if n > 1 else 0.0
        lower95 = mean - 1.96 * sd / math.sqrt(n) if n > 1 else None
        upper95 = mean + 1.96 * sd / math.sqrt(n) if n > 1 else None
        staked = sum(r.stake for r in settled)
        fees = sum(r.fee or 0.0 for r in settled)
        st = _bankroll_state(db, strat)
        blocked = _blocked(st)
        if blocked and blocked.startswith("paused"):
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
        out.append({
            "strategy": strat,
            "description": desc,
            "status": status,
            "blocked": blocked,
            "trades": n,
            "open": len(rows) - n,
            "wins": wins,
            "win_rate": wins / n if n else None,
            "avg_entry": (sum(r.entry_price for r in settled) / n) if n else None,
            "pnl": round(sum(pnls), 2),
            "fees": round(fees, 2),
            "roi": (sum(pnls) / staked) if staked else None,
            "mean_pnl": mean,
            "lower95": lower95,
            "upper95": upper95,
            "bankroll": round(st["bankroll"], 2),
            "peak": round(st["peak"], 2),
            "drawdown": (1 - st["bankroll"] / st["peak"]) if st["peak"] else 0.0,
            "today_pnl": round(st["today_pnl"], 2),
            "by_timeframe": by_tf,
            "needs": max(0, MIN_PROOF_TRADES - n),
            "exits": {k: sum(1 for r in settled if r.exit_reason == k)
                      for k in ("take_profit", "stop_loss", "settled")},
        })
    recent = (db.query(SurvivalTrade).order_by(SurvivalTrade.opened_at.desc()).limit(25).all())
    return {
        "rules": {
            "start_bankroll": START_BANKROLL, "stake_frac": STAKE_FRAC,
            "daily_stop_frac": DAILY_STOP_FRAC, "drawdown_pause_frac": DRAWDOWN_PAUSE_FRAC,
            "min_proof_trades": MIN_PROOF_TRADES, "fee": "shares x 0.07 x p x (1-p)",
            "entry": "ask of the chosen side", "real_money": False,
        },
        "strategies": out,
        "recent": [{
            "strategy": r.strategy, "slug": r.slug, "timeframe": r.timeframe,
            "direction": r.direction, "entry_price": r.entry_price, "stake": r.stake,
            "fee": round(r.fee or 0.0, 4), "result": r.result,
            "exit_price": r.exit_price, "exit_reason": r.exit_reason,
            "pnl": None if r.pnl is None else round(r.pnl, 2),
            "opened_at": r.opened_at.isoformat() + "Z" if r.opened_at else None,
        } for r in recent],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
