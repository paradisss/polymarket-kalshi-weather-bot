"""Real-money mirror on Polymarket US. OFF by default.

Places a real order only when ALL of these hold:
  * US_LIVE_TRADING=true and POLYMARKET_US_KEY_ID / POLYMARKET_US_SECRET_KEY are set
  * the strategy is listed in US_LIVE_STRATEGIES
  * the survival scoreboard marks that strategy PROVEN (>= 200 paper trades, 95% CI of mean P&L > 0),
    or the owner lists it in US_LIVE_ALLOW_UNPROVEN and it is still TESTING (not PAUSED/FAILING/RETIRED)
  * all-time live P&L is above -US_LIVE_MAX_TOTAL_LOSS (kill switch)
  * a one-share probe order (IOC buy at 1c, which normally cannot fill) has confirmed that order
    placement and order lookup work with this key
  * the paper strategy just opened (or closed early) a trade in this cycle
  * today's realized live loss is under US_LIVE_DAILY_LOSS and fewer than US_LIVE_MAX_OPEN are open
Orders are limit + immediate-or-cancel at the current ask (buys) or bid (early sells), at most
US_LIVE_MAX_STAKE dollars each - never a resting order, never more than the cap.
"""
import logging
import math
from datetime import datetime
from typing import List, Optional

from backend.config import settings
from backend.models.database import SessionLocal, LiveTrade

logger = logging.getLogger("trading_bot")
DEFAULT_FEE_COEF_LIVE = 0.0695  # Polymarket US BTC Up/Down feeCoefficient


def _allowed() -> List[str]:
    return [s.strip() for s in (settings.US_LIVE_STRATEGIES or "").split(",") if s.strip()]


def _unproven_ok() -> set:
    return {s.strip() for s in (settings.US_LIVE_ALLOW_UNPROVEN or "").split(",") if s.strip()}


# One-share probe: proves the order + lookup path before any real entry. Retried every 10 min on failure.
_probe = {"ok": None, "at": 0.0, "detail": None}


def _order_id(res) -> Optional[str]:
    if not isinstance(res, dict):
        return None
    oid = res.get("id") or res.get("orderId") or res.get("order_id")
    if not oid and isinstance(res.get("order"), dict):
        oid = res["order"].get("id") or res["order"].get("orderId")
    return str(oid) if oid else None


def live_blockers(include_probe: bool = True) -> List[str]:
    out = []
    if not settings.US_LIVE_TRADING:
        out.append("US_LIVE_TRADING is off")
    if not (settings.POLYMARKET_US_KEY_ID and settings.POLYMARKET_US_SECRET_KEY):
        out.append("no Polymarket US API key")
    if not _allowed():
        out.append("US_LIVE_STRATEGIES is empty")
    if include_probe and _probe["ok"] is False:
        out.append(f"order probe failed: {_probe['detail']}")
    return out


async def _run_probe(client, quotes) -> None:
    import time
    if _probe["ok"] is True or (_probe["ok"] is False and time.time() - _probe["at"] < 600):
        return
    # Only probe a live window whose Up ask is well above 1c, so the 1c IOC buy cannot fill.
    q = next((q for q in quotes if getattr(q, "slug", None) and (getattr(q, "up_ask", None) or 0) >= 0.05
              and getattr(q, "is_active", True)), None)
    if q is None:
        return
    _probe["at"] = time.time()
    try:
        res = await client.create_order(q.slug, "up", "buy", 0.01, 1, tick=getattr(q, "tick", 0.01) or 0.01)
        oid = _order_id(res)
        if not oid:
            _probe.update(ok=False, detail=f"no order id in response (keys: {sorted(res)[:10] if isinstance(res, dict) else type(res).__name__})")
            return
        filled, px, state, reason = await _fill(client, res, oid, "up")
        if state == "unknown":
            _probe.update(ok=False, detail=f"order {oid} placed but its fill could not be read "
                                           f"(no executions in response; keys: {sorted(res)[:10]})")
            return
        if state == "ORDER_STATE_REJECTED":
            _probe.update(ok=False, detail=f"order {oid} rejected: {reason or 'no reason given'}")
            return
        if filled > 0:
            # Shouldn't happen (ask was >= 5c), but if it does, track the position like any live trade.
            db = SessionLocal()
            try:
                db.add(LiveTrade(strategy="probe", slug=q.slug, timeframe=getattr(q, "timeframe", None),
                                 window_end=getattr(q, "window_end", None), direction="up", limit_price=0.01,
                                 quantity=1, order_id=oid, filled=filled, avg_price=px or 0.01,
                                 cost=filled * (px or 0.01), status="filled"))
                db.commit()
            finally:
                db.close()
        _probe.update(ok=True, detail=f"order {oid} state={state} filled={filled} on {q.slug}")
        logger.info(f"Live: probe ok - {_probe['detail']}")
    except Exception as e:
        _probe.update(ok=False, detail=str(e)[:200])
        logger.warning(f"Live: probe failed: {e}")


def _client():
    from backend.data.polymarket_us import PolymarketUSClient
    return PolymarketUSClient(settings.POLYMARKET_US_KEY_ID, settings.POLYMARKET_US_SECRET_KEY)


def _today_live_pnl(db) -> float:
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = db.query(LiveTrade.pnl).filter(LiveTrade.settled == True,  # noqa: E712
                                          LiveTrade.settled_at >= today).all()
    return sum(r[0] or 0.0 for r in rows)


def _num(x) -> Optional[float]:
    if isinstance(x, dict):
        x = x.get("value")
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fill_from_res(res, side: str):
    """(filled shares, average price in the side's terms, state, reject reason) from the create-order
    response. Orders are sent with synchronousExecution, so Polymarket US returns the executions inline;
    an IOC order is closed at once and GET /v1/order/{id} then answers 404. None if no executions."""
    ex = res.get("executions") if isinstance(res, dict) else None
    if not isinstance(ex, list) or not ex:
        return None
    ex = [e for e in ex if isinstance(e, dict)]
    orders = [e["order"] for e in ex if isinstance(e.get("order"), dict)]
    reason = next((e.get("orderRejectReason") or e.get("text") for e in ex
                   if e.get("type") == "EXECUTION_TYPE_REJECTED"), None)
    filled, px, state = 0.0, None, (orders[-1].get("state", "") if orders else "")
    if orders:
        o = max(orders, key=lambda o: float(o.get("cumQuantity") or 0))
        filled, px = float(o.get("cumQuantity") or 0), _num(o.get("avgPx"))
        state = o.get("state", state)
    if filled > 0 and px is None or not orders:
        fills = [(float(e.get("lastShares") or 0), _num(e.get("lastPx"))) for e in ex
                 if e.get("type") in ("EXECUTION_TYPE_FILL", "EXECUTION_TYPE_PARTIAL_FILL")]
        fills = [(q, p) for q, p in fills if q > 0 and p is not None]
        if fills:
            filled = sum(q for q, _ in fills)
            px = sum(q * p for q, p in fills) / filled
    if px is not None and side == "down":
        px = 1 - px  # API reports the long (Up) price
    if reason and not state:
        state = "ORDER_STATE_REJECTED"
    return filled, px, state, reason


async def _fill(client, res, oid, side: str):
    """Fill info from the create response, else from an order lookup (works only while the order exists)."""
    info = _fill_from_res(res, side)
    if info is not None:
        return info
    if not oid:
        return 0.0, None, "unknown", None
    filled, px, state = await _fill_info(client, oid, side)
    return filled, px, state, None


async def _fill_info(client, order_id: str, side: str):
    """(filled shares, average price in the side's terms, state) from the order record."""
    try:
        d = await client.order(order_id)
    except Exception as e:
        logger.warning(f"Live: order lookup failed {order_id}: {e}")
        return 0.0, None, "unknown"
    o = d.get("order", d)
    filled = float(o.get("cumQuantity") or 0)
    px = _num(o.get("avgPx"))
    if px is not None and side == "down":
        px = 1 - px  # API reports the long (Up) price
    return filled, px, o.get("state", "")


def _eligible(db, strat: str) -> Optional[str]:
    """None if the strategy may trade live now, else the reason it may not."""
    from backend.core.survival import strategy_stats
    st = strategy_stats(db, strat)["status"]
    if st == "PROVEN":
        return None
    if strat in _unproven_ok() and st == "TESTING":
        return None
    return f"{strat} is {st}" + ("" if strat in _unproven_ok() else " (not PROVEN)")


def _total_live_pnl(db) -> float:
    """All-time live P&L for the kill switch; unconfirmed fills count as a total loss of their cost."""
    rows = db.query(LiveTrade.pnl).filter(LiveTrade.settled == True).all()  # noqa: E712
    unconf = db.query(LiveTrade.cost).filter(LiveTrade.settled == False,  # noqa: E712
                                             LiveTrade.status == "unconfirmed").all()
    return sum(r[0] or 0.0 for r in rows) - sum(r[0] or 0.0 for r in unconf)


async def _reconcile_unconfirmed(db, client) -> int:
    """Re-read unconfirmed orders from the venue; returns how many are still unconfirmed."""
    left = 0
    for lt in db.query(LiveTrade).filter(LiveTrade.settled == False,  # noqa: E712
                                         LiveTrade.status == "unconfirmed").all():
        if not lt.order_id:
            left += 1
            continue
        filled, px, state = await _fill_info(client, lt.order_id, lt.direction)
        if state == "unknown":
            left += 1
            continue
        lt.filled, lt.avg_price, lt.error = filled, px, None
        if filled > 0 and px is not None:
            coef = DEFAULT_FEE_COEF_LIVE
            lt.cost = filled * px + filled * coef * px * (1 - px)
            lt.status = "filled" if filled >= (lt.quantity or 0) else "partial"
        else:
            lt.status, lt.cost = "unfilled", 0.0
            lt.settled, lt.settled_at, lt.pnl = True, datetime.utcnow(), 0.0
    db.commit()
    return left


async def mirror_live(summary: dict, quotes) -> dict:
    blockers = live_blockers(include_probe=False)
    if blockers:
        return {"enabled": False, "blockers": blockers}
    from backend.core.survival import DEFAULT_FEE_COEF
    by_slug = {q.slug: q for q in quotes}
    allowed = set(_allowed())
    placed, errors, skipped = [], [], []
    db = SessionLocal()
    try:
        client = _client()
        await _run_probe(client, quotes)
        if _probe["ok"] is not True:
            return {"enabled": False, "blockers": live_blockers() or ["order probe pending"]}
        # 1) early exits (take-profit / stop-loss) for live positions of strategies that just exited
        for strat, slug, side, reason, bid in summary.get("exited", []):
            lt = (db.query(LiveTrade).filter(LiveTrade.strategy == strat, LiveTrade.slug == slug,
                                             LiveTrade.settled == False,  # noqa: E712
                                             LiveTrade.exit_order_id.is_(None),
                                             LiveTrade.filled > 0).first())
            if not lt:
                continue
            try:
                q = by_slug.get(slug)
                res = await client.create_order(slug, side, "sell", bid, int(lt.filled),
                                                tick=q.tick if q else 0.01)
                lt.exit_order_id = _order_id(res)
                sold, px, _, _ = await _fill(client, res, lt.exit_order_id, side)
                if sold > 0 and px is not None:
                    coef = q.fee_coef if q else DEFAULT_FEE_COEF
                    proceeds = sold * px - sold * coef * px * (1 - px)
                    lt.exit_price = px
                    lt.pnl = proceeds - lt.cost * (sold / lt.filled)
                    if sold >= lt.filled:
                        lt.settled, lt.settled_at, lt.outcome = True, datetime.utcnow(), reason
                placed.append(("sell", strat, slug, side, bid, int(lt.filled)))
            except Exception as e:
                errors.append(f"sell {slug}: {e}")
            db.commit()

        # 2) new entries - none while any fill is still unconfirmed
        if await _reconcile_unconfirmed(db, client):
            skipped.append("unconfirmed live fill - no new entries until it is reconciled")
            return {"enabled": True, "placed": placed, "errors": errors, "skipped": skipped}
        day_pnl = _today_live_pnl(db)
        total_pnl = _total_live_pnl(db)
        for strat, slug, side, ask in summary.get("opened", []):
            if strat not in allowed:
                continue
            why = _eligible(db, strat)
            if why:
                skipped.append(why)
                continue
            if total_pnl <= -settings.US_LIVE_MAX_TOTAL_LOSS:
                skipped.append("total live loss limit reached (kill switch)")
                break
            if day_pnl <= -settings.US_LIVE_DAILY_LOSS:
                skipped.append("daily live loss limit reached")
                break
            open_n = db.query(LiveTrade).filter(LiveTrade.settled == False, LiveTrade.filled > 0).count()  # noqa: E712
            if open_n >= settings.US_LIVE_MAX_OPEN:
                skipped.append("max open live positions")
                break
            if db.query(LiveTrade.id).filter(LiveTrade.strategy == strat, LiveTrade.slug == slug).first():
                continue
            q = by_slug.get(slug)
            coef = q.fee_coef if q else DEFAULT_FEE_COEF
            qty = int(math.floor(settings.US_LIVE_MAX_STAKE / (ask + coef * ask * (1 - ask))))
            if qty < 1:
                skipped.append(f"{slug}: stake cap below one share")
                continue
            lt = LiveTrade(strategy=strat, slug=slug, timeframe=q.timeframe if q else None,
                           window_end=q.window_end if q else None, direction=side,
                           limit_price=ask, quantity=qty)
            db.add(lt)
            try:
                res = await client.create_order(slug, side, "buy", ask, qty, tick=q.tick if q else 0.01)
                lt.order_id = _order_id(res)
                filled, px, state, reason = await _fill(client, res, lt.order_id, side)
                if reason:
                    lt.error = f"rejected: {reason}"[:500]
                if state == "unknown":
                    # Can't confirm the fill: assume the worst (fully filled at the limit) so the
                    # open-position and loss caps still count it; settlement scores it normally.
                    filled, px = float(qty), ask
                    lt.error = "fill not confirmed - counted as filled at limit for the caps; reconciling"
                lt.filled, lt.avg_price = filled, px
                if filled > 0 and px is not None:
                    lt.cost = filled * px + filled * coef * px * (1 - px)
                lt.status = ("unconfirmed" if state == "unknown" else
                             "filled" if filled >= qty else ("partial" if filled > 0 else "unfilled"))
                if filled <= 0:
                    lt.settled, lt.settled_at, lt.pnl = True, datetime.utcnow(), 0.0
                placed.append(("buy", strat, slug, side, ask, qty, lt.status))
            except Exception as e:
                lt.status, lt.error = "error", str(e)[:500]
                lt.settled, lt.settled_at, lt.pnl = True, datetime.utcnow(), 0.0
                errors.append(f"buy {slug}: {e}")
            db.commit()
    finally:
        db.close()
    return {"enabled": True, "placed": placed, "errors": errors, "skipped": skipped}


async def settle_live_trades(db) -> int:
    """Score filled live positions held to settlement (Polymarket US pays $1/share to the winner)."""
    from backend.data.polymarket_us import fetch_us_resolution
    now = datetime.utcnow()
    from sqlalchemy import or_
    rows = db.query(LiveTrade).filter(LiveTrade.settled == False, LiveTrade.filled > 0,  # noqa: E712
                                      or_(LiveTrade.status.is_(None), LiveTrade.status != "unconfirmed")).all()
    done = 0
    for lt in rows:
        if lt.window_end and lt.window_end > now:
            continue
        ok, val = await fetch_us_resolution(lt.slug)
        if not ok or val is None:
            continue
        lt.outcome = "up" if val == 1.0 else "down"
        won = lt.outcome == lt.direction
        lt.pnl = (lt.filled if won else 0.0) - lt.cost
        lt.settled, lt.settled_at = True, now
        done += 1
    if done:
        db.commit()
    return done


def live_status(db) -> dict:
    from backend.core.survival import strategy_stats, STRATEGIES
    blockers = live_blockers()
    allowed = [s for s in _allowed() if s in STRATEGIES]
    rows = db.query(LiveTrade).order_by(LiveTrade.opened_at.desc()).limit(20).all()
    settled = db.query(LiveTrade.pnl).filter(LiveTrade.settled == True).all()  # noqa: E712
    return {
        "enabled": not blockers,
        "blockers": blockers,
        "strategies": {s: strategy_stats(db, s)["status"] for s in allowed},
        "max_stake": settings.US_LIVE_MAX_STAKE,
        "daily_loss_limit": settings.US_LIVE_DAILY_LOSS,
        "max_open": settings.US_LIVE_MAX_OPEN,
        "max_total_loss": settings.US_LIVE_MAX_TOTAL_LOSS,
        "allow_unproven": sorted(_unproven_ok()),
        "probe": {"ok": _probe["ok"], "detail": _probe["detail"]},
        "today_pnl": round(_today_live_pnl(db), 2),
        "total_pnl": round(sum(r[0] or 0.0 for r in settled), 2),
        "recent": [{
            "strategy": r.strategy, "slug": r.slug, "direction": r.direction,
            "limit_price": r.limit_price, "quantity": r.quantity, "filled": r.filled,
            "avg_price": r.avg_price, "status": r.status, "pnl": r.pnl, "error": r.error,
            "opened_at": r.opened_at.isoformat() + "Z" if r.opened_at else None,
        } for r in rows],
    }


_key_check = {"at": 0.0, "result": None}


async def check_key() -> dict:
    """Read-only check that Polymarket US accepts the API key (signed GET /v1/account/balances).
    Returns only ok/error - never balances, positions or the key. Cached for 5 minutes."""
    import time
    if not (settings.POLYMARKET_US_KEY_ID and settings.POLYMARKET_US_SECRET_KEY):
        return {"key_set": False, "key_ok": False, "error": "no Polymarket US API key set"}
    if _key_check["result"] and time.time() - _key_check["at"] < 300:
        return _key_check["result"]
    try:
        await _client().balances()
        res = {"key_set": True, "key_ok": True, "error": None}
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            msg = "Polymarket US rejected the key (" + ("401" if "401" in msg else "403") + "): " + msg.split(":", 1)[-1].strip()[:120]
        elif "Incorrect padding" in msg or "base64" in msg.lower() or "32 bytes" in msg:
            msg = "the secret key is not in the expected format (base64 Ed25519 key from polymarket.us/developer)"
        res = {"key_set": True, "key_ok": False, "error": msg[:200]}
    _key_check.update(at=time.time(), result=res)
    return res
