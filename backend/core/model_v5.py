"""BTC Up/Down model v5 for Polymarket US: realized-vol backbone + calibrated corrections.

    P(Up) = Phi(w . x)          (probit link; w = [0, 1, 0, ...] is exactly the backbone)

Backbone
    z = ln(S / K) / sqrt(sigma^2 * tau_eff + basis^2)        (all in percent)
    S = BRTI (the index Polymarket US settles on), K = the market's price to beat,
    sigma = realized BRTI 1-minute volatility over the last hour,
    tau_eff = minutes left - 2/3 (the settlement price is the average of the last 60 s of BRTI).
Corrections (x after z)
    z*frac, z*ln(tau)     time in the window
    lead10/30/60          cheap CEX lead-lag: (Binance return - BRTI return) over the last 10/30/60 s, in z units
    z*ln(sig5m/sigma)     short-term vs hourly volatility
    frac-0.5              time in the window (level)
    imbalance             Polymarket US book imbalance near the touch, (bid qty - ask qty) / total, -1..1.
                          Polymarket US keeps no depth history, so this starts at 0 and is learned from the
                          rows the bot records itself (model_v5_rows).
Calibration
    Default weights come from a walk-forward fit on Polymarket US windows (claude/model-backtest-v5.md).
    refit_v5() runs daily: it fills outcomes for recorded rows, fits on everything except the last
    V5_HOLDOUT_DAYS, keeps the new weights only if they beat the current ones on that holdout, then
    refits on all rows and stores them (model_v5_coefs). Ridge pulls toward the current weights.
"""
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

from backend.config import settings

logger = logging.getLogger("trading_bot")

FEATS = ["const", "z", "z*frac", "z*ln_tau", "lead10", "lead30", "lead60", "z*ln_volratio", "frac-0.5", "imbalance"]
BASIS_PCT = 0.003
GATEWAY = "https://gateway.polymarket.us"
BINANCE = "https://data-api.binance.vision/api/v3"
UA = {"User-Agent": "Mozilla/5.0 (polymarket-btc-bot)"}

# Fitted on Polymarket US windows 2026-09-22 .. 2026-11-02 (claude/model-backtest-v5.md). Walk-forward ablation:
# 15m gains from every correction (lead-lag most); 1h gains only from lead-lag, so its other corrections stay 0;
# 1d is the backbone (too few windows to fit). imbalance starts at 0 and is learned from the bot's own rows.
DEFAULT_WEIGHTS: Dict[str, List[float]] = {
    "15m": [-0.0174, 0.4889, -0.3895, -0.0307, 0.1342, 0.174, 0.3703, -0.3505, 0.0133, 0.0],
    "1h": [0.0123, 1.0345, 0.0, 0.0, 0.1641, 0.1685, 0.4122, 0.0, 0.0, 0.0],
    "1d": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
# Features the daily refit may change per timeframe (the rest stay at their current value, normally 0).
REFIT_FEATS: Dict[str, List[str]] = {
    "15m": FEATS,
    "1h": ["const", "z", "lead10", "lead30", "lead60", "imbalance"],
}

_weights_cache: Tuple[float, Dict[str, List[float]], Dict[str, dict]] = (0.0, {}, {})


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ------------------------------------------------------------------ live inputs
async def fetch_brti_seconds(seconds: int = 330) -> List[Tuple[int, float]]:
    """BRTI 1-second (timestamp, close) pairs, oldest first."""
    now = int(time.time())
    try:
        async with httpx.AsyncClient(timeout=10, headers=UA) as c:
            r = await c.get(f"{GATEWAY}/v1/asset-prices/history", params={
                "assets": "crypto:btc", "from": now - seconds, "bucket": "ASSET_PRICE_BUCKET_1S"})
            r.raise_for_status()
            series = r.json().get("series", [])
        return [(int(x["timestamp"]), float(x["close"])) for x in (series[0].get("candles", []) if series else []) if x.get("close")]
    except Exception as e:
        logger.debug(f"v5: BRTI 1s fetch failed: {e}")
        return []


async def fetch_binance_seconds(limit: int = 120) -> Dict[int, float]:
    """Binance BTCUSDT 1-second closes by unix second (cheap CEX lead)."""
    try:
        async with httpx.AsyncClient(timeout=8, headers=UA) as c:
            r = await c.get(f"{BINANCE}/klines", params={"symbol": "BTCUSDT", "interval": "1s", "limit": limit})
            r.raise_for_status()
            return {int(k[0]) // 1000: float(k[4]) for k in r.json()}
    except Exception as e:
        logger.debug(f"v5: Binance 1s fetch failed: {e}")
        return {}


async def fetch_book_imbalance(slug: str, band: float = 0.05) -> Tuple[Optional[float], float, float]:
    """(imbalance, bid qty, ask qty) of the Up book within `band` of the mid."""
    try:
        async with httpx.AsyncClient(timeout=8, headers=UA) as c:
            r = await c.get(f"{GATEWAY}/v1/markets/{slug}/book")
            r.raise_for_status()
            d = r.json().get("marketData", r.json())
        lv = lambda side: [(float(x["px"]["value"]), float(x["qty"])) for x in (d.get(side) or []) if x.get("px")]
        bids, asks = lv("bids"), lv("offers")
        if not bids or not asks:
            return None, 0.0, 0.0
        mid = (max(p for p, _ in bids) + min(p for p, _ in asks)) / 2
        bq = sum(q for p, q in bids if p >= mid - band)
        aq = sum(q for p, q in asks if p <= mid + band)
        tot = bq + aq
        return ((bq - aq) / tot if tot > 0 else None), bq, aq
    except Exception as e:
        logger.debug(f"v5: book fetch failed for {slug}: {e}")
        return None, 0.0, 0.0


def _at(series: Dict[int, float], t: int) -> Optional[float]:
    for k in range(0, 4):
        if t - k in series:
            return series[t - k]
    return None


class LiveInputs:
    """Prices shared by every window in one cycle."""

    def __init__(self, brti_s: List[Tuple[int, float]], binance_s: Dict[int, float], sigma: Optional[float]):
        self.brti = dict(brti_s)
        self.t = brti_s[-1][0] if brti_s else None
        self.S = brti_s[-1][1] if brti_s else None
        self.binance = binance_s
        self.sigma = sigma
        closes = [p for _, p in brti_s[-301:]]
        rets = [math.log(b / a) * 100 for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
        if len(rets) >= 120:
            m = sum(rets) / len(rets)
            self.sigma_short = max(0.002, math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1)) * math.sqrt(60))
        else:
            self.sigma_short = None


async def gather_inputs(sigma: Optional[float]) -> LiveInputs:
    import asyncio
    brti_s, binance_s = await asyncio.gather(fetch_brti_seconds(330), fetch_binance_seconds(120))
    return LiveInputs(brti_s, binance_s, sigma)


def features(inp: LiveInputs, beat: float, minutes_left: float, frac: float,
             imbalance: Optional[float] = None) -> Optional[List[float]]:
    """Feature vector in FEATS order, or None if the backbone can't be computed."""
    if not (inp.S and beat and inp.sigma):
        return None
    tau = max(minutes_left, 0.05)
    sd = math.sqrt(inp.sigma ** 2 * max(tau - 2 / 3, 0.05) + BASIS_PCT ** 2)
    z = math.log(inp.S / beat) * 100 / sd
    leads = []
    for L in (10, 30, 60):
        b0, x0, x1 = _at(inp.brti, inp.t - L), _at(inp.binance, inp.t - L), _at(inp.binance, inp.t)
        leads.append((math.log(x1 / x0) - math.log(inp.S / b0)) * 100 / sd if (b0 and x0 and x1) else 0.0)
    lvr = math.log(inp.sigma_short / inp.sigma) if inp.sigma_short else 0.0
    return [1.0, z, z * frac, z * math.log(tau), *leads, z * lvr, frac - 0.5, float(imbalance or 0.0)]


# ------------------------------------------------------------------ weights
def current_weights() -> Tuple[Dict[str, List[float]], Dict[str, dict]]:
    """Latest accepted weights per timeframe (DB, cached 10 min), falling back to DEFAULT_WEIGHTS."""
    global _weights_cache
    ts, w, info = _weights_cache
    if w and time.time() - ts < 600:
        return w, info
    w = {k: list(v) for k, v in DEFAULT_WEIGHTS.items()}
    info = {k: {"source": "default"} for k in w}
    try:
        from backend.models.database import SessionLocal, ModelCoef
        db = SessionLocal()
        try:
            for tf in w:
                row = (db.query(ModelCoef).filter(ModelCoef.timeframe == tf, ModelCoef.accepted == True)  # noqa: E712
                       .order_by(ModelCoef.fitted_at.desc()).first())
                if row and isinstance(row.weights, list) and len(row.weights) == len(FEATS):
                    w[tf] = row.weights
                    info[tf] = {"source": "refit", "fitted_at": row.fitted_at.isoformat() + "Z", "n_rows": row.n_rows,
                                "metrics": row.metrics}
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"v5: weights load failed: {e}")
    _weights_cache = (time.time(), w, info)
    return w, info


def predict(tf: str, x: Optional[List[float]]) -> Optional[float]:
    if x is None:
        return None
    w, _ = current_weights()
    ww = w.get(tf) or DEFAULT_WEIGHTS["1d"]
    return max(0.01, min(0.99, _ncdf(sum(a * b for a, b in zip(ww, x)))))


# ------------------------------------------------------------------ recording (one row per window per minute)
_last_logged: Dict[str, int] = {}


async def record_rows(quotes) -> int:
    """Store this minute's features, book imbalance, market and both models for each active window."""
    from backend.models.database import SessionLocal, ModelFeatureRow
    minute = int(time.time()) // 60 * 60
    todo = [q for q in quotes if q.is_active and q.v5_x is not None and _last_logged.get(q.slug) != minute]
    if not todo:
        return 0
    db = SessionLocal()
    n = 0
    try:
        for q in todo:
            if db.query(ModelFeatureRow.id).filter(ModelFeatureRow.slug == q.slug, ModelFeatureRow.minute_ts == minute).first():
                _last_logged[q.slug] = minute
                continue
            db.add(ModelFeatureRow(
                slug=q.slug, timeframe=q.timeframe, minute_ts=minute, window_end=q.window_end,
                frac=q.elapsed_frac, x=q.v5_x, imbalance=q.v5_imbalance,
                bid_qty=q.v5_depth[0], ask_qty=q.v5_depth[1],
                market_mid=q.mid, up_ask=q.up_ask, up_bid=q.up_bid, v4=q.model_up, v5=q.model_v5))
            _last_logged[q.slug] = minute
            n += 1
        db.commit()
    except Exception as e:
        logger.warning(f"v5: recording failed: {e}")
        db.rollback()
    finally:
        db.close()
    return n


# ------------------------------------------------------------------ daily walk-forward refit
def _fit(X, y, w0, ridge, offset=None):
    """Ridge-penalized probit fit (penalty toward w0); `offset` is added to X @ w."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import ndtr, log_ndtr
    off = np.zeros(len(X)) if offset is None else offset

    def f(w):
        m = X @ w + off
        ll = np.where(y, log_ndtr(m), log_ndtr(-m))
        pdf = np.exp(-0.5 * m * m) / math.sqrt(2 * math.pi)
        g = np.where(y, pdf / np.maximum(ndtr(m), 1e-300), -pdf / np.maximum(ndtr(-m), 1e-300))
        return -ll.sum() + ridge * np.sum((w - w0) ** 2), -(X.T @ g) + 2 * ridge * (w - w0)
    return minimize(f, w0.copy(), jac=True, method="L-BFGS-B").x


def _logloss(X, y, w):
    import numpy as np
    from scipy.special import ndtr
    p = np.clip(ndtr(X @ w), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


async def fill_outcomes(limit_slugs: int = 400) -> int:
    """Set the outcome on recorded rows whose window has resolved."""
    from backend.models.database import SessionLocal, ModelFeatureRow
    from backend.data.polymarket_us import fetch_us_resolution
    db = SessionLocal()
    filled = 0
    try:
        slugs = [s for (s,) in db.query(ModelFeatureRow.slug).filter(
            ModelFeatureRow.outcome == None, ModelFeatureRow.window_end < datetime.utcnow() - timedelta(minutes=3)  # noqa: E711
        ).distinct().limit(limit_slugs).all()]
        for slug in slugs:
            try:
                resolved, up = await fetch_us_resolution(slug)  # (resolved?, 1.0 Up won / 0.0 Down won)
            except Exception:
                continue
            if not resolved or up is None:
                continue
            filled += db.query(ModelFeatureRow).filter(ModelFeatureRow.slug == slug).update({"outcome": up >= 0.5})
        db.commit()
    except Exception as e:
        logger.warning(f"v5: outcome fill failed: {e}")
        db.rollback()
    finally:
        db.close()
    return filled


async def refit_v5() -> dict:
    """Walk-forward recalibration from the bot's own recorded rows. Returns a per-timeframe report."""
    import numpy as np
    from backend.models.database import SessionLocal, ModelFeatureRow, ModelCoef
    report = {"filled": await fill_outcomes()}
    days = int(getattr(settings, "V5_TRAIN_DAYS", 30))
    hold = int(getattr(settings, "V5_HOLDOUT_DAYS", 2))
    min_rows = int(getattr(settings, "V5_MIN_REFIT_ROWS", 3000))
    ridge = float(getattr(settings, "V5_RIDGE", 5.0))
    cur, _ = current_weights()
    db = SessionLocal()
    try:
        since = datetime.utcnow() - timedelta(days=days)
        for tf in ("15m", "1h"):
            rows = (db.query(ModelFeatureRow).filter(ModelFeatureRow.timeframe == tf, ModelFeatureRow.outcome != None,  # noqa: E711
                                                     ModelFeatureRow.window_end >= since).all())
            rows = [r for r in rows if isinstance(r.x, list) and len(r.x) == len(FEATS)]
            if len(rows) < min_rows:
                report[tf] = {"status": "waiting", "rows": len(rows), "need": min_rows}
                continue
            cut = datetime.utcnow() - timedelta(days=hold)
            tr = [r for r in rows if r.window_end < cut]
            ho = [r for r in rows if r.window_end >= cut]
            X = lambda rs: np.array([r.x for r in rs], dtype=float)
            Y = lambda rs: np.array([bool(r.outcome) for r in rs])
            w0 = np.array(cur[tf], dtype=float)
            free = np.array([FEATS.index(f) for f in REFIT_FEATS.get(tf, FEATS)])

            def fit_masked(Xm, y):
                # fit only the free features; the fixed ones keep w0 (their contribution is a constant offset)
                fixed = np.setdiff1d(np.arange(len(FEATS)), free)
                off = Xm[:, fixed] @ w0[fixed] if len(fixed) else np.zeros(len(Xm))
                w = w0.copy()
                w[free] = _fit(Xm[:, free], y, w0[free], ridge, offset=off)
                return w

            w_new = fit_masked(X(tr), Y(tr)) if len(tr) >= min_rows // 2 else w0
            ll_cur, ll_new = (_logloss(X(ho), Y(ho), w0), _logloss(X(ho), Y(ho), w_new)) if ho else (None, None)
            accept = bool(ho) and ll_new <= ll_cur
            w_all = fit_masked(X(rows), Y(rows)) if accept else w0
            metrics = {"holdout_rows": len(ho), "holdout_logloss_current": ll_cur, "holdout_logloss_refit": ll_new,
                       "train_rows": len(tr)}
            db.add(ModelCoef(timeframe=tf, weights=[float(v) for v in w_all], n_rows=len(rows),
                             accepted=bool(accept), metrics=metrics))
            report[tf] = {"status": "accepted" if accept else "kept current", "rows": len(rows), **metrics,
                          "weights": dict(zip(FEATS, [round(float(v), 4) for v in w_all]))}
        db.commit()
    except Exception as e:
        logger.warning(f"v5: refit failed: {e}")
        db.rollback()
        report["error"] = str(e)
    finally:
        db.close()
    global _weights_cache
    _weights_cache = (0.0, {}, {})
    return report


def status(db) -> dict:
    from backend.models.database import ModelFeatureRow, ModelCoef
    w, info = current_weights()
    out = {"features": FEATS, "weights": {tf: dict(zip(FEATS, [round(float(v), 4) for v in ws])) for tf, ws in w.items()},
           "source": info, "rows": {}, "last_refits": []}
    for tf in ("15m", "1h", "1d"):
        n = db.query(ModelFeatureRow).filter(ModelFeatureRow.timeframe == tf).count()
        s = db.query(ModelFeatureRow).filter(ModelFeatureRow.timeframe == tf, ModelFeatureRow.outcome != None).count()  # noqa: E711
        out["rows"][tf] = {"recorded": n, "settled": s}
    for r in db.query(ModelCoef).order_by(ModelCoef.fitted_at.desc()).limit(6).all():
        out["last_refits"].append({"timeframe": r.timeframe, "fitted_at": r.fitted_at.isoformat() + "Z",
                                   "accepted": r.accepted, "rows": r.n_rows, "metrics": r.metrics})
    return out
