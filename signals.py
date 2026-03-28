# signals.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Detect EMA crossovers. Returns a complete signal record or None.
# RULE: No API calls. No DB calls. Pure logic only.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone
import pandas as pd
from config import ADX_THRESHOLD
from utils import get_prev_signal, update_prev_signal, log_error

# These are the feature keys that should NOT be stored in the database.
# They are needed for crossover detection but have no matching DB column.
_PRIVATE_KEYS = {"_ema_fast_prev", "_ema_slow_prev"}


def detect_signal(features: dict, symbol: str) -> dict:
    """
    Check if the latest candle produced a valid 9/15 EMA crossover signal.

    Uses the private keys _ema_fast_prev and _ema_slow_prev from features.py
    to compare current vs previous EMA values — the only reliable way to
    detect a crossover without reconstructing history.

    Returns a complete dict ready for Supabase, or None if no signal.
    """
    if not features:
        return None

    try:
        # Extract current and previous EMA values
        ema_fast      = features.get("ema_fast_ltf")
        ema_slow      = features.get("ema_slow_ltf")
        ema_fast_prev = features.get("_ema_fast_prev")
        ema_slow_prev = features.get("_ema_slow_prev")
        adx_latest    = features.get("adx_ltf")
        htf_4h_bias   = features.get("htf_4h_bias")
        price         = features.get("price")

        # Guard — can't detect crossover without these values
        if any(v is None for v in [ema_fast, ema_slow, ema_fast_prev, ema_slow_prev, price]):
            return None

        # ── CROSSOVER DETECTION ───────────────────────────────────────────────
        # LONG: fast was below slow, now above — bullish crossover
        cross_up   = (ema_fast_prev < ema_slow_prev) and (ema_fast >= ema_slow)
        # SHORT: fast was above slow, now below — bearish crossover
        cross_down = (ema_fast_prev > ema_slow_prev) and (ema_fast <= ema_slow)

        if not cross_up and not cross_down:
            return None  # No crossover this candle

        # ── FILTERS ───────────────────────────────────────────────────────────
        # ADX filter: only trade in trending markets (ADX >= 25)
        if adx_latest is None or adx_latest < ADX_THRESHOLD:
            return None

        # HTF filter: only take signals that agree with the 4h trend
        # LONG signal needs bullish 4h, SHORT signal needs bearish 4h
        if cross_up   and not htf_4h_bias:
            return None  # LONG against bearish 4h — skip
        if cross_down and htf_4h_bias:
            return None  # SHORT against bullish 4h — skip

        # ── DIRECTION ─────────────────────────────────────────────────────────
        signal_direction = "LONG" if cross_up else "SHORT"

        # ── PREVIOUS SIGNAL METADATA ──────────────────────────────────────────
        prev = get_prev_signal(symbol)
        prev_signal_type  = prev.get("signal") if prev else None
        signal_gap_hours  = None

        if prev and prev.get("checked_at_utc"):
            try:
                prev_ts      = pd.to_datetime(prev["checked_at_utc"], utc=True)
                now_ts       = pd.Timestamp.utcnow()
                signal_gap_hours = (now_ts - prev_ts).total_seconds() / 3600.0
            except Exception:
                signal_gap_hours = None

        # ── BUILD RECORD ──────────────────────────────────────────────────────
        checked_at_utc = datetime.now(timezone.utc).isoformat()

        record = {
            "checked_at_utc":   checked_at_utc,
            "symbol":           symbol,
            "signal":           signal_direction,
            "status":           "pending",
            "prev_signal":      prev_signal_type,
            "signal_gap_hours": round(signal_gap_hours, 2) if signal_gap_hours else None,
        }

        # Copy all feature keys EXCEPT private keys and 'price' (handled above)
        for key, value in features.items():
            if key not in _PRIVATE_KEYS:
                record[key] = value

        # ── UPDATE PREV SIGNAL CACHE ──────────────────────────────────────────
        update_prev_signal(symbol, {
            "signal":         signal_direction,
            "checked_at_utc": checked_at_utc
        })

        return record

    except Exception as e:
        log_error(f"detect_signal error for {symbol}: {repr(e)}")
        return None