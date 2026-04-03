# signals.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Detect EMA crossovers and assemble signal records for storage.
#
# IMPORTANT — TWO MODES:
#   COLLECTION MODE (current — for building the ML dataset):
#       Detects ALL 9/15 EMA crossovers regardless of ADX, HTF, or any filter.
#       Every crossover is a data point. The ML model will learn which ones
#       are worth trading. We cannot pre-filter here or the model has no
#       negative examples to learn from.
#
#   LIVE MODE (future — after model is trained):
#       Only fires when model confidence exceeds threshold.
#       ADX, HTF, and other filters applied at prediction stage.
#
# RULE: No API calls. No DB calls. Pure logic only.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone
import pandas as pd
from utils import get_prev_signal, update_prev_signal, log_error

# Keys that signals.py needs but should NOT be stored in the database
_PRIVATE_KEYS = {"_ema_fast_prev", "_ema_slow_prev"}


def detect_signal(features: dict, symbol: str) -> dict:
    """
    Detect a 9/15 EMA crossover and return a complete signal record.

    COLLECTION MODE — no filters applied:
        Every crossover is recorded regardless of ADX level, HTF bias,
        market conditions, or any other quality measure.
        These quality metrics ARE computed and stored as features so the
        ML model can use them — but they do not gate whether a signal
        is recorded or not.

    A crossover is defined as:
        LONG:  fast EMA (9) crosses from BELOW to ABOVE slow EMA (15)
        SHORT: fast EMA (9) crosses from ABOVE to BELOW slow EMA (15)

    Args:
        features: dict from compute_features() — all 35 features plus
                  private keys _ema_fast_prev and _ema_slow_prev
        symbol:   coin pair e.g. "BTCUSDT"

    Returns:
        Complete signal dict ready for Supabase insert, or None if no crossover.
    """
    if not features:
        return None

    try:
        # Extract EMA values — current and previous candle
        ema_fast      = features.get("ema_fast_ltf")
        ema_slow      = features.get("ema_slow_ltf")
        ema_fast_prev = features.get("_ema_fast_prev")
        ema_slow_prev = features.get("_ema_slow_prev")
        price         = features.get("price")

        # Cannot detect a crossover without these four values
        if any(v is None for v in [ema_fast, ema_slow, ema_fast_prev, ema_slow_prev, price]):
            return None

        # ── CROSSOVER DETECTION ───────────────────────────────────────────────
        # A crossover means the relationship between fast and slow EMAs
        # changed from one candle to the next.

        # LONG: fast was below slow, now at or above — upward cross
        cross_up = (ema_fast_prev < ema_slow_prev) and (ema_fast > ema_slow)

        # SHORT: fast was above slow, now at or below — downward cross
        cross_down = (ema_fast_prev > ema_slow_prev) and (ema_fast < ema_slow)

        # No crossover detected — return None, move to next candle
        if not cross_up and not cross_down:
            return None

        # ── DIRECTION ─────────────────────────────────────────────────────────
        signal_direction = "LONG" if cross_up else "SHORT"

        # ── SIGNAL GAP METADATA ───────────────────────────────────────────────
        # Track time since last signal on this coin.
        # Stored as a feature — the model may learn that signals
        # clustered close together are less reliable.
        prev             = get_prev_signal(symbol)
        prev_signal_type = prev.get("signal") if prev else None
        signal_gap_hours = None

        if prev and prev.get("checked_at_utc"):
            try:
                prev_ts          = pd.to_datetime(prev["checked_at_utc"], utc=True)
                now_ts           = pd.Timestamp.utcnow()
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

        # Copy all feature values into the record EXCEPT private keys
        # Private keys (_ema_fast_prev, _ema_slow_prev) have no DB column
        for key, value in features.items():
            if key not in _PRIVATE_KEYS:
                record[key] = value

        # ── UPDATE PREV SIGNAL CACHE ──────────────────────────────────────────
        # Saved to last_signals.csv so the next candle can compute signal_gap_hours
        update_prev_signal(symbol, {
            "signal":         signal_direction,
            "checked_at_utc": checked_at_utc
        })

        return record

    except Exception as e:
        log_error(f"detect_signal error for {symbol}: {repr(e)}")
        return None