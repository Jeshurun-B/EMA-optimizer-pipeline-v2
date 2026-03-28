# signals.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Detect EMA crossovers and assemble the final signal record.
# RULE: Takes the feature dictionary from features.py. If a crossover is 
#       detected, returns a fully populated dict ready for the database.
#       If no crossover, returns None.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone
import pandas as pd
from utils import get_prev_signal
from config import ADX_THRESHOLD

def detect_signal(features: dict, symbol: str) -> dict | None:
    """
    Checks for a 9/15 EMA crossover using the computed features.
    
    Args:
        features: The dictionary returned by features.compute_features()
        symbol: The coin pair (e.g., "BTCUSDT")
        
    Returns:
        A dictionary formatted for the Supabase 'signals' table, or None.
    """
    if not features:
        return None

    # 1. Extract the values needed for detection
    ema_fast = features["ema_fast_ltf"]
    ema_slow = features["ema_slow_ltf"]
    adx_latest = features["adx_ltf"]
    
    # To detect a cross, we need to know what the EMA was on the previous candle.
    # We can calculate the previous EMA by reversing the slope percentage.
    # slope = (latest - prev) / prev  =>  prev = latest / (1 + slope)
    ema_fast_prev = ema_fast / (1 + (features["ema_fast_slope"] / 100.0))
    ema_slow_prev = ema_slow / (1 + (features["ema_slow_slope"] / 100.0))

    # 2. Define Crossover Conditions
    # LONG: Fast was below Slow, now Fast is above Slow
    ltf_cross_up = (ema_fast_prev < ema_slow_prev) and (ema_fast >= ema_slow)
    
    # SHORT: Fast was above Slow, now Fast is below Slow
    ltf_cross_down = (ema_fast_prev > ema_slow_prev) and (ema_fast <= ema_slow)

    # Filter: Only accept signals if ADX indicates a trending market
    adx_ok = adx_latest >= ADX_THRESHOLD

    is_long = ltf_cross_up and adx_ok
    is_short = ltf_cross_down and adx_ok

    if not (is_long or is_short):
        return None  # No valid signal detected

    # 3. Calculate Gap Hours from the Previous Signal Cache
    prev_record = get_prev_signal(symbol)
    prev_signal_type = prev_record.get("signal") if prev_record else None
    prev_time_str = prev_record.get("checked_at_utc") if prev_record else None
    
    signal_gap_hours = 0.0
    if prev_time_str:
        try:
            prev_ts = pd.to_datetime(prev_time_str, utc=True)
            now_ts = pd.Timestamp.utcnow()
            delta = now_ts - prev_ts
            signal_gap_hours = delta.total_seconds() / 3600.0
        except Exception:
            pass # Keep it at 0.0 if parsing fails

    # 4. Assemble the final record for Supabase
    signal_record = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "signal": "LONG" if is_long else "SHORT",
        "status": "pending",
        "prev_signal": prev_signal_type,
        "signal_gap_hours": round(signal_gap_hours, 2)
    }
    
    # Merge the technical features into the final record
    signal_record.update(features)

    return signal_record