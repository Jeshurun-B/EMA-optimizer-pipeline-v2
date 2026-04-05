# signals.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: Detect EMA crossovers and assemble signal records for database.
#
# CRITICAL — TWO OPERATIONAL MODES:
#
#   MODE 1: COLLECTION (current — building ML dataset)
#       ✅ Detect ALL 9/15 EMA crossovers regardless of quality
#       ✅ No ADX filter, no HTF filter, no volume filter
#       ✅ Every crossover = one data point for the model to learn from
#       ✅ Model needs BOTH good and bad examples
#
#   MODE 2: LIVE TRADING (future — after model trained)
#       ✅ Only fire signals when model confidence > threshold
#       ✅ Apply model's learned filters (ADX, HTF, etc)
#       ✅ Use model prediction to gate signal execution
#
# CROSSOVER MATH (matches Pine Script ta.crossover/crossunder exactly):
#   LONG:  (fast was BELOW slow) AND (fast is now ABOVE slow)
#   SHORT: (fast was ABOVE slow) AND (fast is now BELOW slow)
#
# CRITICAL BUG FIX:
#   ❌ OLD: Used >= and <= (fires on "touch", 15min early)
#   ✅ NEW: Uses > and < (fires on actual cross, correct timing)
#
# PHILOSOPHY:
#   - Pure logic only (no API calls, no database calls)
#   - Input: features dict from compute_features()
#   - Output: complete signal record ready for Supabase insert
#   - Never raises exceptions (returns None on failure)
# ═════════════════════════════════════════════════════════════════════════════

from datetime import datetime, timezone
import pandas as pd
from utils import get_prev_signal, update_prev_signal, log_error


# ══════════════════════════════════════════════════════════════════════════════
#                              PRIVATE KEYS (NOT STORED IN DB)
# ══════════════════════════════════════════════════════════════════════════════
# 
# These keys exist in the features dict but are NOT database columns.
# They're used internally by this file for crossover detection.
# 
# _ema_fast_prev: EMA(9) value from previous candle
# _ema_slow_prev: EMA(15) value from previous candle
# 
# When assembling the final signal record, we filter these out.
# ══════════════════════════════════════════════════════════════════════════════

_PRIVATE_KEYS = {"_ema_fast_prev", "_ema_slow_prev"}


def detect_signal(features: dict, symbol: str) -> dict:
    """
    Detect a 9/15 EMA crossover and return a complete signal record.
    
    COLLECTION MODE — NO FILTERS APPLIED:
        Every crossover is recorded regardless of:
            - ADX level (even if ADX < 25, choppy market)
            - HTF bias (even if counter-trend)
            - Volume (even if volume is low)
            - Fear & Greed (even if extreme values)
        
        These quality metrics ARE computed and stored as features.
        The ML model will learn which conditions make good trades.
        But they do NOT gate whether a signal is recorded.
    
    CROSSOVER DEFINITION:
        LONG (bullish):
            - Previous candle: fast EMA < slow EMA
            - Current candle:  fast EMA > slow EMA
            - Interpretation: momentum shifted from bearish to bullish
        
        SHORT (bearish):
            - Previous candle: fast EMA > slow EMA
            - Current candle:  fast EMA < slow EMA
            - Interpretation: momentum shifted from bullish to bearish
    
    CRITICAL MATH FIX:
        ❌ WRONG: (ema_fast_prev < ema_slow_prev) and (ema_fast >= ema_slow)
        ✅ RIGHT: (ema_fast_prev < ema_slow_prev) and (ema_fast > ema_slow)
        
        Why > instead of >=?
            >= fires when EMAs are EQUAL (the touch candle)
            This happens DURING the crossover, 15min before full cross
            Pine Script's ta.crossover() uses STRICT inequality
            We match that behavior exactly
    
    Args:
        features: dict from compute_features() with all 37 features + private keys
        symbol:   Trading pair (e.g., "BTCUSDT")
    
    Returns:
        dict: Complete signal record ready for db.insert_signal()
              Includes all features + metadata (symbol, timestamp, status, etc)
        
        None: If no crossover detected OR missing required data
    
    Example:
        features = compute_features(df_15m, df_4h, df_1d, ...)
        signal = detect_signal(features, "BTCUSDT")
        
        if signal:
            print(f"CROSSOVER: {signal['signal']} at {signal['price']}")
            insert_signal(signal)
        else:
            print("No crossover")
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              INPUT VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    
    if not features:
        return None  # compute_features() failed — no data to work with
    
    try:
        # ══════════════════════════════════════════════════════════════════════
        #                              EXTRACT VALUES
        # ══════════════════════════════════════════════════════════════════════
        # 
        # We need FOUR values to detect a crossover:
        #   1. ema_fast      — current candle's fast EMA
        #   2. ema_slow      — current candle's slow EMA
        #   3. ema_fast_prev — previous candle's fast EMA
        #   4. ema_slow_prev — previous candle's slow EMA
        # 
        # Plus price (for the signal record)
        # ══════════════════════════════════════════════════════════════════════
        
        ema_fast      = features.get("ema_fast_ltf")
        ema_slow      = features.get("ema_slow_ltf")
        ema_fast_prev = features.get("_ema_fast_prev")  # ← Private key from features.py
        ema_slow_prev = features.get("_ema_slow_prev")  # ← Private key from features.py
        price         = features.get("price")
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              DATA QUALITY CHECK
        # ══════════════════════════════════════════════════════════════════════
        # 
        # CRITICAL BUG FIX:
        #   features.py converts NaN → 0.0 as a safety net.
        #   So we can't just check "is None" — we also check "== 0.0"
        # 
        # Why 0.0 is bad:
        #   - EMA values are NEVER actually 0.0 (they're price-based)
        #   - If we see 0.0, it means data was missing/invalid
        #   - Comparing current EMA to 0.0 would give false crossovers
        # 
        # If ANY required value is None or 0.0 → return None (skip)
        # ══════════════════════════════════════════════════════════════════════
        
        if any(v is None or v == 0.0 for v in [ema_fast, ema_slow, ema_fast_prev, ema_slow_prev]):
            return None  # Missing EMA data — can't detect crossover
        
        if price is None or price <= 0:
            return None  # Missing or invalid price
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              CROSSOVER DETECTION
        # ══════════════════════════════════════════════════════════════════════
        # 
        # CROSSOVER LOGIC (matches Pine Script exactly):
        # 
        # For a BULLISH crossover (LONG signal):
        #   1. Previous state: fast was BELOW slow (ema_fast_prev < ema_slow_prev)
        #   2. Current state:  fast is ABOVE slow (ema_fast > ema_slow)
        #   3. Result: Momentum shifted from bearish to bullish
        # 
        # For a BEARISH crossover (SHORT signal):
        #   1. Previous state: fast was ABOVE slow (ema_fast_prev > ema_slow_prev)
        #   2. Current state:  fast is BELOW slow (ema_fast < ema_slow)
        #   3. Result: Momentum shifted from bullish to bearish
        # 
        # CRITICAL: We use STRICT inequality (> and <) NOT (>= and <=)
        # 
        # Why strict?
        #   - Pine Script's ta.crossover() uses strict inequality
        #   - >= fires when EMAs are EQUAL (the transition candle)
        #   - This causes signals to fire 15 minutes early
        #   - Example timeline:
        #       10:00 AM: fast=100, slow=101 (no cross)
        #       10:15 AM: fast=100.5, slow=100.5 (EQUAL — touch candle)
        #       10:30 AM: fast=101, slow=100 (CROSS — actual signal)
        #   - Using >= would fire at 10:15 (wrong)
        #   - Using >  fires at 10:30 (correct)
        # ══════════════════════════════════════════════════════════════════════
        
        cross_up = (ema_fast_prev < ema_slow_prev) and (ema_fast > ema_slow)
        cross_down = (ema_fast_prev > ema_slow_prev) and (ema_fast < ema_slow)
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              NO CROSSOVER → EXIT EARLY
        # ══════════════════════════════════════════════════════════════════════
        
        if not cross_up and not cross_down:
            return None  # No crossover detected — move to next candle
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              DUPLICATE CHECK
        # ══════════════════════════════════════════════════════════════════════
        # 
        # WHY THIS EXISTS:
        #   If pipeline runs twice within same 15-min window, it will detect
        #   the same crossover twice and try to insert it twice.
        # 
        # STRATEGY:
        #   Check last_signals.csv for this symbol's most recent signal.
        #   If SAME signal type fired in last 30 minutes → skip (duplicate)
        # 
        # Why 30 minutes?
        #   - Crossovers are rare (maybe 1-3 per day per coin)
        #   - Same signal type (LONG→LONG or SHORT→SHORT) can't happen in 30min
        #   - If it does, it's a duplicate detection (pipeline ran 2x)
        # 
        # NOTE: Database has UNIQUE(symbol, checked_at_utc) constraint too.
        #       This is an extra layer of protection at the application level.
        # ══════════════════════════════════════════════════════════════════════
        
        signal_direction = "LONG" if cross_up else "SHORT"
        
        prev = get_prev_signal(symbol)
        if prev:
            try:
                prev_ts = pd.to_datetime(prev["checked_at_utc"], utc=True)
                now_ts = pd.Timestamp.utcnow()
                minutes_since = (now_ts - prev_ts).total_seconds() / 60.0
                
                # Same signal type within 30 minutes → duplicate, skip
                if prev.get("signal") == signal_direction and minutes_since < 30:
                    return None  # Duplicate — don't insert again
            
            except Exception:
                # Time parsing failed — continue anyway (better to risk duplicate than skip real signal)
                pass
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              SIGNAL GAP METADATA
        # ══════════════════════════════════════════════════════════════════════
        # 
        # SIGNAL GAP = Time since last signal on this coin (any direction)
        # 
        # Why we track this:
        #   Stored as a feature (signal_gap_hours).
        #   Model might learn: "signals clustered close together are noise"
        #                      "signals after 24hr gap are fresher/better"
        # 
        # Example:
        #   Last signal: LONG at 10:00 AM
        #   Current signal: SHORT at 2:00 PM
        #   Gap: 4 hours
        # ══════════════════════════════════════════════════════════════════════
        
        prev_signal_type = prev.get("signal") if prev else None
        signal_gap_hours = None
        
        if prev and prev.get("checked_at_utc"):
            try:
                prev_ts = pd.to_datetime(prev["checked_at_utc"], utc=True)
                now_ts = pd.Timestamp.utcnow()
                signal_gap_hours = (now_ts - prev_ts).total_seconds() / 3600.0
            except Exception:
                signal_gap_hours = None
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              BUILD SIGNAL RECORD
        # ══════════════════════════════════════════════════════════════════════
        # 
        # This dict structure EXACTLY matches the Supabase signals table schema.
        # 
        # Required columns:
        #   - checked_at_utc: ISO timestamp (when signal fired)
        #   - symbol: trading pair
        #   - signal: "LONG" or "SHORT"
        #   - status: "pending" (will be "analyzed" after labeler runs)
        #   - All 35 feature columns
        # 
        # Metadata columns (optional but useful):
        #   - prev_signal: what was the last signal? (for pattern analysis)
        #   - signal_gap_hours: time since last signal (feature)
        # ══════════════════════════════════════════════════════════════════════
        
        checked_at_utc = datetime.now(timezone.utc).isoformat()
        
        record = {
            # ── CORE SIGNAL METADATA ──────────────────────────────────────────
            "checked_at_utc":   checked_at_utc,
            "symbol":           symbol,
            "signal":           signal_direction,  # "LONG" or "SHORT"
            "status":           "pending",         # Will be "analyzed" after labeling
            
            # ── SIGNAL GAP TRACKING ───────────────────────────────────────────
            "prev_signal":      prev_signal_type,  # What was last signal? (LONG/SHORT/None)
            "signal_gap_hours": round(signal_gap_hours, 2) if signal_gap_hours else None,
        }
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              COPY ALL FEATURES
        # ══════════════════════════════════════════════════════════════════════
        # 
        # Copy every key from features dict into the record EXCEPT private keys.
        # 
        # Private keys (_ema_fast_prev, _ema_slow_prev):
        #   - Needed for crossover math (we used them above)
        #   - NOT database columns (no column exists for them)
        #   - If we try to insert them → database error
        # 
        # All other features:
        #   - Database has matching columns
        #   - Safe to copy directly
        # ══════════════════════════════════════════════════════════════════════
        
        for key, value in features.items():
            if key not in _PRIVATE_KEYS:  # Skip _ema_fast_prev and _ema_slow_prev
                record[key] = value
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              UPDATE CACHE
        # ══════════════════════════════════════════════════════════════════════
        # 
        # Save this signal to last_signals.csv so next run can:
        #   1. Calculate gap for next signal
        #   2. Detect duplicates
        # 
        # This happens BEFORE database insert (optimistic update).
        # If insert fails, cache is slightly wrong but not critical.
        # ══════════════════════════════════════════════════════════════════════
        
        update_prev_signal(symbol, {
            "signal": signal_direction,
            "checked_at_utc": checked_at_utc
        })
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              RETURN COMPLETE RECORD
        # ══════════════════════════════════════════════════════════════════════
        # 
        # This dict goes straight to db.insert_signal()
        # All keys match database columns (except private keys which we filtered)
        # ══════════════════════════════════════════════════════════════════════
        
        return record
    
    except Exception as e:
        # ══════════════════════════════════════════════════════════════════════
        #                              ERROR HANDLING
        # ══════════════════════════════════════════════════════════════════════
        # 
        # If ANYTHING goes wrong (unexpected data type, math error, etc):
        #   - Log it with symbol context
        #   - Return None (skip this signal)
        #   - Pipeline continues to next coin
        # ══════════════════════════════════════════════════════════════════════
        
        log_error(f"detect_signal error for {symbol}: {repr(e)}")
        return None
