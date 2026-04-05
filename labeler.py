# labeler.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: Label pending signals by recording RAW post-signal price movement.
#
# PHILOSOPHY — NO TRADE SIMULATION:
#   ❌ OLD: Simulate trades with hardcoded R:R ratios (1.5:1 target)
#   ✅ NEW: Record RAW facts about what price did after signal
#
# WHAT WE RECORD:
#   - max_price_after:      Highest price reached
#   - min_price_after:      Lowest price reached
#   - max_move_up_pct:      How far UP from entry (%)
#   - max_move_down_pct:    How far DOWN from entry (%)
#   - time_of_max_price:    When did highest price occur
#   - time_of_min_price:    When did lowest price occur
#   - candles_to_max_price: How many candles until highest price
#   - candles_to_min_price: How many candles until lowest price
#
# WHY NO trade_quality COLUMN:
#   Old approach encoded assumptions: "good trade = hits 1.5R before stop"
#   New approach: let MODEL decide what "good" means from raw data
#
# TRADE WINDOW:
#   From signal timestamp → next opposite signal timestamp
#   Example: LONG at 10:30 AM → SHORT at 4:15 PM (5h 45min window)
# ═════════════════════════════════════════════════════════════════════════════

import time
import pandas as pd
from db import fetch_pending, update_signal_labels, fetch_next_signal_time
from fetcher import fetch_forward_candles
from utils import log_error, api_limit_reached


def label_pending_signals():
    """
    Fetch pending signals, find their exit timestamp, record raw price movement.
    
    WORKFLOW:
        1. Query database for all status='pending' signals
        2. For each pending signal:
           a. Ask DB: when did next signal fire? (trade exit time)
           b. Fetch candles from entry → exit
           c. Find max/min prices in that window
           d. Calculate percentage moves from entry
           e. Count candles to max/min
           f. Update database with outcome data
           g. Set status='analyzed'
    
    SMART EXIT DETECTION:
        Instead of manually fetching 400 candles and recalculating EMAs,
        we ask the database: "When was the next crossover for this coin?"
        This is 1000x faster and uses data we already collected.
    
    STATUS TRANSITIONS:
        pending → analyzed (after labeling)
        pending → pending (if trade still open, no next signal yet)
    
    API SAFETY:
        Checks api_limit_reached() before each fetch.
        Stops early if limit hit (saves state for next run).
    
    OUTPUT:
        Console logs showing:
            - How many signals need labeling
            - Progress per signal
            - Up/down percentages and timing
            - Final counts (labeled, skipped, failed)
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              FETCH PENDING SIGNALS
    # ══════════════════════════════════════════════════════════════════════════
    
    pending = fetch_pending()
    
    if not pending:
        print("No pending signals to label.")
        return
    
    print(f"Labeling {len(pending)} pending signals using raw data collection...\n")
    
    # Counters for summary
    labeled = 0
    skipped = 0
    failed  = 0
    
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              PROCESS EACH SIGNAL
    # ══════════════════════════════════════════════════════════════════════════
    
    for row in pending:
        # ── API SAFETY CHECK ──────────────────────────────────────────────────
        # 
        # Before making ANY fetch, check if we've hit API limit.
        # If yes → stop processing, save state for next run.
        # ──────────────────────────────────────────────────────────────────────
        
        if api_limit_reached():
            print(f"API limit reached — stopping early.")
            break
        
        # ── EXTRACT SIGNAL DATA ───────────────────────────────────────────────
        
        signal_id   = row.get("id")
        symbol      = row.get("symbol")
        signal_dir  = str(row.get("signal", "LONG")).upper()
        entry_price = float(row.get("price") or 0)
        start_time  = row.get("checked_at_utc")
        
        # ── VALIDATE SIGNAL DATA ──────────────────────────────────────────────
        # 
        # If any critical field is missing/invalid → skip this signal
        # ──────────────────────────────────────────────────────────────────────
        
        if not signal_id or not symbol or not start_time or entry_price <= 0:
            continue  # Skip malformed row
        
        # ── FIND TRADE EXIT TIME ──────────────────────────────────────────────
        # 
        # Ask database: "When was the next signal for this coin?"
        # This tells us when the trade ended (opposite signal fired).
        # 
        # Example:
        #   LONG at 10:30 AM → SHORT at 4:15 PM (exit)
        #   Trade window: 10:30 AM → 4:15 PM
        # ──────────────────────────────────────────────────────────────────────
        
        end_time_str = fetch_next_signal_time(symbol, start_time)
        
        if not end_time_str:
            # ── NO EXIT YET — TRADE STILL OPEN ────────────────────────────────
            # 
            # No future signal found in database.
            # This means:
            #   - Trade is still running (no opposite signal yet)
            #   - We can't label it yet (don't know final outcome)
            # 
            # Action: Skip for now, leave status='pending'
            # Next labeler run will check again
            # ──────────────────────────────────────────────────────────────────
            
            skipped += 1
            print(f"  ⏳ {symbol} id={signal_id} still open, waiting for next signal...")
            continue
        
        # ── CONVERT TIMESTAMPS ────────────────────────────────────────────────
        # 
        # Start time: when signal fired (entry)
        # End time:   when next signal fired (exit)
        # ──────────────────────────────────────────────────────────────────────
        
        start_time_dt = pd.to_datetime(start_time, utc=True)
        end_time_dt   = pd.to_datetime(end_time_str, utc=True)
        start_ms      = int(start_time_dt.timestamp() * 1000)
        
        # ── FETCH FUTURE CANDLES ──────────────────────────────────────────────
        # 
        # Get candles from entry time forward.
        # limit=500 → up to 500 candles (5+ days on 15m)
        # 
        # Why 500?
        #   Most trades resolve within 5 days.
        #   If not resolved in 500 candles → trade took too long (skip)
        # ──────────────────────────────────────────────────────────────────────
        
        candles = fetch_forward_candles(symbol, "15m", start_ms, limit=500)
        
        if candles.empty:
            # Fetch failed (network error, API limit, etc)
            # Skip this signal, try again next run
            continue
        
        # ── SLICE TO TRADE WINDOW ─────────────────────────────────────────────
        # 
        # We only care about candles WITHIN the trade window.
        # Candles after exit time are irrelevant.
        # 
        # Filter: timestamp <= end_time_dt
        # ──────────────────────────────────────────────────────────────────────
        
        wave_candles = candles[candles["timestamp"] <= end_time_dt]
        
        if wave_candles.empty:
            # No candles in trade window (shouldn't happen)
            continue
        
        # ══════════════════════════════════════════════════════════════════════
        #                              FIND EXTREMES
        # ══════════════════════════════════════════════════════════════════════
        # 
        # Find the HIGHEST and LOWEST prices during the trade window.
        # Also find WHEN these extremes occurred.
        # 
        # idxmax() → index (row number) of maximum value
        # idxmin() → index (row number) of minimum value
        # ══════════════════════════════════════════════════════════════════════
        
        idx_high = wave_candles["high"].idxmax()  # Row with highest price
        idx_low  = wave_candles["low"].idxmin()   # Row with lowest price
        
        max_price = float(wave_candles.loc[idx_high, "high"])
        min_price = float(wave_candles.loc[idx_low, "low"])
        
        time_of_max = wave_candles.loc[idx_high, "timestamp"]
        time_of_min = wave_candles.loc[idx_low, "timestamp"]
        
        # ══════════════════════════════════════════════════════════════════════
        #                              CALCULATE MOVES
        # ══════════════════════════════════════════════════════════════════════
        # 
        # Percentage move from entry price to extreme prices.
        # 
        # max_move_up_pct:
        #   How far price went UP from entry.
        #   Formula: ((max - entry) / entry) × 100
        #   Example: Entry $1000, Max $1050 → 5% up
        # 
        # max_move_down_pct:
        #   How far price went DOWN from entry.
        #   Formula: ((entry - min) / entry) × 100
        #   Example: Entry $1000, Min $950 → 5% down
        # ══════════════════════════════════════════════════════════════════════
        
        max_move_up_pct   = ((max_price - entry_price) / entry_price) * 100.0
        max_move_down_pct = ((entry_price - min_price) / entry_price) * 100.0
        
        # ══════════════════════════════════════════════════════════════════════
        #                              COUNT CANDLES
        # ══════════════════════════════════════════════════════════════════════
        # 
        # How many candles elapsed before max/min was reached?
        # 
        # idx_high is the row number (0-indexed).
        # Candle 0 = entry candle
        # Candle 5 = max reached after 5 candles
        # ══════════════════════════════════════════════════════════════════════
        
        candles_to_max = int(idx_high)
        candles_to_min = int(idx_low)
        
        # ══════════════════════════════════════════════════════════════════════
        #                              PREPARE UPDATE
        # ══════════════════════════════════════════════════════════════════════
        # 
        # Build dict of columns to update in database.
        # 
        # CRITICAL: We are recording PURE RAW DATA.
        # No trade_quality column (no hardcoded R:R assumptions).
        # The model will learn what "good" means from this data.
        # ══════════════════════════════════════════════════════════════════════
        
        updates = {
            "max_price_after":      round(max_price, 8),
            "min_price_after":      round(min_price, 8),
            "max_move_up_pct":      round(max_move_up_pct, 4),
            "max_move_down_pct":    round(max_move_down_pct, 4),
            "time_of_max_price":    time_of_max.isoformat(),
            "time_of_min_price":    time_of_min.isoformat(),
            "candles_to_max_price": candles_to_max,
            "candles_to_min_price": candles_to_min,
            "status":               "analyzed",  # Mark as complete
        }
        
        # ══════════════════════════════════════════════════════════════════════
        #                              UPDATE DATABASE
        # ══════════════════════════════════════════════════════════════════════
        
        ok = update_signal_labels(signal_id, updates)
        
        if ok:
            labeled += 1
            print(
                f"  🏷️  {symbol} id={signal_id}: "
                f"↑{round(max_move_up_pct, 2)}% "
                f"↓{round(max_move_down_pct, 2)}% | "
                f"{candles_to_max}c to high, {candles_to_min}c to low"
            )
        else:
            failed += 1
            log_error(f"Failed to update labels for signal id={signal_id}")
        
        # ── RATE LIMITING ─────────────────────────────────────────────────────
        # 
        # Sleep 0.3 seconds between signals.
        # Why?
        #   - Prevents hammering Binance too fast
        #   - Spreads load over time
        #   - Reduces chance of rate limit errors
        # ──────────────────────────────────────────────────────────────────────
        
        time.sleep(0.3)
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    
    print(f"\nLabeling complete — {labeled} labeled, {skipped} still open, {failed} failed")


# ══════════════════════════════════════════════════════════════════════════════
#                              ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    label_pending_signals()
