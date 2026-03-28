# labeler.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: The Post-Analysis Engine. This script looks at the 'future' 
#          (candles that happened AFTER our signal) to determine if a 
#          trade was a winner or a loser.
# RULE: This script only processes rows where status='pending'.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from datetime import timezone
from config import RR_TARGET_MULTIPLE, RR_STOP_MULTIPLE
from db import fetch_pending, update_signal_labels
from fetcher import fetch_forward_candles
from utils import log_error
from config import BINANCE_BASE_URL, REQUEST_TIMEOUT
import requests


def label_pending_signals():
    """
    Main loop to find pending signals, fetch their forward-looking price action,
    calculate MAE/MFE, determine trade quality, and save the labels to the DB.
    """
    print("🔎 Starting Post-Analysis Labeling...\n")
    
    # 1. Fetch all signals that haven't been analyzed yet
    pending_signals = fetch_pending()
    
    if not pending_signals:
        print("✅ No pending signals to label.")
        return
        
    print(f"Found {len(pending_signals)} pending signals to process.")
    
    # 2. Process each signal one by one
    for row in pending_signals:
        signal_id = row["id"]
        symbol = row["symbol"]
        signal_dir = row["signal"]  # "LONG" or "SHORT"
        entry_price = float(row["price"])
        
        # The ATR distance tells us how wide our stop loss and take profit should be
        atr_stop_dist = float(row.get("atr_stop_distance", 0.0))
        
        # Safety check: if ATR is missing or 0, we can't calculate risk tiers
        if atr_stop_dist <= 0:
            print(f"⚠️ Row {signal_id} missing ATR stop distance. Skipping.")
            continue
            
        # Parse the exact time the signal fired, convert to milliseconds for Binance API
        signal_time = pd.to_datetime(row["checked_at_utc"])
        start_time_ms = int(signal_time.timestamp() * 1000)
        
        # 3. Fetch the next 100 candles (15m * 100 = 25 hours of price action)
        # This gives the trade 24+ hours to play out and hit a target/stop.
        future_df = fetch_forward_candles(symbol, "15m", start_time_ms, limit=100)
        
        if future_df.empty:
            print(f"⚠️ No future data available yet for {symbol} at {signal_time}. Skipping.")
            continue
            
        # 4. Define our exact price levels for success (Target) and failure (Stop)
        if signal_dir == "LONG":
            target_price = entry_price + (atr_stop_dist * RR_TARGET_MULTIPLE)
            stop_price = entry_price - (atr_stop_dist * RR_STOP_MULTIPLE)
        else: # SHORT
            target_price = entry_price - (atr_stop_dist * RR_TARGET_MULTIPLE)
            stop_price = entry_price + (atr_stop_dist * RR_STOP_MULTIPLE)

        # 5. Initialize tracking variables
        trade_quality = 0      # Default to a losing trade (0)
        max_favorable = 0.0    # Highest profit achieved
        max_adverse = 0.0      # Deepest drawdown suffered
        expected_move_pct = 0.0
        
        # 6. The Candle-by-Candle Simulation
        # We must iterate in chronological order to see which level price hits FIRST.
        # If it hits the stop loss on candle 3, we don't care if it hits the target on candle 10.
        for index, candle in future_df.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            
            # --- LONG LOGIC ---
            if signal_dir == "LONG":
                # Track extreme moves relative to entry
                current_favorable = high - entry_price
                current_adverse = entry_price - low
                
                # Update absolute maximums
                if current_favorable > max_favorable:
                    max_favorable = current_favorable
                if current_adverse > max_adverse:
                    max_adverse = current_adverse
                    
                # Did we hit the stop loss?
                if low <= stop_price:
                    trade_quality = 0
                    break # Trade is over, stop checking future candles
                    
                # Did we hit the take profit target?
                if high >= target_price:
                    trade_quality = 1
                    break # Trade is over, target achieved
                    
            # --- SHORT LOGIC ---
            else: 
                # Track extreme moves relative to entry (inverted for shorts)
                current_favorable = entry_price - low
                current_adverse = high - entry_price
                
                if current_favorable > max_favorable:
                    max_favorable = current_favorable
                if current_adverse > max_adverse:
                    max_adverse = current_adverse
                    
                # Did we hit the stop loss? (Price goes UP)
                if high >= stop_price:
                    trade_quality = 0
                    break # Trade is over
                    
                # Did we hit the take profit? (Price goes DOWN)
                if low <= target_price:
                    trade_quality = 1
                    break # Trade is over

        # 7. Calculate Final Expected Move Percentage
        # This tells us the maximum % gain the trade offered before it ended
        expected_move_pct = (max_favorable / entry_price) * 100.0 if entry_price > 0 else 0.0

        # 8. Prepare the update payload for the database
        updates = {
            "max_adverse_excursion": round(max_adverse, 6),
            "max_favorable_excursion": round(max_favorable, 6),
            "expected_move_pct": round(expected_move_pct, 4),
            "trade_quality": trade_quality,
            "status": "analyzed" # Mark as analyzed so we don't process it again
        }
        
        # 9. Send the updates to Supabase
        success = update_signal_labels(signal_id, updates)
        
        if success:
            print(f"  🏷️ Labeled row {signal_id} ({symbol} {signal_dir}): Quality={trade_quality}, Move={round(expected_move_pct,2)}%")
        else:
            print(f"  ❌ Failed to update row {signal_id} in DB.")

    print("\n✅ Labeling pass complete.")

if __name__ == "__main__":
    label_pending_signals()