# labeler.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Label pending signals using exact Database Timestamps.
#          Calculates MAE (Pain), MFE (Profit), AND the exact time in hours 
#          it took to reach those extreme points.
# ─────────────────────────────────────────────────────────────────────────────

import time
import pandas as pd
from db import fetch_pending, update_signal_labels, fetch_next_signal_time
from fetcher import fetch_forward_candles
from utils import log_error, api_limit_reached

def label_pending_signals():
    """
    Fetch pending signals, find their exit timestamp, and compute outcome labels
    including the time taken to reach maximum excursions.
    """
    pending = fetch_pending()

    if not pending:
        print("No pending signals to label.")
        return

    print(f"Labeling {len(pending)} pending signals using Time-Window logic...\n")
    labeled = 0
    skipped = 0
    failed  = 0

    for row in pending:
        # 1. API Safety Check
        if api_limit_reached():
            print(f"API limit reached — stopping early.")
            break

        # 2. Extract Data from Database Row
        signal_id   = row.get("id")
        symbol      = row.get("symbol")
        signal_dir  = str(row.get("signal", "LONG")).upper()
        entry_price = float(row.get("price") or 0)
        start_time  = row.get("checked_at_utc")

        if not signal_id or not symbol or not start_time or entry_price <= 0:
            continue

        # 3. Ask the DB: When is the exact moment this trade ended?
        end_time_str = fetch_next_signal_time(symbol, start_time)

        # If there is no "next signal", the trade is still live in the market.
        if not end_time_str:
            skipped += 1
            print(f"  ⏳ Trade {symbol} id={signal_id} is still open. Waiting for next signal.")
            continue

        # 4. Convert timestamps for Binance and Pandas
        start_time_dt = pd.to_datetime(start_time, utc=True)
        end_time_dt   = pd.to_datetime(end_time_str, utc=True)
        start_ms      = int(start_time_dt.timestamp() * 1000)

        # 5. Fetch a large chunk of future candles (Limit=500 -> ~5 days on 15m)
        candles = fetch_forward_candles(symbol, "15m", start_ms, limit=500)

        if candles.empty:
            continue

        # 6. SLICE THE DATA: Keep only the candles between Entry and Exit
        wave_candles = candles[candles["timestamp"] <= end_time_dt]

        if wave_candles.empty:
            continue

        # 7. VECTORIZED MATH: Find the extremes AND when they happened
        # .idxmax() and .idxmin() return the index of the row where the extreme occurred
        idx_high = wave_candles["high"].idxmax()
        idx_low  = wave_candles["low"].idxmin()

        highest_price = float(wave_candles.loc[idx_high, "high"])
        lowest_price  = float(wave_candles.loc[idx_low, "low"])

        highest_time = wave_candles.loc[idx_high, "timestamp"]
        lowest_time  = wave_candles.loc[idx_low, "timestamp"]

        # 8. Calculate exactly how many hours it took to hit those prices
        hours_to_high = (highest_time - start_time_dt).total_seconds() / 3600.0
        hours_to_low  = (lowest_time - start_time_dt).total_seconds() / 3600.0

        # Prevent negative zero times if the extreme happened on the entry candle itself
        hours_to_high = max(0.0, hours_to_high)
        hours_to_low  = max(0.0, hours_to_low)

        # 9. Map Excursions and Time based on Trade Direction
        if signal_dir == "LONG":
            max_adverse   = max(0.0, entry_price - lowest_price)
            max_favorable = max(0.0, highest_price - entry_price)
            # For LONG: High is Favorable, Low is Adverse
            time_to_mfe_hrs = hours_to_high
            time_to_mae_hrs = hours_to_low
        else: # SHORT
            max_adverse   = max(0.0, highest_price - entry_price)
            max_favorable = max(0.0, entry_price - lowest_price)
            # For SHORT: Low is Favorable, High is Adverse
            time_to_mfe_hrs = hours_to_low
            time_to_mae_hrs = hours_to_high

        # 10. Convert to Percentages
        mae_pct           = (max_adverse / entry_price) * 100.0 if entry_price > 0 else 0.0
        mfe_pct           = (max_favorable / entry_price) * 100.0 if entry_price > 0 else 0.0
        expected_move_pct = mfe_pct

        # 11. Define Trade Quality (1 = Profit > Pain)
        trade_quality = 1 if mfe_pct > mae_pct else 0

        # 12. Prepare payload for Supabase
        updates = {
            "max_adverse_excursion":   round(mae_pct, 6),
            "max_favorable_excursion": round(mfe_pct, 6),
            "expected_move_pct":       round(expected_move_pct, 6),
            "time_to_mae_hrs":         round(time_to_mae_hrs, 2), # NEW
            "time_to_mfe_hrs":         round(time_to_mfe_hrs, 2), # NEW
            "trade_quality":           trade_quality,
            "status":                  "analyzed",
        }

        # 13. Execute Database Update
        ok = update_signal_labels(signal_id, updates)
        if ok:
            labeled += 1
            print(f"  🏷️ Labeled {symbol} id={signal_id}: Q={trade_quality} | MFE={round(mfe_pct,2)}% ({round(time_to_mfe_hrs, 1)}h) | MAE={round(mae_pct,2)}% ({round(time_to_mae_hrs, 1)}h)")
        else:
            failed += 1
            log_error(f"Failed to update labels for signal id={signal_id}")

        time.sleep(0.1)  # Rate limit safety

    print(f"\nLabeling complete — {labeled} labeled, {skipped} still open, {failed} failed")

if __name__ == "__main__":
    label_pending_signals()