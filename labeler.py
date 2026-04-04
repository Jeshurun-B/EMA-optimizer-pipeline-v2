# labeler.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Label pending signals by recording RAW post-signal price movement.
#          No trade simulation. No hardcoded R:R. Just facts about what happened.
# ─────────────────────────────────────────────────────────────────────────────

import time
import pandas as pd
from db import fetch_pending, update_signal_labels, fetch_next_signal_time
from fetcher import fetch_forward_candles
from utils import log_error, api_limit_reached


def label_pending_signals():
    """
    Fetch pending signals, find their exit timestamp, and record raw price movement.
    """
    pending = fetch_pending()

    if not pending:
        print("No pending signals to label.")
        return

    print(f"Labeling {len(pending)} pending signals using raw data collection...\n")
    labeled = 0
    skipped = 0
    failed  = 0

    for row in pending:
        # API Safety Check
        if api_limit_reached():
            print(f"API limit reached — stopping early.")
            break

        # Extract data from database row
        signal_id   = row.get("id")
        symbol      = row.get("symbol")
        signal_dir  = str(row.get("signal", "LONG")).upper()
        entry_price = float(row.get("price") or 0)
        start_time  = row.get("checked_at_utc")

        if not signal_id or not symbol or not start_time or entry_price <= 0:
            continue

        # Ask DB: when did this trade end (when was next signal)?
        end_time_str = fetch_next_signal_time(symbol, start_time)

        if not end_time_str:
            skipped += 1
            print(f"  ⏳ {symbol} id={signal_id} still open, waiting for next signal...")
            continue

        # Convert timestamps
        start_time_dt = pd.to_datetime(start_time, utc=True)
        end_time_dt   = pd.to_datetime(end_time_str, utc=True)
        start_ms      = int(start_time_dt.timestamp() * 1000)

        # Fetch future candles (limit=500 -> ~5 days on 15m)
        candles = fetch_forward_candles(symbol, "15m", start_ms, limit=500)

        if candles.empty:
            continue

        # Slice to exact time window (entry to exit)
        wave_candles = candles[candles["timestamp"] <= end_time_dt]

        if wave_candles.empty:
            continue

        # Find extremes and when they happened
        idx_high = wave_candles["high"].idxmax()
        idx_low  = wave_candles["low"].idxmin()

        max_price = float(wave_candles.loc[idx_high, "high"])
        min_price = float(wave_candles.loc[idx_low, "low"])

        time_of_max = wave_candles.loc[idx_high, "timestamp"]
        time_of_min = wave_candles.loc[idx_low, "timestamp"]

        # Calculate raw percentage moves from entry
        max_move_up_pct   = ((max_price - entry_price) / entry_price) * 100.0
        max_move_down_pct = ((entry_price - min_price) / entry_price) * 100.0

        # Count candles to each extreme
        candles_to_max = int(idx_high)
        candles_to_min = int(idx_low)

        # Prepare update payload — PURE RAW DATA
        updates = {
            "max_price_after":      round(max_price, 8),
            "min_price_after":      round(min_price, 8),
            "max_move_up_pct":      round(max_move_up_pct, 4),
            "max_move_down_pct":    round(max_move_down_pct, 4),
            "time_of_max_price":    time_of_max.isoformat(),
            "time_of_min_price":    time_of_min.isoformat(),
            "candles_to_max_price": candles_to_max,
            "candles_to_min_price": candles_to_min,
            "status":               "analyzed",
        }

        # Execute database update
        ok = update_signal_labels(signal_id, updates)
        if ok:
            labeled += 1
            print(f"  🏷️  {symbol} id={signal_id}: ↑{round(max_move_up_pct,2)}% ↓{round(max_move_down_pct,2)}% | {candles_to_max}c to high, {candles_to_min}c to low")
        else:
            failed += 1
            log_error(f"Failed to update labels for signal id={signal_id}")

        time.sleep(0.3)

    print(f"\nLabeling complete — {labeled} labeled, {skipped} still open, {failed} failed")


if __name__ == "__main__":
    label_pending_signals()