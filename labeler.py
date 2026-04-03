# labeler.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Label pending signals with trade outcome data.
#          Fetches post-signal candles and computes MAE, MFE, trade_quality.
# RULE: Only processes rows where status='pending'. Never re-labels analyzed rows.
# ─────────────────────────────────────────────────────────────────────────────

import time
import pandas as pd
from config import RR_TARGET_MULTIPLE, RR_STOP_MULTIPLE
from db import fetch_pending, update_signal_labels
from fetcher import fetch_forward_candles
from utils import log_error, api_limit_reached


def label_pending_signals():
    
    """
    Fetch all pending signals, compute outcome labels, update Supabase.

    For each pending signal:
        1. Fetch 200 candles after the signal time (= ~50 hours on 15m)
        2. Simulate the trade candle by candle
        3. Record MAE, MFE, trade_quality, expected_move_pct
        4. Update the row in Supabase and set status=analyzed
    """
    pending = fetch_pending()

    if not pending:
        print("No pending signals to label.")
        return

    print(f"Labeling {len(pending)} pending signals...\n")
    labeled = 0
    failed  = 0

    for row in pending:
        if api_limit_reached():
            print(f"API limit reached — stopping at {labeled} labeled")
            break

        signal_id     = row.get("id")
        symbol        = row.get("symbol")
        signal_dir    = str(row.get("signal", "LONG")).upper()
        entry_price   = float(row.get("price") or 0)
        atr_stop_dist = float(row.get("atr_stop_distance") or 0)
        checked_at    = row.get("checked_at_utc")

        if not signal_id or not symbol or not checked_at:
            continue

        if atr_stop_dist <= 0 or entry_price <= 0:
            # Can't compute R:R without these — mark as analyzed with quality=0
            update_signal_labels(signal_id, {
                "trade_quality": 0,
                "status": "analyzed"
            })
            continue

        # Convert signal time to milliseconds for Binance startTime parameter
        try:
            start_ms = int(pd.to_datetime(checked_at).timestamp() * 1000)
        except Exception:
            continue

        # Fetch post-signal candles
        candles = fetch_forward_candles(symbol, "15m", start_ms, limit=200)

        if candles.empty:
            print(f"  No post-signal candles for {symbol} id={signal_id} — skipping")
            continue

        # Calculate target and stop price levels
        if signal_dir == "LONG":
            target_price = entry_price + (atr_stop_dist * RR_TARGET_MULTIPLE)
            stop_price   = entry_price - (atr_stop_dist * RR_STOP_MULTIPLE)
        else:
            target_price = entry_price - (atr_stop_dist * RR_TARGET_MULTIPLE)
            stop_price   = entry_price + (atr_stop_dist * RR_STOP_MULTIPLE)

        # Iterate candle by candle to find which level was hit first
        max_adverse   = 0.0
        max_favorable = 0.0
        trade_quality = 0  # default: bad trade

        for _, candle in candles.iterrows():
            high = float(candle["high"])
            low  = float(candle["low"])

            if signal_dir == "LONG":
                adverse   = max(0.0, entry_price - low)
                favorable = max(0.0, high - entry_price)

                if high >= target_price:
                    trade_quality = 1
                    break
                if low <= stop_price:
                    trade_quality = 0
                    break
            else:
                adverse   = max(0.0, high - entry_price)
                favorable = max(0.0, entry_price - low)

                if low <= target_price:
                    trade_quality = 1
                    break
                if high >= stop_price:
                    trade_quality = 0
                    break

            max_adverse   = max(max_adverse, adverse)
            max_favorable = max(max_favorable, favorable)

        # Convert absolute price moves to percentages
        mae_pct           = max_adverse   / entry_price * 100 if entry_price > 0 else 0.0
        mfe_pct           = max_favorable / entry_price * 100 if entry_price > 0 else 0.0
        expected_move_pct = mfe_pct

        updates = {
            "max_adverse_excursion":   round(mae_pct, 6),
            "max_favorable_excursion": round(mfe_pct, 6),
            "expected_move_pct":       round(expected_move_pct, 6),
            "trade_quality":           trade_quality,
            "status":                  "analyzed",
        }

        ok = update_signal_labels(signal_id, updates)
        if ok:
            labeled += 1
            print(f"  Labeled {symbol} id={signal_id}: quality={trade_quality} MFE={round(mfe_pct,2)}%")
        else:
            failed += 1
            log_error(f"Failed to update labels for signal id={signal_id}")

        time.sleep(0.1)  # Be gentle with Binance API

    print(f"\nLabeling complete — {labeled} labeled, {failed} failed")


if __name__ == "__main__":
    label_pending_signals()