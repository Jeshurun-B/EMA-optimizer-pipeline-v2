# test_db.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Manual test script for db.py
# Run this to verify your Supabase connection works end to end.
# DELETE this file or move to tests/ folder after Day 2.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone
from db import insert_signal, fetch_pending, update_signal_labels, fetch_all_labeled

# ── Step 1: build a fake signal row with every column ────────────────────────
# This represents what features.py and signals.py will eventually produce.
# All feature values are fake — we just want to verify the DB round-trip.

fake_signal = {
    "checked_at_utc":            datetime.now(timezone.utc).isoformat(),
    "symbol":                    "TESTUSDT",
    "signal":                    "LONG",
    "price":                     100.0,
    "signal_gap_hours":          None,
    "prev_signal":               None,

    # EMA features
    "ema_fast_ltf":              101.0,
    "ema_slow_ltf":              99.0,
    "ema_fast_slope":            0.05,
    "ema_slow_slope":            0.02,
    "ema_separation":            2.0,
    "price_above_both_emas":     True,
    "crossover_candle_strength": 0.8,

    # Trend features
    "adx_ltf":                   30.0,
    "adx_slope":                 1.5,
    "adx_4h":                    28.0,
    "macd_histogram_ltf":        0.3,
    "macd_histogram_4h":         0.1,

    # HTF features
    "htf_4h_bias":               True,
    "htf_1d_bias":               True,
    "ema_separation_4h":         1.5,
    "rsi_4h":                    58.0,

    # Momentum
    "rsi_ltf":                   62.0,
    "roc_ltf":                   0.4,

    # Volatility
    "atr_ltf":                   1.2,
    "atr_pct":                   1.2,
    "bb_width_ltf":              3.5,
    "price_to_atr":              83.3,

    # Volume
    "volume_ratio":              1.4,
    "volume_trend":              0.2,
    "crossover_volume_ratio":    1.6,

    # Context
    "fear_greed_index":          65,
    "btc_trend_bias":            True,
    "hour_of_day":               14,
    "day_of_week":               2,

    # Trade management
    "swing_high":                103.0,
    "swing_low":                 97.0,
    "atr_stop_distance":         1.8,

    # Labels — None at insert time, filled by labeler later
    "max_adverse_excursion":     None,
    "max_favorable_excursion":   None,
    "trade_quality":             None,
    "expected_move_pct":         None,
    "status":                    "pending",
}

# ── Step 2: insert the fake signal ───────────────────────────────────────────
print("\n--- TEST 1: insert_signal ---")
ok = insert_signal(fake_signal)
print(f"insert_signal returned: {ok}")
print("Go check your Supabase dashboard — you should see a TESTUSDT row")
input("Press Enter when you've confirmed it's there...")

# ── Step 3: fetch pending ─────────────────────────────────────────────────────
print("\n--- TEST 2: fetch_pending ---")
pending = fetch_pending()
print(f"fetch_pending returned {len(pending)} rows")
test_row = next((r for r in pending if r.get("symbol") == "TESTUSDT"), None)
if test_row:
    print(f"Found our test row — id: {test_row['id']}, status: {test_row['status']}")
    signal_id = test_row["id"]
else:
    print("ERROR: TESTUSDT row not found in pending results")
    signal_id = None

# ── Step 4: update labels ─────────────────────────────────────────────────────
print("\n--- TEST 3: update_signal_labels ---")
if signal_id:
    updates = {
        "max_adverse_excursion":   0.5,
        "max_favorable_excursion": 2.1,
        "trade_quality":           1,
        "expected_move_pct":       2.1,
        "status":                  "analyzed",
    }
    ok = update_signal_labels(signal_id, updates)
    print(f"update_signal_labels returned: {ok}")
    print("Go check Supabase — the TESTUSDT row should now show status=analyzed")
    input("Press Enter when confirmed...")

# ── Step 5: fetch all labeled ─────────────────────────────────────────────────
print("\n--- TEST 4: fetch_all_labeled ---")
df = fetch_all_labeled()
print(f"fetch_all_labeled returned DataFrame with {len(df)} rows")
if not df.empty:
    print(df[["symbol", "signal", "status", "trade_quality"]].tail(3))

print("\n--- ALL TESTS DONE ---")
print("Go to Supabase dashboard and DELETE the TESTUSDT row before continuing.")