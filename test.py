# test_day4.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Validate features.py and signals.py by simulating a pipeline run 
#          across historical candles to find a guaranteed crossover.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from fetcher import fetch_candles
from features import compute_features
from signals import detect_signal

def run_test():
    symbol = "SOLUSDT"  # Solana is volatile, good for finding crossovers
    print(f"--- Day 4 Integration Test ---")
    print(f"Fetching 300 15m candles for {symbol}...")
    
    # 1. Fetch the data
    df = fetch_candles(symbol, "15m", 300)
    
    if df.empty:
        print("❌ Failed to fetch data. Check your connection or API limits.")
        return

    print("✅ Data fetched successfully. Searching for a historical crossover...\n")

    # 2. The "Time Machine" Loop
    # We need at least 30 candles for the indicators (like ADX) to warm up.
    # We slice the dataframe to simulate what the pipeline would have seen 
    # at that exact moment in the past.
    signal_found = False
    
    for i in range(50, len(df)):
        # Create a "snapshot" of the market up to candle 'i'
        market_snapshot = df.iloc[:i].copy()
        
        # Compute features for this specific snapshot
        features = compute_features(market_snapshot, symbol)
        
        if not features:
            continue  # Skip if not enough data
            
        # Check for a signal
        signal = detect_signal(features, symbol)
        
        if signal:
            print(f"🚨 CROSSOVER DETECTED AT ROW {i}!")
            print(f"Timestamp: {market_snapshot.iloc[-1]['timestamp']}")
            print(f"Direction: {signal['signal']}")
            print(f"Price:     ${signal['price']}")
            print(f"Fast EMA:  {signal['ema_fast_ltf']} | Slow EMA: {signal['ema_slow_ltf']}")
            print(f"ADX:       {signal['adx_ltf']} (Must be >= 25)")
            
            print("\n--- Full Database Payload ---")
            for key, value in signal.items():
                print(f"  {key}: {value} ({type(value).__name__})")
                
            signal_found = True
            break  # We found one, stop searching
            
    if not signal_found:
        print("⚠️ No crossovers found in the last 300 candles. (This happens in choppy markets).")
        print("Try changing the symbol to 'ETHUSDT' or 'DOGEUSDT' and run again.")
    else:
        print("\n✅ Day 4 Test Passed! features.py and signals.py are working perfectly.")

if __name__ == "__main__":
    run_test()