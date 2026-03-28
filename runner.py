# runner.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Orchestrate the pipeline. Ties fetcher, features, signals, and db together.
# ─────────────────────────────────────────────────────────────────────────────

from config import COINS, EMA_FAST, EMA_SLOW
from fetcher import fetch_candles, fetch_fear_greed
from features import compute_features
from signals import detect_signal
from db import insert_signal
from utils import log_error

def _get_btc_bias() -> bool:
    """Helper to get overall market trend (BTC 4h EMA bias) to pass to features."""
    btc_4h = fetch_candles("BTCUSDT", "4h", 50)
    if btc_4h.empty or len(btc_4h) < EMA_SLOW:
        return True # Default to True if we can't fetch it
    
    # Calculate simple EMA to find bias
    fast = btc_4h["close"].ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    slow = btc_4h["close"].ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
    return bool(fast > slow)

def run_pipeline():
    """Executes the Scan Phase of the pipeline for all coins."""
    print("🚀 Starting Pipeline Scan...\n")
    
    # 1. Fetch Context Data ONCE per run (Saves API calls)
    print("🌍 Fetching market context...")
    fgi = fetch_fear_greed()
    btc_bias = _get_btc_bias()
    print(f"   Fear & Greed: {fgi} | BTC 4H Bias: {'BULLISH' if btc_bias else 'BEARISH'}\n")
    
    for symbol in COINS:
        print(f"🔍 Scanning {symbol}...")
        try:
            # 2. Fetch all 3 timeframes
            df_15m = fetch_candles(symbol, "15m", 300)
            df_4h  = fetch_candles(symbol, "4h", 200)
            df_1d  = fetch_candles(symbol, "1d", 100)
            
            if df_15m.empty or df_4h.empty or df_1d.empty:
                print(f"  ⚠️ Missing timeframe data, skipping.")
                continue
                
            # 3. Compute ALL 35 features
            features = compute_features(df_15m, df_4h, df_1d, fgi, btc_bias, symbol)
            
            if not features:
                print(f"  ⚠️ Not enough data to compute features, skipping.")
                continue
                
            # 4. Check for a crossover signal
            signal_record = detect_signal(features, symbol)
            
            # 5. Store if found
            if signal_record:
                print(f"  🚨 {signal_record['signal']} SIGNAL DETECTED!")
                success = insert_signal(signal_record)
                if success:
                    print(f"  ✅ Successfully saved to Supabase.")
                else:
                    print(f"  ❌ Database insert failed.")
            else:
                print(f"  😴 No crossover.")
                
        except Exception as e:
            log_error(f"Error processing {symbol} in runner: {repr(e)}")

    print("\n✅ Scan complete.")

if __name__ == "__main__":
    run_pipeline()

    