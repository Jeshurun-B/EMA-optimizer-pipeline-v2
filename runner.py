# runner.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: Main pipeline orchestrator. Ties everything together.
#
# WORKFLOW:
#   1. Fetch market context (Fear & Greed, BTC bias) ONCE per run
#   2. For each coin in COINS:
#      a. Fetch 15m, 4h, 1d candles
#      b. Compute 37 features
#      c. Detect crossover signal
#      d. Insert to Supabase if signal found
#   3. Handle API limits gracefully
#   4. Log all errors, continue to next coin
#
# This file runs on GitHub Actions every 4 hours.
# ═════════════════════════════════════════════════════════════════════════════

from config import COINS, EMA_FAST, EMA_SLOW
from fetcher import fetch_candles, fetch_fear_greed
from features import compute_features
from signals import detect_signal
from db import insert_signal
from utils import log_error


# ══════════════════════════════════════════════════════════════════════════════
#                              HELPER: GET BTC BIAS
# ══════════════════════════════════════════════════════════════════════════════

def _get_btc_bias() -> bool:
    """
    Calculate overall market trend from BTC 4h chart.
    
    WHY THIS EXISTS:
        BTC is the market leader — when BTC trends up, altcoins often follow.
        We use BTC's 4h trend as a "macro context" feature for ALL coins.
        
        Model might learn:
            "LONG signals on altcoins work better when BTC 4h is bullish"
            "SHORT signals when BTC 4h is bearish are more reliable"
    
    CALCULATION:
        Simple 9/15 EMA crossover on BTC 4h chart.
        If 9 EMA > 15 EMA → bullish bias (True)
        If 9 EMA < 15 EMA → bearish bias (False)
    
    Returns:
        True:  BTC 4h trend is bullish
        False: BTC 4h trend is bearish
        True (default): If fetch fails (safer to assume bullish than crash)
    
    Why default to True?
        Crypto has upward bias long-term.
        If BTC data unavailable, assuming bullish is reasonable.
        Better than crashing the entire pipeline.
    """
    
    # Fetch 50 candles of BTC 4h data
    # Why 50? Need at least 15 candles for EMA(15) to warm up.
    # 50 gives safety margin.
    btc_4h = fetch_candles("BTCUSDT", "4h", 50)
    
    # ── HANDLE FETCH FAILURE ──────────────────────────────────────────────────
    # 
    # If fetch failed (network error, API limit, etc):
    #   - btc_4h.empty will be True
    #   - Can't calculate EMAs from empty data
    #   - Return True (default bullish assumption)
    # ──────────────────────────────────────────────────────────────────────────
    
    if btc_4h.empty or len(btc_4h) < EMA_SLOW:
        return True  # Default to bullish if can't fetch
    
    # ── CALCULATE EMAS ────────────────────────────────────────────────────────
    # 
    # .ewm() = exponential weighted moving average
    # span=EMA_FAST (9) → fast EMA
    # span=EMA_SLOW (15) → slow EMA
    # adjust=False → matches Pine Script and standard TA library behavior
    # .mean() → actually compute the average
    # .iloc[-1] → take the latest value (most recent candle)
    # ──────────────────────────────────────────────────────────────────────────
    
    fast = btc_4h["close"].ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    slow = btc_4h["close"].ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
    
    # ── RETURN BIAS ───────────────────────────────────────────────────────────
    # 
    # True if fast > slow (bullish)
    # False if fast <= slow (bearish)
    # 
    # We cast to bool explicitly for clarity (though > already returns bool)
    # ──────────────────────────────────────────────────────────────────────────
    
    return bool(fast > slow)


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """
    Execute the Scan Phase of the pipeline for all coins.
    
    PHASE 1: MARKET CONTEXT (once per run)
        - Fetch Fear & Greed Index
        - Calculate BTC 4h bias
        - These are shared across all coins (save API calls)
    
    PHASE 2: COIN SCANNING (loop through each coin)
        For each symbol in COINS:
            1. Fetch 15m, 4h, 1d candles
            2. Compute 37 features
            3. Detect crossover signal
            4. Insert to database if signal found
    
    ERROR HANDLING:
        - One coin failing doesn't stop the entire run
        - Every step wrapped in try/except
        - Errors logged, pipeline continues to next coin
    
    API EFFICIENCY:
        - F&G fetched once (not per coin)
        - BTC bias calculated once (not per coin)
        - Each coin: 3 API calls (15m, 4h, 1d)
        - 5 coins = 15 calls total (well under 200 limit)
    
    OUTPUT:
        Console logs showing:
            - Market context
            - Progress per coin
            - Whether signals were found
            - Success/failure of database inserts
    """
    
    print("🚀 Starting Pipeline Scan...\n")
    
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              PHASE 1: MARKET CONTEXT
    # ══════════════════════════════════════════════════════════════════════════
    # 
    # Fetch these ONCE at the start of the run.
    # All coins share the same F&G and BTC bias.
    # 
    # Why fetch once?
    #   - Saves API calls (1 F&G call vs 5)
    #   - Faster execution
    #   - Consistent — all coins see same market context
    # ══════════════════════════════════════════════════════════════════════════
    
    print("🌍 Fetching market context...")
    
    # Fetch Fear & Greed Index
    # Returns int 0-100 or None on failure
    fgi = fetch_fear_greed()
    
    # Calculate BTC market bias
    # Returns bool (True=bullish, False=bearish)
    btc_bias = _get_btc_bias()
    
    # Print context for logging/debugging
    print(f"   Fear & Greed: {fgi if fgi is not None else 'N/A (using 50)'}")
    print(f"   BTC 4H Bias: {'BULLISH ↑' if btc_bias else 'BEARISH ↓'}\n")
    
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              PHASE 2: COIN SCANNING
    # ══════════════════════════════════════════════════════════════════════════
    
    for symbol in COINS:
        print(f"🔍 Scanning {symbol}...")
        
        try:
            # ── STEP 1: FETCH ALL 3 TIMEFRAMES ───────────────────────────────
            # 
            # We need data from 3 different timeframes:
            #   - 15m (300 candles): Where crossovers happen
            #   - 4h (200 candles):  Trend alignment context
            #   - 1d (100 candles):  Market regime context
            # 
            # Why these limits?
            #   - 300 × 15m = 3.125 days (enough for indicator warmup)
            #   - 200 × 4h = 33 days (captures recent trend shifts)
            #   - 100 × 1d = 100 days (captures longer-term regime)
            # ──────────────────────────────────────────────────────────────────
            
            df_15m = fetch_candles(symbol, "15m", 300)
            df_4h  = fetch_candles(symbol, "4h", 200)
            df_1d  = fetch_candles(symbol, "1d", 100)
            
            # ── STEP 2: VALIDATE DATA ────────────────────────────────────────
            # 
            # If ANY timeframe fetch failed → skip this coin
            # Can't compute features without all 3 timeframes
            # ──────────────────────────────────────────────────────────────────
            
            if df_15m.empty or df_4h.empty or df_1d.empty:
                print(f"  ⚠️  Missing timeframe data, skipping.")
                continue  # Move to next coin
            
            # ── STEP 3: COMPUTE FEATURES ──────────────────────────────────────
            # 
            # compute_features() returns:
            #   - dict with 37 features + 2 private keys
            #   - None if insufficient data
            # 
            # We pass:
            #   - All 3 timeframe DataFrames
            #   - Fear & Greed (or None)
            #   - BTC bias (True/False)
            #   - Symbol (for error logging)
            # ──────────────────────────────────────────────────────────────────
            
            features = compute_features(
                df_15m,      # LTF (where signals happen)
                df_4h,       # HTF for trend alignment
                df_1d,       # HTF for market regime
                fgi,         # Market sentiment (0-100 or None)
                btc_bias,    # BTC trend (True=bull, False=bear)
                symbol       # For error logging
            )
            
            if not features:
                print(f"  ⚠️  Feature computation failed, skipping.")
                continue
            
            # ── STEP 4: DETECT CROSSOVER ─────────────────────────────────────
            # 
            # detect_signal() returns:
            #   - dict (complete signal record) if crossover found
            #   - None if no crossover
            # 
            # The dict includes:
            #   - All features (35 of them)
            #   - Signal metadata (symbol, timestamp, direction)
            #   - Previous signal info (for gap calculation)
            # ──────────────────────────────────────────────────────────────────
            
            signal_record = detect_signal(features, symbol)
            
            # ── STEP 5: INSERT TO DATABASE ────────────────────────────────────
            # 
            # If signal found → insert to Supabase
            # If no signal → log and move to next coin
            # ──────────────────────────────────────────────────────────────────
            
            if signal_record:
                # Signal detected! Print which direction
                print(f"  🚨 {signal_record['signal']} SIGNAL DETECTED!")
                
                # Try to insert to database
                success = insert_signal(signal_record)
                
                if success:
                    print(f"  ✅ Successfully saved to Supabase.")
                else:
                    print(f"  ❌ Database insert failed (see error.log)")
            
            else:
                # No crossover on this scan
                print(f"  😴 No crossover.")
        
        except Exception as e:
            # ── CATCH-ALL ERROR HANDLER ───────────────────────────────────────
            # 
            # If ANYTHING goes wrong processing this coin:
            #   - Log the error
            #   - Continue to next coin (don't crash entire pipeline)
            # 
            # Possible errors:
            #   - Unexpected data format from Binance
            #   - Math error in indicators
            #   - Type mismatch
            #   - Memory error (unlikely)
            # ──────────────────────────────────────────────────────────────────
            
            log_error(f"Error processing {symbol} in runner: {repr(e)}")
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              SCAN COMPLETE
    # ══════════════════════════════════════════════════════════════════════════
    
    print("\n✅ Scan complete.")


# ══════════════════════════════════════════════════════════════════════════════
#                              ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
# 
# When this file is run directly (not imported):
#   python runner.py
# 
# Execute the main pipeline function.
# 
# On GitHub Actions, the workflow calls this file.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline()
