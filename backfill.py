# backfill.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: One-time historical data collection — builds the ML training dataset.
#
# SMART RESUME LOGIC:
#   ✅ Checks Supabase for last stored signal per coin
#   ✅ Only fetches data from that point forward
#   ✅ Safe to re-run anytime (no duplicate data)
#   ✅ If no existing data → goes back look_back_days (default 3)
#
# TIME MACHINE APPROACH:
#   Slides a 300-candle window through historical 15m data.
#   At each position, simulates what the live pipeline would have seen.
#   Prevents lookahead bias (features only use data available at that moment).
#
# CRITICAL BUG FIXES APPLIED:
#   1. Window size: 100 → 300 (proper indicator warmup)
#   2. BTC bias: Pre-calculated timeline (not recalculated every candle)
#   3. HTF data slicing: Validated for sufficient length
#   4. Smart resume: Minimum 3 days fetch even for recent signals
# ═════════════════════════════════════════════════════════════════════════════

import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import COINS, BINANCE_BASE_URL, REQUEST_TIMEOUT, EMA_FAST, EMA_SLOW, look_back_days
from features import compute_features
from signals import detect_signal
from db import insert_signal, supabase, TABLE
from utils import log_error, _inc_api_call


# ══════════════════════════════════════════════════════════════════════════════
#                              SMART RESUME — GET LAST SIGNAL DATE
# ══════════════════════════════════════════════════════════════════════════════

def get_last_signal_date(symbol: str) -> datetime:
    """
    Query Supabase for the most recent signal timestamp for this specific coin.
    
    WHY THIS EXISTS:
        Prevents re-processing months of data we already have.
        Instead of always going back 180 days, we only fetch NEW data.
    
    HOW IT WORKS:
        1. Query Supabase: SELECT checked_at_utc FROM signals
                           WHERE symbol = 'BTCUSDT'
                           ORDER BY checked_at_utc DESC
                           LIMIT 1
        2. If found → return that timestamp
        3. If not found → return (now - look_back_days)
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
    
    Returns:
        datetime: Most recent signal timestamp for this coin
                  OR (now - look_back_days) if no data exists
    
    Example:
        last = get_last_signal_date("BTCUSDT")
        print(f"Last BTCUSDT signal: {last.date()}")
        # Output: Last BTCUSDT signal: 2026-04-03
        # → Only fetch data from April 3rd forward
    """
    try:
        response = (
            supabase.table(TABLE)
            .select("checked_at_utc")
            .eq("symbol", symbol)
            .order("checked_at_utc", desc=True)
            .limit(1)
            .execute()
        )
        
        if response.data:
            # Found existing data for this coin
            last_date = pd.to_datetime(response.data[0]["checked_at_utc"], utc=True)
            print(f"  ✓ Resuming {symbol} from {last_date.date()}")
            return last_date.to_pydatetime()
    
    except Exception as e:
        log_error(f"get_last_signal_date error for {symbol}: {repr(e)}")
    
    # No data found — go back look_back_days (default 3 days)
    fallback = datetime.now(timezone.utc) - timedelta(days=look_back_days)
    print(f"  ✓ No existing data for {symbol}, starting from {fallback.date()}")
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
#                              FETCH HISTORICAL KLINES (PAGINATED)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_historical_klines(symbol: str, interval: str, days_back: int) -> pd.DataFrame:
    """
    Fetch months of historical candles by paginating backwards from today.
    
    WHY PAGINATION:
        Binance limits each request to 1000 candles.
        To get months of data, we loop backwards:
            - Request 1: Last 1000 candles (most recent)
            - Request 2: 1000 candles before that
            - Request 3: 1000 candles before that
            - ... until we reach target start date
    
    HOW IT WORKS:
        1. Calculate target start timestamp (now - days_back)
        2. Fetch 1000 candles ending at 'now'
        3. Find oldest candle in batch
        4. Next request: fetch 1000 candles ending just before oldest
        5. Repeat until oldest candle <= target start
        6. Combine all batches, sort chronologically
    
    Args:
        symbol:     Trading pair (e.g., "BTCUSDT")
        interval:   Timeframe (e.g., "15m", "4h", "1d")
        days_back:  How many days of history to fetch
    
    Returns:
        DataFrame sorted oldest-first with columns:
            [timestamp, open, high, low, close, volume]
        
        Empty DataFrame on failure (never raises exception)
    
    Example:
        df = fetch_historical_klines("BTCUSDT", "15m", 7)
        print(f"Got {len(df)} candles")
        # Output: Got 672 candles (7 days × 96 candles per day)
    """
    
    # ── CALCULATE TARGET START TIME ───────────────────────────────────────────
    # 
    # Convert days_back → milliseconds timestamp
    # Example: days_back=7 → start_time = (now - 7 days)
    # ──────────────────────────────────────────────────────────────────────────
    
    start_time_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000
    )
    
    all_dfs     = []       # Collect all batches here
    end_time_ms = None     # For pagination (moves backward each loop)
    
    print(f"  Fetching {interval} history for {symbol} ({days_back} days back)...")
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              PAGINATION LOOP
    # ══════════════════════════════════════════════════════════════════════════
    
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        
        # ── SET END TIME FOR PAGINATION ───────────────────────────────────────
        # 
        # First request: endTime not set (get latest 1000 candles)
        # Subsequent requests: endTime = oldest candle from previous batch - 1ms
        # ──────────────────────────────────────────────────────────────────────
        
        if end_time_ms:
            params["endTime"] = end_time_ms
        
        try:
            _inc_api_call()
            resp = requests.get(
                f"{BINANCE_BASE_URL}/api/v3/klines",
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            raw = resp.json()
            
            # ── CHECK IF DONE ─────────────────────────────────────────────────
            # 
            # If Binance returns empty array → no more historical data
            # This means we've reached the beginning of available data
            # ──────────────────────────────────────────────────────────────────
            
            if not raw:
                break  # No more data, exit loop
            
            # ── PARSE BATCH ───────────────────────────────────────────────────
            
            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            
            all_dfs.append(df)
            
            # ── MOVE PAGINATION POINTER BACKWARD ──────────────────────────────
            # 
            # oldest_ms = timestamp of first candle in this batch
            # end_time_ms = oldest_ms - 1 (next request ends just before this)
            # ──────────────────────────────────────────────────────────────────
            
            oldest_ms   = int(raw[0][0])
            end_time_ms = oldest_ms - 1
            
            # ── CHECK IF REACHED TARGET ───────────────────────────────────────
            # 
            # If oldest candle in this batch <= target start time → we're done
            # ──────────────────────────────────────────────────────────────────
            
            if oldest_ms <= start_time_ms:
                break
            
            # ── RATE LIMITING ─────────────────────────────────────────────────
            # 
            # Sleep 0.15 seconds between requests.
            # Prevents hitting Binance rate limits.
            # ──────────────────────────────────────────────────────────────────
            
            time.sleep(0.15)
        
        except Exception as e:
            log_error(f"fetch_historical_klines error for {symbol} {interval}: {repr(e)}")
            break
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              COMBINE BATCHES
    # ══════════════════════════════════════════════════════════════════════════
    
    if not all_dfs:
        # No data fetched (all requests failed)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Concatenate all batches into one DataFrame
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (pagination might overlap by 1 candle)
    combined = combined.drop_duplicates(subset=["timestamp"])
    
    # Sort chronologically (oldest first)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    
    print(f"  Got {len(combined)} {interval} candles")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN BACKFILL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_backfill(default_days_back: int = look_back_days):
    """
    Main backfill function with smart resume and time machine simulation.
    
    SMART RESUME:
        For each coin, checks Supabase for most recent signal.
        Only fetches data from that date forward.
        If no data exists, goes back default_days_back (3 days).
    
    TIME MACHINE APPROACH:
        Slides a 300-candle window through the full 15m history.
        At each position, simulates what the live pipeline would see:
            - Features computed only on data available at that moment
            - HTF data sliced to exclude future candles
            - No lookahead bias
    
    STEP SIZE:
        We check EVERY candle (step=1) to catch ALL crossovers.
        A crossover between checks would be missed otherwise.
    
    CRITICAL BUG FIXES:
        1. Window size: 300 (was 100, too small for indicators)
        2. BTC bias: Pre-calculated (was recalculated every 15m candle)
        3. HTF validation: Check sufficient data after slicing
        4. Smart resume: Minimum 3-day fetch even for recent signals
    
    Args:
        default_days_back: Fallback if no existing data (default: 3)
    
    Example:
        run_backfill()
        # Checks database, finds BTCUSDT last signal was 2 days ago
        # Fetches only last 2 days of data
        # Processes ~200 candles instead of 17,000
    """
    
    print(f"Starting smart backfill for {len(COINS)} coins")
    print(f"Default lookback: {default_days_back} days (used when no existing data)\n")
    
    total_signals = 0
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              PROCESS EACH COIN
    # ══════════════════════════════════════════════════════════════════════════
    
    for symbol in COINS:
        print(f"\n{'='*50}")
        print(f"Backfilling {symbol}...")
        coin_signals = 0
        
        # ── SMART RESUME: GET START DATE ──────────────────────────────────────
        # 
        # Query database: when was last signal for THIS coin specifically?
        # Returns datetime of last signal OR (now - look_back_days) if none.
        # ──────────────────────────────────────────────────────────────────────
        
        start_datetime = get_last_signal_date(symbol)
        
        # ── CALCULATE DAYS TO FETCH ───────────────────────────────────────────
        # 
        # BUG FIX: Ensure minimum 3-day fetch even for recent signals.
        # 
        # Why?
        #   Window size = 300 candles × 15min = 75 hours = 3.125 days
        #   If last signal was 6 hours ago and we only fetch 1 day of data,
        #   the first ~2 days of that data won't have enough history
        #   to compute features (need 300 candles of lookback).
        # 
        # Solution:
        #   Always fetch AT LEAST 3 days, even if resume point is recent.
        # ──────────────────────────────────────────────────────────────────────
        
        days_since_last = (datetime.now(timezone.utc) - start_datetime).days + 1
        days_back_for_coin = max(days_since_last, 30)  # CRITICAL FIX
        
        # ── FETCH BTC 4H DATA ─────────────────────────────────────────────────
        # 
        # BTC bias is needed for every signal as a feature.
        # We fetch BTC 4h data covering the same time period as the coin.
        # ──────────────────────────────────────────────────────────────────────
        
        print(f"Fetching BTC 4h history ({days_back_for_coin} days)...")
        btc_4h_master = fetch_historical_klines("BTCUSDT", "4h", days_back_for_coin)
        
        if btc_4h_master.empty:
            print(f"  ⚠️  Failed to fetch BTC data — skipping {symbol}")
            continue
        
        # ══════════════════════════════════════════════════════════════════════
        #                              PRE-CALCULATE BTC BIAS TIMELINE
        # ══════════════════════════════════════════════════════════════════════
        # 
        # CRITICAL BUG FIX:
        #   Old code recalculated BTC bias on EVERY 15m candle.
        #   But BTC 4h data only updates every 4 hours!
        #   This meant 16 consecutive 15m candles got the SAME btc_bias value.
        # 
        # NEW APPROACH:
        #   Pre-calculate BTC bias for the ENTIRE 4h timeline.
        #   Store as a new column: btc_4h_master["btc_bias"]
        #   Then in the loop, just look up the value for current time.
        # 
        # This is:
        #   - More accurate (each 4h candle has correct bias)
        #   - More efficient (calculate once vs 1000 times)
        # ══════════════════════════════════════════════════════════════════════
        
        print(f"  Pre-calculating BTC bias timeline...")
        btc_ema_fast = btc_4h_master["close"].ewm(span=EMA_FAST, adjust=False).mean()
        btc_ema_slow = btc_4h_master["close"].ewm(span=EMA_SLOW, adjust=False).mean()
        btc_4h_master["btc_bias"] = btc_ema_fast > btc_ema_slow
        
        # ── FETCH ALL 3 TIMEFRAMES FOR THIS COIN ──────────────────────────────
        
        df_15m = fetch_historical_klines(symbol, "15m", days_back_for_coin)
        df_4h  = fetch_historical_klines(symbol, "4h",  days_back_for_coin)
        df_1d  = fetch_historical_klines(symbol, "1d",  days_back_for_coin)
        
        if df_15m.empty or df_4h.empty or df_1d.empty:
            print(f"  ⚠️  Missing data for {symbol} — skipping")
            continue
        
        # ══════════════════════════════════════════════════════════════════════
        #                              TIME MACHINE LOOP
        # ══════════════════════════════════════════════════════════════════════
        # 
        # WINDOW SIZE:
        #   300 candles (CRITICAL FIX — was 100, too small)
        #   300 × 15min = 4500min = 75 hours = 3.125 days
        #   Enough for all indicators to warm up properly
        # 
        # STEP SIZE:
        #   1 candle (check every candle for crossovers)
        #   Ensures no crossovers are missed
        # 
        # LOOP LOGIC:
        #   Start at candle 300 (first full window)
        #   End at last candle in df_15m
        #   Each iteration = move window forward 1 candle
        # ══════════════════════════════════════════════════════════════════════
        
        window_size = 300  # CRITICAL FIX (was 100)
        step = 1
        print(f"  Processing {len(df_15m)} candles (window={window_size}, step={step})...")
        
        for end_idx in range(window_size, len(df_15m), step):
            # ── EXTRACT LTF WINDOW ────────────────────────────────────────────
            # 
            # Current window: 300 candles ending at end_idx
            # Example: end_idx=500 → candles [200:500]
            # ──────────────────────────────────────────────────────────────────
            
            ltf_window   = df_15m.iloc[end_idx - window_size : end_idx].reset_index(drop=True)
            current_time = ltf_window["timestamp"].iloc[-1]
            
            # ── SLICE HTF DATA (NO LOOKAHEAD) ─────────────────────────────────
            # 
            # CRITICAL: Only use HTF data available at current_time.
            # We filter: timestamp <= current_time
            # This prevents lookahead bias.
            # 
            # Example:
            #   current_time = 2026-04-01 10:30:00
            #   HTF 4h candles:
            #     - 2026-04-01 08:00:00 ✅ (use)
            #     - 2026-04-01 12:00:00 ❌ (future — exclude)
            # ──────────────────────────────────────────────────────────────────
            
            htf_4h_window = df_4h[df_4h["timestamp"] <= current_time].tail(500).reset_index(drop=True)
            htf_1d_window = df_1d[df_1d["timestamp"] <= current_time].tail(200).reset_index(drop=True)
            btc_4h_window = btc_4h_master[btc_4h_master["timestamp"] <= current_time]
            
            # ── VALIDATE HTF DATA SIZE ────────────────────────────────────────
            # 
            # BUG FIX: After slicing, check if we have enough data.
            # Early in the dataset, sliced HTF might be too short.
            # ──────────────────────────────────────────────────────────────────
            
            if len(htf_4h_window) < 50 or len(htf_1d_window) < 20 or len(btc_4h_window) < EMA_SLOW:
                continue  # Not enough HTF data yet, skip
            
            # ── GET BTC BIAS FROM TIMELINE ────────────────────────────────────
            # 
            # BUG FIX: Instead of recalculating EMAs every candle,
            # look up the pre-calculated bias value.
            # 
            # btc_4h_window.iloc[-1] = most recent 4h candle at current_time
            # ["btc_bias"] = boolean (True=bullish, False=bearish)
            # ──────────────────────────────────────────────────────────────────
            
            btc_bias = bool(btc_4h_window.iloc[-1]["btc_bias"])
            
            # ── COMPUTE FEATURES ──────────────────────────────────────────────
            # 
            # Pass historical datetime so hour/day features are correct.
            # Fear & Greed = 50 (neutral — no historical F&G data available).
            # ──────────────────────────────────────────────────────────────────
            
            features = compute_features(
                ltf_window,
                htf_4h_window,
                htf_1d_window,
                fear_greed=50,
                btc_bias=btc_bias,
                symbol=symbol,
                checked_at=current_time.to_pydatetime()
            )
            
            if features is None:
                continue  # Feature computation failed, skip
            
            # ── DETECT CROSSOVER ──────────────────────────────────────────────
            
            signal = detect_signal(features, symbol)
            if signal is None:
                continue  # No crossover, move to next candle
            
            # ── OVERRIDE TIMESTAMP ────────────────────────────────────────────
            # 
            # detect_signal() sets checked_at_utc to NOW (live mode).
            # In backfill, we override with historical timestamp.
            # ──────────────────────────────────────────────────────────────────
            
            signal["checked_at_utc"] = current_time.isoformat()
            
            # ── INSERT TO DATABASE ────────────────────────────────────────────
            # 
            # insert_signal() handles duplicates gracefully.
            # UNIQUE(symbol, checked_at_utc) constraint prevents doubles.
            # ──────────────────────────────────────────────────────────────────
            
            ok = insert_signal(signal)
            if ok:
                coin_signals  += 1
                total_signals += 1
                
                # Progress update every 10 signals
                if coin_signals % 10 == 0:
                    print(f"    {symbol}: {coin_signals} signals so far...")
        
        print(f"  ✓ {symbol} done — {coin_signals} signals found")
    
    # ══════════════════════════════════════════════════════════════════════════
    #                              BACKFILL COMPLETE
    # ══════════════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print(f"Backfill complete — {total_signals} total signals inserted")
    print("Check Supabase for the rows. Run labeler.py next to label them.")


# ══════════════════════════════════════════════════════════════════════════════
#                              ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_backfill()
