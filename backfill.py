# backfill.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Collect months of historical signals to build our ML training dataset.
# MECHANICS: 
#   1. Paginates the Binance API backwards to grab massive chunks of data.
#   2. Simulates a live pipeline by stepping through the data chronologically.
#   3. Inserts any detected historical signals directly into Supabase.
# RULE: This is a heavy, one-time script. We handle rate limits carefully here
#       by sleeping between big requests.
# ─────────────────────────────────────────────────────────────────────────────

import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from config import COINS, BINANCE_BASE_URL
from features import compute_features
from signals import detect_signal
from db import insert_signal
from utils import log_error

# ── 1. HISTORICAL DATA FETCHER ────────────────────────────────────────────────

def fetch_historical_klines(symbol: str, interval: str, days_back: int) -> pd.DataFrame:
    """
    Fetches thousands of historical candles by paginating backwards from today.
    
    Binance limits each API call to 1000 candles. If we need 6 months of 15m 
    candles (approx 17,280 candles), we must make ~18 separate API calls, 
    updating the 'endTime' parameter each time to stitch them together.
    
    Args:
        symbol: e.g., "BTCUSDT"
        interval: e.g., "15m", "4h", "1d"
        days_back: How many days of history to fetch (e.g., 180 for 6 months)
        
    Returns:
        A massive pandas DataFrame containing all requested historical candles,
        sorted chronologically (oldest to newest).
    """
    
    # Calculate the exact timestamp (in milliseconds) for our cutoff date.
    # We subtract 'days_back' from the current UTC time.
    start_time_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    start_time_ms = int(start_time_dt.timestamp() * 1000)
    
    # We will store all the individual dataframes in this list, then combine them.
    all_dfs = []
    
    # We start fetching backwards from "now". So the first endTime is None (which means 'latest').
    end_time_ms = None
    
    # We set a hard limit of 1000 candles per request to maximize efficiency.
    limit = 1000
    
    # Loop continuously until we reach our target start date.
    while True:
        # Build the URL and the query parameters
        url = f"{BINANCE_BASE_URL}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        
        # If we have an end_time (from the previous loop iteration), add it to params
        if end_time_ms:
            params["endTime"] = end_time_ms
            
        try:
            # Make the HTTP GET request to Binance
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw_data = resp.json()
            
            # If Binance returns an empty list, we've run out of data. Break the loop.
            if not raw_data:
                break
                
            # Convert the raw JSON list into a pandas DataFrame.
            # We explicitly define the 12 columns Binance returns so we can map them.
            df = pd.DataFrame(raw_data, columns=[
                "open_time", "open", "high", "low", "close", "volume", 
                "close_time", "quote_asset_volume", "trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            
            # Convert the UNIX millisecond timestamp to a readable UTC datetime object
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            
            # Convert the string price and volume data into floating-point numbers
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
            # Filter the DataFrame to only keep the 6 columns we actually use
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            
            # Add this chunk of data to our master list
            all_dfs.append(df)
            
            # Update the end_time_ms for the NEXT loop iteration.
            # We set it to the open_time of the very FIRST (oldest) candle in this chunk, 
            # minus 1 millisecond, so the next fetch picks up exactly where this one left off.
            oldest_candle_time = raw_data[0][0]
            end_time_ms = oldest_candle_time - 1
            
            # Check if the oldest candle we just fetched is older than our target start date.
            # If it is, we have successfully fetched enough data and can break the loop.
            if end_time_ms < start_time_ms:
                break
                
            # Sleep for a tiny fraction of a second to avoid getting IP-banned by Binance
            time.sleep(0.1)
            
        except Exception as e:
            # If a network error occurs, log it and break. We will just use whatever data
            # we successfully fetched up to this point.
            print(f"Error fetching historical data for {symbol}: {e}")
            break

    # If the loop finished but we got no data at all, return an empty DataFrame
    if not all_dfs:
        return pd.DataFrame()
        
    # Combine all the smaller chunk DataFrames into one massive DataFrame
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Because we fetched backwards, our data is currently ordered newest-to-oldest.
    # Time-series analysis requires oldest-to-newest, so we drop duplicates and sort it.
    final_df = final_df.drop_duplicates(subset=["timestamp"])
    final_df = final_df.sort_values("timestamp").reset_index(drop=True)
    
    return final_df


# ── 2. MAIN BACKFILL LOGIC ────────────────────────────────────────────────────

def run_backfill():
    """
    The orchestrator for the backfill process.
    Downloads months of data for all coins, simulates time passing, and saves signals.
    """
    print("🚀 Starting Historical Backfill (Time Machine Mode)\n")
    
    # We will fetch 180 days (approx 6 months) of data. 
    # This should yield roughly 50-100 signals per coin, giving us our 500 target.
    DAYS_BACK = 180 
    
    # Fetch historical BTC 4h data first. 
    # We need this to calculate the 'btc_trend_bias' feature for all altcoins.
    print("🌍 Fetching Master BTC 4H Bias History...")
    btc_4h_master = fetch_historical_klines("BTCUSDT", "4h", DAYS_BACK)
    
    # Loop through every coin defined in our config.py
    for symbol in COINS:
        print(f"\n========================================")
        print(f"📥 Fetching 6 months of data for {symbol}...")
        
        # 1. Fetch the massive historical datasets for all 3 timeframes
        # This will take a few seconds per coin.
        df_15m = fetch_historical_klines(symbol, "15m", DAYS_BACK)
        df_4h  = fetch_historical_klines(symbol, "4h", DAYS_BACK)
        df_1d  = fetch_historical_klines(symbol, "1d", DAYS_BACK)
        
        # Safety check: if the API failed, skip this coin and move to the next.
        if df_15m.empty or df_4h.empty or df_1d.empty:
            print(f"⚠️ Missing data for {symbol}. Skipping.")
            continue
            
        print(f"✅ Downloaded {len(df_15m)} 15m candles.")
        print(f"⚙️ Running Time Machine simulation...")
        
        signals_found = 0
        
        # 2. The Time Machine Loop
        # We cannot just run compute_features() on the whole dataframe at once, 
        # because the indicators would use "future" data (Lookahead Bias).
        # We must step through the 15m candles one by one, starting at index 200 
        # (to give the moving averages time to warm up).
        for i in range(200, len(df_15m)):
            
            # Slice the 15m dataframe to represent "all data available UP TO this exact moment"
            # It physically chops off the future candles so the logic can't cheat.
            current_15m_window = df_15m.iloc[:i].copy()
            
            # Get the exact timestamp of the current "latest" candle in our simulation
            current_time = current_15m_window.iloc[-1]["timestamp"]
            
            # We only want to run the heavy logic when a 15m candle actually closes.
            # We slice the 4H and 1D dataframes to only include candles that occurred 
            # BEFORE or EXACTLY AT our current simulated time.
            current_4h_window = df_4h[df_4h["timestamp"] <= current_time].copy()
            current_1d_window = df_1d[df_1d["timestamp"] <= current_time].copy()
            
            # Also slice the BTC master dataframe to get the bias up to this exact moment
            current_btc_4h = btc_4h_master[btc_4h_master["timestamp"] <= current_time].copy()
            
            # If our sliced higher timeframes don't have enough data yet, skip to the next 15m candle
            if len(current_4h_window) < 50 or len(current_1d_window) < 20 or len(current_btc_4h) < 50:
                continue
            
                
            # Calculate the BTC 4H bias using the sliced BTC data
            # EMA 9 > EMA 15 = True (Bullish), else False (Bearish)
            fast_btc = current_btc_4h["close"].ewm(span=9, adjust=False).mean().iloc[-1]
            slow_btc = current_btc_4h["close"].ewm(span=15, adjust=False).mean().iloc[-1]
            btc_bias = bool(fast_btc > slow_btc)
            
            # For historical runs, we don't have historical Fear & Greed data easily aligned.
            # We default to 50 (Neutral) to prevent errors. Our model will handle this baseline.
            historical_fgi = 50 
            
            # 3. Compute Features and Detect Signals
            # Pass our perfectly sliced, lookahead-bias-free data into our pure logic functions
            features = compute_features(
                current_15m_window, 
                current_4h_window, 
                current_1d_window, 
                historical_fgi, 
                btc_bias, 
                symbol
            )
            
            # If features computed successfully, pass them to the signal detector
            if features:
                signal_record = detect_signal(features, symbol)
                
                # 4. Save to Database
                # If a crossover occurred at this historical moment, insert it!
                if signal_record:
                    # Overwrite the 'checked_at_utc' with the actual historical timestamp 
                    # of when this crossover happened, rather than 'now'.
                    signal_record["checked_at_utc"] = current_time.isoformat()
                    
                    # Call db.py to insert it into Supabase
                    success = insert_signal(signal_record)
                    
                    if success:
                        signals_found += 1
                        print(f"  🚨 {signal_record['signal']} found at {current_time.date()} — Saved.")
                        
        print(f"🏁 Finished {symbol}. Total signals found and stored: {signals_found}")

    print("\n🎉 BACKFILL COMPLETE! Check your Supabase dashboard.")

if __name__ == "__main__":
    run_backfill()