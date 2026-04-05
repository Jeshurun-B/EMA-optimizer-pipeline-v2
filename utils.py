# utils.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: Shared helper functions used across the entire pipeline.
# 
# CONTAINS:
#   - Error logging (file + console)
#   - Binance API fetcher (with rate limit tracking)
#   - API call counter (prevents rate limit bans)
#   - Run state persistence (resume after timeout)
#   - Previous signal cache (deduplication + gap calculation)
# 
# PHILOSOPHY:
#   - No business logic here (that belongs in features.py, signals.py, etc)
#   - Pure utility functions only
#   - Every function handles its own exceptions (never raises to caller)
#   - Simple return types (bool, list, DataFrame, dict)
# ═════════════════════════════════════════════════════════════════════════════

import os
import json
import pandas as pd
import requests
from datetime import datetime, timezone
from config import (
    REQUEST_TIMEOUT, LOG_FILE, API_CALL_LIMIT, CANDLE_LIMIT,
    LAST_SIGNALS_FILE, RUN_STATE_FILE, BINANCE_BASE_URL
)


# ══════════════════════════════════════════════════════════════════════════════
#                              API CALL COUNTER
# ══════════════════════════════════════════════════════════════════════════════
# 
# WHY WE NEED THIS:
#   - Binance has rate limits (weight-based system)
#   - Exceed limit = temporary IP ban (1-60 minutes)
#   - We track calls manually to stop BEFORE hitting the limit
# 
# HOW IT WORKS:
#   - Module-level variable (shared across all files that import utils)
#   - Every fetch increments this counter via _inc_api_call()
#   - Before each fetch, we check: api_limit_reached() → True/False
#   - If True → save state and exit gracefully
# 
# IMPORTANT:
#   - This counter resets on every pipeline run (it's in-memory only)
#   - On GitHub Actions, each job is a fresh Python process
#   - So each run gets a clean 0→200 call budget
# ══════════════════════════════════════════════════════════════════════════════

API_CALLS = 0  # Global counter — starts at 0 every run


def _inc_api_call() -> int:
    """
    Increment the API call counter by 1 and return the new total.
    
    Called internally by fetch_binance_klines() before EVERY Binance request.
    Prefixed with _ to signal "private helper, don't call directly from other files"
    
    Returns:
        The new total number of API calls made in this run.
    
    Example:
        API_CALLS = 0
        _inc_api_call()  # returns 1
        _inc_api_call()  # returns 2
        _inc_api_call()  # returns 3
    """
    global API_CALLS  # Tell Python we want to MODIFY the global, not create a local copy
    API_CALLS += 1
    return API_CALLS


def api_limit_reached() -> bool:
    """
    Check if we've hit or exceeded the allowed API call limit for this run.
    
    Called before EVERY fetch to decide: should we continue or stop?
    
    Returns:
        True  → Stop fetching, save state, exit gracefully
        False → Safe to continue fetching
    
    Why we return False on error:
        If there's a bug in the counter logic, we don't want it to STOP
        the pipeline from running. Better to risk hitting rate limit
        (which is a soft ban) than to stop collecting data entirely.
    
    Example usage:
        if api_limit_reached():
            save_run_state({"phase": "scan", "last_symbol_index": i})
            print("API limit reached — stopping early")
            exit(0)
    """
    try:
        return API_CALLS >= API_CALL_LIMIT
    except Exception as e:
        # This should NEVER happen (comparing two ints is bulletproof)
        # But if it does, log it and return False (continue running)
        log_error(f"api_limit_reached error: {repr(e)}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#                              ERROR LOGGING
# ══════════════════════════════════════════════════════════════════════════════
# 
# PHILOSOPHY:
#   - Pipeline should NEVER crash from a single coin failing
#   - Every fetch, every insert, every calculation is wrapped in try/except
#   - On error → log it, skip that coin, continue to next
#   - At end of run → review error.log to see what went wrong
# 
# FORMAT:
#   [2026-04-04T10:30:15.123456+00:00] fetch_binance_klines error for BTCUSDT 15m: HTTPError(429)
#   [timestamp in ISO 8601 format] descriptive message with context
# ══════════════════════════════════════════════════════════════════════════════

def log_error(msg: str) -> None:
    """
    Write a timestamped error message to LOG_FILE and print to console.
    
    This is the ONLY error logging function in the entire pipeline.
    Every try/except block calls this on failure.
    
    Args:
        msg: Error description with context
             Good: "fetch_binance_klines error for BTCUSDT 15m: Timeout"
             Bad:  "Error" (no context)
    
    Format:
        [2026-04-04T10:30:15.123456+00:00] your message here
    
    Behavior:
        - Always writes to LOG_FILE (append mode)
        - Always prints to console with [ERROR] prefix
        - Never raises exceptions (logging failures are silent)
    
    Why we never raise:
        If the log file is locked, permissions wrong, disk full, etc —
        we still want the pipeline to continue. The console print is
        a backup. Missing one log line is better than crashing.
    
    Example:
        try:
            result = risky_operation()
        except Exception as e:
            log_error(f"risky_operation failed: {repr(e)}")
            return None  # Return safe default and continue
    """
    # Get current UTC time as ISO 8601 string
    # Example: "2026-04-04T10:30:15.123456+00:00"
    # We use UTC everywhere to avoid timezone confusion
    ts = datetime.now(timezone.utc).isoformat()
    
    # Build the log line with timestamp prefix
    log_line = f"[{ts}] {msg}\n"
    
    try:
        # Append to log file (creates file if doesn't exist)
        # 'a' mode = append (don't overwrite existing logs)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception:
        # File write failed — maybe permissions, disk full, path invalid
        # We silently ignore because logging should never crash the pipeline
        # The console print below is our backup
        pass
    
    # Always print to console regardless of file write success
    # On GitHub Actions, this appears in the job logs
    # Locally, you see it in your terminal
    print(f"[ERROR] {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#                              BINANCE KLINES FETCHER
# ══════════════════════════════════════════════════════════════════════════════
# 
# WHAT THIS DOES:
#   Fetches OHLCV candlestick data from Binance public API.
# 
# WHY THIS EXISTS:
#   - Centralizes all Binance fetching logic in one place
#   - Handles rate limiting, error handling, type conversion
#   - Every file that needs candle data calls this function
# 
# RETURN PHILOSOPHY:
#   - On success: DataFrame with exactly 5 columns (timestamp, OHLC, volume)
#   - On failure: EMPTY DataFrame (not None, not exception)
#   - Why? Caller can check df.empty and skip gracefully
# ══════════════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from Binance public API.
    
    This is the ONLY function that talks to Binance in the entire pipeline.
    Every other file imports this from utils.
    
    Args:
        symbol:   Trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Candle timeframe (e.g., "15m", "4h", "1d")
                  Valid: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        limit:    Number of candles to fetch (default 300, max 1000)
    
    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
        
        timestamp: pandas Timestamp (UTC timezone aware)
        open/high/low/close/volume: float64
        
        Rows are sorted chronologically (oldest candle first).
        
        On ANY error (network, rate limit, invalid symbol, etc):
            Returns EMPTY DataFrame with same 5 columns.
    
    Example:
        df = fetch_binance_klines("BTCUSDT", "15m", 100)
        if df.empty:
            print("Fetch failed")
            return
        
        latest_price = df["close"].iloc[-1]
        print(f"Latest BTC price: ${latest_price:,.2f}")
    
    Rate Limiting:
        - Checks api_limit_reached() BEFORE making request
        - If limit hit → returns empty DataFrame (no API call made)
        - Increments counter via _inc_api_call() before request
    
    Error Handling:
        - Network timeout (slow connection) → empty DataFrame
        - HTTP 429 (rate limit) → empty DataFrame, logged
        - HTTP 4xx (bad symbol, invalid interval) → empty DataFrame, logged
        - HTTP 5xx (Binance server error) → empty DataFrame, logged
        - JSON parse error → empty DataFrame, logged
        - Type conversion error → empty DataFrame, logged
    """
    
    # ── STEP 1: CHECK API LIMIT ───────────────────────────────────────────────
    # 
    # Before making ANY request, check if we've hit our self-imposed limit.
    # This prevents getting banned by Binance for exceeding their rate limits.
    # 
    # If we're at/over limit:
    #   - Log a message (for debugging)
    #   - Return empty DataFrame (caller sees df.empty and skips)
    #   - Do NOT increment counter (we didn't make a request)
    # ──────────────────────────────────────────────────────────────────────────
    
    if api_limit_reached():
        log_error(
            f"API call limit reached ({API_CALLS}/{API_CALL_LIMIT}). "
            f"Skipping fetch for {symbol} {interval}."
        )
        # Return empty DataFrame with correct column structure
        # This lets caller use df.empty check consistently
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    
    # ── STEP 2: ENFORCE CANDLE LIMIT ──────────────────────────────────────────
    # 
    # Binance hard limit: 1000 candles per request
    # Our config.CANDLE_LIMIT: 300 (or whatever you set in .env)
    # 
    # Why we enforce this:
    #   - Prevents accidentally requesting 5000 candles (would fail)
    #   - Keeps request size reasonable (less data = faster, less failure-prone)
    #   - If config says 300, we respect that globally
    # 
    # We take the MINIMUM of (user's limit request, config limit)
    # Example: fetch_binance_klines("BTCUSDT", "15m", 500)
    #          If CANDLE_LIMIT=300 → we request 300 (not 500)
    # ──────────────────────────────────────────────────────────────────────────
    
    req_limit = min(limit, CANDLE_LIMIT)
    
    
    # ── STEP 3: BUILD REQUEST ─────────────────────────────────────────────────
    # 
    # Binance klines endpoint:
    #   GET https://data-api.binance.vision/api/v3/klines
    # 
    # Required parameters:
    #   symbol:   BTCUSDT (no spaces, uppercase)
    #   interval: 15m (lowercase)
    #   limit:    300 (integer 1-1000)
    # 
    # Optional parameters we DON'T use:
    #   startTime: fetch from specific timestamp forward
    #   endTime:   fetch up to specific timestamp
    #   (we use these in backfill.py for historical data)
    # 
    # No authentication needed — this is public market data
    # ──────────────────────────────────────────────────────────────────────────
    
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,      # Trading pair
        "interval": interval,  # Candle timeframe
        "limit": req_limit     # Number of candles
    }
    
    try:
        # ── STEP 4: INCREMENT COUNTER (BEFORE REQUEST) ────────────────────────
        # 
        # We count this call BEFORE making it.
        # Why? If request fails, we still "spent" an API call attempt.
        # Binance counts failed requests toward rate limit too.
        # ──────────────────────────────────────────────────────────────────────
        
        _inc_api_call()
        
        
        # ── STEP 5: MAKE REQUEST ──────────────────────────────────────────────
        # 
        # requests.get() parameters:
        #   url:     full endpoint URL
        #   params:  dict of query parameters (auto URL-encoded)
        #   timeout: max seconds to wait (from config.REQUEST_TIMEOUT)
        # 
        # timeout behavior:
        #   If Binance doesn't respond within timeout seconds,
        #   requests raises a Timeout exception (caught below)
        # 
        # This is a BLOCKING call — code waits here until:
        #   - Response received (success)
        #   - Timeout expires (exception)
        #   - Network error (exception)
        # ──────────────────────────────────────────────────────────────────────
        
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        
        
        # ── STEP 6: CHECK HTTP STATUS ─────────────────────────────────────────
        # 
        # raise_for_status() checks if HTTP status is 4xx or 5xx.
        # If so, raises an HTTPError exception (caught below).
        # 
        # Common status codes:
        #   200: Success (continue)
        #   429: Rate limit exceeded (wait and retry)
        #   400: Bad request (wrong symbol/interval)
        #   500: Binance server error (temporary issue)
        # 
        # If status is 200, this does nothing (no exception raised)
        # ──────────────────────────────────────────────────────────────────────
        
        resp.raise_for_status()
        
        
        # ── STEP 7: PARSE JSON ────────────────────────────────────────────────
        # 
        # Binance returns a JSON array of arrays:
        # [
        #   [1617235200000, "50000.00", "51000.00", "49500.00", "50500.00", "1234.56", ...],
        #   [1617236100000, "50500.00", "50700.00", "50300.00", "50600.00", "2345.67", ...],
        #   ...
        # ]
        # 
        # Each inner array has 12 elements:
        #   [0]  open_time (milliseconds since epoch)
        #   [1]  open price (string, not float!)
        #   [2]  high price
        #   [3]  low price
        #   [4]  close price
        #   [5]  volume
        #   [6]  close_time
        #   [7]  quote_asset_volume
        #   [8]  number_of_trades
        #   [9]  taker_buy_base_volume
        #   [10] taker_buy_quote_volume
        #   [11] ignore (unused by Binance)
        # 
        # resp.json() parses this into Python list of lists
        # ──────────────────────────────────────────────────────────────────────
        
        raw = resp.json()
        
        
        # ── STEP 8: BUILD DATAFRAME ───────────────────────────────────────────
        # 
        # Convert list of lists into pandas DataFrame with named columns.
        # 
        # We provide ALL 12 column names even though we only keep 5.
        # Why? So DataFrame structure matches Binance response exactly.
        # Makes debugging easier if something goes wrong.
        # ──────────────────────────────────────────────────────────────────────
        
        df = pd.DataFrame(raw, columns=[
            "open_time",            # [0] milliseconds timestamp
            "open",                 # [1] opening price this candle
            "high",                 # [2] highest price this candle
            "low",                  # [3] lowest price this candle
            "close",                # [4] closing price this candle
            "volume",               # [5] base asset volume (e.g., BTC amount)
            "close_time",           # [6] milliseconds timestamp
            "quote_asset_volume",   # [7] quote asset volume (e.g., USDT amount)
            "number_of_trades",     # [8] how many trades in this candle
            "taker_buy_base_volume",  # [9] buy volume (base)
            "taker_buy_quote_volume", # [10] buy volume (quote)
            "ignore"                # [11] unused
        ])
        
        
        # ── STEP 9: CONVERT TYPES ─────────────────────────────────────────────
        # 
        # Problem: Binance returns numbers as STRINGS ("50000.00")
        # Solution: Convert to proper types (int, float, datetime)
        # 
        # TIMESTAMP CONVERSION:
        #   open_time is milliseconds since epoch (e.g., 1617235200000)
        #   pd.to_datetime() with unit='ms' converts to datetime
        #   utc=True makes it timezone-aware (avoids timezone bugs later)
        # 
        # PRICE/VOLUME CONVERSION:
        #   pd.to_numeric() converts string → float
        #   errors='coerce' → if conversion fails, use NaN (not crash)
        # 
        # Why coerce instead of raise?
        #   If Binance sends malformed data (rare but possible),
        #   we get NaN instead of crashing. We can filter NaN rows later.
        # ──────────────────────────────────────────────────────────────────────
        
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        
        # ── STEP 10: RETURN CLEAN DATA ────────────────────────────────────────
        # 
        # We only return the 5 columns needed by the rest of the pipeline:
        #   timestamp, open, high, low, close, volume
        # 
        # All other columns (close_time, number_of_trades, etc) are dropped.
        # 
        # Why only 5 columns?
        #   - Keeps DataFrame small (less memory)
        #   - Standardized format across entire pipeline
        #   - Every function expects these exact 5 columns
        # 
        # Rows are already sorted chronologically (Binance returns oldest first)
        # ──────────────────────────────────────────────────────────────────────
        
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
        
    
    except Exception as e:
        # ── ERROR HANDLING ────────────────────────────────────────────────────
        # 
        # ANY exception (network, timeout, HTTP error, JSON parse, etc)
        # is caught here and logged.
        # 
        # We include:
        #   - symbol and interval (context — which request failed?)
        #   - repr(e) (full exception class and message)
        # 
        # Then we return empty DataFrame (same 5 columns as success case)
        # 
        # Why return empty instead of None?
        #   - Caller can use df.empty consistently
        #   - No need for "if df is None" checks everywhere
        #   - Empty DataFrame is truthy-false: if not df.empty works
        # ──────────────────────────────────────────────────────────────────────
        
        log_error(f"fetch_binance_klines error for {symbol} {interval}: {repr(e)}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


# ══════════════════════════════════════════════════════════════════════════════
#                              RUN STATE PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
# 
# WHY THIS EXISTS:
#   - Pipeline might hit API limit mid-run (scanned 3 coins, need to scan 5)
#   - Pipeline might hit time limit (GitHub Actions 5min timeout)
#   - We need to save: "I was on coin #3, phase scan"
#   - Next run loads this and resumes from coin #3 (not start from #0 again)
# 
# FILE FORMAT (JSON):
#   {
#     "phase": "scan",              ← What we were doing (scan or label)
#     "last_symbol_index": 2,       ← Last coin we COMPLETED (0-indexed)
#     "timestamp": "2026-04-04..."  ← When we stopped
#   }
# 
# LIFECYCLE:
#   1. Start of run: load_run_state() → get resume point
#   2. During run:   for i in range(resume_index, len(COINS))
#   3. End of run:   save_run_state() → save progress for next time
# ══════════════════════════════════════════════════════════════════════════════

def load_run_state() -> dict:
    """
    Load saved pipeline state from RUN_STATE_FILE.
    
    Called at the START of runner.py to check: should we resume or start fresh?
    
    Returns:
        dict with keys: phase, last_symbol_index, timestamp
        
        If file doesn't exist or can't be parsed → returns {}
        Caller should treat {} as "start from beginning"
    
    Example:
        state = load_run_state()
        if state:
            resume_index = state.get("last_symbol_index", 0) + 1
            print(f"Resuming from coin #{resume_index}")
        else:
            resume_index = 0
            print("Starting fresh scan")
    
    Error Handling:
        - File not found → return {} (not an error, just first run)
        - JSON malformed → return {} (log error, start fresh)
        - Permission denied → return {} (log error, start fresh)
    """
    
    # Check if file exists before trying to open
    # os.path.exists() returns True/False (never raises)
    if not os.path.exists(RUN_STATE_FILE):
        return {}  # First run ever — no state to load
    
    try:
        # Open file in read mode, parse JSON
        with open(RUN_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)  # Parses JSON → Python dict
        return state
    
    except Exception as e:
        # File exists but can't be read/parsed
        # Possible causes:
        #   - File corrupted (half-written during crash)
        #   - Not valid JSON (manual edit broke it)
        #   - Permission denied (rare)
        # 
        # We log the error and return {} (start fresh)
        # This is safe — worst case we re-scan coins we already did
        log_error(f"load_run_state error: {repr(e)}")
        return {}


def save_run_state(state: dict) -> None:
    """
    Save current pipeline state to RUN_STATE_FILE.
    
    Called when:
        - API limit reached (save progress before exit)
        - Time limit reached (save progress before exit)
        - Pipeline completes fully (save for next run)
    
    Args:
        state: dict with keys:
               - phase: "scan" or "label"
               - last_symbol_index: integer (0-indexed coin position)
               - timestamp: ISO string (when we stopped)
    
    Example:
        save_run_state({
            "phase": "scan",
            "last_symbol_index": 3,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    Error Handling:
        - File write fails → logged, but pipeline continues
        - Why continue? Saving state is nice-to-have, not critical
        - If save fails, next run starts from scratch (inefficient but safe)
    """
    
    try:
        # Write dict as JSON to file
        # 'w' mode = overwrite (we want latest state, not append)
        with open(RUN_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)  # indent=2 for human readability
    
    except Exception as e:
        # File write failed — log it but don't crash
        # Possible causes:
        #   - Disk full (rare on cloud runners)
        #   - Permission denied
        #   - Path invalid (RUN_STATE_FILE misconfigured)
        # 
        # Impact: Next run starts from scratch instead of resuming
        # Not ideal, but not catastrophic
        log_error(f"save_run_state error: {repr(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#                              PREVIOUS SIGNAL CACHE
# ══════════════════════════════════════════════════════════════════════════════
# 
# WHY THIS EXISTS:
#   1. DEDUPLICATION: Prevent inserting same signal twice
#   2. GAP CALCULATION: Compute time since last signal (stored as feature)
# 
# FILE FORMAT (CSV):
#   symbol,signal,checked_at_utc
#   BTCUSDT,LONG,2026-04-04T10:30:00+00:00
#   ETHUSDT,SHORT,2026-04-04T09:15:00+00:00
#   SOLUSDT,LONG,2026-04-04T11:00:00+00:00
# 
# LIFECYCLE:
#   1. signals.py calls get_prev_signal(symbol) before inserting
#   2. If same signal fired <30min ago → skip (duplicate)
#   3. If different signal or >30min ago → insert to DB
#   4. signals.py calls update_prev_signal(symbol, {...}) after insert
# 
# IMPORTANT:
#   - One row per symbol (not one row per signal)
#   - We only track the MOST RECENT signal per coin
#   - Old signals are overwritten (we don't need history in this file)
# ══════════════════════════════════════════════════════════════════════════════

def get_prev_signal(symbol: str) -> dict:
    """
    Get the most recent signal record for a specific symbol.
    
    Used by signals.py to:
        1. Check if current signal is a duplicate (same signal <30min ago)
        2. Calculate signal_gap_hours (time since last signal on this coin)
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
    
    Returns:
        dict: {"symbol": ..., "signal": ..., "checked_at_utc": ...}
        None: If no previous signal exists for this symbol
    
    Example:
        prev = get_prev_signal("BTCUSDT")
        if prev:
            print(f"Last signal was {prev['signal']} at {prev['checked_at_utc']}")
        else:
            print("No previous signal for BTCUSDT")
    
    Error Handling:
        - File doesn't exist → return None (first run ever)
        - Symbol not in file → return None (first signal for this coin)
        - CSV malformed → return None (log error, continue)
    """
    
    # Check if cache file exists
    if not os.path.exists(LAST_SIGNALS_FILE):
        return None  # File doesn't exist yet (first run)
    
    try:
        # Read entire CSV into DataFrame
        df = pd.read_csv(LAST_SIGNALS_FILE)
        
        # Filter for this specific symbol (case-insensitive)
        # Why case-insensitive? User might have "btcusdt" in file but query "BTCUSDT"
        df = df[df["symbol"].str.upper() == symbol.upper()]
        
        # If no rows match → symbol never seen before
        if df.empty:
            return None
        
        # Get the last row (most recent entry for this symbol)
        # .iloc[-1] = last row
        # .to_dict() = convert Series → dict
        last_row = df.iloc[-1].to_dict()
        
        # Return standardized dict format
        return {
            "symbol": last_row["symbol"],
            "signal": last_row["signal"],
            "checked_at_utc": last_row["checked_at_utc"]
        }
    
    except Exception as e:
        # CSV read failed (corrupted file, wrong columns, etc)
        log_error(f"get_prev_signal error for {symbol}: {repr(e)}")
        return None


def update_prev_signal(symbol: str, rec: dict) -> None:
    """
    Update the most recent signal for a symbol in LAST_SIGNALS_FILE.
    
    This maintains a running cache of the latest signal per coin.
    If symbol already has a row → replace it (not append).
    If symbol is new → add a new row.
    
    Called by signals.py AFTER successfully inserting a signal to database.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        rec:    dict with keys:
                - signal: "LONG" or "SHORT"
                - checked_at_utc: ISO timestamp string (optional)
    
    Example:
        update_prev_signal("BTCUSDT", {
            "signal": "LONG",
            "checked_at_utc": "2026-04-04T10:30:00+00:00"
        })
    
    Error Handling:
        - File write fails → logged, but pipeline continues
        - CSV corrupted → logged, file recreated with just this one record
    """
    
    try:
        # Build the record to save
        record = {
            "symbol": symbol,
            "signal": rec.get("signal"),
            "checked_at_utc": rec.get(
                "checked_at_utc",
                datetime.now(timezone.utc).isoformat()  # Default to now if not provided
            )
        }
        
        # ── CASE 1: FILE DOESN'T EXIST YET ────────────────────────────────────
        # 
        # This is the first signal EVER recorded.
        # Create the file with just this one row.
        # ──────────────────────────────────────────────────────────────────────
        
        if not os.path.exists(LAST_SIGNALS_FILE):
            pd.DataFrame([record]).to_csv(LAST_SIGNALS_FILE, index=False)
            return  # Done — file created with one row
        
        
        # ── CASE 2: FILE EXISTS — UPDATE OR APPEND ────────────────────────────
        # 
        # Strategy:
        #   1. Read entire CSV
        #   2. Remove any existing row for this symbol
        #   3. Append new record
        #   4. Write back to CSV
        # 
        # Why remove + append instead of update in place?
        #   - CSV doesn't support "UPDATE WHERE" like SQL
        #   - Easier to filter out old row and add new one
        #   - File is tiny (max ~50 rows for 50 coins) so performance doesn't matter
        # ──────────────────────────────────────────────────────────────────────
        
        df = pd.read_csv(LAST_SIGNALS_FILE)
        
        # Remove any existing row(s) for this symbol (case-insensitive)
        # Keep all OTHER symbols' rows
        df = df[df["symbol"].str.upper() != symbol.upper()]
        
        # Append the new record
        # pd.concat() is the modern way (replaces deprecated df.append())
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        
        # Write back to CSV
        # index=False → don't write row numbers as a column
        df.to_csv(LAST_SIGNALS_FILE, index=False)
    
    except Exception as e:
        # File operation failed
        # Impact: Next signal for this coin won't have gap calculation
        # Not critical — log and continue
        log_error(f"update_prev_signal error for {symbol}: {repr(e)}")
