# utils.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Shared helper functions used across the entire pipeline.
# CONTAINS: logging, Binance fetcher, API call counter,
#           run state persistence, previous signal cache.
# RULE: No business logic here. Pure utility functions only.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import pandas as pd
import requests
from datetime import datetime, timezone
from config import (
    REQUEST_TIMEOUT, LOG_FILE, API_CALL_LIMIT, CANDLE_LIMIT,
    LAST_SIGNALS_FILE, RUN_STATE_FILE, BINANCE_BASE_URL
)


# ── API CALL COUNTER ───────────────────────────────────────Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser───────────────────
# Tracks total Binance requests made in this process run.
# Module-level so it's shared across all files that import utils.
# We check this before every fetch to avoid hitting rate limits mid-run.

API_CALLS = 0


def _inc_api_call() -> int:
    """
    Increment the API call counter by 1 and return the new value.
    Called internally before every Binance request.
    """
    # TODO: declare API_CALLS as global so we can modify it
    global API_CALLS
    # TODO: increment API_CALLS by 1
    API_CALLS += 1
    # TODO: return API_CALLS
    return API_CALLS


def api_limit_reached() -> bool:
    """
    Return True if we have reached or exceeded the allowed API call limit.
    Called before every fetch — if True, skip the fetch and save state.
    """
    # TODO: return True if API_CALLS >= API_CALL_LIMIT
    # if API_CALLS >= API_CALL_LIMIT:
    #     return True
    # return False
    # TODO: wrap in try/except — return False if anything errors
    # This is a safety measure — we don't want   a bug in the counter to stop the pipeline from running.:
    try:
        return API_CALLS >= API_CALL_LIMIT
    except Exception as e:
        log_error(f"api_limit_reached error: {repr(e)}")
        return False    
# ── LOGGING ───────────────────────────────────────────────────────────────────

def log_error(msg: str) -> None:
    """
    Write a timestamped error message to the log file and print it.

    Format: [2026-03-25T10:00:00+00:00] message here

    Never raises — if the file write fails we still print to console.
    The pipeline should never crash because of a logging failure.
    """
    # TODO: get current UTC time using datetime.now(timezone.utc).isoformat()
    ts = datetime.now(timezone.utc).isoformat()
    # TODO: build the log line: f"[{ts}] {msg}\n"
    log_line = f"[{ts}] {msg}\n"
    # TODO: try to open LOG_FILE in append mode and write the line
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_line)
    except Exception as e:
        # If logging to file fails, we still want to print the error to console.
        print(f"[ERROR] Failed to write to log file: {repr(e)}")
    # TODO: always print("[ERROR]", msg) at the end regardless
    print(f"[ERROR] {msg}")


# ── BINANCE KLINES FETCHER ────────────────────────────────────────────────────

def fetch_binance_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from Binance public API.

    Args:
        symbol:   trading pair e.g. "BTCUSDT"
        interval: candle timeframe e.g. "15m", "4h", "1d"
        limit:    how many candles to fetch (max 1000)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        On ANY failure — returns empty DataFrame, never raises an exception.

    Why we return empty DataFrame instead of raising:
        The pipeline checks df.empty after every fetch.
        A crash here would stop ALL coins from being scanned.
        An empty return lets the caller decide what to do.
    """
    # Step 1 — check API limit before making any request
    # TODO: if api_limit_reached(), log a message and return empty DataFrame
    # hint: empty = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    if api_limit_reached():
        log_error(f"API call limit reached ({API_CALLS} calls). Skipping fetch for {symbol} {interval}.")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    # Step 2 — enforce candle limit
    # TODO: set req_limit = min(limit, CANDLE_LIMIT)
    # This prevents accidentally requesting more than our config allows
    req_limit = min(limit, CANDLE_LIMIT)

    # Step 3 — build request
    # TODO: set url = f"{BINANCE_BASE_URL}/api/v3/klines"
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    # TODO: set params = {"symbol": symbol, "interval": interval, "limit": req_limit}
    params = {"symbol": symbol, "interval": interval, "limit": req_limit}
    
    try:
        # Step 4 — make the request
        # TODO: call _inc_api_call() to count this request
        _inc_api_call()
        # TODO: resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        # TODO: resp.raise_for_status()  — raises exception on 4xx/5xx errors
        resp.raise_for_status()
        # TODO: raw = resp.json()
        raw = resp.json()
        # Step 5 — parse the response
        # Binance returns a list of lists. Each inner list has exactly 12 items:
        # index 0:  open_time (milliseconds timestamp)
        # index 1:  open price
        # index 2:  high price
        # index 3:  low price
        # index 4:  close price
        # index 5:  volume
        # index 6:  close_time
        # index 7:  quote_asset_volume
        # index 8:  number of trades
        # index 9:  taker_buy_base_volume
        # index 10: taker_buy_quote_volume
        # index 11: ignore

        # TODO: create DataFrame from raw with all 12 column names listed above

        df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"])
        # Step 6 — convert types
        # TODO: df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        # TODO: for each of open, high, low, close, volume:
        #       df[col] = pd.to_numeric(df[col], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")  
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        
        # Step 7 — return only the columns we need
        # TODO: return df[["timestamp", "open", "high", "low", "close", "volume"]]
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
        

    except Exception as e:
        # TODO: log the error — include symbol and interval in the message
        log_error(f"fetch_binance_klines error for {symbol} {interval}: {repr(e)}")
        # TODO: return empty DataFrame
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        


# ── RUN STATE PERSISTENCE ─────────────────────────────────────────────────────
# run_state.json tracks where the pipeline stopped if it hit a time/API limit.
# Next run reads this file and resumes from the same position.
#
# Structure:
# {
#   "phase": "scan",          <- "scan" or "label"
#   "last_symbol_index": 2,   <- which coin index we got to
#   "timestamp": "2026-..."   <- when we stopped
# }

def load_run_state() -> dict:
    #am not sure about this function,
    """
    Load the saved run state from disk.
    Returns empty dict {} if file doesn't exist or can't be parsed.
    """
    # TODO: if RUN_STATE_FILE does not exist (use os.path.exists), return {}
    if not os.path.exists(RUN_STATE_FILE):
        return {}
    # TODO: open the file in read mode, use json.load() to parse it, return the dict

    # TODO: wrap 
    # everything in try/except — on any error, log it and return {}
    try:
        with open(RUN_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Error loading run state: {repr(e)}")
        return {}


def save_run_state(state: dict) -> None:
    """
    Save the current run state dict to disk as JSON.
    Called before any graceful exit so the next run can resume.
    Never raises.
    """
    # TODO: open RUN_STATE_FILE in write mode ("w")
    with open(RUN_STATE_FILE, "w") as f:
        # TODO: json.dump(state, f) to write the dict as JSON
        json.dump(state, f)

    # TODO: wrap in try/except — on failure log the error
    try:
        with open(RUN_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        log_error(f"Error saving run state: {repr(e)}")


# ── PREVIOUS SIGNAL CACHE ─────────────────────────────────────────────────────
# last_signals.csv stores the most recent signal detected per coin.
# One row per coin. Used for two things:
#   1. Calculating signal_gap_hours (time since last signal on this coin)
#   2. Preventing duplicate inserts (same signal fired again before next candle)
#
# CSV columns: symbol, signal, checked_at_utc

def get_prev_signal(symbol: str) -> dict:
    """
    Return the last saved signal record for this symbol as a dict.
    Returns None if no previous signal exists for this coin.

    Return format: {"symbol": ..., "signal": ..., "checked_at_utc": ...}
    """
    # TODO: if LAST_SIGNALS_FILE doesn't exist, return None
    if not os.path.exists(LAST_SIGNALS_FILE):
        return None
    # TODO: read the CSV into a DataFrame
    df = pd.read_csv(LAST_SIGNALS_FILE)

    # TODO: filter rows where the symbol column matches (compare uppercase to uppercase)
    df = df[df["symbol"].str.upper() == symbol.upper()]
    # TODO: if DataFrame is empty after filtering, return None
    if df.empty:
        return None
    # TODO: take the last row: df.iloc[-1].to_dict()
    last_row = df.iloc[-1].to_dict()
    # TODO: return only the 3 keys: symbol, signal, checked_at_utc
    
    ## i don't know what to do here
    # TODO: wrap everything in try/except — on failure log error and return None
    try:
        return {
            "symbol": last_row["symbol"],
            "signal": last_row["signal"],
            "checked_at_utc": last_row["checked_at_utc"]
        }
    except Exception as e:
        log_error(f"Error processing previous signal for {symbol}: {repr(e)}")
        return None


def update_prev_signal(symbol: str, rec: dict) -> None:
    """
    Update the last signal record for this symbol in last_signals.csv.
    If a row for this symbol already exists it is replaced, not duplicated.

    Args:
        symbol: e.g. "BTCUSDT"
        rec:    dict containing at least {"signal": ..., "checked_at_utc": ...}
    """
    try:
        # Build the record to save
        record = {
            "symbol": symbol,
            "signal": rec.get("signal"),
            "checked_at_utc": rec.get(
                "checked_at_utc",
                datetime.now(timezone.utc).isoformat()
            )
        }

        # TODO: if LAST_SIGNALS_FILE does not exist yet:
        #       create it by writing just this one record as a DataFrame
        #       pd.DataFrame([record]).to_csv(LAST_SIGNALS_FILE, index=False)
        #       then return early
        if not os.path.exists(LAST_SIGNALS_FILE):
            pd.DataFrame([record]).to_csv(LAST_SIGNALS_FILE, index=False)
            return
        # TODO: if file exists, read it into a DataFrame
        df = pd.read_csv(LAST_SIGNALS_FILE)
        # TODO: remove any existing row(s) for this symbol
        #       filter OUT rows where symbol matches (case-insensitive)
        df = df[df["symbol"].str.upper() != symbol.upper()] 
        # TODO: append the new record using pd.concat
        #       hint: pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        # TODO: save the updated DataFrame back to CSV with index=False
        df.to_csv(LAST_SIGNALS_FILE, index=False)   
    except Exception as e:
        log_error(f"update_prev_signal error: {repr(e)}")
