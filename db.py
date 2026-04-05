# db.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: All database operations. The ONLY file that talks to Supabase.
#
# PHILOSOPHY:
#   - Single Responsibility: No other file imports supabase directly
#   - Simple Returns: Every function returns bool, list, or DataFrame (never complex types)
#   - Never Crash: Every function handles its own exceptions, never raises to caller
#   - Graceful Degradation: Database errors are logged but don't stop the pipeline
#
# FUNCTIONS:
#   - insert_signal()         → Insert one signal row
#   - fetch_pending()         → Get all pending signals for labeler
#   - update_signal_labels()  → Update a signal with outcome data
#   - fetch_all_labeled()     → Get all analyzed signals for training
#   - fetch_next_signal_time()→ Find when next signal fired (for labeler)
# ═════════════════════════════════════════════════════════════════════════════

import pandas as pd
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from utils import log_error


# ══════════════════════════════════════════════════════════════════════════════
#                              CLIENT SETUP
# ══════════════════════════════════════════════════════════════════════════════
# 
# We create ONE Supabase client at module load time.
# Every function below reuses this same client instance.
# 
# Why one client?
#   - Connection pooling (reusing connections is faster)
#   - Authenticates once, not on every request
#   - Less memory overhead
# 
# If SUPABASE_URL or SUPABASE_KEY is None (missing from .env):
#   - This line crashes IMMEDIATELY when the file is imported
#   - This is CORRECT behavior (fail fast, fail loud)
#   - Nothing works without a database, so we want to know right away
# ══════════════════════════════════════════════════════════════════════════════

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Table name constant — change here to rename table globally
TABLE = "signals"


# ══════════════════════════════════════════════════════════════════════════════
#                              INSERT SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def insert_signal(record: dict) -> bool:
    """
    Insert one signal row into the Supabase signals table.
    
    This is called by runner.py every time a crossover is detected.
    
    Args:
        record: Dictionary where keys = column names, values = data
                Must include: symbol, signal, checked_at_utc, status, price
                Plus all 35 feature columns
    
    Returns:
        True  → Row inserted successfully
        False → Insert failed (duplicate, network error, schema mismatch, etc)
    
    Example:
        signal = {
            "symbol": "BTCUSDT",
            "signal": "LONG",
            "checked_at_utc": "2026-04-04T10:30:00+00:00",
            "price": 65000.0,
            "adx_ltf": 32.5,
            ... all other features
        }
        success = insert_signal(signal)
        if success:
            print("✅ Signal saved")
        else:
            print("❌ Insert failed (see error.log)")
    
    Error Cases (all return False):
        - Duplicate: (symbol, checked_at_utc) already exists in table
        - Network timeout: Supabase unreachable
        - Schema mismatch: record has columns that don't exist in table
        - Bad data type: e.g., string in a float column
        - RLS violation: Row Level Security policy blocked insert (shouldn't happen — RLS disabled)
    
    Why return False instead of raising?
        One coin's insert failing shouldn't crash the entire pipeline.
        We log the error and move to the next coin.
    """
    try:
        # ── SUPABASE INSERT PATTERN ───────────────────────────────────────────
        # 
        # .table(TABLE)     → Select which table to work with
        # .insert(record)   → Insert this dict as one row
        # .execute()        → Actually send the request to Supabase
        # 
        # What happens under the hood:
        #   1. Supabase converts dict → SQL INSERT statement
        #   2. PostgreSQL validates schema and constraints
        #   3. If UNIQUE constraint violated → exception raised
        #   4. If successful → new row written, ID auto-generated
        # ──────────────────────────────────────────────────────────────────────
        
        supabase.table(TABLE).insert(record).execute()
        return True  # Success
    
    except Exception as e:
        # ── ERROR HANDLING ────────────────────────────────────────────────────
        # 
        # Common errors:
        #   - UniqueViolation: (symbol, checked_at_utc) duplicate
        #   - ConnectionError: Network timeout
        #   - SchemaError: Column doesn't exist in table
        # 
        # We log the error with symbol for debugging context.
        # Then return False so caller knows insert failed.
        # ──────────────────────────────────────────────────────────────────────
        
        log_error(f"insert_signal failed for {record.get('symbol')}: {repr(e)}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#                              FETCH PENDING SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_pending() -> list:
    """
    Get all signals where status = 'pending' from Supabase.
    
    'pending' = signal detected and stored, but NOT YET LABELED.
    labeler.py calls this to find signals that need outcome analysis.
    
    Returns:
        list of dicts — each dict is one pending signal row
        [] (empty list) — if no pending signals OR if error occurred
    
    Example:
        pending = fetch_pending()
        if pending:
            print(f"Found {len(pending)} signals to label")
            for signal in pending:
                print(f"  {signal['symbol']} {signal['signal']} at {signal['checked_at_utc']}")
        else:
            print("No pending signals")
    
    Why list of dicts (not DataFrame)?
        labeler.py processes one signal at a time in a loop.
        for signal in pending: ...
        Iterating dicts is simpler than DataFrame.iterrows()
    
    Status values:
        'pending'  → Not yet labeled
        'analyzed' → Labeled with outcome data
    """
    try:
        # ── SUPABASE SELECT WITH FILTER ───────────────────────────────────────
        # 
        # .table(TABLE)           → Which table
        # .select("*")            → All columns (* = everything)
        # .eq("status", "pending")→ WHERE status = 'pending'
        # .execute()              → Send request
        # 
        # Response structure:
        #   response.data = [
        #       {"id": 1, "symbol": "BTCUSDT", "signal": "LONG", ...},
        #       {"id": 2, "symbol": "ETHUSDT", "signal": "SHORT", ...},
        #   ]
        # ──────────────────────────────────────────────────────────────────────
        
        response = supabase.table(TABLE).select("*").eq("status", "pending").execute()
        return response.data  # List of dicts
    
    except Exception as e:
        log_error(f"fetch_pending failed: {repr(e)}")
        return []  # Return empty list (safe fallback)


# ══════════════════════════════════════════════════════════════════════════════
#                              UPDATE SIGNAL LABELS
# ══════════════════════════════════════════════════════════════════════════════

def update_signal_labels(signal_id: int, updates: dict) -> bool:
    """
    Update specific columns on one signal row by its ID.
    
    Called by labeler.py after computing trade outcome.
    Updates the row with:
        - max_price_after, min_price_after
        - max_move_up_pct, max_move_down_pct
        - time_of_max_price, time_of_min_price
        - candles_to_max_price, candles_to_min_price
        - status = "analyzed"
    
    Args:
        signal_id: Primary key (integer ID of the row)
        updates:   Dict of {column_name: new_value}
    
    Returns:
        True  → Update successful
        False → Update failed (row not found, network error, etc)
    
    Example:
        updates = {
            "max_price_after": 65300.0,
            "min_price_after": 64800.0,
            "max_move_up_pct": 0.46,
            "max_move_down_pct": 0.31,
            "status": "analyzed"
        }
        success = update_signal_labels(42, updates)
    
    Why identify by ID (not symbol + timestamp)?
        - ID is primary key (guaranteed unique)
        - Faster lookup (indexed)
        - Simpler query (one column match vs two)
    """
    try:
        # ── SUPABASE UPDATE PATTERN ───────────────────────────────────────────
        # 
        # .table(TABLE)             → Which table
        # .update(updates)          → Set these columns to new values
        # .eq("id", signal_id)      → WHERE id = signal_id
        # .execute()                → Send request
        # 
        # What happens:
        #   1. Supabase finds row with this ID
        #   2. Updates ONLY the columns in updates dict
        #   3. Other columns unchanged
        #   4. If ID not found → no error, just no rows updated
        # ──────────────────────────────────────────────────────────────────────
        
        supabase.table(TABLE).update(updates).eq("id", signal_id).execute()
        return True
    
    except Exception as e:
        log_error(f"update_signal_labels failed for id={signal_id}: {repr(e)}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#                              FETCH ALL LABELED SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_labeled() -> pd.DataFrame:
    """
    Get ALL signals where status = 'analyzed' as a pandas DataFrame.
    
    'analyzed' = signal has been labeled with outcome data (trade results).
    These are the rows used for ML model training.
    
    Called by train.py (runs on Google Colab) to download full dataset.
    
    Returns:
        DataFrame with all analyzed signals, sorted chronologically (oldest first)
        Empty DataFrame if no labeled signals exist OR on error
    
    Why sort by time (oldest first)?
        ML train/test split is TIME-BASED (not random shuffle).
        We train on old data, test on recent data.
        Data MUST be in chronological order before split.
        If shuffled → lookahead bias (train on future, test on past).
    
    Example:
        df = fetch_all_labeled()
        if not df.empty:
            print(f"Downloaded {len(df)} labeled signals")
            print(f"Date range: {df['checked_at_utc'].min()} to {df['checked_at_utc'].max()}")
        else:
            print("No labeled data yet — run labeler.py first")
    
    Columns in DataFrame:
        - id, symbol, signal, checked_at_utc, price
        - All 35 feature columns
        - max_price_after, min_price_after, max_move_up_pct, max_move_down_pct
        - time_of_max_price, time_of_min_price
        - candles_to_max_price, candles_to_min_price
        - status (will be "analyzed" for all rows)
    """
    try:
        # ── FETCH ALL ANALYZED SIGNALS ────────────────────────────────────────
        
        response = supabase.table(TABLE).select("*").eq("status", "analyzed").execute()
        df = pd.DataFrame(response.data)
        
        # ── HANDLE EMPTY RESULT ───────────────────────────────────────────────
        # 
        # If no signals analyzed yet → response.data = []
        # pd.DataFrame([]) creates empty DataFrame with no columns
        # We return it as-is (caller checks df.empty)
        # ──────────────────────────────────────────────────────────────────────
        
        if df.empty:
            return df  # Empty DataFrame (no data to sort)
        
        # ── CONVERT TIMESTAMP COLUMN ──────────────────────────────────────────
        # 
        # checked_at_utc comes from DB as string: "2026-04-04T10:30:00+00:00"
        # We convert to pandas datetime for:
        #   - Proper sorting (chronological order)
        #   - Time-based train/test split
        #   - Feature engineering (hour, day of week)
        # 
        # utc=True makes it timezone-aware (avoids timezone bugs later)
        # ──────────────────────────────────────────────────────────────────────
        
        df["checked_at_utc"] = pd.to_datetime(df["checked_at_utc"], utc=True)
        
        # ── SORT CHRONOLOGICALLY ──────────────────────────────────────────────
        # 
        # ascending=True → oldest rows first
        # This is CRITICAL for time-based train/test split
        # ──────────────────────────────────────────────────────────────────────
        
        df = df.sort_values("checked_at_utc", ascending=True)
        
        # ── RESET INDEX ───────────────────────────────────────────────────────
        # 
        # After sorting, row indices are out of order (might be 5, 2, 8, 1, ...)
        # Reset to 0, 1, 2, 3, ... for clean DataFrame
        # drop=True → don't keep old index as a new column
        # ──────────────────────────────────────────────────────────────────────
        
        df = df.reset_index(drop=True)
        
        return df
    
    except Exception as e:
        log_error(f"fetch_all_labeled failed: {repr(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error


# ══════════════════════════════════════════════════════════════════════════════
#                              FETCH NEXT SIGNAL TIME
# ══════════════════════════════════════════════════════════════════════════════

def fetch_next_signal_time(symbol: str, current_time_utc: str) -> str | None:
    """
    Find the timestamp of the NEXT signal that fired for a specific coin.
    
    WHY THIS EXISTS:
        labeler.py needs to know: when did this trade END?
        Trade ends when opposite signal fires (LONG ends when SHORT fires).
        
        Instead of fetching 400 future candles and recalculating EMAs,
        we just ask the database: "When was the next crossover for this coin?"
        
        This is 1000x faster and uses the data we already collected.
    
    Args:
        symbol:           Coin pair (e.g., "BTCUSDT")
        current_time_utc: ISO timestamp of entry signal (e.g., "2026-04-04T10:30:00+00:00")
    
    Returns:
        str:  ISO timestamp of next signal (e.g., "2026-04-04T16:45:00+00:00")
        None: No future signal found (trade still open)
    
    Example:
        Entry signal: LONG at 10:30 AM
        next_time = fetch_next_signal_time("BTCUSDT", "2026-04-04T10:30:00+00:00")
        
        if next_time:
            print(f"Trade closed at {next_time}")
            # Fetch candles from 10:30 to next_time, measure outcome
        else:
            print("Trade still open — skip labeling for now")
    
    Logic:
        1. Filter for this exact symbol
        2. Filter for timestamps AFTER current_time
        3. Sort chronologically (oldest first)
        4. Take first result (the very next signal)
    """
    try:
        # ── QUERY PATTERN ─────────────────────────────────────────────────────
        # 
        # .select("checked_at_utc")     → Only fetch timestamp column (faster)
        # .eq("symbol", symbol)         → WHERE symbol = 'BTCUSDT'
        # .gt("checked_at_utc", ...)    → WHERE checked_at_utc > current_time
        # .order("checked_at_utc", ...)  → ORDER BY checked_at_utc ASC
        # .limit(1)                     → LIMIT 1 (only first result)
        # 
        # gt = "greater than" (Supabase query filter)
        # desc=False = ascending order (oldest first)
        # ──────────────────────────────────────────────────────────────────────
        
        response = (
            supabase.table(TABLE)
            .select("checked_at_utc")
            .eq("symbol", symbol)
            .gt("checked_at_utc", current_time_utc)
            .order("checked_at_utc", desc=False)
            .limit(1)
            .execute()
        )
        
        # ── HANDLE RESULT ─────────────────────────────────────────────────────
        # 
        # If query found a future signal:
        #   response.data = [{"checked_at_utc": "2026-04-04T16:45:00+00:00"}]
        # 
        # If no future signal exists:
        #   response.data = []
        # ──────────────────────────────────────────────────────────────────────
        
        if response.data:
            return response.data[0]["checked_at_utc"]
        
        return None  # No future signal found (trade still open)
    
    except Exception as e:
        log_error(f"fetch_next_signal_time error for {symbol}: {repr(e)}")
        return None  # Return None on error (safer than crashing)
