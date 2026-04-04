# db.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: All database operations. The ONLY file that talks to Supabase.
# RULE: No other file imports supabase directly — everything goes through here.
#       Every function returns a simple type (bool, list, DataFrame).
#       Every function handles its own exceptions — never raises to caller.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from utils import log_error

# ── CLIENT SETUP ──────────────────────────────────────────────────────────────
# We create ONE Supabase client here at the top of the file.
# Every function below reuses this same client — we don't create a new one
# inside each function because that would be slow and wasteful.
#
# create_client() takes two arguments:
#   - SUPABASE_URL: your project URL e.g. "https://xxxx.supabase.co"
#   - SUPABASE_KEY: your publishable API key
#
# If either of these is None (missing from .env), this line will crash loudly
# when the file is imported — which is correct behaviour. Nothing works without
# a database connection, so we want to fail immediately and clearly.
#
# TODO: 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# This is the name of our table in Supabase.
# We store it as a constant so if we ever rename the table,
# we only need to change it in one place.
TABLE = "signals"


# ── INSERT SIGNAL ─────────────────────────────────────────────────────────────

def insert_signal(record: dict) -> bool:
    """
    Insert one signal row into the signals table in Supabase.

    'record' is a Python dictionary where:
        - keys   = column names in the signals table
        - values = the data to store for this signal

    Example of what record looks like:
        {
            "symbol": "BTCUSDT",
            "signal": "LONG",
            "price": 65000.0,
            "adx_ltf": 32.5,
            "rsi_ltf": 61.0,
            "status": "pending",
            ... all other columns
        }

    Returns:
        True  — row was inserted successfully
        False — something went wrong (logged automatically)

    Why we use try/except:
        If Supabase is down, or the record has a duplicate (same symbol +
        same timestamp), or a column name is wrong — it will raise an error.
        We catch that here and return False so the pipeline can continue
        scanning other coins instead of crashing entirely.
    """
    try:
        # This is the Supabase pattern for inserting a row.
        # .table(TABLE)  — tells Supabase which table to use
        # .insert(record) — the dict of data to insert as one row
        # .execute()      — actually sends the request to Supabase
        #
        # TODO: write the insert call using the pattern above
        supabase.table(TABLE).insert(record).execute()

        # TODO: return True to tell the caller the insert worked
        return True
        pass

    except Exception as e:
        # If anything went wrong, log it so we can see what failed.
        # We include the symbol so we know which coin caused the issue.
        # Then return False so the pipeline keeps running.
        #
        # TODO: log the error — use this pattern:
        log_error(f"insert_signal failed for {record.get('symbol')}: {repr(e)}")
        #
        #TODO: return False
        return False
        pass


# ── FETCH PENDING ─────────────────────────────────────────────────────────────

def fetch_pending() -> list:
    """
    Get all signals from Supabase where status = 'pending'.

    'pending' means the signal was detected and stored, but we haven't yet
    checked what happened to the price afterwards (no labels computed yet).
    labeler.py calls this function to find signals that need to be labeled.

    Returns:
        A list of dicts — each dict is one row from the database.
        Example: [{"id": 1, "symbol": "BTCUSDT", "signal": "LONG", ...}, ...]

        Returns [] (empty list) if there are no pending signals or if
        something goes wrong. Never raises an exception.

    Why list of dicts instead of DataFrame:
        labeler.py processes one signal at a time in a loop.
        Iterating over a list of dicts is simpler than iterating DataFrame rows.
        fetch_all_labeled() returns a DataFrame because train.py needs to do
        maths across all rows at once — different use case.
    """
    try:
        # Supabase query pattern:
        # .table(TABLE)          — which table
        # .select("*")           — select all columns (* means everything)
        # .eq("status","pending")— WHERE status = 'pending'
        # .execute()             — send the request
        #
        # The response object has a .data attribute.
        # response.data is a list of dicts — one dict per matching row.
        #
        # TODO: write the query using the pattern above
        response = supabase.table(TABLE).select("*").eq("status", "pending").execute()
        #
        # TODO: return response.data
        return response.data

    except Exception as e:
        # TODO: log the error
        log_error(f"fetch_pending failed: {repr(e)}")
        #
        # TODO: return empty list so labeler.py gets [] and skips gracefully
        # return []
        return []


# ── UPDATE SIGNAL LABELS ──────────────────────────────────────────────────────

def update_signal_labels(signal_id: int, updates: dict) -> bool:
    """
    Update specific columns on one signal row, identified by its id.

    Called by labeler.py after it has computed the outcome of a trade signal.
    For example, after checking what the price did after the signal fired,
    labeler.py calls this to save:
        - max_adverse_excursion  (how far price went against us)
        - max_favorable_excursion (how far price went in our favour)
        - trade_quality          (1 = good trade, 0 = bad trade)
        - expected_move_pct      (actual % move achieved)
        - status                 (set to "analyzed" so we don't process it again)

    Args:
        signal_id: the integer 'id' of the row to update (primary key)
        updates:   dict of column names and their new values
                   e.g. {"trade_quality": 1, "status": "analyzed"}

    Returns:
        True  — update was successful
        False — something went wrong (logged automatically)

    Why we identify rows by id and not by symbol+timestamp:
        'id' is the primary key — guaranteed unique.
        Using symbol+timestamp would be more complex and slightly slower.
    """
    try:
        # Supabase update pattern:
        # .table(TABLE)            — which table
        # .update(updates)         — dict of columns to change and their new values
        # .eq("id", signal_id)     — WHERE id = signal_id (only update this one row)
        # .execute()               — send the request
        #
        # TODO: write the update call using the pattern above
        supabase.table(TABLE).update(updates).eq("id", signal_id).execute()
        #
        # TODO: return True
        return True

    except Exception as e:
        # TODO: log the error — include signal_id so we know which row failed
        log_error(f"update_signal_labels failed for id={signal_id}: {repr(e)}")
        #
        # TODO: return False
        return False


# ── FETCH ALL LABELED ─────────────────────────────────────────────────────────

def fetch_all_labeled() -> pd.DataFrame:
    """
    Get ALL signals where status = 'analyzed' as a pandas DataFrame.

    'analyzed' means the signal has been fully labeled — it has trade_quality,
    MAE, MFE, and expected_move_pct filled in. These are the rows we use
    to train our ML models.

    This function is called by train.py running on Google Colab to pull
    the complete labeled dataset for model training.

    Returns:
        pandas DataFrame with all analyzed signals, sorted oldest first.
        Returns empty DataFrame if no labeled signals exist or on failure.

    Why we sort by time (oldest first):
        Our train/test split is time-based — we train on old data and test
        on recent data. The data MUST be in chronological order before we
        split it. If it's shuffled, we accidentally train on future data
        which makes the model look better than it really is (lookahead bias).
    """
    try:
        # TODO: query Supabase for all rows where status = "analyzed"
        # Same pattern as fetch_pending but with "analyzed" instead of "pending"
        response = supabase.table(TABLE).select("*").eq("status","analyzed").execute()

        # TODO: convert response.data (list of dicts) into a pandas DataFrame
        df = pd.DataFrame(response.data)

        # TODO: if the DataFrame is empty (no labeled signals yet), return it as-is
        if df.empty:
            return df

        # TODO: convert the checked_at_utc column from string to proper datetime
        #This is needed so we can sort by time correctly
        df["checked_at_utc"] = pd.to_datetime(df["checked_at_utc"], utc=True)

        #TODO: sort the DataFrame by checked_at_utc ascending (oldest row first)
        df = df.sort_values("checked_at_utc", ascending=True)

        #TODO: reset the index so rows are numbered 0, 1, 2, 3... after sorting
        df = df.reset_index(drop=True)

        # TODO: return the DataFrame
        return df

    except Exception as e:
        # TODO: log the error
        log_error(f"fetch_all_labeled failed: {repr(e)}")
        #
        # TODO: return empty DataFrame — same columns as the signals table
        return pd.DataFrame()


# ── FETCH NEXT SIGNAL TIME (Used by Labeler) ──────────────────────────────────

def fetch_next_signal_time(symbol: str, current_time_utc: str) -> str | None:
    """
    Finds the exact timestamp of the NEXT signal that fired for a specific coin.
    
    Why this exists: 
    Instead of making the labeler manually fetch 400 future candles and 
    recalculate the EMAs to find when a trade ends, it's 1000x faster to just 
    ask the database: "When was the next time you recorded a crossover for this coin?"
    
    Args:
        symbol: The coin we are checking (e.g., "BTCUSDT")
        current_time_utc: The exact timestamp of our entry signal.
        
    Returns:
        The ISO string timestamp of the next signal, or None if the trade 
        is still open (meaning no opposite signal has fired yet).
    """
    try:
        # 1. Query the Supabase 'signals' table
        # 2. Filter for the exact coin we are analyzing (.eq)
        # 3. Filter for timestamps STRICTLY GREATER THAN our entry time (.gt)
        # 4. Sort them chronologically, oldest first (.order desc=False)
        # 5. Grab only the very first one it finds (.limit 1)
        response = supabase.table(TABLE).select("checked_at_utc") \
            .eq("symbol", symbol) \
            .gt("checked_at_utc", current_time_utc) \
            .order("checked_at_utc", desc=False) \
            .limit(1) \
            .execute()
            
        # If the query found a future signal, return its timestamp
        if response.data:
            return response.data[0]["checked_at_utc"]
            
        # If response.data is empty, it means no signal has fired since our entry.
        # The trade is still live.
        return None 
        
    except Exception as e:
        # If the database connection fails, log it and return None 
        # so the pipeline doesn't crash.
        log_error(f"fetch_next_signal_time error: {repr(e)}")
        return None