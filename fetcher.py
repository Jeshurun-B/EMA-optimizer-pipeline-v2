# fetcher.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: All external API calls. The ONLY file that fetches from the internet.
#
# PHILOSOPHY:
#   - Wrapper around utils.fetch_binance_klines for consistency
#   - Simple return types (DataFrame or int/None)
#   - Never raises exceptions (returns safe defaults)
#   - Retry logic for transient failures
#
# FUNCTIONS:
#   - fetch_candles()         → Fetch OHLCV for one coin, one timeframe
#   - fetch_fear_greed()      → Get Crypto Fear & Greed Index (0-100)
#   - fetch_forward_candles() → Fetch candles from specific timestamp forward (for labeler)
# ═════════════════════════════════════════════════════════════════════════════

from time import sleep
import requests
import pandas as pd
from config import FEAR_GREED_URL, REQUEST_TIMEOUT, BINANCE_BASE_URL
from utils import fetch_binance_klines, log_error, _inc_api_call


# ══════════════════════════════════════════════════════════════════════════════
#                              FETCH CANDLES (WRAPPER)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_candles(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data for one coin on one timeframe.
    
    This is just a thin wrapper around utils.fetch_binance_klines().
    Exists for consistency — all external fetches go through fetcher.py.
    
    Args:
        symbol:   Trading pair (e.g., "BTCUSDT")
        interval: Timeframe (e.g., "15m", "4h", "1d")
        limit:    Number of candles (default 300, max 1000)
    
    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
        Empty DataFrame on failure (never raises exception)
    
    Example:
        df = fetch_candles("BTCUSDT", "15m", 100)
        if not df.empty:
            print(f"Got {len(df)} candles")
            print(f"Latest close: ${df['close'].iloc[-1]}")
    """
    return fetch_binance_klines(symbol, interval, limit)


# ══════════════════════════════════════════════════════════════════════════════
#                              FETCH FEAR & GREED INDEX
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fear_greed() -> int:
    """
    Fetch the Crypto Fear & Greed Index from Alternative.me.
    
    What it is:
        Free API, no authentication needed.
        Returns integer 0-100:
            0-25:  Extreme Fear (panic selling, potential buying opportunity)
            25-45: Fear (cautious market)
            45-55: Neutral
            55-75: Greed (market confident)
            75-100: Extreme Greed (euphoria, potential bubble)
    
    Why we use it:
        Market sentiment indicator.
        Model might learn: "LONG signals work better during Fear"
                          "SHORT signals work better during Greed"
        Or maybe it has no predictive power — model will tell us.
    
    Returns:
        int (0-100) on success
        None on failure (network error, API down, etc)
    
    Example:
        fgi = fetch_fear_greed()
        if fgi is not None:
            if fgi < 30:
                print("Market is FEARFUL — potential buying opportunity")
            elif fgi > 70:
                print("Market is GREEDY — potential bubble")
        else:
            print("Failed to fetch F&G — using neutral default (50)")
    
    API Response Format:
        {
          "data": [
            {
              "value": "45",
              "value_classification": "Fear",
              "timestamp": "1617235200",
              "time_until_update": "..."
            }
          ]
        }
    """
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()  # Raise exception on 4xx/5xx
        data = resp.json()
        
        # Extract value from nested structure
        value = data["data"][0]["value"]
        return int(value)
    
    except Exception as e:
        log_error(f"fetch_fear_greed failed: {repr(e)}")
        return None  # Return None (caller will use default of 50)


# ══════════════════════════════════════════════════════════════════════════════
#                              FETCH FORWARD CANDLES (FOR LABELER)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_forward_candles(
    symbol: str,
    interval: str,
    start_time_ms: int,
    limit: int = 500
) -> pd.DataFrame:
    """
    Fetch candles starting from a specific timestamp going FORWARD in time.
    
    WHY THIS EXISTS:
        labeler.py needs to look into the FUTURE after a signal fired.
        Example: Signal at 10:30 AM → fetch next 500 candles to see what happened.
        
        Normal fetch_candles() gets MOST RECENT candles (ending at now).
        This function gets HISTORICAL candles starting at a specific time.
    
    DIFFERENCE FROM REGULAR FETCH:
        Regular:  Get 300 latest candles (ending at current time)
        This:     Get 500 candles starting at start_time_ms (going forward)
    
    RETRY LOGIC:
        Binance sometimes times out under load.
        We retry up to 3 times with 2-second delays.
        This prevents losing labels due to temporary network glitches.
    
    Args:
        symbol:        Trading pair (e.g., "BTCUSDT")
        interval:      Timeframe (e.g., "15m")
        start_time_ms: Unix timestamp in MILLISECONDS (e.g., 1617235200000)
        limit:         Number of candles to fetch (default 500)
    
    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
        Empty DataFrame on failure after all retries
    
    Example:
        # Signal fired at 2026-04-04 10:30:00 UTC
        signal_time = datetime(2026, 4, 4, 10, 30, tzinfo=timezone.utc)
        start_ms = int(signal_time.timestamp() * 1000)
        
        future = fetch_forward_candles("BTCUSDT", "15m", start_ms, 500)
        if not future.empty:
            print(f"Got {len(future)} candles after signal")
            max_price = future["high"].max()
            print(f"Highest price reached: ${max_price}")
    
    Why 500 candles?
        500 × 15min = 7500 minutes = 125 hours = 5.2 days
        Most trades resolve within 5 days.
        If not, labeler skips (trade still open).
    """
    
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_time_ms,  # Fetch FROM this time forward
        "limit":     limit,
    }
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # ── INCREMENT API COUNTER ─────────────────────────────────────────
            _inc_api_call()
            
            # ── MAKE REQUEST WITH LONGER TIMEOUT ──────────────────────────────
            # 
            # We use 15 seconds instead of default 10 seconds.
            # Why? This request might return more data (500 candles).
            # Binance needs more time to process and serialize the response.
            # 
            # 15 seconds is still reasonable:
            #   - Short enough to detect actual failures quickly
            #   - Long enough to handle slow responses
            # ──────────────────────────────────────────────────────────────────
            
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            
            # ── HANDLE EMPTY RESPONSE ─────────────────────────────────────────
            # 
            # If no candles available for this time range:
            #   raw = []
            # 
            # This happens if:
            #   - start_time is in the future (can't fetch future data)
            #   - Coin didn't exist yet at start_time
            #   - Time range has no trading activity
            # ──────────────────────────────────────────────────────────────────
            
            if not raw:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # ── PARSE RESPONSE ────────────────────────────────────────────────
            # 
            # Same structure as regular Binance klines.
            # 12 columns per candle, we keep 5.
            # ──────────────────────────────────────────────────────────────────
            
            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "num_trades",
                "taker_base_vol", "taker_quote_vol", "ignore"
            ])
            
            # Convert timestamp (milliseconds → datetime)
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            
            # Convert prices and volume (string → float)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Return only the 5 columns we need
            return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        
        except requests.exceptions.ReadTimeout:
            # ── TIMEOUT — RETRY LOGIC ─────────────────────────────────────────
            # 
            # Binance timed out (didn't respond within 15 seconds).
            # This is usually temporary (server overload).
            # 
            # Strategy:
            #   - Wait 2 seconds (let server recover)
            #   - Try again (up to 3 total attempts)
            #   - Log on each retry so we can see it happening
            # ──────────────────────────────────────────────────────────────────
            
            print(f"    ⚠️  Timeout on {symbol} (Attempt {attempt + 1}/{max_retries}). Retrying in 2s...")
            sleep(2)  # Wait before retry
        
        except Exception as e:
            # ── OTHER ERROR — STOP RETRYING ───────────────────────────────────
            # 
            # Non-timeout errors (404, 500, JSON parse error, etc).
            # These won't be fixed by retrying.
            # Log and return empty DataFrame immediately.
            # ──────────────────────────────────────────────────────────────────
            
            log_error(f"fetch_forward_candles error for {symbol}: {repr(e)}")
            break  # Exit retry loop, return empty DataFrame
    
    # ── ALL RETRIES EXHAUSTED ─────────────────────────────────────────────────
    # 
    # If we get here, either:
    #   - 3 timeouts in a row (Binance is struggling)
    #   - Non-timeout error occurred
    # 
    # Return empty DataFrame (labeler will skip this signal)
    # ──────────────────────────────────────────────────────────────────────────
    
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
