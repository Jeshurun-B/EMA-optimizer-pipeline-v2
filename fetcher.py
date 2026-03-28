# fetcher.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: All external API calls. The only file that fetches data from
#          the internet. Nothing else lives here.
# CONTAINS: Binance candle fetching, Fear & Greed index fetching.
# RULE: Every function returns a simple type (DataFrame or int or None).
#       Every function handles its own exceptions — never raises to caller.
# ─────────────────────────────────────────────────────────────────────────────

import requests
import pandas as pd
from config import FEAR_GREED_URL, REQUEST_TIMEOUT
from utils import fetch_binance_klines, log_error
from config import BINANCE_BASE_URL, REQUEST_TIMEOUT

# ── FETCH CANDLES ─────────────────────────────────────────────────────────────

def fetch_candles(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data for one coin on one timeframe.

    This is a clean wrapper around utils.fetch_binance_klines.
    All the actual HTTP logic lives in utils — this function just calls it
    and returns the result. The reason we have this wrapper is that
    ALL external data fetching must go through fetcher.py, not scattered
    across multiple files.

    Args:
        symbol:   coin pair e.g. "BTCUSDT", "ETHUSDT"
        interval: candle size e.g. "15m", "4h", "1d"
        limit:    how many candles to fetch — default 300

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        Returns empty DataFrame on any failure — never raises.

    Usage in runner.py will look like:
        ltf_df  = fetch_candles("BTCUSDT", "15m", 300)
        htf_df  = fetch_candles("BTCUSDT", "4h",  200)
        day_df  = fetch_candles("BTCUSDT", "1d",  100)
    """
    # TODO: call fetch_binance_klines(symbol, interval, limit) from utils
    result = fetch_binance_klines(symbol, interval, limit)
    # TODO: return the result directly — utils already handles all errors
    return result


# ── FETCH FEAR & GREED INDEX ──────────────────────────────────────────────────

def fetch_fear_greed() -> int:
    """
    Fetch the current Crypto Fear & Greed Index from Alternative.me.
    This is a free API — no authentication needed.

    The index is a single number from 0 to 100:
        0-24   = Extreme Fear  (market is very bearish, people are panic selling)
        25-49  = Fear
        50-74  = Greed
        75-100 = Extreme Greed (market is very bullish, people are overconfident)

    Why this matters for our model:
        EMA crossover signals behave differently depending on market sentiment.
        A LONG signal during Extreme Fear has different odds than during Greed.
        This gives the model context about the broader market environment.

    The API response looks like this:
        {
            "data": [
                {
                    "value": "65",
                    "value_classification": "Greed",
                    "timestamp": "1234567890"
                }
            ]
        }

    Returns:
        Integer between 0 and 100 — the current fear/greed score.
        Returns None if the request fails for any reason.
    """
    try:
        # TODO: make a GET request to FEAR_GREED_URL with timeout=REQUEST_TIMEOUT
        resp = requests.get(FEAR_GREED_URL, timeout=REQUEST_TIMEOUT)

        # TODO: call resp.raise_for_status() to catch HTTP errors (4xx, 5xx)
        resp.raise_for_status()
        # TODO: parse the JSON response
        data = resp.json()

        # TODO: extract the value from the nested structure
        # the value is at data["data"][0]["value"]
        # it comes back as a STRING — cast it to int
        value = int(data["data"][0]["value"])

        # TODO: return the value
        return value

    except Exception as e:
        # TODO: log the error
        log_error(f"fetch_fear_greed failed: {repr(e)}")

        # TODO: return None — the pipeline will use None as the fear_greed_index
        # and the model will handle missing values during preprocessing
        return None
def fetch_forward_candles(symbol: str, interval: str, start_time_ms: int, limit: int = 100) -> pd.DataFrame:
    """
    Fetches candles starting from a specific historical timestamp moving forward.
    This is critical for labeling — we need to see what happened AFTER the signal.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    
    # We pass 'startTime' instead of 'endTime' to paginate forward into the future
    params = {
        "symbol": symbol, 
        "interval": interval, 
        "startTime": start_time_ms, 
        "limit": limit
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw_data = resp.json()
        
        if not raw_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(raw_data, columns=[
            "open_time", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "trades", 
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        
        # Convert timestamp to a proper datetime object for easy sorting
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        
        # Ensure our pricing data is strictly numeric so we can do math on it
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        return df[["timestamp", "open", "high", "low", "close"]]
        
    except Exception as e:
        log_error(f"fetch_forward_candles error for {symbol}: {repr(e)}")
        return pd.DataFrame()