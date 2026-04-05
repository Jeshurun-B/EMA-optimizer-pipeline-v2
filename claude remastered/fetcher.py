# fetcher.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: All external API calls. The only file that fetches from the internet.
# RULE: Every function returns a simple type. Never raises to caller.
# ─────────────────────────────────────────────────────────────────────────────


from time import time

import requests
import pandas as pd
from config import FEAR_GREED_URL, REQUEST_TIMEOUT, BINANCE_BASE_URL
from utils import fetch_binance_klines, log_error, _inc_api_call


def fetch_candles(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV candles for one coin on one timeframe.
    Wrapper around utils.fetch_binance_klines.
    Returns empty DataFrame on failure — never raises.
    """
    return fetch_binance_klines(symbol, interval, limit)


def fetch_fear_greed() -> int:
    """
    Fetch the Crypto Fear & Greed Index from Alternative.me (free, no auth).
    Returns integer 0-100, or None on failure.
    """
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return int(data["data"][0]["value"])
    except Exception as e:
        log_error(f"fetch_fear_greed failed: {repr(e)}")
        return None


def fetch_forward_candles(
    symbol: str,
    interval: str,
    start_time_ms: int,
    limit: int = 500
) -> pd.DataFrame:
    """
    Fetch candles starting from a specific timestamp going forward in time.
    Includes a retry mechanism for handling Binance API timeouts gracefully.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_time_ms,
        "limit":     limit,
    }
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            _inc_api_call()
            # We override the 10s default with 15s to give Binance more breathing room
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "num_trades",
                "taker_base_vol", "taker_quote_vol", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

        except requests.exceptions.ReadTimeout:
            # If Binance times out, wait 2 seconds and try again
            print(f"    ⚠️ Timeout on {symbol} (Attempt {attempt + 1}/{max_retries}). Retrying in 2s...")
            time.sleep(2)
            
        except Exception as e:
            log_error(f"fetch_forward_candles error for {symbol}: {repr(e)}")
            break
            
    # If all retries fail, return empty DataFrame safely
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])