# fetcher.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: All external API calls. The only file that fetches from the internet.
# RULE: Every function returns a simple type. Never raises to caller.
# ─────────────────────────────────────────────────────────────────────────────

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
    limit: int = 200
) -> pd.DataFrame:
    """
    Fetch candles starting from a specific timestamp going forward in time.
    Used by labeler.py to see what happened AFTER a signal fired.

    Args:
        symbol:        coin pair e.g. "BTCUSDT"
        interval:      candle size e.g. "15m"
        start_time_ms: UNIX timestamp in milliseconds — fetch candles from here
        limit:         how many candles to fetch after start_time

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        Returns empty DataFrame on failure.
    """
    try:
        _inc_api_call()
        resp = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/klines",
            params={
                "symbol":    symbol,
                "interval":  interval,
                "startTime": start_time_ms,
                "limit":     limit,
            },
            timeout=REQUEST_TIMEOUT
        )
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

    except Exception as e:
        log_error(f"fetch_forward_candles error for {symbol}: {repr(e)}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])