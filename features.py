# features.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Compute all 35 technical indicators and features for our ML model.
# RULE: Pure functions only. Takes DataFrames, returns a dictionary.
#       No API calls, no database calls. Just math.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import ta
import math
from config import EMA_FAST, EMA_SLOW, ADX_LEN, ATR_STOP_MULTIPLIER
from utils import log_error
from fetcher import fetch_forward_candles
from config import BINANCE_BASE_URL, REQUEST_TIMEOUT
from datetime import datetime, timezone 
import requests 


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _adx_series(df: pd.DataFrame, length: int) -> pd.Series:
    ind = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=length, fillna=True)
    return ind.adx()

def _rsi_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    ind = ta.momentum.RSIIndicator(close=df["close"], window=length, fillna=True)
    return ind.rsi()

def _atr_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    ind = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=length, fillna=True)
    return ind.average_true_range()

def _roc_series(series: pd.Series, length: int = 9) -> pd.Series:
    ind = ta.momentum.ROCIndicator(close=series, window=length, fillna=True)
    return ind.roc()

def _macd_hist_series(series: pd.Series) -> pd.Series:
    ind = ta.trend.MACD(close=series, fillna=True)
    return ind.macd_diff() # The histogram

def _bb_width_series(series: pd.Series) -> pd.Series:
    ind = ta.volatility.BollingerBands(close=series, fillna=True)
    return ind.bollinger_wband()


# ── MAIN FEATURE COMPUTATION ──────────────────────────────────────────────────

def compute_features(ltf_df: pd.DataFrame, htf_4h_df: pd.DataFrame, htf_1d_df: pd.DataFrame, fear_greed: int, btc_bias: bool, symbol: str) -> dict:
    """
    Computes all 35 features across 3 timeframes and context data.
    Returns a dictionary, or None if there isn't enough data.
    """
    # Check if we have enough data across all timeframes
    if ltf_df.empty or htf_4h_df.empty or htf_1d_df.empty:
        return None
    if len(ltf_df) < 30 or len(htf_4h_df) < 30 or len(htf_1d_df) < 20:
        return None

    # ── 1. LTF (15m) COMPUTATIONS ──
    ema_fast_ltf_s = _ema(ltf_df["close"], EMA_FAST)
    ema_slow_ltf_s = _ema(ltf_df["close"], EMA_SLOW)
    adx_ltf_s      = _adx_series(ltf_df, ADX_LEN)
    rsi_ltf_s      = _rsi_series(ltf_df, 14)
    atr_ltf_s      = _atr_series(ltf_df, 14)
    roc_ltf_s      = _roc_series(ltf_df["close"], 9)
    macd_hist_ltf_s= _macd_hist_series(ltf_df["close"])
    bb_width_ltf_s = _bb_width_series(ltf_df["close"])
    vol_ma_ltf_s   = ltf_df["volume"].rolling(window=20, min_periods=1).mean()

    # Extract LTF latest/prev
    latest_ltf     = ltf_df.iloc[-1]
    prev_ltf       = ltf_df.iloc[-2]
    
    close_ltf      = float(latest_ltf["close"])
    low_ltf        = float(latest_ltf["low"])
    high_ltf       = float(latest_ltf["high"])
    vol_ltf        = float(latest_ltf["volume"])
    
    timestamp_utc  = pd.to_datetime(latest_ltf["timestamp"]) # For time context features

    ema_fast_ltf   = float(ema_fast_ltf_s.iloc[-1])
    ema_slow_ltf   = float(ema_slow_ltf_s.iloc[-1])
    ema_fast_prev  = float(ema_fast_ltf_s.iloc[-2])
    ema_slow_prev  = float(ema_slow_ltf_s.iloc[-2])
    atr_ltf        = float(atr_ltf_s.iloc[-1])
    adx_ltf        = float(adx_ltf_s.iloc[-1])

    # ── 2. HTF (4h) COMPUTATIONS ──
    ema_fast_4h_s  = _ema(htf_4h_df["close"], EMA_FAST)
    ema_slow_4h_s  = _ema(htf_4h_df["close"], EMA_SLOW)
    adx_4h_s       = _adx_series(htf_4h_df, ADX_LEN)
    rsi_4h_s       = _rsi_series(htf_4h_df, 14)
    macd_hist_4h_s = _macd_hist_series(htf_4h_df["close"])
    
    ema_fast_4h    = float(ema_fast_4h_s.iloc[-1])
    ema_slow_4h    = float(ema_slow_4h_s.iloc[-1])

    # ── 3. HTF (1D) COMPUTATIONS ──
    ema_fast_1d_s  = _ema(htf_1d_df["close"], EMA_FAST)
    ema_slow_1d_s  = _ema(htf_1d_df["close"], EMA_SLOW)
    
    ema_fast_1d    = float(ema_fast_1d_s.iloc[-1])
    ema_slow_1d    = float(ema_slow_1d_s.iloc[-1])

    # ── 4. DERIVED FEATURES & TRADE MANAGEMENT ──
    candle_range = high_ltf - low_ltf
    crossover_candle_strength = (close_ltf - low_ltf) / candle_range if candle_range > 0 else 0.5
    
    # Swing High/Low over the last 10 candles
    swing_high = float(ltf_df["high"].tail(10).max())
    swing_low  = float(ltf_df["low"].tail(10).max()) # Typo fix: Should be .min() but mirroring standard practice
    swing_low  = float(ltf_df["low"].tail(10).min())
    
    atr_stop_distance = atr_ltf * ATR_STOP_MULTIPLIER

    # ── 5. ASSEMBLE DICTIONARY ──
    features = {
        "price": close_ltf, # Required for signals.py and db
        
        # Group 1: EMA
        "ema_fast_ltf": round(ema_fast_ltf, 6),
        "ema_slow_ltf": round(ema_slow_ltf, 6),
        "ema_fast_slope": round(((ema_fast_ltf - ema_fast_prev) / ema_fast_prev) * 100.0 if ema_fast_prev else 0.0, 4),
        "ema_slow_slope": round(((ema_slow_ltf - ema_slow_prev) / ema_slow_prev) * 100.0 if ema_slow_prev else 0.0, 4),
        "ema_separation": round(((ema_fast_ltf - ema_slow_ltf) / ema_slow_ltf) * 100.0 if ema_slow_ltf else 0.0, 4),
        "price_above_both_emas": bool(close_ltf > ema_fast_ltf and close_ltf > ema_slow_ltf),
        "crossover_candle_strength": round(crossover_candle_strength, 4),

        # Group 2: Trend
        "adx_ltf": round(adx_ltf, 2),
        "adx_slope": round(adx_ltf - float(adx_ltf_s.iloc[-2]), 2),
        "adx_4h": round(float(adx_4h_s.iloc[-1]), 2),
        "macd_histogram_ltf": round(float(macd_hist_ltf_s.iloc[-1]), 6),
        "macd_histogram_4h": round(float(macd_hist_4h_s.iloc[-1]), 6),

        # Group 3: HTF Alignment
        "htf_4h_bias": bool(ema_fast_4h > ema_slow_4h),
        "htf_1d_bias": bool(ema_fast_1d > ema_slow_1d),
        "ema_separation_4h": round(((ema_fast_4h - ema_slow_4h) / ema_slow_4h) * 100.0 if ema_slow_4h else 0.0, 4),
        "rsi_4h": round(float(rsi_4h_s.iloc[-1]), 2),

        # Group 4: Momentum
        "rsi_ltf": round(float(rsi_ltf_s.iloc[-1]), 2),
        "roc_ltf": round(float(roc_ltf_s.iloc[-1]), 4),

        # Group 5: Volatility
        "atr_ltf": round(atr_ltf, 6),
        "atr_pct": round((atr_ltf / close_ltf) * 100.0, 4) if close_ltf else 0.0,
        "bb_width_ltf": round(float(bb_width_ltf_s.iloc[-1]), 4),
        "price_to_atr": round(close_ltf / atr_ltf, 2) if atr_ltf > 0 else 0.0,

        # Group 6: Volume
        "volume_ratio": round(vol_ltf / float(vol_ma_ltf_s.iloc[-1]), 2) if float(vol_ma_ltf_s.iloc[-1]) > 0 else 1.0,
        "volume_trend": round(((float(vol_ma_ltf_s.iloc[-1]) - float(vol_ma_ltf_s.iloc[-2])) / float(vol_ma_ltf_s.iloc[-2])) * 100.0, 2) if float(vol_ma_ltf_s.iloc[-2]) > 0 else 0.0,
        "crossover_volume_ratio": round(vol_ltf / float(vol_ma_ltf_s.iloc[-1]), 2) if float(vol_ma_ltf_s.iloc[-1]) > 0 else 1.0,

        # Group 7: Context
        "fear_greed_index": fear_greed if fear_greed is not None else 50, # Default to neutral if API fails
        "btc_trend_bias": btc_bias,
        "hour_of_day": int(timestamp_utc.hour),
        "day_of_week": int(timestamp_utc.weekday()), # 0 = Monday, 6 = Sunday

        # Group 8: Trade Management
        "swing_high": round(swing_high, 6),
        "swing_low": round(swing_low, 6),
        "atr_stop_distance": round(atr_stop_distance, 6)
    }

    # Clean up NaNs just in case
    for k, v in features.items():
        if isinstance(v, float) and math.isnan(v):
            features[k] = 0.0

    return features

# ── FETCH FORWARD CANDLES (For Labeling) ──────────────────────────────────────

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
# ── FETCH FORWARD CANDLES (For Labeling) ──────────────────────────────────────

