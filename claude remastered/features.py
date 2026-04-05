# features.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Compute all 35 technical indicators and features for our ML model.
# RULE: Pure functions only. Takes DataFrames, returns a dictionary.
#       No API calls, no database calls. Just math.
# ─────────────────────────────────────────────────────────────────────────────

import math
import pandas as pd
import ta
from datetime import datetime, timezone
from config import EMA_FAST, EMA_SLOW, ADX_LEN, ATR_STOP_MULTIPLIER, LOOKBACK_SL
from utils import log_error


# ── PRIVATE INDICATOR HELPERS ─────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average — weights recent prices more than older ones."""
    return series.ewm(span=span, adjust=False).mean()

def _adx_series(df: pd.DataFrame, length: int) -> pd.Series:
    """ADX — measures trend strength, not direction. Above 25 = trending."""
    try:
        ind = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"],
            window=length, fillna=True
        )
        return ind.adx()
    except Exception as e:
        log_error(f"_adx_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(df), index=df.index)

def _rsi_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """RSI — momentum indicator. Below 30 = oversold, above 70 = overbought."""
    try:
        ind = ta.momentum.RSIIndicator(close=df["close"], window=length, fillna=True)
        return ind.rsi()
    except Exception as e:
        log_error(f"_rsi_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(df), index=df.index)

def _atr_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR — average true range, measures volatility per candle."""
    try:
        ind = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=length, fillna=True
        )
        return ind.average_true_range()
    except Exception as e:
        log_error(f"_atr_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(df), index=df.index)

def _roc_series(series: pd.Series, length: int = 9) -> pd.Series:
    """Rate of Change — raw momentum as % change over N candles."""
    try:
        ind = ta.momentum.ROCIndicator(close=series, window=length, fillna=True)
        return ind.roc()
    except Exception as e:
        log_error(f"_roc_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(series), index=series.index)

def _macd_hist_series(series: pd.Series) -> pd.Series:
    """MACD Histogram — positive and growing = bullish momentum accelerating."""
    try:
        ind = ta.trend.MACD(close=series, fillna=True)
        return ind.macd_diff()
    except Exception as e:
        log_error(f"_macd_hist_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(series), index=series.index)

def _bb_width_series(series: pd.Series) -> pd.Series:
    """Bollinger Band width — wide = volatile/expanding, narrow = consolidating."""
    try:
        ind = ta.volatility.BollingerBands(close=series, window=20, window_dev=2, fillna=True)
        return ind.bollinger_wband()
    except Exception as e:
        log_error(f"_bb_width_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(series), index=series.index)


# ── MAIN FEATURE COMPUTATION ──────────────────────────────────────────────────

def compute_features(
    ltf_df: pd.DataFrame,
    htf_4h_df: pd.DataFrame,
    htf_1d_df: pd.DataFrame,
    fear_greed: int,
    btc_bias: bool,
    symbol: str,
    checked_at: datetime = None
) -> dict:
    """
    Compute all 35 features from raw OHLCV candle data across three timeframes.

    Args:
        ltf_df:     15m candles DataFrame
        htf_4h_df:  4h candles DataFrame
        htf_1d_df:  1D candles DataFrame
        fear_greed: integer 0-100 from fetch_fear_greed(), or None
        btc_bias:   bool — is BTC 4h trend bullish?
        symbol:     coin pair e.g. "BTCUSDT"
        checked_at: datetime of signal — defaults to now if None

    Returns:
        dict with all 35 feature keys, or None if data is insufficient.
    """
    try:
        # Guard — need enough candles for indicators to warm up
        if ltf_df.empty or len(ltf_df) < 50:
            log_error(f"compute_features: insufficient 15m data for {symbol} ({len(ltf_df)} rows)")
            return None
        if htf_4h_df.empty or len(htf_4h_df) < 30:
            log_error(f"compute_features: insufficient 4h data for {symbol}")
            return None
        if htf_1d_df.empty or len(htf_1d_df) < 20:
            log_error(f"compute_features: insufficient 1D data for {symbol}")
            return None

        if checked_at is None:
            checked_at = datetime.now(timezone.utc)

        # ── LTF (15m) INDICATORS ─────────────────────────────────────────────
        ema_fast_s    = _ema(ltf_df["close"], EMA_FAST)
        ema_slow_s    = _ema(ltf_df["close"], EMA_SLOW)
        adx_s         = _adx_series(ltf_df, ADX_LEN)
        rsi_s         = _rsi_series(ltf_df)
        atr_s         = _atr_series(ltf_df)
        roc_s         = _roc_series(ltf_df["close"])
        macd_hist_s   = _macd_hist_series(ltf_df["close"])
        bb_width_s    = _bb_width_series(ltf_df["close"])
        vol_ma_s      = ltf_df["volume"].rolling(window=20, min_periods=1).mean()

        # Extract latest values (iloc[-1]) and previous candle (iloc[-2])
        price         = float(ltf_df["close"].iloc[-1])
        candle_open   = float(ltf_df["open"].iloc[-1])
        candle_high   = float(ltf_df["high"].iloc[-1])
        candle_low    = float(ltf_df["low"].iloc[-1])
        volume_latest = float(ltf_df["volume"].iloc[-1])

        ema_fast      = float(ema_fast_s.iloc[-1])
        ema_slow      = float(ema_slow_s.iloc[-1])
        ema_fast_prev = float(ema_fast_s.iloc[-2])  # Previous candle — used by signals.py
        ema_slow_prev = float(ema_slow_s.iloc[-2])  # Previous candle — used by signals.py
        adx_latest    = float(adx_s.iloc[-1])
        adx_prev      = float(adx_s.iloc[-2])
        rsi_latest    = float(rsi_s.iloc[-1])
        atr_latest    = float(atr_s.iloc[-1])
        macd_latest   = float(macd_hist_s.iloc[-1])
        bb_latest     = float(bb_width_s.iloc[-1])
        roc_latest    = float(roc_s.iloc[-1])
        vol_ma_latest = float(vol_ma_s.iloc[-1])
        vol_ma_prev   = float(vol_ma_s.iloc[-2])

        # ── LTF DERIVED FEATURES ─────────────────────────────────────────────
        ema_separation = (ema_fast - ema_slow) / ema_slow * 100 if ema_slow != 0 else 0.0
        ema_fast_slope = (ema_fast - ema_fast_prev) / ema_fast_prev * 100 if ema_fast_prev != 0 else 0.0
        ema_slow_slope = (ema_slow - ema_slow_prev) / ema_slow_prev * 100 if ema_slow_prev != 0 else 0.0
        adx_slope      = adx_latest - adx_prev

        price_above_both_emas = bool(price > ema_fast and price > ema_slow)

        atr_pct       = atr_latest / price * 100 if price > 0 else 0.0
        price_to_atr  = price / atr_latest if atr_latest > 0 else 0.0
        atr_stop_dist = atr_latest * ATR_STOP_MULTIPLIER

        candle_range  = candle_high - candle_low
        crossover_candle_strength = (
            (candle_open - candle_low) / candle_range if candle_range > 0 else 0.5
        )

        volume_ratio          = volume_latest / vol_ma_latest if vol_ma_latest > 0 else 1.0
        crossover_volume_ratio = volume_ratio  # Same candle's volume ratio
        volume_trend          = (
            (vol_ma_latest - vol_ma_prev) / vol_ma_prev * 100 if vol_ma_prev > 0 else 0.0
        )

        swing_high = float(ltf_df["high"].iloc[-LOOKBACK_SL:].max())
        swing_low  = float(ltf_df["low"].iloc[-LOOKBACK_SL:].min())

        # ── 4H INDICATORS ────────────────────────────────────────────────────
        ema_fast_4h_s  = _ema(htf_4h_df["close"], EMA_FAST)
        ema_slow_4h_s  = _ema(htf_4h_df["close"], EMA_SLOW)
        adx_4h_s       = _adx_series(htf_4h_df, ADX_LEN)
        rsi_4h_s       = _rsi_series(htf_4h_df)
        macd_4h_s      = _macd_hist_series(htf_4h_df["close"])

        ema_fast_4h    = float(ema_fast_4h_s.iloc[-1])
        ema_slow_4h    = float(ema_slow_4h_s.iloc[-1])
        htf_4h_bias    = bool(ema_fast_4h > ema_slow_4h)
        ema_sep_4h     = (ema_fast_4h - ema_slow_4h) / ema_slow_4h * 100 if ema_slow_4h != 0 else 0.0
        adx_4h         = float(adx_4h_s.iloc[-1])
        rsi_4h         = float(rsi_4h_s.iloc[-1])
        macd_4h        = float(macd_4h_s.iloc[-1])

        # ── 1D INDICATORS ────────────────────────────────────────────────────
        ema_fast_1d_s  = _ema(htf_1d_df["close"], EMA_FAST)
        ema_slow_1d_s  = _ema(htf_1d_df["close"], EMA_SLOW)

        ema_fast_1d    = float(ema_fast_1d_s.iloc[-1])
        ema_slow_1d    = float(ema_slow_1d_s.iloc[-1])
        htf_1d_bias    = bool(ema_fast_1d > ema_slow_1d)

        # ── CONTEXT FEATURES ─────────────────────────────────────────────────
        hour_of_day  = int(checked_at.hour) if hasattr(checked_at, 'hour') else 0
        day_of_week  = int(checked_at.weekday()) if hasattr(checked_at, 'weekday') else 0

        # ── ASSEMBLE DICT ─────────────────────────────────────────────────────
        features = {
            # Price — needed by signals.py and stored in DB
            "price": round(price, 8),

            # Private keys — needed by signals.py for crossover detection
            # Prefixed with _ so signals.py knows not to store them in DB
            "_ema_fast_prev": ema_fast_prev,
            "_ema_slow_prev": ema_slow_prev,

            # Group 1: EMA
            "ema_fast_ltf":              round(ema_fast, 6),
            "ema_slow_ltf":              round(ema_slow, 6),
            "ema_fast_slope":            round(ema_fast_slope, 4),
            "ema_slow_slope":            round(ema_slow_slope, 4),
            "ema_separation":            round(ema_separation, 4),
            "price_above_both_emas":     price_above_both_emas,
            "crossover_candle_strength": round(crossover_candle_strength, 4),

            # Group 2: Trend
            "adx_ltf":             round(adx_latest, 2),
            "adx_slope":           round(adx_slope, 2),
            "adx_4h":              round(adx_4h, 2),
            "macd_histogram_ltf":  round(macd_latest, 6),
            "macd_histogram_4h":   round(macd_4h, 6),

            # Group 3: HTF Alignment
            "htf_4h_bias":        htf_4h_bias,
            "htf_1d_bias":        htf_1d_bias,
            "ema_separation_4h":  round(ema_sep_4h, 4),
            "rsi_4h":             round(rsi_4h, 2),

            # Group 4: Momentum
            "rsi_ltf": round(rsi_latest, 2),
            "roc_ltf": round(roc_latest, 4),

            # Group 5: Volatility
            "atr_ltf":      round(atr_latest, 6),
            "atr_pct":      round(atr_pct, 4),
            "bb_width_ltf": round(bb_latest, 4),
            "price_to_atr": round(price_to_atr, 2),

            # Group 6: Volume
            "volume_ratio":           round(volume_ratio, 4),
            "volume_trend":           round(volume_trend, 4),
            "crossover_volume_ratio": round(crossover_volume_ratio, 4),

            # Group 7: Context
            "fear_greed_index": fear_greed if fear_greed is not None else 50,
            "btc_trend_bias":   btc_bias,
            "hour_of_day":      hour_of_day,
            "day_of_week":      day_of_week,

            # Group 8: Trade Management
            "swing_high":        round(swing_high, 6),
            "swing_low":         round(swing_low, 6),
            "atr_stop_distance": round(atr_stop_dist, 6),
        }

        # Replace any NaN with 0.0 — safety net
        for k, v in features.items():
            if isinstance(v, float) and math.isnan(v):
                features[k] = 0.0

        return features

    except Exception as e:
        log_error(f"compute_features error for {symbol}: {repr(e)}")
        return None