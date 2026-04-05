# features.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: Compute all 37 technical indicators and features for ML model.
#
# PHILOSOPHY:
#   - Pure functions only (no API calls, no database calls, just math)
#   - Input: 3 DataFrames (15m, 4h, 1d candles) + context variables
#   - Output: Single dict with 37 feature keys
#   - Never raises exceptions (returns None on failure)
#
# FEATURE GROUPS (9 total):
#   1. EMA (9 features)     — crossover detection, slopes, separation
#   2. Trend (5 features)   — ADX, MACD on multiple timeframes
#   3. HTF Alignment (4)    — are we trading with bigger trend?
#   4. Momentum (2)         — RSI, ROC
#   5. Volatility (4)       — ATR, Bollinger Bands
#   6. Volume (3)           — volume spikes, trends
#   7. Context (4)          — Fear & Greed, BTC bias, time of day
#   8. Trade Mgmt (3)       — swing high/low, ATR stop distance
#   9. Private (2)          — _ema_fast_prev, _ema_slow_prev (for crossover math)
#
# CRITICAL RULE — NO LOOKAHEAD BIAS:
#   Every feature uses ONLY data available at signal time.
#   We never peek into the future.
#   Example: If signal fires at 10:30 AM, we only use candles BEFORE 10:30 AM.
# ═════════════════════════════════════════════════════════════════════════════

import math
import pandas as pd
import ta  # Technical Analysis library for indicators
from datetime import datetime, timezone
from config import EMA_FAST, EMA_SLOW, ADX_LEN, ATR_STOP_MULTIPLIER, LOOKBACK_SL
from utils import log_error


# ══════════════════════════════════════════════════════════════════════════════
#                              PRIVATE INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
# 
# These are internal calculation functions.
# Prefixed with _ to signal "don't call from other files"
# 
# Each function:
#   - Takes a Series or DataFrame
#   - Returns a Series of indicator values
#   - On error: returns Series of NaN (not crash)
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average
    
    What it is:
        A weighted moving average that gives MORE weight to recent prices.
        Unlike SMA (simple moving average), EMA reacts faster to price changes.
    
    Formula (simplified):
        EMA_today = (Price_today × K) + (EMA_yesterday × (1 - K))
        where K = 2 / (span + 1)
    
    Parameters:
        series: Price series (usually close prices)
        span:   Number of periods (9 for fast, 15 for slow)
    
    Returns:
        Series of EMA values (same length as input)
    
    Example:
        close_prices = df["close"]
        ema9 = _ema(close_prices, 9)
        print(f"Latest 9-EMA: {ema9.iloc[-1]}")
    """
    return series.ewm(span=span, adjust=False).mean()


def _adx_series(df: pd.DataFrame, length: int) -> pd.Series:
    """
    ADX — Average Directional Index
    
    What it measures:
        Trend STRENGTH (not direction)
        0-100 scale:
            0-20:  Weak/no trend (choppy, ranging market)
            20-25: Trend starting
            25-50: Strong trend
            50+:   Very strong trend (rare)
    
    Why we use it:
        The 9/15 EMA crossover strategy works ONLY when market is trending.
        In choppy markets (ADX < 25), crossovers are noise (whipsaws).
        ADX helps model learn: "only trade signals when ADX shows a real trend"
    
    Calculation:
        Complex — uses +DI, -DI, and smoothing
        ta library handles the math for us
    
    Parameters:
        df:     DataFrame with high, low, close columns
        length: Lookback period (14 is standard)
    
    Returns:
        Series of ADX values (0-100)
        On error: Series of NaN
    """
    try:
        ind = ta.trend.ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=length,
            fillna=True  # Fill initial NaN with 0 (before enough data)
        )
        return ind.adx()
    except Exception as e:
        log_error(f"_adx_series error: {repr(e)}")
        # Return Series of NaN (same length as df)
        # This prevents downstream errors while flagging the issue
        return pd.Series([float("nan")] * len(df), index=df.index)


def _rsi_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    RSI — Relative Strength Index
    
    What it measures:
        Momentum oscillator (overbought/oversold indicator)
        0-100 scale:
            < 30:  Oversold (price might bounce up)
            > 70:  Overbought (price might drop)
            50:    Neutral
    
    Why we use it:
        Helps model understand if signal fired after big move.
        Example: LONG signal with RSI=80 → already ran up, risky
                 LONG signal with RSI=40 → fresher move, better entry
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over N periods
    
    Parameters:
        df:     DataFrame with close prices
        length: Lookback period (14 is standard)
    
    Returns:
        Series of RSI values (0-100)
    """
    try:
        ind = ta.momentum.RSIIndicator(
            close=df["close"],
            window=length,
            fillna=True
        )
        return ind.rsi()
    except Exception as e:
        log_error(f"_rsi_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(df), index=df.index)


def _atr_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    ATR — Average True Range
    
    What it measures:
        Volatility (how much price moves per candle)
        Measured in price units (e.g., $500 for BTC)
    
    Why we use it:
        1. Stop loss placement: stop = entry ± (ATR × 1.5)
        2. Position sizing: riskier (high ATR) = smaller position
        3. Model learns if volatility predicts trade success
    
    Formula:
        True Range = max of:
            - high - low
            - abs(high - prev_close)
            - abs(low - prev_close)
        ATR = Moving average of True Range over N periods
    
    Parameters:
        df:     DataFrame with high, low, close
        length: Lookback period (14 is standard)
    
    Returns:
        Series of ATR values (in price units)
    """
    try:
        ind = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=length,
            fillna=True
        )
        return ind.average_true_range()
    except Exception as e:
        log_error(f"_atr_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(df), index=df.index)


def _roc_series(series: pd.Series, length: int = 9) -> pd.Series:
    """
    ROC — Rate of Change
    
    What it measures:
        Raw momentum as percentage change over N candles
    
    Formula:
        ROC = ((close_today - close_N_periods_ago) / close_N_periods_ago) × 100
    
    Why we use it:
        Pure momentum without smoothing (unlike MACD which uses EMAs).
        Positive ROC = upward momentum
        Negative ROC = downward momentum
        
        Model learns: do signals after strong momentum work better?
    
    Parameters:
        series: Close prices
        length: Lookback period (9 matches our EMA fast)
    
    Returns:
        Series of ROC values (percentage)
    """
    try:
        ind = ta.momentum.ROCIndicator(
            close=series,
            window=length,
            fillna=True
        )
        return ind.roc()
    except Exception as e:
        log_error(f"_roc_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(series), index=series.index)


def _macd_hist_series(series: pd.Series) -> pd.Series:
    """
    MACD Histogram
    
    What it is:
        MACD = EMA(12) - EMA(26)
        Signal Line = EMA(MACD, 9)
        Histogram = MACD - Signal Line
    
    What it measures:
        Momentum acceleration/deceleration
        Positive & growing = bullish momentum accelerating
        Positive & shrinking = bullish momentum fading
        Negative & growing = bearish momentum accelerating
    
    Why we use histogram (not raw MACD):
        Histogram shows the CHANGE in momentum.
        A crossover often happens when histogram flips from negative → positive.
        
        Model learns: is MACD confirming the EMA crossover?
    
    Parameters:
        series: Close prices
    
    Returns:
        Series of MACD histogram values
    """
    try:
        ind = ta.trend.MACD(close=series, fillna=True)
        return ind.macd_diff()  # This is the histogram
    except Exception as e:
        log_error(f"_macd_hist_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(series), index=series.index)


def _bb_width_series(series: pd.Series) -> pd.Series:
    """
    Bollinger Band Width
    
    What it is:
        BBands = SMA(20) ± (2 × StdDev)
        Width = (Upper Band - Lower Band) / SMA
    
    What it measures:
        Volatility and consolidation
        Wide bands = high volatility (price swinging)
        Narrow bands = low volatility (consolidation → potential breakout)
    
    Why we use it:
        When bands narrow (low width) → market is coiling
        Crossover during low width → potential big move starting
        Crossover during high width → might be late (move already happened)
        
        Model learns: does BB width predict trade quality?
    
    Parameters:
        series: Close prices
    
    Returns:
        Series of BB width values (normalized, 0-1 scale)
    """
    try:
        ind = ta.volatility.BollingerBands(
            close=series,
            window=20,       # Standard period
            window_dev=2,    # Standard deviations
            fillna=True
        )
        return ind.bollinger_wband()  # Normalized width
    except Exception as e:
        log_error(f"_bb_width_series error: {repr(e)}")
        return pd.Series([float("nan")] * len(series), index=series.index)


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(
    ltf_df: pd.DataFrame,      # 15m candles
    htf_4h_df: pd.DataFrame,   # 4h candles
    htf_1d_df: pd.DataFrame,   # 1d candles
    fear_greed: int,           # 0-100 from API (or None)
    btc_bias: bool,            # Is BTC 4h trend bullish?
    symbol: str,               # e.g., "BTCUSDT"
    checked_at: datetime = None  # Timestamp of signal (for backfill)
) -> dict:
    """
    Compute ALL 37 features from raw OHLCV data across three timeframes.
    
    This is the HEART of the pipeline.
    Every signal recorded in the database has these exact features.
    The ML model trains on this exact feature set.
    
    Args:
        ltf_df:     15-minute candles (at least 50 rows for warmup)
        htf_4h_df:  4-hour candles (at least 30 rows)
        htf_1d_df:  Daily candles (at least 20 rows)
        fear_greed: Crypto Fear & Greed Index (0-100), or None
        btc_bias:   True if BTC 4h trend is bullish (9EMA > 15EMA)
        symbol:     Trading pair (e.g., "BTCUSDT")
        checked_at: Datetime of signal (defaults to now)
    
    Returns:
        dict with 37 feature keys + 2 private keys (_ema_fast_prev, _ema_slow_prev)
        
        Returns None if insufficient data for calculation
    
    Feature Categories:
        Group 1 (EMA):        ema_fast_ltf, ema_slow_ltf, ema_fast_slope, ...
        Group 2 (Trend):      adx_ltf, adx_slope, adx_4h, macd_histogram_ltf, ...
        Group 3 (HTF):        htf_4h_bias, htf_1d_bias, ema_separation_4h, ...
        Group 4 (Momentum):   rsi_ltf, roc_ltf
        Group 5 (Volatility): atr_ltf, atr_pct, bb_width_ltf, price_to_atr
        Group 6 (Volume):     volume_ratio, volume_trend, crossover_volume_ratio
        Group 7 (Context):    fear_greed_index, btc_trend_bias, hour_of_day, ...
        Group 8 (Trade Mgmt): swing_high, swing_low, atr_stop_distance
        Group 9 (Private):    _ema_fast_prev, _ema_slow_prev
    
    Example:
        features = compute_features(df_15m, df_4h, df_1d, 45, True, "BTCUSDT")
        if features:
            print(f"ADX: {features['adx_ltf']}")
            print(f"EMA Fast: {features['ema_fast_ltf']}")
    """
    
    try:
        # ══════════════════════════════════════════════════════════════════════
        #                              DATA VALIDATION
        # ══════════════════════════════════════════════════════════════════════
        # 
        # CRITICAL: We need enough candles to warm up all indicators.
        # 
        # Why 50 candles for 15m?
        #   - EMA(15) needs 15+ candles to stabilize
        #   - ADX(14) needs 14+ candles
        #   - BBands(20) needs 20+ candles
        #   - Volume MA(20) needs 20+ candles
        #   - We need 2 candles minimum (current + previous for slope)
        #   - 50 gives us safety margin
        # 
        # Why 30 for 4h, 20 for 1d?
        #   - Same logic, but these timeframes move slower
        #   - 30 × 4h = 5 days of data
        #   - 20 × 1d = 20 days of data
        # 
        # If insufficient data → return None
        # Caller checks: if features is None, skip this coin
        # ══════════════════════════════════════════════════════════════════════
        
        if ltf_df.empty or len(ltf_df) < 50:
            log_error(
                f"compute_features: insufficient 15m data for {symbol} "
                f"({len(ltf_df)} rows, need 50+)"
            )
            return None
        
        if htf_4h_df.empty or len(htf_4h_df) < 30:
            log_error(f"compute_features: insufficient 4h data for {symbol}")
            return None
        
        if htf_1d_df.empty or len(htf_1d_df) < 20:
            log_error(f"compute_features: insufficient 1D data for {symbol}")
            return None
        
        # Default timestamp to now if not provided (live mode)
        # In backfill mode, checked_at is the historical timestamp
        if checked_at is None:
            checked_at = datetime.now(timezone.utc)
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              LTF INDICATORS (15-MINUTE)
        # ══════════════════════════════════════════════════════════════════════
        # 
        # These are calculated on the 15-minute timeframe.
        # Each _XXX_series() function returns a Series (one value per candle).
        # We extract the LATEST value (iloc[-1]) for the current signal.
        # ══════════════════════════════════════════════════════════════════════
        
        ema_fast_s    = _ema(ltf_df["close"], EMA_FAST)   # 9-period EMA
        ema_slow_s    = _ema(ltf_df["close"], EMA_SLOW)   # 15-period EMA
        adx_s         = _adx_series(ltf_df, ADX_LEN)      # 14-period ADX
        rsi_s         = _rsi_series(ltf_df)               # 14-period RSI
        atr_s         = _atr_series(ltf_df)               # 14-period ATR
        roc_s         = _roc_series(ltf_df["close"])      # 9-period ROC
        macd_hist_s   = _macd_hist_series(ltf_df["close"]) # MACD histogram
        bb_width_s    = _bb_width_series(ltf_df["close"])  # Bollinger Band width
        vol_ma_s      = ltf_df["volume"].rolling(window=20, min_periods=1).mean()
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              EXTRACT LATEST VALUES
        # ══════════════════════════════════════════════════════════════════════
        # 
        # .iloc[-1] = last row (most recent candle)
        # .iloc[-2] = second-to-last row (previous candle)
        # 
        # We need previous values for:
        #   - Slope calculation (change from prev to current)
        #   - Crossover detection (was fast < slow, now fast > slow?)
        # ══════════════════════════════════════════════════════════════════════
        
        # Current candle OHLCV
        price         = float(ltf_df["close"].iloc[-1])
        candle_open   = float(ltf_df["open"].iloc[-1])
        candle_high   = float(ltf_df["high"].iloc[-1])
        candle_low    = float(ltf_df["low"].iloc[-1])
        volume_latest = float(ltf_df["volume"].iloc[-1])
        
        # Current candle indicators (latest value)
        ema_fast      = float(ema_fast_s.iloc[-1])
        ema_slow      = float(ema_slow_s.iloc[-1])
        adx_latest    = float(adx_s.iloc[-1])
        rsi_latest    = float(rsi_s.iloc[-1])
        atr_latest    = float(atr_s.iloc[-1])
        macd_latest   = float(macd_hist_s.iloc[-1])
        bb_latest     = float(bb_width_s.iloc[-1])
        roc_latest    = float(roc_s.iloc[-1])
        vol_ma_latest = float(vol_ma_s.iloc[-1])
        
        # Previous candle indicators (for slope/crossover)
        ema_fast_prev = float(ema_fast_s.iloc[-2])  # ← CRITICAL for crossover detection
        ema_slow_prev = float(ema_slow_s.iloc[-2])  # ← CRITICAL for crossover detection
        adx_prev      = float(adx_s.iloc[-2])
        vol_ma_prev   = float(vol_ma_s.iloc[-2])
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              LTF DERIVED FEATURES
        # ══════════════════════════════════════════════════════════════════════
        
        # EMA Separation — how far apart are the EMAs?
        # Positive = fast above slow (bullish)
        # Negative = fast below slow (bearish)
        # Large absolute value = strong trend
        ema_separation = (
            (ema_fast - ema_slow) / ema_slow * 100 if ema_slow != 0 else 0.0
        )
        
        # EMA Slopes — are EMAs rising or falling?
        # Positive slope = EMA pointing up
        # Negative slope = EMA pointing down
        ema_fast_slope = (
            (ema_fast - ema_fast_prev) / ema_fast_prev * 100 if ema_fast_prev != 0 else 0.0
        )
        ema_slow_slope = (
            (ema_slow - ema_slow_prev) / ema_slow_prev * 100 if ema_slow_prev != 0 else 0.0
        )
        
        # ADX Slope — is trend strengthening or weakening?
        # Positive = trend getting stronger
        # Negative = trend fading
        adx_slope = adx_latest - adx_prev
        
        # Price Position — is price above BOTH EMAs?
        # True = bullish setup (price leading the EMAs)
        # False = price below or between EMAs
        price_above_both_emas = bool(price > ema_fast and price > ema_slow)
        
        # ATR as percentage of price — normalizes volatility
        # BTC at $50k with $500 ATR = 1% ATR
        # Altcoin at $1 with $0.05 ATR = 5% ATR (more volatile)
        atr_pct = atr_latest / price * 100 if price > 0 else 0.0
        
        # Price to ATR ratio — how many ATRs is the current price?
        # Used for position sizing: higher ratio = less risk per ATR
        price_to_atr = price / atr_latest if atr_latest > 0 else 0.0
        
        # ATR-based stop distance
        # Example: ATR=$500, multiplier=1.5 → stop is $750 away
        atr_stop_dist = atr_latest * ATR_STOP_MULTIPLIER
        
        # Crossover candle strength — where in the candle did we close?
        # 1.0 = closed at high (strong bullish candle)
        # 0.0 = closed at low (strong bearish candle)
        # 0.5 = closed mid-range (weak candle)
        candle_range = candle_high - candle_low
        crossover_candle_strength = (
            (candle_open - candle_low) / candle_range if candle_range > 0 else 0.5
        )
        
        # Volume features
        volume_ratio = (
            volume_latest / vol_ma_latest if vol_ma_latest > 0 else 1.0
        )
        crossover_volume_ratio = volume_ratio  # Same as volume_ratio (on crossover candle)
        volume_trend = (
            (vol_ma_latest - vol_ma_prev) / vol_ma_prev * 100 if vol_ma_prev > 0 else 0.0
        )
        
        # Swing high/low — support/resistance levels
        # Used for stop loss placement
        swing_high = float(ltf_df["high"].iloc[-LOOKBACK_SL:].max())
        swing_low  = float(ltf_df["low"].iloc[-LOOKBACK_SL:].min())
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              4H INDICATORS (HIGHER TIMEFRAME)
        # ══════════════════════════════════════════════════════════════════════
        # 
        # These provide trend context from a bigger picture.
        # If 4h trend is bullish and 15m fires LONG → "aligned trade"
        # If 4h trend is bearish and 15m fires LONG → "counter-trend trade"
        # 
        # Model learns: do aligned trades work better?
        # ══════════════════════════════════════════════════════════════════════
        
        ema_fast_4h_s  = _ema(htf_4h_df["close"], EMA_FAST)
        ema_slow_4h_s  = _ema(htf_4h_df["close"], EMA_SLOW)
        adx_4h_s       = _adx_series(htf_4h_df, ADX_LEN)
        rsi_4h_s       = _rsi_series(htf_4h_df)
        macd_4h_s      = _macd_hist_series(htf_4h_df["close"])
        
        ema_fast_4h    = float(ema_fast_4h_s.iloc[-1])
        ema_slow_4h    = float(ema_slow_4h_s.iloc[-1])
        
        # 4h trend bias — is the 4h chart bullish or bearish?
        htf_4h_bias = bool(ema_fast_4h > ema_slow_4h)
        
        # 4h EMA separation
        ema_sep_4h = (
            (ema_fast_4h - ema_slow_4h) / ema_slow_4h * 100 if ema_slow_4h != 0 else 0.0
        )
        
        adx_4h   = float(adx_4h_s.iloc[-1])
        rsi_4h   = float(rsi_4h_s.iloc[-1])
        macd_4h  = float(macd_4h_s.iloc[-1])
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              1D INDICATORS (DAILY TIMEFRAME)
        # ══════════════════════════════════════════════════════════════════════
        # 
        # Even bigger picture — daily trend direction.
        # Helps model understand macro market regime.
        # ══════════════════════════════════════════════════════════════════════
        
        ema_fast_1d_s  = _ema(htf_1d_df["close"], EMA_FAST)
        ema_slow_1d_s  = _ema(htf_1d_df["close"], EMA_SLOW)
        
        ema_fast_1d    = float(ema_fast_1d_s.iloc[-1])
        ema_slow_1d    = float(ema_slow_1d_s.iloc[-1])
        
        # Daily trend bias
        htf_1d_bias = bool(ema_fast_1d > ema_slow_1d)
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              CONTEXT FEATURES
        # ══════════════════════════════════════════════════════════════════════
        # 
        # Time-based patterns and market sentiment.
        # Model might learn: "signals at 2 AM work poorly" (low liquidity)
        #                    "signals on Monday work better" (week start momentum)
        # ══════════════════════════════════════════════════════════════════════
        
        hour_of_day = int(checked_at.hour) if hasattr(checked_at, 'hour') else 0
        day_of_week = int(checked_at.weekday()) if hasattr(checked_at, 'weekday') else 0
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              ASSEMBLE FINAL DICT
        # ══════════════════════════════════════════════════════════════════════
        # 
        # This dict structure EXACTLY matches the Supabase table columns.
        # Every key here becomes a column in the database.
        # 
        # IMPORTANT: _ema_fast_prev and _ema_slow_prev are PRIVATE keys.
        # They're used by signals.py for crossover detection but NOT stored in DB.
        # ══════════════════════════════════════════════════════════════════════
        
        features = {
            # ── PRICE (needed by signals.py and stored in DB) ────────────────
            "price": round(price, 8),
            
            # ── PRIVATE KEYS (used by signals.py, NOT stored in DB) ──────────
            "_ema_fast_prev": ema_fast_prev,  # ← DO NOT REMOVE — signals.py needs this
            "_ema_slow_prev": ema_slow_prev,  # ← DO NOT REMOVE — signals.py needs this
            
            # ── GROUP 1: EMA ──────────────────────────────────────────────────
            "ema_fast_ltf":              round(ema_fast, 6),
            "ema_slow_ltf":              round(ema_slow, 6),
            "ema_fast_slope":            round(ema_fast_slope, 4),
            "ema_slow_slope":            round(ema_slow_slope, 4),
            "ema_separation":            round(ema_separation, 4),
            "price_above_both_emas":     price_above_both_emas,
            "crossover_candle_strength": round(crossover_candle_strength, 4),
            
            # ── GROUP 2: TREND ────────────────────────────────────────────────
            "adx_ltf":             round(adx_latest, 2),
            "adx_slope":           round(adx_slope, 2),
            "adx_4h":              round(adx_4h, 2),
            "macd_histogram_ltf":  round(macd_latest, 6),
            "macd_histogram_4h":   round(macd_4h, 6),
            
            # ── GROUP 3: HTF ALIGNMENT ────────────────────────────────────────
            "htf_4h_bias":        htf_4h_bias,
            "htf_1d_bias":        htf_1d_bias,
            "ema_separation_4h":  round(ema_sep_4h, 4),
            "rsi_4h":             round(rsi_4h, 2),
            
            # ── GROUP 4: MOMENTUM ─────────────────────────────────────────────
            "rsi_ltf": round(rsi_latest, 2),
            "roc_ltf": round(roc_latest, 4),
            
            # ── GROUP 5: VOLATILITY ───────────────────────────────────────────
            "atr_ltf":      round(atr_latest, 6),
            "atr_pct":      round(atr_pct, 4),
            "bb_width_ltf": round(bb_latest, 4),
            "price_to_atr": round(price_to_atr, 2),
            
            # ── GROUP 6: VOLUME ───────────────────────────────────────────────
            "volume_ratio":           round(volume_ratio, 4),
            "volume_trend":           round(volume_trend, 4),
            "crossover_volume_ratio": round(crossover_volume_ratio, 4),
            
            # ── GROUP 7: CONTEXT ──────────────────────────────────────────────
            "fear_greed_index": fear_greed if fear_greed is not None else 50,
            "btc_trend_bias":   btc_bias,
            "hour_of_day":      hour_of_day,
            "day_of_week":      day_of_week,
            
            # ── GROUP 8: TRADE MANAGEMENT ─────────────────────────────────────
            "swing_high":        round(swing_high, 6),
            "swing_low":         round(swing_low, 6),
            "atr_stop_distance": round(atr_stop_dist, 6),
        }
        
        
        # ══════════════════════════════════════════════════════════════════════
        #                              SAFETY CHECK — REPLACE NaN
        # ══════════════════════════════════════════════════════════════════════
        # 
        # Sometimes indicator calculations return NaN (not enough data warmup).
        # We replace NaN with 0.0 as a safe fallback.
        # 
        # This prevents:
        #   1. Database insert errors (PostgreSQL can't store NaN)
        #   2. Model training errors (most ML algos can't handle NaN)
        # 
        # Trade-off:
        #   - 0.0 is not always semantically correct (e.g., RSI=0 means something)
        #   - But it's better than crashing or skipping good data
        #   - If warmup is proper (50+ candles), we shouldn't get NaN anyway
        # ══════════════════════════════════════════════════════════════════════
        
        for k, v in features.items():
            if isinstance(v, float) and math.isnan(v):
                features[k] = 0.0
        
        return features
    
    except Exception as e:
        # ══════════════════════════════════════════════════════════════════════
        #                              CATCH-ALL ERROR HANDLER
        # ══════════════════════════════════════════════════════════════════════
        # 
        # If ANYTHING goes wrong (division by zero, invalid data type, etc),
        # we log it and return None.
        # 
        # Caller checks: if features is None → skip this coin
        # Pipeline continues to next coin instead of crashing entirely.
        # ══════════════════════════════════════════════════════════════════════
        
        log_error(f"compute_features error for {symbol}: {repr(e)}")
        return None
