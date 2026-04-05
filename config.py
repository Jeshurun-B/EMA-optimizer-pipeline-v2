# config.py
# ═════════════════════════════════════════════════════════════════════════════
# PURPOSE: Single source of truth for ALL constants and settings.
# 
# PHILOSOPHY:
#   - Nothing is hardcoded anywhere else in the project
#   - Every value lives here, loaded from environment variables
#   - Default values only provided where absolutely safe
#   - If a secret is missing, we want the app to crash immediately and loudly
# ═════════════════════════════════════════════════════════════════════════════

import os

# ── ENVIRONMENT LOADER ────────────────────────────────────────────────────────
# 
# load_dotenv() reads the .env file when running locally on your machine.
# On GitHub Actions, secrets are injected as real environment variables,
# so load_dotenv() does nothing (which is fine).
# 
# We wrap it in try/except because python-dotenv might not be installed
# in the GitHub Actions runner environment (and that's okay).
# ──────────────────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed — we're probably on GitHub Actions
    # where secrets are already in the environment. No problem.
    pass


# ══════════════════════════════════════════════════════════════════════════════
#                              BINANCE API SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# Base URL for Binance public market data endpoints.
# We use data-api.binance.vision instead of api.binance.com because:
#   - No API key required (free tier)
#   - More lenient rate limits for historical data
#   - Designed for data collection use cases like ours
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")

# Maximum number of Binance API calls allowed in a single pipeline run.
# Why we need this:
#   - Binance rate limits are per IP, not per API key (since we don't use one)
#   - Hitting the limit = temporary ban (1-60 minutes depending on severity)
#   - 200 calls is safe for scanning 5 coins across 3 timeframes
#   - The pipeline tracks calls and gracefully saves state before hitting limit
API_CALL_LIMIT = int(os.getenv("API_CALL_LIMIT") or 200)

# Maximum candles we request per single API call.
# Binance hard limit: 1000 candles per request
# Our default: 300 candles
# Why 300?
#   - 300 × 15min = 3.125 days of data per call
#   - Enough to warm up all indicators (ADX, EMAs, BBands all need <100 candles)
#   - Lower number = more API calls but safer (less data lost if request fails)
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT") or 300)

# HTTP timeout in seconds for each API request.
# If Binance doesn't respond within this time, we abort the request.
# 10 seconds is reasonable:
#   - Fast enough to detect network issues quickly
#   - Long enough to handle occasional Binance slowness
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT") or 10)


# ══════════════════════════════════════════════════════════════════════════════
#                              TIMEFRAME SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# LTF = Lower TimeFrame
# This is where we DETECT crossover signals.
# Every 15 minutes, we check: did the 9 EMA cross the 15 EMA?
# Valid values: "1m", "5m", "15m", "30m", "1h", "4h", "1d"
LTF_INTERVAL = os.getenv("LTF_INTERVAL", "15m")

# HTF = Higher TimeFrame (4 hour)
# Used for TREND ALIGNMENT — are we trading with or against the bigger trend?
# Example: if 4h trend is bullish (9 EMA > 15 EMA on 4h), then LONG signals
#          on 15m are "aligned" and SHORT signals are "counter-trend"
# The model learns whether aligned trades work better.
HTF_4H_INTERVAL = os.getenv("HTF_4H_INTERVAL", "4h")

# HTF = Higher TimeFrame (daily)
# Used for MARKET REGIME context — is the overall daily trend up or down?
# Example: if 1D trend is bearish, maybe 15m LONG signals are riskier
# This gives the model macro context beyond just the 15m chart.
HTF_1D_INTERVAL = os.getenv("HTF_1D_INTERVAL", "1d")


# ══════════════════════════════════════════════════════════════════════════════
#                              COIN SELECTION
# ══════════════════════════════════════════════════════════════════════════════

# List of trading pairs to scan for crossover signals.
# Format: "SOLUSDT,ETHUSDT,BTCUSDT,XRPUSDT,DOGEUSDT"
# 
# Why these coins?
#   - High liquidity (tight spreads, easy to enter/exit)
#   - 24/7 trading (crypto never sleeps)
#   - Different market caps (BTC = large, DOGE = small)
#   - The model can learn if strategy works better on certain coins
# 
# How to add more coins:
#   - Update your .env: COINS="BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,MATICUSDT"
#   - Or edit the default here
# 
# IMPORTANT: More coins = more API calls. Stay under API_CALL_LIMIT.
#            Each coin × 3 timeframes = 3 API calls per scan.
#            5 coins = 15 calls per run. Safe.
#            20 coins = 60 calls per run. Still safe.
#            50 coins = 150 calls. Getting close to limit.

COINS = [
    coin.strip().strip('"')  # Remove whitespace and quotes from each coin
    for coin in os.getenv("COINS", "XRPUSDT,BTCUSDT,SOLUSDT,ETHUSDT,DOGEUSDT").split(",")
]


# ══════════════════════════════════════════════════════════════════════════════
#                              EMA SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# Fast EMA period — how many candles to average for the "fast" moving average.
# Default: 9 candles
# In Pine Script: ta.ema(close, 9)
# 
# Why 9?
#   - Reacts quickly to price changes (small period = sensitive)
#   - Classic short-term momentum indicator
#   - Paired with 15 EMA, this is a proven crossover system in forex/crypto
EMA_FAST = int(os.getenv("EMA_FAST") or 9)

# Slow EMA period — how many candles to average for the "slow" moving average.
# Default: 15 candles
# In Pine Script: ta.ema(close, 15)
# 
# Why 15?
#   - Slower to react (larger period = smoother, filters noise)
#   - When fast crosses slow = momentum shift detected
#   - The gap between 9 and 15 is small enough to catch moves early,
#     but large enough to avoid whipsaws (false crosses)
EMA_SLOW = int(os.getenv("EMA_SLOW") or 15)


# ══════════════════════════════════════════════════════════════════════════════
#                              ADX SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# ADX = Average Directional Index
# Measures trend STRENGTH (not direction).
# Scale: 0-100
#   - Below 20: weak trend / choppy / ranging market
#   - 20-25: trend starting to develop
#   - 25-50: strong trend
#   - Above 50: very strong trend (rare)

# ADX calculation period
# Default: 14 candles (industry standard)
# This is how many candles are used to calculate the ADX value.
ADX_LEN = int(os.getenv("ADX_LEN") or 14)

# ADX threshold for signal filtering
# 
# CRITICAL SETTING — IMPACTS DATA COLLECTION PHILOSOPHY:
# 
# OLD THINKING (wrong):
#   ADX_THRESHOLD = 25.0
#   Only record signals when ADX > 25 (trending market)
#   Problem: No negative examples for the model to learn from.
#            Model never sees "what does a bad signal look like?"
# 
# NEW THINKING (correct):
#   ADX_THRESHOLD = 0.0
#   Record ALL crossovers regardless of ADX level.
#   Store ADX as a feature. Let the MODEL decide if ADX matters.
#   
# Why 0.0 is correct:
#   - Collection phase = gather all data, no filtering
#   - ADX is stored as a feature anyway (adx_ltf, adx_4h)
#   - The ML model will learn: "if ADX < 25, ignore this signal"
#   - We get negative examples (choppy market signals that fail)
#   - This makes the model SMARTER, not dumber
# 
# When to use 25.0:
#   - LIVE TRADING only (after model is trained)
#   - At prediction time, if model says "only trade when ADX > X"
#   - But NOT during data collection
ADX_THRESHOLD = float(os.getenv("ADX_THRESHOLD") or 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#                              TRADE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

# LOOKBACK_SL = Swing high/low lookback period
# 
# How it works:
#   - When a signal fires, we look back N candles
#   - swing_high = highest price in last N candles
#   - swing_low  = lowest price in last N candles
# 
# Why we track this:
#   - These are natural support/resistance levels
#   - swing_low might be used as a stop loss (for LONG trades)
#   - swing_high might be used as a stop loss (for SHORT trades)
#   - The model can learn if this stop placement method works
# 
# Default: 10 candles
# On 15m timeframe: 10 candles = 2.5 hours of price action
LOOKBACK_SL = int(os.getenv("LOOKBACK_SL") or 10)

# ATR_STOP_MULTIPLIER = How many ATRs away to place stop loss
# 
# ATR = Average True Range (volatility measure)
# If ATR = $100, and multiplier = 1.5, then stop is $150 away from entry.
# 
# Why ATR-based stops?
#   - Adapts to volatility (wide stops in volatile markets, tight in calm)
#   - More scientific than "always use 2%" stop
#   - We store this as a feature (atr_stop_distance)
#   - Model learns if this stop distance is too tight/too wide
# 
# Default: 1.5 ATRs
# Industry standard range: 1.0 - 3.0 ATRs
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER") or 1.5)

# ──────────────────────────────────────────────────────────────────────────────
# DEPRECATED — These R:R settings are NO LONGER USED
# 
# Old labeler.py simulated trades with hardcoded R:R ratios (1.5:1 target).
# Problem: This encoded our ASSUMPTIONS into the labels.
#          "A good trade = hits 1.5R before stop"
#          But what if 1.5R is wrong? What if 2.5R works better?
# 
# New labeler.py records RAW PRICE MOVEMENT:
#   - max_move_up_pct: how far did price go UP after signal?
#   - max_move_down_pct: how far did price go DOWN after signal?
#   - candles_to_max_price: how long until highest point?
#   - candles_to_min_price: how long until lowest point?
# 
# The MODEL decides what "good" means from this raw data.
# 
# We keep these variables for backwards compatibility (old code might reference)
# but they DO NOTHING in the current pipeline.
# ──────────────────────────────────────────────────────────────────────────────
RR_TARGET_MULTIPLE = float(os.getenv("RR_TARGET_MULTIPLE") or 1.5)
RR_STOP_MULTIPLE   = float(os.getenv("RR_STOP_MULTIPLE")   or 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#                              SUPABASE DATABASE
# ══════════════════════════════════════════════════════════════════════════════

# Supabase Project URL
# Format: "https://xxxxxxxxxxxxx.supabase.co"
# 
# Where to find it:
#   1. Go to your Supabase dashboard
#   2. Select your project
#   3. Settings → API
#   4. Look for "Project URL"
# 
# CRITICAL: This MUST be set in .env or GitHub Secrets
# If missing, create_client() will crash immediately (which is correct behavior)
SUPABASE_URL = os.getenv("SUPABASE_URL")

# Supabase API Key (publishable/anon key)
# 
# Which key to use?
#   - NOT the service_role key (that's for admin operations)
#   - YES the anon/publishable key (starts with "eyJ...")
# 
# Why publishable is safe:
#   - Row Level Security (RLS) disabled on our table
#   - Insert-only operations (we never delete via API)
#   - Key is in GitHub Secrets, not hardcoded
# 
# Where to find it:
#   1. Supabase dashboard → your project
#   2. Settings → API
#   3. Look for "Project API keys" → anon/public
# 
# CRITICAL: This MUST be set in .env or GitHub Secrets
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


# ══════════════════════════════════════════════════════════════════════════════
#                              TELEGRAM NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

# Telegram Bot Token
# Format: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
# 
# How to get one:
#   1. Open Telegram
#   2. Search for @BotFather
#   3. Send /newbot
#   4. Follow instructions
#   5. Copy the token it gives you
# 
# Optional: If not set, pipeline runs without notifications (which is fine)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Telegram Chat ID (where to send messages)
# Format: "-1001234567890" (group) or "123456789" (personal)
# 
# How to find your chat ID:
#   1. Add @RawDataBot to your chat
#   2. Send any message
#   3. Bot replies with chat details
#   4. Copy the "chat" → "id" value
# 
# Optional: If not set, pipeline runs without notifications
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ══════════════════════════════════════════════════════════════════════════════
#                              EXTERNAL APIs (FREE)
# ══════════════════════════════════════════════════════════════════════════════

# Alternative.me Crypto Fear & Greed Index
# 
# What it is:
#   - Free API (no key needed)
#   - Returns integer 0-100
#   - 0 = Extreme Fear (market panic, maybe buy opportunity)
#   - 100 = Extreme Greed (market euphoria, maybe sell opportunity)
# 
# How we use it:
#   - Fetch once per pipeline run (not per coin)
#   - Store as "fear_greed_index" feature
#   - Model learns if market sentiment correlates with trade success
# 
# Example: Maybe LONG signals work better when F&G is low (contrarian)
#          Or maybe they work better when F&G is high (momentum)
#          The model figures this out from data.
FEAR_GREED_URL = "https://api.alternative.me/fng/"


# ══════════════════════════════════════════════════════════════════════════════
#                              RUNTIME LIMITS
# ══════════════════════════════════════════════════════════════════════════════

# Maximum minutes a single pipeline run is allowed before graceful shutdown
# 
# Why we need this:
#   - GitHub Actions has a 6-minute default timeout per job
#   - If we hit that limit, the job is KILLED (state not saved)
#   - We set our own limit at 5 minutes
#   - At 5min, we GRACEFULLY save state and exit
#   - Next run picks up exactly where we left off
# 
# Safety margin:
#   - GitHub limit: 6 minutes (hard kill)
#   - Our limit: 5 minutes (graceful save)
#   - This gives us 60 seconds to save state before being killed
RUNTIME_LIMIT_MINUTES = int(os.getenv("RUNTIME_LIMIT_MINUTES") or 5)


# ══════════════════════════════════════════════════════════════════════════════
#                              LOCAL FILES (PERSISTENCE)
# ══════════════════════════════════════════════════════════════════════════════
# 
# These are small CSV/JSON files stored in the project directory.
# They persist state WITHIN a single run, but are NOT cloud storage.
# On GitHub Actions, these files are destroyed after the job ends.
# That's fine — they're just for tracking progress during a run.
# ══════════════════════════════════════════════════════════════════════════════

# LAST_SIGNALS_FILE = Tracks most recent signal per coin
# 
# Format: CSV with columns [symbol, signal, checked_at_utc]
# Example:
#   symbol,signal,checked_at_utc
#   BTCUSDT,LONG,2026-04-04T10:30:00+00:00
#   ETHUSDT,SHORT,2026-04-04T09:15:00+00:00
# 
# Why we need this:
#   1. DEDUPLICATION: Don't insert same signal twice if pipeline runs 2x
#   2. SIGNAL GAP: Calculate time since last signal (stored as feature)
# 
# How it works:
#   - signals.py calls get_prev_signal(symbol) before inserting
#   - If same signal fired <30min ago → skip (duplicate)
#   - Otherwise, insert to DB and update this file via update_prev_signal()
LAST_SIGNALS_FILE = os.getenv("LAST_SIGNALS_FILE") or "last_signals.csv"

# RUN_STATE_FILE = Tracks pipeline progress for resume
# 
# Format: JSON
# Example:
#   {
#     "phase": "scan",
#     "last_symbol_index": 2,
#     "timestamp": "2026-04-04T10:30:00+00:00"
#   }
# 
# Why we need this:
#   - If we hit API limit mid-run → save which coin we were on
#   - Next run loads this file and resumes from coin #3 instead of #0
#   - Prevents re-scanning same coins over and over
# 
# Phases:
#   - "scan" = detecting signals (runner.py)
#   - "label" = labeling pending signals (labeler.py)
RUN_STATE_FILE = os.getenv("RUN_STATE_FILE") or "run_state.json"

# LOG_FILE = Error log destination
# 
# Format: Plain text with timestamps
# Example:
#   [2026-04-04T10:30:15+00:00] fetch_binance_klines error for BTCUSDT 15m: HTTPError(429)
#   [2026-04-04T10:31:42+00:00] insert_signal failed for ETHUSDT: UniqueViolation
# 
# Why we need this:
#   - Every error in the pipeline is logged here via log_error()
#   - On GitHub Actions, these logs are visible in the Actions tab
#   - Locally, you can tail -f error.log to watch errors in real-time
# 
# Important: We NEVER raise exceptions to crash the pipeline.
#            Instead, we log_error() and continue to next coin.
#            This file is your audit trail of what went wrong.
LOG_FILE = os.getenv("LOG_FILE") or "error.log"


# ══════════════════════════════════════════════════════════════════════════════
#                              BACKFILL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# look_back_days = How far back to go when NO existing data for a coin
# 
# Used by: backfill.py
# 
# How it works:
#   - backfill.py checks Supabase: "when was the last signal for BTCUSDT?"
#   - If Supabase has data → resume from that date (smart resume)
#   - If Supabase is EMPTY for that coin → go back look_back_days
# 
# Why 3 days?
#   - Enough to catch recent crossovers
#   - Not too much data (keeps API calls low)
#   - You can manually increase this to 30, 90, 180 for initial backfill
# 
# Initial setup recommendation:
#   - First run: set to 180 (6 months of history)
#   - After that: set to 3 (only fetch new data since last run)
look_back_days = int(os.getenv("LOOK_BACK_DAYS") or 180)
