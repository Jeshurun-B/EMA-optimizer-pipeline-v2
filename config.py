# config.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Single source of truth for ALL constants and settings.
# RULE: Nothing is hardcoded anywhere else in the project.
#       Every value lives here, loaded from environment variables.
#       Default values are only provided where safe to do so.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

# Loads your .env file when running locally.
# On GitHub Actions, secrets are injected as real env vars — load_dotenv() is harmless there.
load_dotenv()
# load_dotenv() reads the .env file when running locally.
# On GitHub Actions, secrets are real environment variables so this is not needed.
# We wrap it in a try/except so the pipeline doesn't crash if python-dotenv
# is not installed in the GitHub Actions environment.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── BINANCE API ───────────────────────────────────────────────────────────────

# Base URL for Binance public market data (no API key needed)
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")

# Maximum Binance API calls allowed per single pipeline run
# Prevents rate limit errors. 200 is safe for 5 coins across 3 timeframes.
API_CALL_LIMIT = int(os.getenv("API_CALL_LIMIT") or 200)

# Maximum candles per single klines request
# Binance hard cap is 1000. 300 gives ~3 days of 15m data per call.
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT") or 300)

# HTTP timeout in seconds for each API request
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT") or 10)


# ── TIMEFRAMES ────────────────────────────────────────────────────────────────

# LTF = Lower TimeFrame — this is where crossover signals are detected
LTF_INTERVAL = os.getenv("LTF_INTERVAL", "15m")

# TODO: add HTF_4H_INTERVAL — higher timeframe for trend alignment
#       load from env with os.getenv(), default value should be "4h"
HTF_4H_INTERVAL = os.getenv("HTF_4H_INTERVAL", "4h")

# TODO: add HTF_1D_INTERVAL — daily timeframe for market regime context
#       load from env with os.getenv(), default value should be "1d"
HTF_1D_INTERVAL = os.getenv("HTF_1D_INTERVAL", "1d")

# ── COINS ─────────────────────────────────────────────────────────────────────

# TODO: define COINS as a Python list of coin pair strings
#       load from os.getenv("COINS", "SOLUSDT,ETHUSDT,BTCUSDT,XRPUSDT,DOGEUSDT")
#       split by comma, and strip whitespace + quote characters from each item
#       hint: look at how the existing config.py handled this — same pattern
COINS = [coin.strip().strip('"') for coin in os.getenv("COINS", "SOLUSDT,ETHUSDT,BTCUSDT,XRPUSDT,DOGEUSDT").split(",")]


# ── EMA SETTINGS ──────────────────────────────────────────────────────────────

# TODO: add EMA_FAST — the fast EMA period
#       load from env, default 9, cast to int
EMA_FAST = int(os.getenv("EMA_FAST") or 9)

# TODO: add EMA_SLOW — the slow EMA period
#       load from env, default 15, cast to int
EMA_SLOW = int(os.getenv("EMA_SLOW") or 15)


# ── ADX SETTINGS ──────────────────────────────────────────────────────────────

# TODO: add ADX_LEN — the ADX calculation period
#       load from env, default 14, cast to int
ADX_LEN = int(os.getenv("ADX_LEN") or 14)

# TODO: add ADX_THRESHOLD — minimum ADX value to consider market trending
#       load from env, default 25.0, cast to float
#       NOTE: old code used 0.0 — we changed to 25.0 because below 25 = choppy market
ADX_THRESHOLD = float(os.getenv("ADX_THRESHOLD") or 0.0)


# ── TRADE MANAGEMENT ──────────────────────────────────────────────────────────

# Number of candles to look back when calculating swing high / swing low
LOOKBACK_SL = int(os.getenv("LOOKBACK_SL") or 10)

# Stop loss distance multiplier: stop_distance = ATR * ATR_STOP_MULTIPLIER
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER") or 1.5)

# Trade quality label thresholds (used in labeler.py)
# trade_quality = 1 if price hits (entry + RR_TARGET_MULTIPLE * stop_distance)
#                   before (entry - RR_STOP_MULTIPLE * stop_distance)
RR_TARGET_MULTIPLE = float(os.getenv("RR_TARGET_MULTIPLE") or 1.5)
RR_STOP_MULTIPLE   = float(os.getenv("RR_STOP_MULTIPLE")   or 1.0)


# ── SUPABASE ──────────────────────────────────────────────────────────────────

# TODO: add SUPABASE_URL — load from os.getenv("SUPABASE_URL")
#       NO default value — this must exist in .env or GitHub Secrets
SUPABASE_URL = os.getenv("SUPABASE_URL")


# TODO: add SUPABASE_KEY — load from os.getenv("SUPABASE_KEY")
#       NO default value — this must exist in .env or GitHub Secrets
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ── TELEGRAM ──────────────────────────────────────────────────────────────────

# TODO: add TELEGRAM_BOT_TOKEN — load from os.getenv, no default
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# TODO: add TELEGRAM_CHAT_ID   — load from os.getenv, no default
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")



# ── EXTERNAL APIs ─────────────────────────────────────────────────────────────

# Alternative.me Fear & Greed index — completely free, no auth needed
FEAR_GREED_URL = "https://api.alternative.me/fng/"


# ── RUNTIME LIMITS ────────────────────────────────────────────────────────────

# Max minutes a single pipeline run can take before saving state and exiting
# GitHub Actions default job timeout is 6 minutes — this keeps us safely under
RUNTIME_LIMIT_MINUTES = int(os.getenv("RUNTIME_LIMIT_MINUTES") or 5)


# ── LOCAL PERSISTENCE FILES ───────────────────────────────────────────────────
# These are small local files that persist between pipeline steps within one run.
# They are NOT stored in Supabase — just local workspace files.

# Stores the last signal seen per coin — used for dedup and gap calculation
LAST_SIGNALS_FILE = os.getenv("LAST_SIGNALS_FILE") or "last_signals.csv"

# Stores pipeline phase and resume index — used to continue after a timeout
RUN_STATE_FILE = os.getenv("RUN_STATE_FILE") or "run_state.json"

# Error log file path
LOG_FILE = os.getenv("LOG_FILE") or "error.log"