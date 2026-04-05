# backfill.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: One-time historical data collection — builds the ML training dataset.
# SMART RESUME: Checks Supabase for the last stored signal per coin and only
#               fetches data from that point forward. Safe to re-run anytime.
# ─────────────────────────────────────────────────────────────────────────────

import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import COINS, BINANCE_BASE_URL, REQUEST_TIMEOUT, EMA_FAST, EMA_SLOW , look_back_days
from features import compute_features
from signals import detect_signal
from db import insert_signal, supabase, TABLE
from utils import log_error, _inc_api_call
 # Default days back if no existing data for a coin

def get_last_signal_date(symbol: str) -> datetime:
    """
    Query Supabase for the most recent checked_at_utc for this symbol.
    If no data exists, default to 180 days ago.
    Returns a datetime object.
    
    This allows backfill to resume from where it left off instead of
    re-processing months of data that's already stored.
    """
    try:
        response = (
            supabase.table(TABLE)
            .select("checked_at_utc")
            .eq("symbol", symbol)
            .order("checked_at_utc", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            last_date = pd.to_datetime(response.data[0]["checked_at_utc"], utc=True)
            print(f"  ✓ Resuming {symbol} from {last_date.date()}")
            return last_date.to_pydatetime()
    except Exception as e:
        log_error(f"get_last_signal_date error for {symbol}: {repr(e)}")
    
    # No data found — go back 180 days (6 months)
    fallback = datetime.now(timezone.utc) - timedelta(days=look_back_days)
    print(f"  ✓ No existing data for {symbol}, starting from {fallback.date()}")
    return fallback


def fetch_historical_klines(symbol: str, interval: str, days_back: int) -> pd.DataFrame:
    """
    Fetch months of historical candles by paginating backwards from today.

    Binance limits each request to 1000 candles. We loop backwards,
    setting endTime to just before the oldest candle in each batch,
    until we reach our target start date.

    Returns DataFrame sorted oldest first, or empty DataFrame on failure.
    """
    start_time_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000
    )
    all_dfs     = []
    end_time_ms = None

    print(f"  Fetching {interval} history for {symbol} ({days_back} days back)...")

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time_ms:
            params["endTime"] = end_time_ms

        try:
            _inc_api_call()
            resp = requests.get(
                f"{BINANCE_BASE_URL}/api/v3/klines",
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                break

            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            all_dfs.append(df)

            oldest_ms   = int(raw[0][0])
            end_time_ms = oldest_ms - 1

            if oldest_ms <= start_time_ms:
                break

            time.sleep(0.15)

        except Exception as e:
            log_error(f"fetch_historical_klines error for {symbol} {interval}: {repr(e)}")
            break

    if not all_dfs:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    print(f"  Got {len(combined)} {interval} candles")
    return combined


def run_backfill(default_days_back: int = look_back_days):
    """
    Main backfill function. Fetches historical data for all coins,
    slides a window through the data, detects signals, and inserts to Supabase.

    SMART RESUME LOGIC:
        For each coin, checks Supabase for the most recent signal.
        Only fetches data from that date forward.
        If no data exists for a coin, goes back default_days_back (180 days).

    The time machine approach:
        We slide a 300-candle window through the full 15m history.
        At each position we simulate what the live pipeline would have seen —
        features and crossovers computed only on data available at that moment.
        This prevents lookahead bias in historical signals.

    Step size:
        We check every candle (step=1) to catch all crossovers.
        A crossover that happened between checks would be missed otherwise.
    """
    print(f"Starting smart backfill for {len(COINS)} coins")
    print(f"Default lookback: {default_days_back} days (used when no existing data)\n")
    total_signals = 0

    for symbol in COINS:
        print(f"\n{'='*50}")
        print(f"Backfilling {symbol}...")
        coin_signals = 0

        # SMART RESUME: Get the last signal date for THIS coin specifically
        start_datetime = get_last_signal_date(symbol)
        days_back_for_coin = (datetime.now(timezone.utc) - start_datetime).days + 1  # +1 for safety

        # Also fetch BTC 4h data for the same time period (needed for btc_bias feature)
        print(f"Fetching BTC 4h history ({days_back_for_coin} days)...")
        btc_4h_master = fetch_historical_klines("BTCUSDT", "4h", days_back_for_coin)
        if not btc_4h_master.empty:
            btc_ema_fast = btc_4h_master["close"].ewm(span=EMA_FAST, adjust=False).mean()
            btc_ema_slow = btc_4h_master["close"].ewm(span=EMA_SLOW, adjust=False).mean()
            btc_4h_master["btc_bias"] = btc_ema_fast > btc_ema_slow

        # Fetch full history for all three timeframes for this coin
        df_15m = fetch_historical_klines(symbol, "15m", days_back_for_coin)
        df_4h  = fetch_historical_klines(symbol, "4h",  days_back_for_coin)
        df_1d  = fetch_historical_klines(symbol, "1d",  days_back_for_coin)

        if df_15m.empty or df_4h.empty or df_1d.empty:
            print(f"  ⚠️  Missing data for {symbol} — skipping")
            continue

        window_size = 300 # candles needed to warm up indicators

        step = 1    # check every candle — catches all crossovers
        print(f"  Processing {len(df_15m)} candles (window={window_size}, step={step})...")

        for end_idx in range(window_size, len(df_15m), step):

            # Current window: 300 candles ending at end_idx
            ltf_window   = df_15m.iloc[end_idx - window_size : end_idx].reset_index(drop=True)
            current_time = ltf_window["timestamp"].iloc[-1]

            # Slice HTF data to only what was available at current_time
            # This prevents lookahead bias — we only use data that existed at that moment
            htf_4h_window = df_4h[df_4h["timestamp"] <= current_time].tail(500).reset_index(drop=True)
            htf_1d_window = df_1d[df_1d["timestamp"] <= current_time].tail(200).reset_index(drop=True)
            btc_4h_window = btc_4h_master[btc_4h_master["timestamp"] <= current_time]

            if len(htf_4h_window) < 50 or len(htf_1d_window) < 20 or len(btc_4h_window) < EMA_SLOW:
                continue

            # Compute BTC bias from sliced data
            btc_bias = bool(btc_4h_window.iloc[-1]["btc_bias"]) if not btc_4h_window.empty else True

            # Compute features — pass historical datetime so hour/day features are correct
            features = compute_features(
                ltf_window, htf_4h_window, htf_1d_window,
                fear_greed=50,  # Neutral — no historical F&G data available
                btc_bias=btc_bias,
                symbol=symbol,
                checked_at=current_time.to_pydatetime()
            )

            if features is None:
                continue

            signal = detect_signal(features, symbol)
            if signal is None:
                continue

            # Override checked_at_utc with historical timestamp
            signal["checked_at_utc"] = current_time.isoformat()

            # insert_signal handles duplicate (symbol, checked_at_utc) gracefully
            ok = insert_signal(signal)
            if ok:
                coin_signals  += 1
                total_signals += 1
                if coin_signals % 10 == 0:
                    print(f"    {symbol}: {coin_signals} signals so far...")

        print(f"  ✓ {symbol} done — {coin_signals} signals found")

    print(f"\n{'='*50}")
    print(f"Backfill complete — {total_signals} total signals inserted")
    print("Check Supabase for the rows. Run labeler.py next to label them.")


if __name__ == "__main__":
    run_backfill()