# runner.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Orchestrate the full pipeline. Called by GitHub Actions on schedule.
# FLOW: scan phase (detect signals) → label phase (compute outcomes)
# ─────────────────────────────────────────────────────────────────────────────

import time
from datetime import datetime, timezone
from config import COINS, EMA_FAST, EMA_SLOW, LTF_INTERVAL, HTF_4H_INTERVAL, HTF_1D_INTERVAL, RUNTIME_LIMIT_MINUTES
from fetcher import fetch_candles, fetch_fear_greed
from features import compute_features
from signals import detect_signal
from db import insert_signal
from utils import load_run_state, save_run_state, api_limit_reached, log_error


def _get_btc_bias() -> bool:
    """
    Fetch BTC 4h candles and return True if 9 EMA is above 15 EMA.
    Used as a market-wide trend filter for all altcoin signals.
    Defaults to True (bullish) if fetch fails.
    """
    btc_4h = fetch_candles("BTCUSDT", "4h", 50)
    if btc_4h.empty or len(btc_4h) < EMA_SLOW:
        return True
    fast = btc_4h["close"].ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    slow = btc_4h["close"].ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
    return bool(fast > slow)


def run_scan():
    """
    Scan all coins for EMA crossover signals.
    Saves run state on exit so the next GitHub Actions job can resume
    if we hit the runtime or API limit mid-scan.
    """
    start_time = time.time()
    state      = load_run_state()
    last_idx   = int(state.get("last_symbol_index", 0))

    print(f"Scan start: {datetime.now(timezone.utc).isoformat()}")
    print(f"Scanning {len(COINS)} coins from index {last_idx}\n")

    # Fetch context once — same values used for all coins this run
    fear_greed = fetch_fear_greed()
    btc_bias   = _get_btc_bias()
    print(f"Fear & Greed: {fear_greed} | BTC 4h bias: {'BULLISH' if btc_bias else 'BEARISH'}\n")

    for i in range(last_idx, len(COINS)):
        symbol = COINS[i]

        # Runtime limit check
        elapsed_min = (time.time() - start_time) / 60.0
        if elapsed_min >= RUNTIME_LIMIT_MINUTES:
            print(f"Runtime limit reached ({elapsed_min:.1f} min) — saving state.")
            state["last_symbol_index"] = i
            state["phase"]             = "scan"
            state["timestamp"]         = datetime.now(timezone.utc).isoformat()
            save_run_state(state)
            return

        # API limit check
        if api_limit_reached():
            print(f"API limit reached — saving state.")
            state["last_symbol_index"] = i
            state["phase"]             = "scan"
            state["timestamp"]         = datetime.now(timezone.utc).isoformat()
            save_run_state(state)
            return

        print(f"Scanning {symbol}...")

        try:
            # Fetch all three timeframes
            ltf_df    = fetch_candles(symbol, LTF_INTERVAL, 300)
            htf_4h_df = fetch_candles(symbol, HTF_4H_INTERVAL, 200)
            htf_1d_df = fetch_candles(symbol, HTF_1D_INTERVAL, 100)

            if ltf_df.empty or htf_4h_df.empty or htf_1d_df.empty:
                log_error(f"Missing candle data for {symbol} — skipping")
                continue

            # Compute features
            features = compute_features(ltf_df, htf_4h_df, htf_1d_df, fear_greed, btc_bias, symbol)
            if features is None:
                print(f"  Feature computation failed for {symbol} — skipping")
                continue

            # Detect signal
            signal = detect_signal(features, symbol)
            if signal is None:
                print(f"  No signal for {symbol}")
                continue

            # Insert to Supabase
            ok = insert_signal(signal)
            if ok:
                print(f"  {signal['signal']} signal inserted for {symbol}")
            else:
                log_error(f"  Failed to insert signal for {symbol}")

        except Exception as e:
            log_error(f"run_scan error for {symbol}: {repr(e)}")

    # Finished all coins — move to label phase
    state["last_symbol_index"] = 0
    state["phase"]             = "label"
    state["timestamp"]         = datetime.now(timezone.utc).isoformat()
    save_run_state(state)
    print("\nScan complete.")


def run_once():
    """
    Main entry point — called by GitHub Actions.
    Runs scan phase then label phase in sequence.
    """
    state = load_run_state()
    phase = state.get("phase", "scan")

    print(f"Runner start — phase: {phase}")

    if phase == "scan":
        run_scan()
        state = load_run_state()
        phase = state.get("phase", "label")

    if phase == "label":
        from labeler import label_pending_signals
        label_pending_signals()
        # Reset for next scheduled run
        state["phase"]             = "scan"
        state["last_symbol_index"] = 0
        save_run_state(state)

    print("Runner finished.")


if __name__ == "__main__":
    run_once()