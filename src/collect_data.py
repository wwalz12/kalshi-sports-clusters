"""
Step 1: Data Collection
Pull all resolved NBA markets from Kalshi API and save raw data.
Run from project root: python3 src/collect_data.py
"""

import os, sys
# Ensure we always save relative to project root, not CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

import requests
import pandas as pd
import time
import json
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# NBA series tickers and their market type labels (deduplicated)
NBA_SERIES = [
    ("KXNBAGAME",       "game_winner"),
    ("KXNBATOTAL",      "total_points"),
    ("KXNBASERIES",     "series_outcome"),
    ("KXNBAPTS",        "player_points"),
    ("KXNBAPA",         "player_prop"),
    ("KXNBAPR",         "player_prop"),
    ("KXNBAPRA",        "player_prop"),
    ("KXNBAREB",        "player_prop"),
    ("KXNBAAST",        "player_prop"),
    ("KXNBA3PT",        "player_prop"),
    ("KXNBABLK",        "player_prop"),
    ("KXNBASTL",        "player_prop"),
    ("KXNBA2D",         "player_prop"),
    ("KXNBA1HWINNER",   "half_winner"),
    ("KXNBA2HWINNER",   "half_winner"),
    ("KXNBA1HTOTAL",    "half_total"),
    ("KXNBA2HTOTAL",    "half_total"),
    ("KXNBA1HSPREAD",   "half_spread"),
    ("KXNBA2HSPREAD",   "half_spread"),
    ("KXNBA1QWINNER",   "quarter_winner"),
    ("KXNBATEAMTOTAL",  "team_total"),
    ("KXNBAOVERTIME",   "overtime"),
    ("KXNBAPLAYOFFPTS", "player_points"),
    ("KXNBASERIESSCORE","series_outcome"),
]

MAX_PAGES_PER_SERIES = 10   # up to 2000 markets per series
SLEEP_BETWEEN_PAGES   = 0.5
SLEEP_BETWEEN_SERIES  = 1.0
SLEEP_CANDLESTICKS    = 0.4


def parse_ts(ts_str: str) -> int:
    if not ts_str:
        return 0
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return 0


def get_markets_for_series(series_ticker: str) -> list:
    markets = []
    cursor = None
    for page in range(MAX_PAGES_PER_SERIES):
        params = {"limit": 200, "status": "settled", "series_ticker": series_ticker}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  [RATE LIMIT] Sleeping 10s then retrying {series_ticker} page {page}...")
                time.sleep(10)
                try:
                    resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e2:
                    print(f"  [ERROR] {series_ticker} page {page}: {e2}")
                    break
            else:
                print(f"  [ERROR] {series_ticker} page {page}: {e}")
                break
        except Exception as e:
            print(f"  [ERROR] {series_ticker} page {page}: {e}")
            break

        batch = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(SLEEP_BETWEEN_PAGES)

    return markets


def get_candlesticks(series_ticker: str, market_ticker: str, open_ts: int, close_ts: int) -> list:
    url = f"{BASE_URL}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    params = {"period_interval": 60, "start_ts": open_ts, "end_ts": close_ts}
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            print(f"  [RATE LIMIT] candlesticks {market_ticker}, sleeping 10s...")
            time.sleep(10)
            resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json().get("candlesticks", [])
    except Exception as e:
        print(f"  [ERROR] candlesticks {market_ticker}: {e}")
        return []


def collect_all_data(output_dir: str = "data") -> tuple:
    Path(output_dir).mkdir(exist_ok=True)

    all_markets = []
    all_candles = []

    print("=" * 60)
    print("STEP 1: DATA COLLECTION")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)

    for i, (series_ticker, market_type) in enumerate(NBA_SERIES):
        print(f"\n[{i+1}/{len(NBA_SERIES)}] Pulling {series_ticker} ({market_type})...")
        raw_markets = get_markets_for_series(series_ticker)
        valid = 0
        for m in raw_markets:
            ticker  = m.get("ticker", "")
            result  = m.get("result", "")
            open_ts = parse_ts(m.get("open_time", ""))
            close_ts = parse_ts(m.get("close_time", ""))
            if not ticker or result not in ("yes", "no"):
                continue
            duration_h = (close_ts - open_ts) / 3600 if close_ts > open_ts else 0
            all_markets.append({
                "market_id":     ticker,
                "event_ticker":  m.get("event_ticker", ""),
                "series_ticker": series_ticker,
                "market_type":   market_type,
                "title":         m.get("title", ""),
                "result":        result,
                "open_ts":       open_ts,
                "close_ts":      close_ts,
                "open_time":     m.get("open_time", ""),
                "close_time":    m.get("close_time", ""),
                "volume":        m.get("volume_fp", 0),
                "duration_h":    round(duration_h, 2),
            })
            valid += 1
        print(f"  -> {valid} valid resolved markets (of {len(raw_markets)} total)")
        time.sleep(SLEEP_BETWEEN_SERIES)

    if not all_markets:
        print("\n[WARNING] No markets collected.")
        return pd.DataFrame(), pd.DataFrame()

    markets_df = pd.DataFrame(all_markets).drop_duplicates(subset=["market_id"])
    print(f"\nTotal unique markets: {len(markets_df)}")
    print("By series:\n", markets_df["series_ticker"].value_counts().to_string())

    # Save raw markets immediately
    markets_df.to_csv(f"{output_dir}/raw_markets.csv", index=False)
    print(f"\nSaved: {output_dir}/raw_markets.csv")

    # Only pull candlesticks for markets with >= 24h duration (pre-filter)
    eligible = markets_df[markets_df["duration_h"] >= 24].reset_index(drop=True)
    print(f"\nMarkets >= 24h duration: {len(eligible)} / {len(markets_df)}")
    print("Pulling hourly candlesticks for eligible markets...")

    n = len(eligible)
    for idx, row in eligible.iterrows():
        if idx % 100 == 0:
            print(f"  Candlesticks: {idx}/{n} ...")

        candles = get_candlesticks(
            row["series_ticker"],
            row["market_id"],
            int(row["open_ts"]),
            int(row["close_ts"]),
        )
        for c in candles:
            price_info = c.get("price", {})
            close_d = price_info.get("close_dollars")
            if close_d is None:
                close_d = c.get("yes_bid", {}).get("close_dollars")
            try:
                close_price = float(close_d) if close_d is not None else None
            except (ValueError, TypeError):
                close_price = None
            if close_price is None:
                continue

            def _f(val, fallback=close_price):
                try: return float(val) if val is not None else fallback
                except: return fallback

            all_candles.append({
                "market_id": row["market_id"],
                "ts":        c.get("end_period_ts", 0),
                "close":     close_price,
                "high":      _f(price_info.get("high_dollars")),
                "low":       _f(price_info.get("low_dollars")),
                "open":      _f(price_info.get("open_dollars")),
                "volume":    _f(c.get("volume_fp", 0), 0.0),
            })
        time.sleep(SLEEP_CANDLESTICKS)

    candles_df = pd.DataFrame(all_candles)
    candles_df.to_csv(f"{output_dir}/raw_price_history.csv", index=False)
    print(f"\nSaved: {output_dir}/raw_markets.csv ({len(markets_df)} rows)")
    print(f"Saved: {output_dir}/raw_price_history.csv ({len(candles_df)} rows)")

    return markets_df, candles_df


if __name__ == "__main__":
    collect_all_data()
