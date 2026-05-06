"""
Targeted candlestick collection: read raw_markets.csv, sample smartly,
pull hourly price histories with robust rate-limit handling.
Run from project root: python3 src/collect_candlesticks.py
"""

import os, sys, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

import requests
import pandas as pd
import numpy as np
from pathlib import Path

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MIN_DURATION_H = 24
MIN_OBS_EXPECTED = 10  # at least 10 hourly candles expected
MAX_PER_TYPE = 300     # sample up to 300 markets per market_type
SLEEP_OK     = 0.6     # seconds between successful calls
SLEEP_RETRY  = 15.0    # seconds after 429
MAX_RETRIES  = 3


def get_candlesticks(series_ticker: str, market_ticker: str,
                     open_ts: int, close_ts: int) -> list:
    url = f"{BASE_URL}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    params = {"period_interval": 60, "start_ts": open_ts, "end_ts": close_ts}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                wait = SLEEP_RETRY * (attempt + 1)
                print(f"    [429] sleeping {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("candlesticks", [])
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    [FAIL] {market_ticker}: {e}")
                return []
            time.sleep(SLEEP_RETRY)
    return []


def sample_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select a stratified sample: up to MAX_PER_TYPE markets per market_type,
    restricted to >= 24h duration markets with a valid result.
    Preserve diversity across series tickers within each type.
    """
    eligible = markets_df[
        (markets_df["duration_h"] >= MIN_DURATION_H) &
        (markets_df["result"].isin(["yes", "no"])) &
        (markets_df["open_ts"] > 0) &
        (markets_df["close_ts"] > 0)
    ].copy()

    sampled = []
    for mtype, group in eligible.groupby("market_type"):
        n = min(len(group), MAX_PER_TYPE)
        # Stratify across series tickers within type for diversity
        sampled.append(group.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42)


def collect_candlesticks(output_dir: str = "data") -> pd.DataFrame:
    Path(output_dir).mkdir(exist_ok=True)
    markets_path = f"{output_dir}/raw_markets.csv"
    candles_path = f"{output_dir}/raw_price_history.csv"

    if not Path(markets_path).exists():
        print(f"[ERROR] {markets_path} not found. Run collect_data.py first.")
        sys.exit(1)

    markets_df = pd.read_csv(markets_path)
    print(f"Loaded {len(markets_df)} markets from {markets_path}")

    # Check if we can resume from existing candles file
    already_done = set()
    if Path(candles_path).exists():
        existing = pd.read_csv(candles_path)
        already_done = set(existing["market_id"].unique())
        print(f"Resuming: {len(already_done)} markets already have candlestick data")
    else:
        existing = pd.DataFrame()

    sample = sample_markets(markets_df)
    print(f"\nStratified sample: {len(sample)} markets")
    print(sample["market_type"].value_counts().to_string())

    # Skip already collected
    todo = sample[~sample["market_id"].isin(already_done)].reset_index(drop=True)
    print(f"To collect: {len(todo)} (skipping {len(already_done)} already done)")

    all_candles = []
    n = len(todo)

    for idx, row in todo.iterrows():
        if idx % 100 == 0:
            print(f"  [{idx}/{n}] series={row['series_ticker']}, type={row['market_type']}")

        candles = get_candlesticks(
            row["series_ticker"],
            row["market_id"],
            int(row["open_ts"]),
            int(row["close_ts"]),
        )
        if not candles:
            time.sleep(SLEEP_OK)
            continue

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

            def _f(val, fb=close_price):
                try: return float(val) if val is not None else fb
                except: return fb

            all_candles.append({
                "market_id": row["market_id"],
                "ts":        c.get("end_period_ts", 0),
                "close":     close_price,
                "high":      _f(price_info.get("high_dollars")),
                "low":       _f(price_info.get("low_dollars")),
                "open":      _f(price_info.get("open_dollars")),
                "volume":    _f(c.get("volume_fp", 0), 0.0),
            })
        time.sleep(SLEEP_OK)

        # Save checkpoint every 500 markets
        if len(all_candles) > 0 and idx % 500 == 499:
            new_df = pd.DataFrame(all_candles)
            combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
            combined.to_csv(candles_path, index=False)
            print(f"  [Checkpoint] Saved {len(combined)} candlestick rows")
            existing = combined
            all_candles = []

    # Final save
    if all_candles:
        new_df = pd.DataFrame(all_candles)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    else:
        combined = existing

    combined.to_csv(candles_path, index=False)
    print(f"\nDone. Saved {len(combined)} candlestick rows to {candles_path}")
    print(f"Unique markets with price data: {combined['market_id'].nunique()}")
    return combined


if __name__ == "__main__":
    collect_candlesticks()
