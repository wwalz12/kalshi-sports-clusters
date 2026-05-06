"""
Step 2 & 3: Feature Engineering and Preprocessing
Compute per-market behavioral features from price history.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def compute_features(markets_df: pd.DataFrame, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Compute behavioral features for each market from its price history."""

    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    if candles_df.empty or markets_df.empty:
        print("[ERROR] Empty input DataFrames.")
        return pd.DataFrame()

    candles_df = candles_df.copy()
    candles_df["ts"] = pd.to_numeric(candles_df["ts"], errors="coerce")
    candles_df["close"] = pd.to_numeric(candles_df["close"], errors="coerce")
    candles_df = candles_df.dropna(subset=["ts", "close"])
    candles_df = candles_df.sort_values(["market_id", "ts"])

    features = []
    skipped_min_hours = 0
    skipped_min_obs = 0

    for market_id, group in candles_df.groupby("market_id"):
        group = group.sort_values("ts").reset_index(drop=True)
        n_obs = len(group)

        if n_obs < 2:
            skipped_min_obs += 1
            continue

        prices = group["close"].values
        timestamps = group["ts"].values

        # Market lifetime in hours
        duration_hours = (timestamps[-1] - timestamps[0]) / 3600.0
        duration_days = duration_hours / 24.0

        # final_prob: price in final hour (last observation before resolution)
        final_prob = float(prices[-1])

        # change_3h: price change in final 3 hours (last 3 candles vs 4th-from-last)
        if n_obs >= 4:
            change_3h = float(prices[-1] - prices[-4])
        else:
            change_3h = float(prices[-1] - prices[0])

        # change_24h: price change in final 24 hours (last 24 candles)
        if n_obs >= 25:
            change_24h = float(prices[-1] - prices[-25])
        else:
            change_24h = float(prices[-1] - prices[0])

        # total_range: max - min over lifetime
        total_range = float(prices.max() - prices.min())

        # volatility: std of hourly price changes
        price_changes = np.diff(prices)
        volatility = float(np.std(price_changes)) if len(price_changes) > 0 else 0.0

        # drift_slope: linear regression slope over final 24 hours
        tail = group.tail(24)
        if len(tail) >= 3:
            x = np.arange(len(tail))
            slope = float(np.polyfit(x, tail["close"].values, 1)[0])
        else:
            x = np.arange(n_obs)
            slope = float(np.polyfit(x, prices, 1)[0]) if n_obs >= 2 else 0.0
        drift_slope = slope

        # time_near_50: fraction of lifetime spent between 40% and 60%
        near_50 = ((prices >= 0.40) & (prices <= 0.60))
        time_near_50 = float(near_50.sum() / n_obs)

        features.append({
            "market_id":   market_id,
            "n_obs":       n_obs,
            "duration_hours": duration_hours,
            "final_prob":  final_prob,
            "change_3h":   change_3h,
            "change_24h":  change_24h,
            "total_range": total_range,
            "volatility":  volatility,
            "drift_slope": drift_slope,
            "days_open":   duration_days,
            "time_near_50": time_near_50,
        })

    features_df = pd.DataFrame(features)
    print(f"Computed features for {len(features_df)} markets")
    print(f"  Skipped (< 2 obs): {skipped_min_obs}")

    # Merge with market metadata
    meta_cols = ["market_id", "series_ticker", "market_type", "title", "result", "open_ts", "close_ts"]
    meta = markets_df[meta_cols].drop_duplicates(subset=["market_id"])
    features_df = features_df.merge(meta, on="market_id", how="left")

    return features_df


def preprocess(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Filter markets and normalize features. Returns (filtered_df, scaled_array_df, scaler)."""

    print("\n" + "=" * 60)
    print("STEP 3: PREPROCESSING")
    print("=" * 60)

    initial_count = len(features_df)

    # Drop markets with fewer than 24 hours of price history
    features_df = features_df[features_df["duration_hours"] >= 24].copy()
    after_hours = len(features_df)
    print(f"After dropping < 24h duration: {initial_count} -> {after_hours}")

    # Drop markets with fewer than 10 price observations
    features_df = features_df[features_df["n_obs"] >= 10].copy()
    after_obs = len(features_df)
    print(f"After dropping < 10 observations: {after_hours} -> {after_obs}")

    # Drop rows with NaN in feature columns
    feature_cols = [
        "final_prob", "change_3h", "change_24h", "total_range",
        "volatility", "drift_slope", "days_open", "time_near_50",
    ]
    features_df = features_df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"After dropping NaN features: {after_obs} -> {len(features_df)}")
    print(f"\nMarkets remaining for clustering: {len(features_df)}")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[feature_cols].values)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    return features_df, scaled_df, scaler


if __name__ == "__main__":
    markets_df = pd.read_csv("data/raw_markets.csv")
    candles_df = pd.read_csv("data/raw_price_history.csv")

    features_df = compute_features(markets_df, candles_df)
    features_df.to_csv("data/features.csv", index=False)
    print("\nSaved: data/features.csv")

    filtered_df, scaled_df, scaler = preprocess(features_df)
    print(f"\nFinal dataset: {len(filtered_df)} markets")
