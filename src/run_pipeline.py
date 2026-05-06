"""
Full pipeline runner: reads saved CSVs and runs Steps 2-9.
Run from project root: python3 src/run_pipeline.py
"""

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import pandas as pd
import numpy as np
from pathlib import Path

from features  import compute_features, preprocess
from cluster   import run_clustering, characterize_clusters, structural_analysis, mispricing_analysis
from visualize import generate_all

FEATURE_COLS = [
    "final_prob", "change_3h", "change_24h", "total_range",
    "volatility", "drift_slope", "days_open", "time_near_50",
]


def main():
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    # ── Step 2: Feature Engineering ──────────────────────────────────────────
    if Path("data/features.csv").exists():
        print("Loading cached features from data/features.csv...")
        features_df = pd.read_csv("data/features.csv")
    else:
        print("Computing features from raw data...")
        markets_df = pd.read_csv("data/raw_markets.csv")
        candles_df = pd.read_csv("data/raw_price_history.csv")
        features_df = compute_features(markets_df, candles_df)
        features_df.to_csv("data/features.csv", index=False)
        print(f"Saved: data/features.csv ({len(features_df)} rows)")

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    filtered_df, scaled_df, scaler = preprocess(features_df)

    if len(filtered_df) < 30:
        print(f"[ERROR] Only {len(filtered_df)} markets after filtering. Need at least 30.")
        return

    # ── Step 4: Clustering ────────────────────────────────────────────────────
    labeled_df, best_k = run_clustering(filtered_df, scaled_df)

    # ── Step 5: Cluster Characterization ─────────────────────────────────────
    cluster_summaries = characterize_clusters(labeled_df, best_k)

    # ── Step 6: Structural Analysis ───────────────────────────────────────────
    structural_analysis(labeled_df)

    # ── Step 7: Mispricing Analysis ───────────────────────────────────────────
    misprice_df = mispricing_analysis(labeled_df)

    # ── Step 8: Visualizations ────────────────────────────────────────────────
    generate_all(labeled_df, scaled_df, misprice_df)

    # ── Step 9: Summary Table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: FINAL OUTPUT")
    print("=" * 60)

    # Merge cluster summaries with mispricing
    # cluster_summaries already has yes_rate; misprice_df also has yes_rate.
    # Drop yes_rate from summaries to avoid the clash.
    cs_cols = [c for c in cluster_summaries.columns if c != "yes_rate"]
    summary = cluster_summaries[cs_cols].merge(
        misprice_df[["cluster", "n", "avg_final_prob", "yes_rate", "mispricing", "p_value", "significant"]],
        on="cluster", how="left",
    )
    dominant_sport = labeled_df.groupby("cluster")["market_type"].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index().rename(columns={"market_type": "dominant_sport"})
    summary = summary.merge(dominant_sport, on="cluster", how="left")

    summary_out = summary[[
        "cluster", "size", "avg_final_prob", "yes_rate",
        "mispricing", "dominant_sport", "behavioral_description",
    ]].rename(columns={"size": "n_markets"})
    summary_out.to_csv("outputs/summary_table.csv", index=False)
    print("Saved: outputs/summary_table.csv")
    print(summary_out.to_string(index=False))

    print("\n=== PIPELINE COMPLETE ===")
    print("Output files:")
    for f in sorted(Path("outputs").iterdir()):
        print(f"  {f}")


if __name__ == "__main__":
    main()
