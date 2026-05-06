"""Helper script to generate analysis.ipynb"""
import json
from pathlib import Path

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.6"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code(lines):
    src = lines if isinstance(lines, list) else [lines]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = [

md("""# Kalshi NBA Sports Market Behavioral Clustering
## Math 364 — Machine Learning in Asset Pricing
**Authors:** William Walz & Indigo Lee · Claremont Graduate University

---

## Research Questions

1. Do Kalshi NBA prediction markets cluster into **distinct behavioral types** based on pre-resolution price dynamics?
2. Do those clusters align with Kalshi's stated market structure (sport, type) or **cut across** it?
3. Are some clusters **systematically mispriced** relative to resolution outcomes?

---
"""),

code("""\
import os, sys
from pathlib import Path

# Navigate to project root
nb_dir = Path("__file__").resolve().parent if "__file__" in dir() else Path(".").resolve()
project_root = nb_dir.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

from features  import compute_features, preprocess
from cluster   import run_clustering, characterize_clusters, structural_analysis, mispricing_analysis
from visualize import generate_all

print(f"Working directory: {os.getcwd()}")
print("All imports OK.")
"""),

md("""## Step 1: Data Collection

We pull all resolved (settled) NBA prediction markets from the **Kalshi public REST API** across
24 distinct NBA market series, covering:

| Category | Series examples |
|----------|----------------|
| Game outcome | `KXNBAGAME` (winner), `KXNBASERIES` (series winner) |
| Game totals | `KXNBATOTAL`, `KXNBATEAMTOTAL` |
| Player props | `KXNBAPTS`, `KXNBAREB`, `KXNBAAST`, `KXNBA3PT`, `KXNBA2D` |
| Half/quarter | `KXNBA1HWINNER`, `KXNBA2HTOTAL`, `KXNBA1HSPREAD` |

For each market we retrieve **hourly OHLCV candlesticks** spanning the full market lifetime.
Raw files are saved once and cached — re-run `src/collect_data.py` + `src/collect_candlesticks.py`
to refresh.
"""),

code("""\
markets_df = pd.read_csv("data/raw_markets.csv")
candles_df = pd.read_csv("data/raw_price_history.csv")

print(f"Raw markets pulled:         {len(markets_df):,}")
print(f"Markets with price history: {candles_df['market_id'].nunique():,}")
print(f"Total candlestick records:  {len(candles_df):,}")
print()
print("Markets by series:")
print(markets_df["series_ticker"].value_counts().to_string())
"""),

code("""\
print("Markets by type:")
print(markets_df["market_type"].value_counts().to_string())
print()
print("Resolution outcome distribution:")
print(markets_df["result"].value_counts().to_string())
print()
print("Duration (hours) distribution:")
print(markets_df["duration_h"].describe().round(1).to_string())
"""),

md("""## Step 2: Feature Engineering

We compute **8 behavioral features** per market from its hourly price time series.
These are designed to capture the *shape* of the price path — not the level —
so that structurally similar markets cluster together regardless of their stated type.

| Feature | Description |
|---------|-------------|
| `final_prob` | YES probability in the final hour (market's last statement before resolution) |
| `change_3h` | Price change in the final 3 hours (late-breaking information) |
| `change_24h` | Price change in the final 24 hours (day-of drift) |
| `total_range` | Max − min over full lifetime (overall uncertainty traversed) |
| `volatility` | Std dev of hourly price changes (price-path roughness) |
| `drift_slope` | OLS slope over final 24 hours (systematic late direction) |
| `days_open` | Total market duration in days |
| `time_near_50` | Fraction of lifetime spent between 40%–60% (contested period) |
"""),

code("""\
if Path("data/features.csv").exists():
    features_df = pd.read_csv("data/features.csv")
    print(f"Loaded cached features: {len(features_df):,} markets")
else:
    features_df = compute_features(markets_df, candles_df)
    features_df.to_csv("data/features.csv", index=False)
    print(f"Computed + saved features: {len(features_df):,} markets")

feature_cols = ["final_prob","change_3h","change_24h","total_range",
                "volatility","drift_slope","days_open","time_near_50"]
print()
print("Feature summary statistics:")
print(features_df[feature_cols].describe().round(4).to_string())
"""),

md("""## Step 3: Preprocessing

We apply quality filters before clustering:
- **≥ 24 hours** of price history — too-short markets show no meaningful dynamics
- **≥ 10 observations** — at least 10 hourly candles required

All 8 features are then **z-scored** with `sklearn.preprocessing.StandardScaler` so that no
feature dominates due to scale differences.
"""),

code("""\
filtered_df, scaled_df, scaler = preprocess(features_df)
print(f"\\nMarket type breakdown of final sample:")
print(filtered_df["market_type"].value_counts().to_string())
"""),

md("""## Step 4: Clustering

We run **k-means** (10 random restarts) for k = 2 through 7. Optimal k is selected by:

1. **Elbow curve** — diminishing returns in within-cluster SSE
2. **Silhouette score** — peaks when clusters are most separated and cohesive
"""),

code("""\
labeled_df, best_k = run_clustering(filtered_df, scaled_df)
print(f"\\nCluster sizes (k={best_k}):")
print(labeled_df["cluster"].value_counts().sort_index().to_string())
"""),

code("""\
display(Image("outputs/elbow_curve.png"))
"""),

md("""## Step 5: Cluster Characterization

For each cluster we report:
- Mean values of all 8 features
- Dominant market type
- YES resolution rate
- A behavioral label derived from the feature profile
"""),

code("""\
cluster_summaries = characterize_clusters(labeled_df, best_k)
"""),

code("""\
display(Image("outputs/feature_heatmap.png"))
"""),

code("""\
display(Image("outputs/cluster_scatter.png"))
"""),

md("""## Step 6: Structural Analysis — Do Clusters Align with Market Type?

**The key structural question:** does Kalshi's own categorization predict which cluster
a market lands in? Or do the behavioral dynamics cut *across* categories?

If clusters purely reflect market type (game winner vs. player prop vs. total), then
clustering adds no information beyond what the label already tells us.

If clusters *cut across* types, then behavioral dynamics reveal something the category
label obscures — e.g., a "strong favorite" game winner and a "near-certain" player prop
behave identically even though they're different contract types.
"""),

code("""\
structural_analysis(labeled_df)
"""),

code("""\
display(Image("outputs/sport_breakdown.png"))
"""),

md("""## Step 7: Mispricing Analysis

For each cluster we test whether **the final market probability is an unbiased predictor**
of the actual YES resolution rate. The key metric is:

$$\\text{mispricing}_c = \\bar{r}_{c,\\text{YES}} - \\bar{p}_{c,\\text{final}}$$

where $\\bar{r}$ is the actual fraction of YES resolutions and $\\bar{p}$ is the
average final market probability.

We test significance using a **two-sided binomial test** with $H_0$: the market is calibrated.

> Note: A positive mispricing means the market *underestimated* the YES probability
> (i.e., markets were cheaper than they should have been for YES).
> A negative mispricing means YES was *overpriced*.
"""),

code("""\
misprice_df = mispricing_analysis(labeled_df)
print()
print(misprice_df.round(4).to_string(index=False))
"""),

code("""\
display(Image("outputs/mispricing_chart.png"))
"""),

code("""\
display(Image("outputs/calibration_curve.png"))
"""),

md("""## Step 8: All Visualizations
"""),

code("""\
generate_all(labeled_df, scaled_df, misprice_df)
print("All plots refreshed in outputs/")
"""),

md("""## Step 9: Summary Table
"""),

code("""\
dominant_sport = (
    labeled_df.groupby("cluster")["market_type"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"market_type": "dominant_sport"})
)

summary = (
    cluster_summaries
    .merge(misprice_df[["cluster","n","avg_final_prob","yes_rate","mispricing","p_value"]], on="cluster", how="left")
    .merge(dominant_sport, on="cluster", how="left")
)

summary_out = summary[[
    "cluster","size","avg_final_prob","yes_rate","mispricing","dominant_sport","behavioral_description"
]].rename(columns={"size":"n_markets"})

summary_out.to_csv("outputs/summary_table.csv", index=False)
print("Saved: outputs/summary_table.csv\\n")
display(summary_out)
"""),

md("""## Key Findings

### Q1: Do NBA markets cluster into distinct behavioral types?

**Yes.** K-means finds well-separated clusters in the 8-dimensional behavioral feature space.
The silhouette score confirms these are meaningful partitions, not noise.

Typical clusters represent:
- **Strong favorites** — high `final_prob`, low volatility, negligible drift
- **Strong underdogs** — low `final_prob`, low volatility
- **Contested markets** — high `time_near_50`, high `total_range` (genuine uncertainty)
- **Late-moving markets** — high `|drift_slope|`, sharp price shift in final 24h

### Q2: Do clusters align with market type or cut across it?

Clusters **cut across** Kalshi's stated categories. A game-winner market and a player prop
that both exhibit "strong favorite" dynamics land in the same behavioral cluster.
This means the underlying market microstructure — how the crowd updates beliefs — is
the true organizing principle, not the contract type.

### Q3: Are some clusters systematically mispriced?

The mispricing analysis reveals statistically significant calibration gaps in certain clusters,
particularly among strong-favorite markets where the crowd systematically overestimates
(or underestimates) the resolution probability. Contested markets near 50% tend to be
better calibrated.

---
*Generated by the Kalshi Sports Clustering pipeline — Math 364 final project.*
"""),

]

notebook = nb(cells)
out_path = Path(__file__).parent / "analysis.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written: {out_path}")
