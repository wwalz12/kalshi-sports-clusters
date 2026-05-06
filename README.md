# Kalshi NBA Sports Market Behavioral Clustering

**Math 364 — Machine Learning in Asset Pricing**  
William Walz & Indigo Lee · Claremont Graduate University

---

## Research Questions

1. Do Kalshi NBA prediction markets cluster into **distinct behavioral types** based on pre-resolution price dynamics?
2. Do those clusters align with Kalshi's stated market structure (sport, type) or **cut across** it?
3. Are some clusters **systematically mispriced** relative to resolution outcomes?

---

## Project Structure

```
kalshi-sports-clusters/
├── data/
│   ├── raw_markets.csv        # All pulled markets with metadata
│   ├── raw_price_history.csv  # Hourly OHLCV candlesticks per market
│   ├── features.csv           # Engineered behavioral features
│   └── clustered_markets.csv  # Markets with cluster assignments
├── notebooks/
│   └── analysis.ipynb         # Full end-to-end analysis notebook
├── outputs/
│   ├── elbow_curve.png        # K-means elbow + silhouette
│   ├── cluster_scatter.png    # PCA 2D scatter colored by cluster
│   ├── feature_heatmap.png    # Cluster feature profiles (z-scored)
│   ├── sport_breakdown.png    # Market type composition per cluster
│   ├── mispricing_chart.png   # Mispricing with confidence intervals
│   ├── calibration_curve.png  # Calibration per cluster
│   ├── mispricing_table.csv   # Statistical test results
│   └── summary_table.csv      # One-row-per-cluster summary
├── src/
│   ├── collect_data.py        # Pull markets from Kalshi API
│   ├── collect_candlesticks.py # Pull price histories (stratified sample)
│   ├── features.py            # Feature engineering & preprocessing
│   ├── cluster.py             # K-means clustering + characterization
│   ├── visualize.py           # All visualizations
│   └── run_pipeline.py        # One-shot Steps 2–9 runner
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1: Collect Data (one-time)

```bash
# Pull all resolved NBA markets (saves data/raw_markets.csv)
python3 src/collect_data.py

# Pull price histories for a stratified sample of markets
# (saves data/raw_price_history.csv)
python3 src/collect_candlesticks.py
```

These scripts use the **Kalshi public REST API** — no authentication required.
Rate limits are handled automatically with exponential backoff.

### Steps 2–9: Run Analysis

```bash
python3 src/run_pipeline.py
```

Or open `notebooks/analysis.ipynb` in Jupyter and run all cells.

---

## Data Sources

- **API:** [Kalshi REST API v2](https://api.elections.kalshi.com/trade-api/v2)
- **Endpoint for markets:** `/markets?status=settled&series_ticker=KXNBA...`
- **Endpoint for price history:** `/series/{series}/markets/{ticker}/candlesticks`

### NBA Series Pulled

| Series Ticker | Type |
|---------------|------|
| `KXNBAGAME` | Game winner (binary: home/away) |
| `KXNBATOTAL` | Total points (over/under) |
| `KXNBATEAMTOTAL` | Team total points |
| `KXNBASERIES` | Playoff series winner |
| `KXNBAPTS` | Player points per game |
| `KXNBAREB` | Player rebounds |
| `KXNBAAST` | Player assists |
| `KXNBA3PT` | Player three-pointers |
| `KXNBA2D` | Player double-doubles |
| `KXNBABLK` | Player blocks |
| `KXNBASTL` | Player steals |
| `KXNBA1HWINNER` | 1st half winner |
| `KXNBA2HWINNER` | 2nd half winner |
| `KXNBA1HTOTAL` | 1st half total |
| `KXNBA2HTOTAL` | 2nd half total |
| `KXNBA1HSPREAD` | 1st half spread |
| `KXNBA2HSPREAD` | 2nd half spread |
| `KXNBA1QWINNER` | Quarter winner |
| `KXNBAOVERTIME` | Overtime |
| `KXNBASERIESSCORE` | Series score |

---

## Features

| Feature | Description |
|---------|-------------|
| `final_prob` | YES probability in final hour before resolution |
| `change_3h` | Price change in final 3 hours |
| `change_24h` | Price change in final 24 hours |
| `total_range` | Max − min over market lifetime |
| `volatility` | Std dev of hourly price changes |
| `drift_slope` | OLS slope over final 24 hours |
| `days_open` | Market lifetime in days |
| `time_near_50` | Fraction of lifetime between 40%–60% |

---

## Methods

- **Clustering:** K-means (k = 2–7, 10 random restarts), optimal k via silhouette score
- **Preprocessing:** StandardScaler normalization
- **Dimensionality reduction:** PCA (2 components) for visualization
- **Calibration:** `sklearn.calibration.calibration_curve` per cluster
- **Mispricing test:** Two-sided binomial test, $H_0$: market is calibrated

---

## Key Findings

**Clusters cut across market types.** A game-winner market and a player prop that both
exhibit "strong favorite" dynamics land in the same behavioral cluster. This means
pre-resolution price dynamics — not the contract category — organize the true structure.

**Certain clusters are systematically mispriced.** Strong-favorite and strong-underdog
clusters show statistically significant gaps between final market probability and actual
resolution rates. Contested markets near 50% exhibit better calibration.

---

## Stack

- Python 3.12
- `kalshi-python` for API type definitions
- `pandas`, `numpy` for data manipulation
- `scikit-learn` for k-means, StandardScaler, PCA, calibration
- `scipy` for binomial tests
- `matplotlib`, `seaborn` for visualization
- `jupyter` for interactive notebook
