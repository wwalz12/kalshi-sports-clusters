# Kalshi Factor Structure Project

## What this is
Math 364 (Machine Learning in Asset Pricing) final project at Claremont Graduate University.
Authors: William Walz and Indigo Lee.

## Research question
How many independent bets are there really on Kalshi, and do contracts that look related actually move together?

## Approach
- PCA on daily log-odds changes of ~80–150 Kalshi contracts to estimate effective dimensionality
- K-means on PCA loadings to cluster contracts; compare to Kalshi's stated categories

## Tech stack
- **Python** for all data pulling, panel construction, PCA, and clustering
- **R** for visualization only (using ggplot2 and a custom theme_kalshi())
- Outputs from Python land in `results/` as CSV/JSON; R reads from there

## Directory structure
kalshi-factors/
├── data/         # raw_candles.csv, panel.csv
├── python/       # 01_pull_data.py, 02_build_panel.py, 03_pca.py, 04_cluster.py
├── r/            # plots.R
├── results/      # pca_results.json, clusters.csv
└── figures/      # output PNGs

## Conventions
- Python: pandas, numpy, scikit-learn, requests. No deep learning frameworks.
- File naming: numbered prefixes (`01_`, `02_`) so execution order is obvious.
- Data files: CSV for tabular, JSON for nested results.
- Returns: daily log-odds changes, computed as log(p/(1-p))_t − log(p/(1-p))_{t-1}, with prices clipped to [0.01, 0.99] before transformation.
- Panel: wide format, rows=dates, columns=contract tickers.

## Data source
Kalshi public REST API: https://api.elections.kalshi.com/trade-api/v2
Endpoints used: `/markets` for discovery, `/series/{series}/markets/{ticker}/candlesticks` for prices.

## Sample universe
- Time range: June 1, 2024 to November 1, 2025
- Categories: Politics, Macro/Economics, Sports, Weather, Crypto/Markets
- Filters: ≥80% trading day coverage, average daily volume ≥100 contracts, resolved only

## Style preferences for code
- Short, direct comments. No corporate-style docstrings.
- Print informative progress for long-running pulls.
- Save intermediate outputs so we don't re-pull on every run.
- Don't over-engineer error handling — log and skip is fine for this project.

## Style preferences for output text
- Casual, direct tone in any markdown or comments.
- No AI-sounding phrasing or filler.
- Short sentences.