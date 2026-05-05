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
- **Python** for data pulling, panel construction, PCA, clustering  
- **Streamlit** for development console / dashboard  
- **SQLite (sqlite3)** for local API cache  
- **R** for visualization only (ggplot2 + `theme_kalshi()`)  
- Outputs from Python land in `results/` as CSV/JSON; R reads from there  

## Directory structure
kalshi-factors/  
├── console.py              # Streamlit dashboard (primary dev interface)  
├── kalshi_lib/             # shared backend logic  
│   ├── api.py              # API wrapper + caching  
│   ├── panel.py            # panel construction  
│   ├── pca.py  
│   └── cluster.py  
├── python/                 # final scripts (exported from kalshi_lib)  
│   ├── 01_pull_data.py  
│   ├── 02_build_panel.py  
│   ├── 03_pca.py  
│   └── 04_cluster.py  
├── data/  
│   ├── kalshi_cache.db     # SQLite cache  
│   ├── raw_candles.csv  
│   └── panel.csv  
├── r/  
│   └── plots.R  
├── results/                # pca_results.json, clusters.csv  
├── figures/                # output PNGs  
└── CLAUDE.md  

## Development workflow
- Primary work happens in `console.py` (Streamlit dashboard)  
- Dashboard is used to:
  - Browse contracts  
  - Run data pulls with live progress  
  - Inspect panel quality  
  - Preview PCA results  
- All core logic lives in `kalshi_lib/`  
- Final scripts in `python/` just import and run that logic  
- Do not duplicate logic between console and scripts  

## Console features
- **Contract browser**  
  - Query `/markets` with filters  
  - View tickers, categories, volume, open interest  
  - Inspect individual price series  

- **Pull progress monitor**  
  - Live status: tickers attempted, success/fail, errors  
  - Candles per ticker, elapsed time  
  - Incremental saving to disk  

- **Panel diagnostics**  
  - Missing data heatmap (date × ticker)  
  - Distribution of returns  
  - Flag bad series (flat, zero, jumps)  

- **Quick PCA preview**  
  - Scree plot  
  - Top loadings  
  - Fast sanity check on factor structure  

## Conventions
- Python: pandas, numpy, scikit-learn, requests, sqlite3  
- File naming: numbered prefixes (`01_`, `02_`)  
- Data: CSV (tabular), JSON (nested results)  
- Panel: wide format (rows = dates, columns = tickers)  

## Returns definition
Daily log-odds changes:  
log(p/(1-p))_t − log(p/(1-p))_{t-1}  
- Prices clipped to [0.01, 0.99] before transform  

## Data source
Kalshi public REST API:  
https://api.elections.kalshi.com/trade-api/v2  

Endpoints used:  
- `/markets` → discovery  
- `/series/{series}/markets/{ticker}/candlesticks` → price data  

## Sample universe
- Time range: June 1, 2024 → November 1, 2025  
- Categories: Politics, Macro/Economics, Sports, Weather, Crypto/Markets  
- Filters:  
  - ≥80% trading day coverage  
  - Avg daily volume ≥100  
  - Resolved contracts only  

## API quirks (Kalshi-specific)
- Resolved contracts disappear after some time → pull + cache early  
- Candlestick endpoint has time-window limits → must paginate  
- Discovery requires multiple endpoints → cache results  
- Rate limits are unclear → add small delay (~0.3s/request)  
- All responses cached in `data/kalshi_cache.db` to avoid re-pulls  

## Style preferences for code
- Short, direct comments  
- Print progress for long pulls  
- Save intermediates aggressively  
- Minimal error handling (log + skip is fine)  

## Style preferences for output text
- Casual, direct tone  
- No AI-sounding phrasing  
- Short sentences  