# Kalshi Sports Clustering Project

## What this is
Math 364 (Machine Learning in Asset Pricing) final project at Claremont Graduate University.
Authors: William Walz and Indigo Lee.

## Research question
Do sports prediction markets on Kalshi cluster into distinct behavioral types based on their pre-resolution price dynamics, and do those clusters reflect Kalshi's stated sport/league/market structure or cut across it? Bonus: are some clusters systematically mispriced?

## Approach
- Pull resolved Kalshi sports contracts via REST API
- Engineer ~6 behavioral features per contract: volatility, directional drift, time-to-resolution decay, end-jump magnitude, path linearity, closing price level
- K-means clustering on the standardized feature matrix
- Compare clusters to sport/league/market-type metadata via confusion matrices
- Bonus: per-cluster calibration analysis (do prices match outcomes?)

## Tech stack
- **Python** for all data pulling, feature engineering, clustering, and the dashboard
- **R** for visualization only (using ggplot2 and a custom theme_kalshi())
- **Streamlit** for the local data console (development environment)
- **SQLite** for local API response caching
- Outputs from Python land in `results/` as CSV/JSON; R reads from there

## Directory structure