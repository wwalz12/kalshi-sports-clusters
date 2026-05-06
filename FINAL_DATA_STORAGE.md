# FINAL DATA STORAGE

**Project:** Kalshi NBA Sports Market Behavioral Clustering  
**Course:** Math 364 — Machine Learning in Asset Pricing  
**Authors:** William Walz & Indigo Lee · Claremont Graduate University  
**Last Updated:** 2026-05-06

This document is the authoritative reference for every data artifact in this repository. It covers raw API sources, intermediate processed files, final outputs, exact column schemas, transformation formulas, data lineage, and full reproduction instructions.

---

## Table of Contents

1. [Data Architecture Overview](#1-data-architecture-overview)
2. [API Source](#2-api-source)
3. [Raw Data: `data/raw_markets.csv`](#3-raw-data-dataraw_marketscsv)
4. [Raw Data: `data/raw_price_history.csv`](#4-raw-data-dataraw_price_historycsv)
5. [Intermediate: `data/features.csv`](#5-intermediate-datafeaturescsv)
6. [Intermediate: `data/clustered_markets.csv`](#6-intermediate-dataclustered_marketscsv)
7. [Output: `outputs/summary_table.csv`](#7-output-outputssummary_tablecsv)
8. [Output: `outputs/mispricing_table.csv`](#8-output-outputsmispricing_tablecsv)
9. [Output Figures](#9-output-figures)
10. [Data Lineage Table](#10-data-lineage-table)
11. [Row Count Funnel](#11-row-count-funnel)
12. [Field Cross-Reference Index](#12-field-cross-reference-index)
13. [Reproducing the Data from Scratch](#13-reproducing-the-data-from-scratch)

---

## 1. Data Architecture Overview

The pipeline moves through four distinct data layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 0 — SOURCE                                                        │
│  Kalshi REST API v2  (public, no auth)                                   │
│  https://api.elections.kalshi.com/trade-api/v2                          │
└─────────────────────┬──────────────────────────┬────────────────────────┘
                      │                          │
          /markets    │                          │  /series/{s}/markets/
          ?status=    │                          │  {m}/candlesticks
          settled     │                          │  ?period_interval=60
                      ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — RAW                                                           │
│  data/raw_markets.csv          19,406 unique resolved NBA markets        │
│  data/raw_price_history.csv    49,015 hourly candlestick records         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │  src/features.py → compute_features()
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — FEATURES                                                      │
│  data/features.csv             996 markets × 18 columns                 │
│                                (8 behavioral features + metadata)        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │  src/features.py → preprocess()
                                  │  + src/cluster.py → run_clustering()
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — CLUSTERS                                                      │
│  data/clustered_markets.csv    946 markets × 19 columns                 │
│                                (features + cluster assignment)           │
└────────────┬────────────────────┬────────────────────┬──────────────────┘
             │                    │                    │
    cluster. │           cluster. │          visualize │
    py       │           py       │          .py       │
    mispricing_          summary_ │                    │
    analysis()           table    │                    │
             │                    │                    │
             ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐
│  LAYER 4 — OUTPUT│  │  LAYER 4 — OUTPUT│  │  LAYER 4 — OUTPUT            │
│  outputs/        │  │  outputs/        │  │  outputs/ (6 PNG figures)    │
│  mispricing_     │  │  summary_        │  │  elbow_curve.png             │
│  table.csv       │  │  table.csv       │  │  cluster_scatter.png         │
│  4 rows          │  │  4 rows          │  │  feature_heatmap.png         │
└──────────────────┘  └──────────────────┘  │  sport_breakdown.png         │
                                             │  mispricing_chart.png        │
                                             │  calibration_curve.png       │
                                             └──────────────────────────────┘
```

**Design principle:** Each layer is a checkpoint. You can reproduce any downstream layer from the layer immediately above it without re-hitting the API.

---

## 2. API Source

### Base URL

```
https://api.elections.kalshi.com/trade-api/v2
```

No authentication is required for market data or price history. The API is publicly accessible.

### Markets Endpoint

```
GET /markets
  ?status=settled          Required. Only "settled" returns resolved markets.
  &series_ticker=KXNBAGAME Required. Filter to one NBA series at a time.
  &limit=200               Max page size.
  &cursor={cursor}         Pagination token returned in each response.
```

**Pagination:** Cursor-based. Each response includes a `cursor` field. Pass it as `?cursor=` in the next request. An absent or empty `cursor` means the last page has been reached. The pipeline caps at 10 pages per series (up to 2,000 markets per series).

**Rate limits:** The API enforces rate limits that trigger HTTP 429 responses. The collection scripts handle this by sleeping 10–15 seconds and retrying up to 3 times per page.

### Candlesticks Endpoint

```
GET /series/{series_ticker}/markets/{market_ticker}/candlesticks
  ?period_interval=60      Candle width in minutes. 60 = hourly.
  &start_ts={unix_seconds} Required. Start of time range.
  &end_ts={unix_seconds}   Required. End of time range.
```

Returns a JSON array `candlesticks[...]`, each element containing:

| JSON path | Description |
|-----------|-------------|
| `end_period_ts` | End of the 1-hour candle window (Unix seconds) |
| `price.close_dollars` | YES closing price (primary) |
| `price.high_dollars` | YES high price |
| `price.low_dollars` | YES low price |
| `price.open_dollars` | YES opening price |
| `volume_fp` | Volume traded in this candle (contracts) |
| `yes_bid.close_dollars` | Fallback close when `price.close_dollars` is missing |

All dollar values are in the range `[0.0000, 1.0000]` and represent the probability that the market resolves YES.

### NBA Series Pulled

The following 24 series were queried. The `market_type` label is assigned by the pipeline, not by Kalshi.

| Series Ticker | Kalshi Title | `market_type` Label |
|--------------|--------------|---------------------|
| `KXNBAGAME` | Professional Basketball Game | `game_winner` |
| `KXNBATOTAL` | Pro Basketball Total Points | `total_points` |
| `KXNBASERIES` | Professional Basketball Series | `series_outcome` |
| `KXNBAPTS` | Pro Basketball Player Points | `player_points` |
| `KXNBAPA` | Pro Basketball Points + Assists | `player_prop` |
| `KXNBAPR` | Pro Basketball Points + Rebounds | `player_prop` |
| `KXNBAPRA` | Pro Basketball Points + Rebounds + Assists | `player_prop` |
| `KXNBAREB` | Pro Basketball Player Rebounds | `player_prop` |
| `KXNBAAST` | Pro Basketball Player Assists | `player_prop` |
| `KXNBA3PT` | Pro Basketball Player Threes | `player_prop` |
| `KXNBABLK` | Pro Basketball Player Blocks | `player_prop` |
| `KXNBASTL` | Pro Basketball Player Steals | `player_prop` |
| `KXNBA2D` | Pro Basketball Double Double | `player_prop` |
| `KXNBA1HWINNER` | Pro Basketball 1st Half Winner | `half_winner` |
| `KXNBA2HWINNER` | Pro Basketball 2nd Half Winner | `half_winner` |
| `KXNBA1HTOTAL` | NBA 1st Half Total Points | `half_total` |
| `KXNBA2HTOTAL` | NBA 2nd Half Total Points | `half_total` |
| `KXNBA1HSPREAD` | Pro Basketball 1st Half Spread | `half_spread` |
| `KXNBA2HSPREAD` | Pro Basketball 2nd Half Spread | `half_spread` |
| `KXNBA1QWINNER` | NBA 1st Quarter Winner | `quarter_winner` |
| `KXNBATEAMTOTAL` | NBA Team Total | `team_total` |
| `KXNBAOVERTIME` | NBA Overtime | `overtime` |
| `KXNBAPLAYOFFPTS` | NBA Playoffs Player Points | `player_points` |
| `KXNBASERIESSCORE` | NBA Series Exact Score | `series_outcome` |

---

## 3. Raw Data: `data/raw_markets.csv`

**Produced by:** `src/collect_data.py` → `collect_all_data()`  
**Row count:** 19,406 (unique markets after deduplication on `market_id`)  
**File size:** ~3.9 MB

### Column Schema

| Column | dtype | Range / Values | Description |
|--------|-------|----------------|-------------|
| `market_id` | string | e.g. `KXNBAGAME-26MAY05CLEDET-DET` | **Primary key.** Kalshi's unique ticker for this binary contract. Format: `{SERIES}-{EVENT_CODE}-{OUTCOME_CODE}`. Always unique. |
| `event_ticker` | string | e.g. `KXNBAGAME-26MAY05CLEDET` | Identifies the parent event (game/series). Multiple markets (YES side, NO side) share an `event_ticker`. |
| `series_ticker` | string | see table above | The NBA series this market belongs to. Used to build candlestick URLs. |
| `market_type` | string | `game_winner`, `total_points`, `series_outcome`, `player_points`, `player_prop`, `half_winner`, `half_total`, `half_spread`, `quarter_winner`, `team_total`, `overtime` | Pipeline-assigned label derived from `series_ticker`. Used for cluster cross-tabulation in Step 6. |
| `title` | string | free text | Human-readable market description, e.g. `"Game 1: Cleveland at Detroit Winner?"`. |
| `result` | string | `"yes"` or `"no"` | The resolved outcome of the binary contract. `"yes"` means the contract resolved YES (the stated condition was met). Markets without a result in `{"yes", "no"}` are excluded during collection. |
| `open_ts` | int64 | Unix seconds (e.g. `1777907100`) | Market open time as Unix timestamp in seconds. Parsed from `open_time` ISO string. |
| `close_ts` | int64 | Unix seconds | Market close/resolution time as Unix timestamp in seconds. Parsed from `close_time` ISO string. |
| `open_time` | string | ISO 8601, UTC, e.g. `"2026-05-04T15:05:00Z"` | Market open time as returned by the API. Redundant with `open_ts`; kept for human readability. |
| `close_time` | string | ISO 8601, UTC | Market close/resolution time as returned by the API. |
| `volume` | float64 | `≥ 0` (e.g. `8154592.45`) | Total trading volume for this contract over its lifetime, in contract units as returned by `volume_fp` in the API. |
| `duration_h` | float64 | `≥ 0` (e.g. `34.59`) | `(close_ts − open_ts) / 3600`. Market lifetime in hours. Computed during collection; 0 if timestamps are invalid. |

### `market_type` Value Distribution

The following counts reflect the full 19,406-market dataset:

| `market_type` | Count | % |
|--------------|-------|---|
| `half_spread` | 2,000 | 10.3% |
| `team_total` | 2,000 | 10.3% |
| `player_prop` | ~5,800 | ~29.9% |
| `player_points` | ~1,400 | ~7.2% |
| `total_points` | 1,600 | 8.2% |
| `half_winner` | ~1,650 | ~8.5% |
| `half_total` | ~1,470 | ~7.6% |
| `game_winner` | 820 | 4.2% |
| `overtime` | 54 | 0.3% |
| `series_outcome` | ~84 | ~0.4% |
| `quarter_winner` | — | — |

### Timestamp Conventions

- `open_ts` and `close_ts` are **Unix epoch seconds** (integer). Convert to datetime: `datetime.fromtimestamp(ts, tz=timezone.utc)`.
- `open_time` and `close_time` are **ISO 8601 strings** ending in `Z` (UTC). Parse via `datetime.fromisoformat(s.replace("Z", "+00:00"))`.
- The API always returns UTC. No timezone conversion is applied.
- `duration_h` is always `(close_ts − open_ts) / 3600`. If either timestamp is 0 (parse failure), `duration_h` is 0.

### Key Notes

- Markets where `result` is not `"yes"` or `"no"` are dropped at collection time (covers provisional/cancelled markets).
- The `market_id` field is Kalshi's `ticker` field in the API response (not `event_ticker`).
- Two contracts typically exist per event (home team and away team), so `event_ticker` groups them: e.g., `KXNBAGAME-26MAY05CLEDET` has both `KXNBAGAME-26MAY05CLEDET-DET` and `KXNBAGAME-26MAY05CLEDET-CLE`.

---

## 4. Raw Data: `data/raw_price_history.csv`

**Produced by:** `src/collect_candlesticks.py` → `collect_candlesticks()`  
**Row count:** 49,015 hourly candlestick records  
**Unique markets covered:** 996  
**File size:** ~3.3 MB

### Column Schema

| Column | dtype | Range / Values | Description |
|--------|-------|----------------|-------------|
| `market_id` | string | e.g. `KXNBATOTAL-26APR07SACGSW-234` | **Foreign key** to `raw_markets.csv.market_id`. Links each candlestick to its parent market. |
| `ts` | int64 | Unix seconds | **End** of the 1-hour candle window. The candle covers `[ts − 3600, ts]`. |
| `close` | float64 | `[0.0, 1.0]` | YES closing price at end of this 1-hour window. Primary price column used for all feature computations. |
| `high` | float64 | `[0.0, 1.0]` | YES high price within this 1-hour window. Falls back to `close` if missing from API. |
| `low` | float64 | `[0.0, 1.0]` | YES low price within this 1-hour window. Falls back to `close` if missing from API. |
| `open` | float64 | `[0.0, 1.0]` | YES opening price at start of this 1-hour window. Falls back to `close` if missing from API. |
| `volume` | float64 | `≥ 0` | Contracts traded within this 1-hour window. From `volume_fp` in API. |

### Candlestick Semantics

- **Window width:** exactly 60 minutes (`period_interval=60`).
- **Timestamp convention:** `ts` marks the **end** of the window. The candle at `ts=1775538000` covers the hour `1775534400` to `1775538000`.
- **Price space:** All values are in `[0.0, 1.0]`. A price of `0.59` means the market believed a 59% probability of YES at that moment.
- **Volume units:** Contract units as returned by `volume_fp`. Not normalized.
- **Missing candles:** If no trades occurred in an hour, the API may omit that candlestick entirely. Gaps in `ts` sequences are normal for illiquid markets.

### Coverage Policy

Only markets with `duration_h ≥ 24` in `raw_markets.csv` are eligible for candlestick collection. The `collect_candlesticks.py` script applies stratified sampling within that eligible pool:
- Up to 300 markets per `market_type`
- Shuffled randomly (`random_state=42`) across series tickers within each type
- Checkpointed every 500 markets to `data/raw_price_history.csv` to survive interruptions

### Price Fallback Logic

The API returns `price.close_dollars` as the primary close price. If that field is null or missing, the script falls back to `yes_bid.close_dollars`. If both are missing, the candlestick is silently dropped. This affects a small fraction of records in very early or very late market windows.

### Relationship to `raw_markets.csv`

```
raw_markets.csv              raw_price_history.csv
market_id (PK) ─────────────► market_id (FK)
                   1 : many
                 (one market has 10–720 hourly candles)
```

---

## 5. Intermediate: `data/features.csv`

**Produced by:** `src/features.py` → `compute_features(markets_df, candles_df)`  
**Row count:** 996 markets  
**File size:** ~235 KB  
**Dependencies:** `raw_markets.csv` + `raw_price_history.csv`

This file is the primary analytical unit. Each row represents one resolved NBA market, augmented with 8 behavioral features computed from its full price history.

### Column Schema

#### Computed Feature Columns

| Column | dtype | Formula | Range | Interpretation |
|--------|-------|---------|-------|----------------|
| `n_obs` | int64 | `len(group)` | `≥ 1` | Number of hourly candlesticks collected for this market. |
| `duration_hours` | float64 | `(ts[-1] − ts[0]) / 3600` | `≥ 0` | Elapsed hours between first and last observed candlestick. Note: this is the observed span of price data, not the full `close_ts − open_ts` market lifetime. |
| `final_prob` | float64 | `prices[-1]` | `[0.0, 1.0]` | The YES closing price in the final candlestick before resolution. This is the market's last stated probability. |
| `change_3h` | float64 | `prices[-1] − prices[-4]` if `n_obs ≥ 4`; else `prices[-1] − prices[0]` | `[−1.0, 1.0]` | Price move in the final 3 hours. Captures last-minute information. Positive = moved toward YES. |
| `change_24h` | float64 | `prices[-1] − prices[-25]` if `n_obs ≥ 25`; else `prices[-1] − prices[0]` | `[−1.0, 1.0]` | Price move in the final 24 hours. Captures day-of drift. |
| `total_range` | float64 | `max(prices) − min(prices)` | `[0.0, 1.0]` | Full price excursion over market lifetime. High values indicate genuine uncertainty; low values indicate consistently priced markets. |
| `volatility` | float64 | `std(np.diff(prices))` | `[0.0, ∞)` | Standard deviation of hourly price changes. Measures roughness of the price path. 0 = price never moved. |
| `drift_slope` | float64 | `np.polyfit(np.arange(n), tail[-24:].close, 1)[0]` | `(−∞, ∞)` | OLS slope (price change per hour) over the final 24 candlesticks, or the full history if fewer than 24 candles are available. Positive = trending toward YES; negative = trending toward NO. |
| `days_open` | float64 | `duration_hours / 24` | `[0.0, ∞)` | Market lifetime expressed in days. Long-duration markets (> 7 days) are typically series-level outcomes. |
| `time_near_50` | float64 | `sum(0.40 ≤ close ≤ 0.60) / n_obs` | `[0.0, 1.0]` | Fraction of the market's life spent in the "contested" probability zone (40%–60%). High values indicate genuine uncertainty that lasted through most of the market's life. |

#### Columns Inherited from `raw_markets.csv`

| Column | Source column | Notes |
|--------|--------------|-------|
| `market_id` | `market_id` | Joins features back to metadata |
| `series_ticker` | `series_ticker` | Used for structural analysis |
| `market_type` | `market_type` | Used for cluster cross-tabulation |
| `title` | `title` | Human-readable label |
| `result` | `result` | Ground truth outcome (`"yes"` / `"no"`) |
| `open_ts` | `open_ts` | Market open Unix timestamp |
| `close_ts` | `close_ts` | Market close Unix timestamp |

### Funnel: Why 996 Markets?

Starting from 19,406 raw markets:

1. `raw_price_history.csv` contains candlesticks for **996 unique `market_id` values** (the stratified sample with `duration_h ≥ 24`).
2. `compute_features()` requires `n_obs ≥ 2` to compute features. All 996 sampled markets met this criterion.
3. Therefore: **996 rows** in `features.csv`.

Markets in `raw_markets.csv` that are not in `raw_price_history.csv` (18,410 markets) received no candlestick data and are absent from all downstream files.

### Feature Computation Notes

- All features are computed from the **`close` column** of `raw_price_history.csv` only. `high`, `low`, `open` are stored but not used in features.
- Candlesticks are sorted ascending by `ts` before computation.
- `drift_slope` uses `np.polyfit(x, y, 1)` where `x = np.arange(len(tail))` (integer index, not actual timestamps). The slope unit is "probability change per candle" (one candle = 1 hour).
- `volatility` uses `np.std(np.diff(prices))` — population std dev (ddof=0 by default in NumPy).

---

## 6. Intermediate: `data/clustered_markets.csv`

**Produced by:** `src/cluster.py` → `run_clustering()` (called via `src/run_pipeline.py`)  
**Row count:** 946 markets  
**File size:** ~226 KB  
**Dependencies:** `data/features.csv`

This file is `features.csv` with three rows removed (preprocessing filters) and one column added (`cluster`).

### Preprocessing Filters Applied

Before clustering, `src/features.py → preprocess()` applies these sequential filters:

| Filter | Threshold | Rationale | Markets removed |
|--------|-----------|-----------|----------------|
| Duration | `duration_hours ≥ 24` | Short-lived markets lack enough price dynamics for behavioral features to be meaningful | 37 markets |
| Observations | `n_obs ≥ 10` | Fewer than 10 hourly candles is insufficient to estimate slope, volatility, or time_near_50 reliably | 13 markets |
| NaN guard | No NaN in any of 8 feature columns | Catches edge cases where formula inputs produced NaN | 0 markets |

**Net:** 996 → **946 markets**

### StandardScaler Normalization

Before k-means, all 8 feature columns are z-scored:

```
X_scaled[i, j] = (X[i, j] − mean(X[:, j])) / std(X[:, j])
```

Computed by `sklearn.preprocessing.StandardScaler`. The scaler is **not saved to disk** — it is recomputed on each run. The `clustered_markets.csv` file retains the **original (unscaled) feature values**.

### Column Schema

All columns from `features.csv` are preserved. One column is added:

| Column | dtype | Values | Description |
|--------|-------|--------|-------------|
| `cluster` | int64 | `0`, `1`, `2`, `3` | K-means cluster assignment. Integer label 0–(k−1). Labels have no inherent ordering; they are assigned arbitrarily by k-means. In the current run with k=4, labels 0–3 correspond to the four behavioral types documented in `outputs/summary_table.csv`. |

### Clustering Methodology

- **Algorithm:** `sklearn.cluster.KMeans`
- **k range searched:** 2 through 7
- **Selection criterion:** highest silhouette score among k ≥ 4. The minimum k=4 is enforced to ensure at least four meaningfully distinct behavioral types (k=2 optimizes silhouette globally but merely separates YES-side from NO-side markets).
- **Parameters:** `random_state=42`, `n_init=10` (ten independent random initializations; best result kept)
- **Input:** 946 × 8 z-scored feature matrix
- **Silhouette scores by k** (current run):

| k | Inertia | Silhouette |
|---|---------|-----------|
| 2 | 4,979.9 | 0.3669 |
| 3 | 4,031.6 | 0.3354 |
| **4** | **3,341.7** | **0.3589** ← selected |
| 5 | 2,935.8 | 0.3161 |
| 6 | 2,644.9 | 0.3353 |
| 7 | 2,385.3 | 0.3090 |

### Cluster Profiles (Current Run, k=4)

| Cluster | n | Behavior | `final_prob` | `change_24h` | `total_range` | `time_near_50` | `days_open` | `yes_rate` |
|---------|---|----------|------------|------------|------------|--------------|-----------|----------|
| 0 | 453 | Stable underdog | 0.163 | −0.017 | 0.226 | 0.016 | 2.15 | 14.1% |
| 1 | 286 | Late-converging favorite | 0.854 | +0.475 | 0.734 | 0.275 | 2.03 | 83.9% |
| 2 | 177 | Volatile near-miss underdog | 0.138 | −0.287 | 0.591 | 0.460 | 2.04 | 15.8% |
| 3 | 30 | Long-duration pre-series | 0.328 | −0.021 | 0.487 | 0.120 | 13.63 | 26.7% |

**Important:** Cluster labels (0–3) are deterministic given `random_state=42` and `n_init=10`, but they are not semantically ordered. Re-running with different seeds may reassign labels. The behavioral interpretations hold regardless of label assignment.

---

## 7. Output: `outputs/summary_table.csv`

**Produced by:** `src/run_pipeline.py` → `main()` (Step 9)  
**Row count:** 4 (one per cluster)  
**File size:** ~1.1 KB

### Column Schema

| Column | dtype | Description |
|--------|-------|-------------|
| `cluster` | int64 | Cluster label (0–3). Primary key. |
| `n_markets` | int64 | Number of markets assigned to this cluster. Sourced from `cluster_summaries["size"]` in `characterize_clusters()`. |
| `avg_final_prob` | float64 | Mean of `final_prob` across all markets in this cluster. Represents the cluster's typical closing probability. |
| `yes_rate` | float64 | `sum(result == "yes") / n_markets`. The actual fraction of markets in this cluster that resolved YES. Used to compute `mispricing`. |
| `mispricing` | float64 | `yes_rate − avg_final_prob`. Positive = market systematically underpriced YES (actual rate > predicted prob). Negative = market overpriced YES. |
| `dominant_sport` | string | The `market_type` value with the highest count in this cluster (plurality). |
| `behavioral_description` | string | Natural-language summary generated by rule-based logic in `characterize_clusters()`. Encodes the cluster's most distinguishing features. |

### Full Table

| cluster | n_markets | avg_final_prob | yes_rate | mispricing | dominant_sport | behavioral_description |
|---------|-----------|----------------|----------|-----------|----------------|------------------------|
| 0 | 453 | 0.1627 | 0.1413 | −0.0214 | half_spread | Stable underdog: consistently priced below 25%, resolved NO ~86% of the time. Low volatility (0.040), almost no time near 50% (2%). Outcome was largely anticipated from market open. |
| 1 | 286 | 0.8536 | 0.8392 | −0.0145 | team_total | Strong late-converging favorite: started near 50/50, resolved YES ~84% of the time with a large intraday range (0.73) and decisive final-day move (+0.48 in 24h). |
| 2 | 177 | 0.1377 | 0.1582 | +0.0205 | half_total | Volatile near-miss underdog: spent 46% of lifetime near 50% before resolving NO ~84% of the time. Large range (0.59), strong late-day drift (−0.29 in 24h). These markets appeared contested before the crowd shifted decisively toward NO. |
| 3 | 30 | 0.3277 | 0.2667 | −0.0610 | series_outcome | Long-duration markets (~14 days open). Dominant type: series_outcome. Low short-term volatility (0.046); final prob=0.33; YES rate=27%. |

### Derivation of `behavioral_description`

The description is programmatically generated in `cluster.py → characterize_clusters()` via this rule hierarchy (first matching rule wins):

```python
if days_open > 7:
    → "Long-duration markets (~{days:.0f} days open)..."
elif final_prob > 0.75 and total_range > 0.5:
    → "Strong late-converging favorite: started near 50/50..."
elif final_prob > 0.75:
    → "Stable favorite: consistently priced above 75%..."
elif final_prob < 0.25 and time_near_50 > 0.30:
    → "Volatile near-miss underdog: spent {t50:.0%} of lifetime near 50%..."
elif final_prob < 0.25:
    → "Stable underdog: consistently priced below 25%..."
elif time_near_50 > 0.4:
    → "Contested/uncertain: spent {t50:.0%} of lifetime near 50%..."
elif drift_slope > 0.01:
    → "Late-surging YES..."
elif drift_slope < -0.01:
    → "Late-fading NO..."
else:
    → "Stable/anchored ({dominant_type})..."
```

---

## 8. Output: `outputs/mispricing_table.csv`

**Produced by:** `src/cluster.py` → `mispricing_analysis()`  
**Row count:** 4 (one per cluster)  
**File size:** ~427 bytes

### Column Schema

| Column | dtype | Description |
|--------|-------|-------------|
| `cluster` | int64 | Cluster label (0–3). Primary key. |
| `n` | int64 | Number of markets in this cluster. |
| `avg_final_prob` | float64 | Mean final YES probability across cluster. |
| `yes_rate` | float64 | Actual YES resolution rate. |
| `mispricing` | float64 | `yes_rate − avg_final_prob`. See note above. |
| `p_value` | float64 | Two-sided binomial test p-value. See methodology below. |
| `significant` | bool | `True` if `p_value < 0.05`. |

### Statistical Methodology

For each cluster c with n markets, observed YES count k, and average final probability p:

**Null hypothesis H₀:** The market is calibrated — the final probability p is an unbiased estimate of the true resolution probability.

**Test:** Two-sided binomial test via `scipy.stats.binomtest(k, n, p, alternative="two-sided")`.

- Under H₀, the number of YES resolutions is `Binomial(n, p)`.
- The p-value is the probability of observing a YES count at least as extreme as k, given H₀.
- Significance threshold: α = 0.05 (two-sided).

### Full Table

| cluster | n | avg_final_prob | yes_rate | mispricing | p_value | significant |
|---------|---|----------------|----------|-----------|---------|-------------|
| 0 | 453 | 0.1627 | 0.1413 | −0.0214 | 0.2270 | False |
| 1 | 286 | 0.8536 | 0.8392 | −0.0145 | 0.5030 | False |
| 2 | 177 | 0.1377 | 0.1582 | +0.0205 | 0.4444 | False |
| 3 | 30 | 0.3277 | 0.2667 | −0.0610 | 0.5628 | False |

### Interpretation

No cluster shows statistically significant mispricing at the 5% level. The largest absolute mispricing is Cluster 3 (−6.1 pp), but with only 30 markets the test has low power. The overall finding is that **Kalshi NBA markets are well-calibrated** across all behavioral cluster types.

---

## 9. Output Figures

All figures are saved to `outputs/` at 150 DPI. All are generated by `src/visualize.py → generate_all()`.

### `outputs/elbow_curve.png`

**What it shows:** Two side-by-side panels.
- Left panel: Within-cluster sum of squared errors (inertia) vs. k for k ∈ {2,…,7}. The "elbow" — where the inertia curve bends — suggests a good k.
- Right panel: Silhouette score vs. k. The selected k (4) is the maximum among k ≥ 4.

**Inputs used:** Scaled 946 × 8 feature matrix. KMeans results for k = 2 through 7.

### `outputs/cluster_scatter.png`

**What it shows:** 2D PCA projection of all 946 markets, colored by cluster assignment (4 colors).

**Inputs used:** Scaled feature matrix → `PCA(n_components=2, random_state=42)`. X-axis = PC1, Y-axis = PC2. Axis labels include the percentage of total variance explained by each component.

**Design:** Alpha=0.55, point size=22. Cluster identity determined by k-means labels.

### `outputs/feature_heatmap.png`

**What it shows:** Heatmap of cluster mean feature values, z-scored across clusters for visual contrast.

**Rows:** One per cluster (0–3).  
**Columns:** 8 behavioral features.  
**Color scale:** RdYlGn diverging — red = above global cluster mean, green = below. Values in each cell are the z-score of that cluster's mean for that feature.

**Inputs used:** `labeled_df.groupby("cluster")[FEATURE_COLS].mean()`, then row-normalized: `(cluster_means − cluster_means.mean()) / (cluster_means.std() + 1e-9)`.

### `outputs/sport_breakdown.png`

**What it shows:** Stacked horizontal bar chart. Each bar = one cluster, normalized to 100%. Segments = `market_type` values. Shows whether cluster composition mirrors Kalshi's market categories or mixes them.

**Inputs used:** `pd.crosstab(labeled_df["cluster"], labeled_df["market_type"], normalize="index")`.

**Key finding illustrated:** All four clusters contain markets from multiple types → clusters cut across Kalshi's stated categories.

### `outputs/mispricing_chart.png`

**What it shows:** Vertical bar chart with one bar per cluster. Bar height = `mispricing` (actual YES rate − avg final prob). Error bars = 95% Wald confidence interval: `±1.96 × sqrt(yes_rate × (1−yes_rate) / n)`. Red asterisk (*) over bar if p_value < 0.05.

**Inputs used:** `outputs/mispricing_table.csv`.

### `outputs/calibration_curve.png`

**What it shows:** Calibration curves per cluster. X-axis = mean predicted probability (binned by quantile). Y-axis = actual fraction of YES resolutions in that bin. The dashed diagonal = perfect calibration.

**Inputs used:** `sklearn.calibration.calibration_curve(y_true, y_prob, n_bins, strategy="quantile")` per cluster. `y_true = (result == "yes")`, `y_prob = final_prob.clip(0.001, 0.999)`. Minimum 20 markets required per cluster to plot a curve.

**n_bins:** `min(8, max(3, n_cluster // 20))`.

---

## 10. Data Lineage Table

| File | Produced by | Consumed by | Row count | Key transformation |
|------|-------------|-------------|-----------|-------------------|
| `data/raw_markets.csv` | `src/collect_data.py` | `src/collect_candlesticks.py`, `src/features.py` | 19,406 | API pull → deduplicate on `market_id` → parse timestamps → compute `duration_h` |
| `data/raw_price_history.csv` | `src/collect_candlesticks.py` | `src/features.py` | 49,015 | Stratified sample (≤300/type, ≥24h duration) → candlestick API → OHLCV extraction |
| `data/features.csv` | `src/features.py :: compute_features()` | `src/features.py :: preprocess()`, `src/run_pipeline.py` | 996 | Candlestick time-series → 8 behavioral features per market |
| `data/clustered_markets.csv` | `src/cluster.py :: run_clustering()` | `src/cluster.py` (characterize, structural, mispricing), `src/visualize.py`, `src/run_pipeline.py` | 946 | Filter (24h, 10 obs) → StandardScale → KMeans(k=4) → add `cluster` column |
| `outputs/summary_table.csv` | `src/run_pipeline.py :: main()` | Human review, course submission | 4 | Merge `cluster_summaries` + `misprice_df` + `dominant_sport` per cluster |
| `outputs/mispricing_table.csv` | `src/cluster.py :: mispricing_analysis()` | `src/visualize.py`, `src/run_pipeline.py` | 4 | Per-cluster binomial test of calibration |
| `outputs/elbow_curve.png` | `src/cluster.py :: run_clustering()` | Human review | 1 figure | KMeans inertia + silhouette for k=2..7 |
| `outputs/cluster_scatter.png` | `src/visualize.py :: cluster_scatter()` | Human review | 1 figure | PCA(2) projection of scaled features, colored by cluster |
| `outputs/feature_heatmap.png` | `src/visualize.py :: feature_heatmap()` | Human review | 1 figure | Z-scored cluster mean features as heatmap |
| `outputs/sport_breakdown.png` | `src/visualize.py :: sport_breakdown()` | Human review | 1 figure | Row-normalized cross-tab: cluster × market_type |
| `outputs/mispricing_chart.png` | `src/visualize.py :: mispricing_chart()` | Human review | 1 figure | Bar chart of mispricing ± 95% CI per cluster |
| `outputs/calibration_curve.png` | `src/visualize.py :: calibration_curve_plot()` | Human review | 1 figure | Quantile-binned calibration curve per cluster |

---

## 11. Row Count Funnel

```
Kalshi API (24 NBA series, cursor-paginated)
         │
         │  src/collect_data.py
         │  • pull up to 2,000 markets per series
         │  • keep only result ∈ {"yes", "no"}
         │  • deduplicate on market_id
         ▼
  data/raw_markets.csv
  ┌──────────────────────┐
  │  19,406 unique       │
  │  resolved NBA markets│
  └──────────────────────┘
         │
         │  src/collect_candlesticks.py
         │  • filter: duration_h ≥ 24   → eligible pool
         │  • stratified sample: ≤ 300 per market_type
         │  • pull hourly candlesticks via API
         ▼
  data/raw_price_history.csv
  ┌──────────────────────┐
  │  49,015 candlestick  │
  │  records across      │
  │  996 markets         │
  └──────────────────────┘
         │
         │  src/features.py :: compute_features()
         │  • require n_obs ≥ 2 (all 996 pass)
         │  • compute 8 behavioral features
         ▼
  data/features.csv
  ┌──────────────────────┐
  │  996 markets         │
  │  × 18 columns        │
  └──────────────────────┘
         │
         │  src/features.py :: preprocess()
         │  • filter: duration_hours ≥ 24 → removes 37
         │  • filter: n_obs ≥ 10          → removes 13
         │  • filter: no NaN in features  → removes  0
         │  • StandardScaler (not saved)
         │
         │  src/cluster.py :: run_clustering()
         │  • KMeans(k=4, random_state=42, n_init=10)
         ▼
  data/clustered_markets.csv
  ┌──────────────────────┐
  │  946 markets         │
  │  × 19 columns        │
  │  (+ cluster label)   │
  └──────────────────────┘
         │
         │  src/cluster.py :: characterize_clusters()
         │                   mispricing_analysis()
         │  src/run_pipeline.py :: main()
         ▼
  outputs/summary_table.csv      outputs/mispricing_table.csv
  ┌──────────────────────┐       ┌──────────────────────┐
  │  4 rows (1/cluster)  │       │  4 rows (1/cluster)  │
  │  × 7 columns         │       │  × 7 columns         │
  └──────────────────────┘       └──────────────────────┘
```

### Why 18,410 Markets Have No Price History

- All 19,406 raw markets are in `raw_markets.csv`.
- Only 996 appear in `raw_price_history.csv`.
- Reason: `collect_candlesticks.py` applies stratified sampling (≤300 per market_type) before pulling candlesticks. This was necessary to avoid exhausting the Kalshi API rate limits during data collection.
- The 18,410 un-sampled markets are valid data and can be sampled in future runs by re-running `collect_candlesticks.py` (it is resume-friendly and will skip markets already in `raw_price_history.csv`).

---

## 12. Field Cross-Reference Index

This table tracks every column that appears in multiple files, showing its origin and propagation path.

| Field | Origin | Present in | Type in downstream files |
|-------|--------|-----------|--------------------------|
| `market_id` | `raw_markets.csv` | `raw_price_history.csv` (FK), `features.csv`, `clustered_markets.csv` | Same string throughout. Primary key in markets files; FK in price history. |
| `series_ticker` | `raw_markets.csv` | `features.csv`, `clustered_markets.csv` | Unchanged. Used for cross-tabulation in Step 6. |
| `market_type` | `raw_markets.csv` (pipeline-assigned) | `features.csv`, `clustered_markets.csv`, `summary_table.csv` (as `dominant_sport`) | Unchanged until `summary_table.csv` where it is aggregated to plurality. |
| `title` | `raw_markets.csv` | `features.csv`, `clustered_markets.csv` | Unchanged. |
| `result` | `raw_markets.csv` | `features.csv`, `clustered_markets.csv`, `summary_table.csv` (as `yes_rate`), `mispricing_table.csv` (as `yes_rate`) | Raw: string `"yes"`/`"no"`. In aggregates: converted to binary (0/1) for mean. |
| `open_ts` | `raw_markets.csv` | `features.csv`, `clustered_markets.csv` | Unchanged Unix seconds. |
| `close_ts` | `raw_markets.csv` | `features.csv`, `clustered_markets.csv` | Unchanged Unix seconds. |
| `duration_h` | `raw_markets.csv` | Used by `collect_candlesticks.py` for filtering; not propagated further as a column | Float (hours). |
| `duration_hours` | Computed in `features.py` | `features.csv`, `clustered_markets.csv` | Float (hours from price data, not market metadata). Distinct from `duration_h`. |
| `final_prob` | Computed in `features.py` | `features.csv`, `clustered_markets.csv`, `summary_table.csv` (as `avg_final_prob`), `mispricing_table.csv` (as `avg_final_prob`) | Raw: float per market. Aggregated: cluster mean. |
| `cluster` | Computed in `cluster.py` | `clustered_markets.csv`, `summary_table.csv`, `mispricing_table.csv` | Integer 0–3. |
| `yes_rate` | Computed in `mispricing_analysis()` | `summary_table.csv`, `mispricing_table.csv` | Float [0,1]. Same value in both files. |
| `mispricing` | Computed in `mispricing_analysis()` | `summary_table.csv`, `mispricing_table.csv` | Same float value in both files. |
| `avg_final_prob` | Computed in `mispricing_analysis()` | `summary_table.csv`, `mispricing_table.csv` | Same float value in both files. |

### Note: `duration_h` vs `duration_hours`

These are subtly different:

| Field | Source | Definition | Used for |
|-------|--------|-----------|----------|
| `duration_h` | `raw_markets.csv` | `(close_ts − open_ts) / 3600` — full market lifetime from metadata | Pre-filtering in `collect_candlesticks.py` |
| `duration_hours` | `features.csv` | `(ts[-1] − ts[0]) / 3600` — observed span of collected candlesticks | Post-filtering in `preprocess()` |

The two values can differ because: (a) candlestick collection may start slightly after market open (API warmup), and (b) the API returns candles only when trades occur, so sparse markets may have gaps that shrink the observed span.

---

## 13. Reproducing the Data from Scratch

### Prerequisites

```bash
cd kalshi-sports-clusters
pip install -r requirements.txt
```

Python 3.12+ required. No API credentials needed.

### Step 1a: Pull Market Metadata

```bash
python3 src/collect_data.py
```

**Produces:** `data/raw_markets.csv`  
**Runtime:** ~8–15 minutes (24 series × up to 10 pages × pagination + rate limit sleeps)  
**Rate limit behavior:** Auto-sleeps 10s on HTTP 429, retries up to 1 time per page. If a page fails twice, collection for that series stops at the last successful page.  
**Determinism:** Output depends on which markets have settled at time of collection. Running again will add new markets from recently completed games.

### Step 1b: Pull Price Histories

```bash
python3 src/collect_candlesticks.py
```

**Requires:** `data/raw_markets.csv` (Step 1a)  
**Produces:** `data/raw_price_history.csv`  
**Runtime:** ~20–40 minutes (up to 2,100 markets × 0.6s/market + rate limit overhead)  
**Resume-friendly:** Re-running will skip markets already in `raw_price_history.csv`.  
**Rate limit behavior:** Exponential backoff on 429 (15s × attempt number), max 3 retries.  
**Checkpointing:** Saves partial results every 500 markets. Data is not lost if the script is interrupted.

### Steps 2–9: Run Full Analysis Pipeline

```bash
python3 src/run_pipeline.py
```

**Requires:** `data/raw_markets.csv`, `data/raw_price_history.csv`  
**Produces:** `data/features.csv`, `data/clustered_markets.csv`, `outputs/summary_table.csv`, `outputs/mispricing_table.csv`, and all 6 PNG figures  
**Runtime:** ~30–60 seconds  
**Caching:** If `data/features.csv` already exists, feature computation is skipped. Delete it to force recomputation.  
**Determinism:** Fully deterministic given the same input CSVs (`random_state=42` throughout).

### Alternative: Interactive Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Runs all steps interactively with inline visualizations. Equivalent to `run_pipeline.py` but with markdown commentary between each step.

### Expected Output Files After Full Reproduction

```
data/
  raw_markets.csv              ~3.9 MB   (19,406 rows)
  raw_price_history.csv        variable  (depends on sample)
  features.csv                 ~235 KB   (996 rows)
  clustered_markets.csv        ~226 KB   (946 rows)

outputs/
  summary_table.csv            ~1.1 KB   (4 rows)
  mispricing_table.csv         ~427 B    (4 rows)
  elbow_curve.png              ~80 KB
  cluster_scatter.png          ~179 KB
  feature_heatmap.png          ~69 KB
  sport_breakdown.png          ~51 KB
  mispricing_chart.png         ~37 KB
  calibration_curve.png        ~105 KB
```

### Re-running With More Data

To expand the sample of markets with price histories:

1. Re-run `collect_candlesticks.py`. It will skip already-collected markets and add new ones.
2. Delete `data/features.csv` to force recomputation with the expanded dataset.
3. Run `python3 src/run_pipeline.py`.

### Changing the Clustering Parameters

To try a different k:

- Edit `src/cluster.py`, line ~68: change the `eligible_k` filter from `k >= 4` to your desired minimum.
- Or change `range(2, 8)` to `range(2, 10)` to evaluate k up to 9.
- Re-run `python3 src/run_pipeline.py`.

---

*End of FINAL_DATA_STORAGE.md*
