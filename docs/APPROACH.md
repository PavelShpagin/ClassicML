# Technical Approach Documentation

## Overview

This document details the technical approach used to achieve a 0.56248 score on the China Real Estate Demand Prediction competition.

## Problem Analysis

### Key Challenges

1. **Extreme Metric Sensitivity**: The two-stage MAPE metric assigns score = 0 if less than 30% of predictions have APE ≤ 100%
2. **Zero Value Handling**: 21% of training samples are zeros; predicting non-zero for these causes catastrophic penalty
3. **Temporal Dependencies**: Real estate transactions show strong seasonal patterns
4. **Missing Sectors**: Sector 95 appears in test but not training data

## Solution Evolution

### Phase 1: Traditional ML Approach (Failed)

**Approach**: Ridge Regression, XGBoost with extensive feature engineering

**Features**:

- 12 lagged values for each sector
- Rolling means (3, 6, 12 months)
- Rolling standard deviations
- Sector-specific indicators

**Result**: Local CV ~0.35-0.50, Public LB: 0.00000

**Why it Failed**:

- Models occasionally predicted small non-zero values for zero sectors
- Even one bad prediction (APE > 100%) could cascade to zero score
- Complex models couldn't learn the conservative behavior required

### Phase 2: Conservative Baseline (Partial Success)

**Approach**: Simple median-based predictions

**Method**:

```python
# For each sector, use historical median
# Apply zero guard for sectors with recent zeros
```

**Result**: Public LB: 0.21591

**Learning**: Conservative approaches work better with this metric

### Phase 3: Geometric Mean Baseline (Success)

**Approach**: Geometric mean of recent months

**Method**:

```python
# 1. Calculate geometric mean of last 6 months
geo_mean = exp(mean(log(last_6_months)))

# 2. Zero guard: if any of last 6 months = 0, predict 0
if min(last_6_months) == 0:
    prediction = 0
```

**Result**: Public LB: 0.55528

**Why it Worked**:

- Geometric mean naturally handles skewed distributions
- Zero guard prevents catastrophic errors
- Simple enough to be robust

### Phase 4: Seasonality Enhancement (Best)

**Approach**: Geometric mean + December boost

**Method**:

```python
# Base prediction from geometric mean
base_pred = geometric_mean_with_zero_guard()

# Apply seasonality
if month == 'December':
    prediction = base_pred * 1.3
else:
    prediction = base_pred
```

**Result**: Public LB: 0.56248

**Seasonality Analysis**:

- December shows 30% higher transactions on average
- Boost factor calculated from historical December/non-December ratios
- Capped at 2x to avoid extreme predictions

## Implementation Details

### Data Processing

```python
# Time encoding
train['time'] = (year - 2019) * 12 + month - 1

# Wide format for easier calculation
amount_matrix = train.pivot(index='time', columns='sector_id', values='amount')

# Handle missing sector 95
if 95 not in amount_matrix.columns:
    amount_matrix[95] = 0
```

### Zero Guard Logic

```python
def apply_zero_guard(predictions, historical_data, lookback=6):
    """
    Set prediction to 0 if any recent month was 0
    """
    for sector in sectors:
        recent_min = historical_data[sector].tail(lookback).min()
        if recent_min == 0:
            predictions[sector] = 0
    return predictions
```

### Geometric Mean Calculation

```python
def geometric_mean(values):
    """
    Calculate geometric mean, handling zeros
    """
    # Replace zeros with NaN for calculation
    non_zero = values.replace(0, np.nan)

    # Geometric mean
    log_mean = np.log(non_zero).mean(skipna=True)
    geo_mean = np.exp(log_mean)

    # Fill NaN with 0
    return geo_mean if not np.isnan(geo_mean) else 0
```

## Hyperparameter Selection

### Geometric Mean Parameters

| Parameter         | Value | Rationale                             |
| ----------------- | ----- | ------------------------------------- |
| Lookback months   | 6     | Balance between recency and stability |
| Zero guard window | 6     | Same as lookback for consistency      |
| December boost    | 1.3   | Historical average boost factor       |

### Why These Values?

- **6 months**: Captures half-year trends without being too sensitive to outliers
- **1.3x boost**: Conservative multiplier that improves December without risking extreme predictions

## Validation Strategy

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=12)
for train_idx, val_idx in tscv.split(X):
    # Train on train_idx
    # Validate on val_idx (12 months forward)
```

### Local vs Public Score Correlation

| Approach   | Local CV | Public LB | Gap    |
| ---------- | -------- | --------- | ------ |
| XGBoost    | 0.45     | 0.00      | -0.45  |
| Ridge      | 0.38     | 0.00      | -0.38  |
| Geometric  | 0.52     | 0.555     | +0.035 |
| Geo+Season | 0.53     | 0.562     | +0.032 |

**Key Insight**: Simple methods showed better local-public correlation

## Code Quality & Optimization

### Modular Design

```
src/
├── data.py       # Data loading and parsing
├── features.py   # Feature engineering
└── models.py     # Model definitions and metrics
```

### Performance Optimizations

1. **Vectorized Operations**: Used pandas/numpy operations instead of loops
2. **Caching**: Stored computed features to avoid recalculation
3. **Memory Management**: Used appropriate data types (int16 for sectors)

## Lessons for Future Competitions

1. **Understand the Metric First**: Spend time analyzing metric behavior before modeling
2. **Start Simple**: Establish baselines before complex models
3. **Domain Knowledge Matters**: Seasonality insight was crucial
4. **Validate Carefully**: Ensure local validation mimics competition evaluation
5. **Conservative Wins**: With sensitive metrics, robust predictions beat optimized ones
