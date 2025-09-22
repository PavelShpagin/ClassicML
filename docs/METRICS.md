# Competition Metric Deep Dive

## Understanding the Two-Stage MAPE Metric

The competition uses a custom metric that severely penalizes poor predictions. This document explains how it works and why it's so challenging.

## Metric Formula

### Stage 1: Calculate APE for each prediction

```python
APE = |actual - predicted| / max(actual, 1)
```

Note: Uses `max(actual, 1)` to avoid division by zero

### Stage 2: Two-stage scoring

```python
def competition_score(y_true, y_pred):
    # Calculate APE for each prediction
    ape = np.abs(y_true - y_pred) / np.maximum(y_true, 1)
    
    # Identify "good" predictions (APE ≤ 100%)
    good_mask = ape <= 1.0
    good_rate = good_mask.mean()
    
    # Stage 1: Check if enough predictions are good
    if good_rate < 0.3:
        return 0.0  # Catastrophic failure!
    
    # Stage 2: Score based on good predictions only
    good_ape = ape[good_mask]
    score = good_rate * np.mean(1 / (1 + good_ape))
    
    return score
```

## Why This Metric is Challenging

### 1. The 30% Threshold Cliff

If less than 30% of predictions have APE ≤ 100%, the score immediately becomes 0.

**Example**:
- 100 predictions total
- 29 have APE ≤ 100% (excellent predictions)
- 71 have APE > 100% (poor predictions)
- **Score = 0** (below 30% threshold)

### 2. Zero Value Sensitivity

When actual = 0:
- Any non-zero prediction gives APE = |predicted| / 1 = predicted
- Predicting 2 when actual is 0 gives APE = 200%
- This single prediction can push you below the 30% threshold

**Example Impact**:
```python
# Scenario: 96 sectors × 12 months = 1152 predictions
# If 21% are zeros (242 predictions)
# And we predict avg value (25000) for these:
# APE = 25000 / 1 = 2500000% for each!
# Result: Score = 0
```

### 3. Asymmetric Penalty

The metric penalizes overestimation on zeros much more than underestimation on large values:

| Actual | Predicted | APE | Impact |
|--------|-----------|-----|--------|
| 0 | 100 | 10000% | Catastrophic |
| 100 | 0 | 100% | Acceptable |
| 10000 | 5000 | 50% | Good |
| 10000 | 20000 | 100% | Borderline |

## Strategic Implications

### 1. Conservative Predictions Win

Better to predict 0 when unsure:
- If actual = 0: Perfect (APE = 0%)
- If actual > 0: At worst APE = 100%

### 2. Identify Zero Patterns

Critical to identify which sectors will be zero:
- Historical zeros often repeat
- Sectors with declining trends likely to hit zero
- New sectors (like 95) safer to predict as zero

### 3. Avoid Middle Ground

The metric pushes toward extreme strategies:
- Very conservative (many zeros)
- Very accurate (high confidence non-zeros)
- Middle ground (small non-zero values) is dangerous

## Practical Examples

### Example 1: Why XGBoost Failed

```python
# XGBoost prediction for a zero-sector
actual = 0
predicted = 15.7  # Small "safe" prediction
APE = 15.7 / 1 = 1570%

# Just 30 such predictions out of 1152 total:
# good_rate = (1152 - 30) / 1152 = 0.974
# Still okay!

# But if 350 predictions have APE > 100%:
# good_rate = (1152 - 350) / 1152 = 0.696 < 0.7
# Score drops significantly!

# If 400 predictions have APE > 100%:
# good_rate = (1152 - 400) / 1152 = 0.653 < 0.7
# Score = 0!
```

### Example 2: Why Geometric Mean Worked

```python
# Geometric mean with zero guard
historical = [0, 100, 200, 0, 150, 0]  # Last 6 months

# Step 1: Check for zeros
has_zero = min(historical) == 0  # True

# Step 2: Apply zero guard
prediction = 0  # Safe choice

# Result:
# If actual = 0: APE = 0% ✓
# If actual = 500: APE = 100% (still acceptable)
```

### Example 3: Seasonality Boost Calculation

```python
# Historical December vs non-December
december_values = [45000, 52000, 48000, 51000]
other_months = [35000, 33000, 37000, 34000, ...]

december_mean = 49000
other_mean = 35000

boost_factor = december_mean / other_mean = 1.4

# Applied conservatively
final_boost = min(boost_factor, 1.3) = 1.3  # Cap at 30%
```

## Optimal Strategy Summary

1. **Zero Guard First**: If any recent month is zero, predict zero
2. **Geometric Mean**: More robust than arithmetic mean for skewed data
3. **Conservative Seasonality**: Apply boosts carefully, cap at reasonable levels
4. **No Small Values**: Either predict 0 or a confident non-zero
5. **Simple Models**: Complex models can't handle the metric's non-linearity

## Metric Visualization

```
Score vs Good Rate:
1.0 |     _______________
    |    /
    |   /
0.5 |  /
    | /
0.0 |__|________________
    0  0.3            1.0
       Good Rate

Key: Cliff at 0.3 - below this, score = 0
```

## Conclusion

This metric design fundamentally changes the modeling approach:
- Traditional optimization (minimize RMSE/MAE) doesn't work
- Need to optimize for "good enough" predictions, not perfect ones
- Conservative strategies outperform aggressive optimization
- Understanding the metric is more important than model sophistication
