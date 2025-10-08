# Technical Concepts Guide

## Overview

This document explains key machine learning and time series concepts used in this project. It's designed for quick reference and understanding.

---

## Feature Engineering Concepts

### 1. Lags (Time-Shifted Features)

**Definition:** Previous values shifted back in time

**Example:**

```
Predicting August 2024 for Sector 1:

Month      | Actual  | lag_1  | lag_2  | lag_3
-----------|---------|--------|--------|--------
May 2024   | 32,000  | 35,000 | 28,000 | 30,000
Jun 2024   | 31,000  | 32,000 | 35,000 | 28,000
Jul 2024   | 29,000  | 31,000 | 32,000 | 35,000
Aug 2024   | ???     | 29,000 | 31,000 | 32,000  ← Features

lag_1 = value 1 month ago (July: 29,000)
lag_2 = value 2 months ago (June: 31,000)
lag_3 = value 3 months ago (May: 32,000)
```

**Why useful:** Recent history strongly predicts future values in time series

**Implementation:**

```python
df['lag_1'] = df.groupby('sector_id')['amount'].shift(1)
df['lag_2'] = df.groupby('sector_id')['amount'].shift(2)
```

---

### 2. Rolling Mean (Moving Average)

**Definition:** Average of last N values in a sliding window

**Example:**

```
roll_mean_3 (3-month window):

Month      | Amount  | roll_mean_3
-----------|---------|------------------
Mar 2024   | 28,000  | NaN (not enough data)
Apr 2024   | 35,000  | NaN
May 2024   | 32,000  | (28k+35k+32k)/3 = 31,667
Jun 2024   | 31,000  | (35k+32k+31k)/3 = 32,667
Jul 2024   | 29,000  | (32k+31k+29k)/3 = 30,667
           |         |    ↑    ↑    ↑
           |         |  Sliding window moves each month
```

**Why useful:** Smooths noise, reveals underlying trends

**Implementation:**

```python
df['roll_mean_3'] = df.groupby('sector_id')['amount'].rolling(3).mean()
df['roll_mean_6'] = df.groupby('sector_id')['amount'].rolling(6).mean()
```

---

### 3. Geometric Mean

**Definition:** The nth root of the product of n numbers

**Formula:**

```
Geometric Mean = (x₁ × x₂ × ... × xₙ)^(1/n)
              = exp(mean(log(x₁), log(x₂), ..., log(xₙ)))
```

**Example:**

```python
values = [10000, 20000, 15000]

# Arithmetic mean
arithmetic = (10000 + 20000 + 15000) / 3 = 15,000

# Geometric mean
geometric = (10000 × 20000 × 15000)^(1/3) = 14,422

# Why different?
# Geometric mean is less sensitive to outliers
```

**Why useful for this project:**

- Real estate amounts have skewed distributions
- Geometric mean handles outliers better
- More robust than arithmetic mean for multiplicative processes

---

### 4. POI Features (Points of Interest)

**Definition:** Static geographic features about each sector

**Examples:**

```python
{
    'poi_school_count': 15,      # Number of schools
    'poi_hospital_count': 3,     # Number of hospitals
    'poi_metro_count': 2,        # Number of metro stations
    'poi_mall_count': 5,         # Shopping malls
    'poi_park_count': 8          # Parks
}
```

**Theory:** More amenities → Higher quality of life → More demand

**Reality in this project:** **Did NOT help!**

- POI features are static (don't change over time)
- Time trends matter more than location
- Removed from final models

---

## Cross-Validation Concepts

### Time Series Cross-Validation (5 Folds)

**Definition:** Validation that respects temporal order

**WRONG (Traditional CV - data leakage!):**

```
Fold 1: Train [1,2,4,5] Test [3]  ❌ Uses future to predict past!
Fold 2: Train [1,3,4,5] Test [2]  ❌ Data leakage!
```

**CORRECT (Time Series CV):**

```
Data: [Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec]

Fold 1: Train [Jan Feb Mar]          Test [Apr]
Fold 2: Train [Jan...Apr]            Test [May Jun]
Fold 3: Train [Jan...Jun]            Test [Jul Aug]
Fold 4: Train [Jan...Aug]            Test [Sep Oct]
Fold 5: Train [Jan...Oct]            Test [Nov Dec]

✓ Training always BEFORE testing
✓ Each fold tests on progressively newer data
✓ Mimics real-world prediction scenario
```

**Why 5 folds?**

- Balance between training data size and validation quality
- Industry standard for time series
- Provides robust performance estimates

**Implementation:**

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # Train and validate
```

---

## XGBoost Concepts

### 1. n_estimators (Number of Trees)

**Definition:** How many decision trees to build in the ensemble

**Example:**

- `n_estimators = 50` → 50 trees
- `n_estimators = 100` → 100 trees
- `n_estimators = 500` → 500 trees

**Trade-offs:**

| Trees   | Training Time | Accuracy     | Risk      |
| ------- | ------------- | ------------ | --------- |
| 50      | Fast          | Low          | Underfits |
| 100-200 | Medium        | High         | ✓ Optimal |
| 500+    | Slow          | High (train) | Overfits  |

**Process:**

```
Tree 1: Learns basic pattern       → Prediction₁
Tree 2: Refines Tree 1             → Prediction₂
Tree 3: Refines Tree 1+2           → Prediction₃
...
Tree N: Final refinement           → PredictionN

Final = Prediction₁ + Prediction₂ + ... + PredictionN
```

---

### 2. subsample (Row Sampling Ratio)

**Definition:** Randomly use only X% of training data per tree

**Example with 1000 training rows:**

```
subsample = 1.0  → Use all 1000 rows per tree
subsample = 0.8  → Use 800 random rows per tree
subsample = 0.6  → Use 600 random rows per tree

Full data: [Row1, Row2, Row3, ..., Row1000]

Tree 1 (subsample=0.8): [Row3, Row7, Row15, ...] (800 random)
Tree 2 (subsample=0.8): [Row2, Row5, Row18, ...] (different 800)
Tree 3 (subsample=0.8): [Row1, Row9, Row22, ...] (different 800)
```

**Why subsample?**

1. **Prevents overfitting** - Randomness helps generalization
2. **Faster training** - Less data per tree
3. **Diverse trees** - Each tree sees different patterns

**Best practice:** `subsample = 0.6 to 0.9`

---

### 3. Gradient Boosting: Why Predict Residuals?

**Key Difference:**

**Traditional ML (Ridge, RandomForest):**

```
Model: features → target directly
Predict 30,000 yuan in one shot
```

**Gradient Boosting (XGBoost):**

```
Model: features → error iteratively
Predict base + correction₁ + correction₂ + ... = 30,000
```

---

**Step-by-Step Example:**

```
ACTUAL TARGET: 30,000 yuan

─────────────────────────────────────────────────────────
STEP 0: Initial Guess
─────────────────────────────────────────────────────────
Prediction: 20,000 (mean of all training data)
Error (residual): 30,000 - 20,000 = +10,000

─────────────────────────────────────────────────────────
STEP 1: Train Tree 1 to predict the +10,000 error
─────────────────────────────────────────────────────────
Input: [lag_1=29k, lag_2=31k, sector=1, time=67]
Tree1 learns: "This pattern needs +8,000"
Tree1 output: +8,000

New prediction: 20,000 + 8,000 = 28,000
New error: 30,000 - 28,000 = +2,000

─────────────────────────────────────────────────────────
STEP 2: Train Tree 2 to predict the +2,000 error
─────────────────────────────────────────────────────────
Tree2 learns: "Still need +1,500"
Tree2 output: +1,500

New prediction: 28,000 + 1,500 = 29,500
New error: 30,000 - 29,500 = +500

─────────────────────────────────────────────────────────
STEP 3: Train Tree 3 to predict the +500 error
─────────────────────────────────────────────────────────
Tree3 learns: "Final adjustment +400"
Tree3 output: +400

Final prediction: 29,500 + 400 = 29,900 yuan
Final error: 30,000 - 29,900 = 100 ✓ (close enough!)
```

---

**Visual Analogy:**

**Traditional ML (Ridge):**

```
Start (0) --------[ONE BIG JUMP]--------> Target (30,000)
                   Might overshoot/undershoot!
```

**Gradient Boosting (XGBoost):**

```
Start (0) --[+20k]--> 20k --[+8k]--> 28k
          --[+1.5k]--> 29.5k --[+0.4k]--> 29.9k ✓
          Many small corrections = more precise!
```

---

**Why This Works Better:**

1. **Focus on Mistakes**

   - Each tree fixes previous errors
   - Gradually improves accuracy

2. **Specialization**

   - Early trees: learn big patterns
   - Later trees: learn subtle details

3. **Gradual Improvement**
   - Small steps prevent overfitting
   - More stable than one big prediction

**Formula:**

```
Final Prediction = base + Σ(tree_i)
                 = 20,000 + 8,000 + 1,500 + 400 + ...
                 = 29,900

Each tree: residual_i = actual - prediction_(i-1)
```

---

**IMPORTANT: Training vs Prediction**

**Training Phase (Iterative - builds model):**
- Input: Training data (features + labels)
- Process: Iteratively train trees on residuals
  - Iteration 1: Train Tree1 on `residual₀ = actual - base`
  - Iteration 2: Calculate `residual₁ = actual - (base + Tree1)`, train Tree2
  - Iteration 3: Calculate `residual₂ = actual - (base + Tree1 + Tree2)`, train Tree3
  - Continue for N iterations
- Output: Saved model `[base, Tree1, Tree2, ..., TreeN]`
- Time: Seconds to hours (training is slow)

**Prediction Phase (Single pass - uses model):**
- Input: Test data (features only, NO labels)
- Process: Apply all pre-trained trees in one pass
  - `prediction = base + Tree1(features) + Tree2(features) + ... + TreeN(features)`
  - Each tree independently evaluates features (no iteration!)
  - Just tree traversal: `if lag_1 > 30k, go left, else go right...`
- Output: Final predictions
- Time: Milliseconds (inference is fast)

**Key Point:** Iterative error correction happens ONLY during training to BUILD the trees. During prediction, all trees just run in parallel and sum their outputs. No actual labels needed for prediction, so no error correction possible!

```python
# Training (iterative)
model = XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)  # Builds 100 trees iteratively

# Prediction (single pass)
predictions = model.predict(X_test)  # Sums 100 tree outputs
```

---

**Why XGBoost Failed in This Project:**

1. **Not enough data** - Only 5,472 rows (57 months × 96 sectors)
2. **Harsh metric** - Two-stage MAPE punishes mistakes severely
3. **Overfitting** - Learned training noise, not real patterns
4. **Too complex** - Seasonality is simpler than 100 trees can capture

**Result:** Geometric mean (0 trees, 0 features) beats XGBoost (100 trees, 20+ features)!

---

## Summary Table

| Concept                 | Category            | Used In          | Impact       |
| ----------------------- | ------------------- | ---------------- | ------------ |
| **Lags**                | Feature Engineering | Ridge, XGBoost   | Medium       |
| **Rolling Mean**        | Feature Engineering | Ridge, XGBoost   | Medium       |
| **Geometric Mean**      | Statistics          | Best baselines   | ✅ High      |
| **POI Features**        | Feature Engineering | XGBoost (failed) | ❌ None      |
| **Time Series CV**      | Validation          | All models       | ✅ Critical  |
| **n_estimators**        | Hyperparameter      | XGBoost          | Low (failed) |
| **subsample**           | Hyperparameter      | XGBoost          | Low (failed) |
| **Residual Prediction** | Algorithm           | XGBoost          | Low (failed) |

---

## Key Takeaway

Understanding these concepts helps explain why **simple heuristics beat complex ML** in this project:

- **Limited data** (57 months) → Simple patterns only
- **Harsh metric** (two-stage MAPE) → Conservative wins
- **Domain knowledge** (December boost) → Beats learned patterns
- **Zero guard** (metric-aware rule) → Beats ML predictions

**Lesson:** Sometimes the best feature engineering is _no_ feature engineering! 🎯
