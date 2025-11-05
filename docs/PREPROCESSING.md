# Data Preprocessing Explained with Examples

This document explains all preprocessing steps with concrete examples from the codebase.

## Table of Contents

1. [Time Encoding](#time-encoding)
2. [Format Conversion (Long to Wide)](#format-conversion-long-to-wide)
3. [Zero Handling](#zero-handling)
4. [Missing Sector Handling](#missing-sector-handling)
5. [Lags Explained](#lags-explained)
6. [Where Preprocessing Happens](#where-preprocessing-happens)

---

## Time Encoding

### What It Is

Converting date strings like `"2019-Jan"` into a numeric time index starting from 0.

### Formula

```python
time = (year - 2019) * 12 + month_num - 1
```

### Examples

| Original Month String | Year | Month Num | Time Index | Explanation                   |
| --------------------- | ---- | --------- | ---------- | ----------------------------- |
| `"2019-Jan"`          | 2019 | 1         | **0**      | (2019-2019)\*12 + 1 - 1 = 0   |
| `"2019-Feb"`          | 2019 | 2         | **1**      | (2019-2019)\*12 + 2 - 1 = 1   |
| `"2019-Dec"`          | 2019 | 12        | **11**     | (2019-2019)\*12 + 12 - 1 = 11 |
| `"2020-Jan"`          | 2020 | 1         | **12**     | (2020-2019)\*12 + 1 - 1 = 12  |
| `"2024-Jul"`          | 2024 | 7         | **66**     | (2024-2019)\*12 + 7 - 1 = 66  |

### Code Location

**File**: `src/utils.py` (lines 36-40)

```python
# Parse month into year, month_num, and time index from 2019 Jan = 0
df["year"] = df["month"].astype(str).str.split("-").str[0].astype(int)
df["month_name"] = df["month"].astype(str).str.split("-").str[1]
df["month_num"] = df["month_name"].map(MONTH_CODES)
df["time"] = (df["year"] - 2019) * 12 + df["month_num"] - 1
```

**Also in**: `src/data.py` (lines 81-86) - `split_month_sector()` function

### Why?

- Makes time series operations easier (arithmetic on time)
- Consistent indexing starting from 0
- Easy to identify December: `time % 12 == 11`

---

## Format Conversion (Long to Wide)

### What It Is

Converting from **long format** (one row per sector-month) to **wide format** (one row per time, columns are sectors).

### Before (Long Format)

```
month        | sector      | amount_new_house_transactions
-------------|-------------|------------------------------
2019-Jan     | sector 1    | 25000
2019-Jan     | sector 2    | 18000
2019-Jan     | sector 3    | 32000
2019-Feb     | sector 1    | 26000
2019-Feb     | sector 2    | 19000
2019-Feb     | sector 3    | 31000
```

### After (Wide Format)

```
time | sector_1 | sector_2 | sector_3 | ... | sector_96
-----|----------|----------|----------|-----|----------
0    | 25000    | 18000    | 32000    | ... | 0
1    | 26000    | 19000    | 31000    | ... | 0
```

### Code Location

**File**: `src/utils.py` (lines 42-44)

```python
# Pivot to wide format
amount = df.set_index(["time", "sector_id"])
amount = amount["amount_new_house_transactions"].unstack()
amount = amount.fillna(0)
```

**Function**: `build_amount_wide()` - converts long format DataFrame to wide format matrix

### Why?

- **Efficient operations**: Can easily access `amount[time, sector]` or `amount.loc[time, sector]`
- **Vectorized calculations**: Compute statistics across sectors or times easily
- **Geometric mean**: Can compute `amount.tail(6).mean(axis=0)` to get per-sector means

### Example Usage

```python
# Load data (long format)
nht = load_all_training_tables(paths)['new_house_transactions']
# Convert to wide format
amount = build_amount_wide(nht)
# Now: amount.shape = (67, 96)  # 67 months × 96 sectors
# Access: amount.loc[0, 1] = transaction amount for time 0, sector 1
```

---

## Zero Handling

### What It Is

Handling sectors that have zero transactions in some months (21% of training data are zeros).

### Why Important?

The competition metric severely penalizes predicting non-zero when actual is zero:

```
Actual = 0, Predicted = 25,000
APE = |0 - 25000| / max(0, ε) = 25000 / 1 = 25,000% (catastrophic!)
```

### Zero Guard Mechanism

**Rule**: If **any** of the last 6 months had zero transactions, predict zero.

### Example

**Sector 42 - Last 6 months:**

```
Time | Amount
-----|-------
61   | 15,000
62   | 18,000
63   | 0        ← Zero detected!
64   | 12,000
65   | 14,000
66   | 16,000
```

**Result**: Prediction = 0 (because time 63 = 0)

**If no zeros:**

```
Time | Amount
-----|-------
61   | 15,000
62   | 18,000
63   | 12,000
64   | 12,000
65   | 14,000
66   | 16,000
```

**Result**: Prediction = geometric_mean([15k, 18k, 12k, 12k, 14k, 16k]) = 14,422

### Code Location

**File**: `src/utils.py` (lines 53-67)

```python
def geometric_mean_with_zero_guard(amount_wide, lookback_months=6, zero_guard_window=6):
    last_months = amount_wide.tail(lookback_months)
    geo = np.exp(np.log(last_months.replace(0, np.nan)).mean(axis=0, skipna=True))
    geo = geo.fillna(0)
    zero_mask = amount_wide.tail(zero_guard_window).min(axis=0) == 0
    geo[zero_mask] = 0  # ← Zero guard applied here
    return geo
```

**Key line**: `zero_mask = amount_wide.tail(zero_guard_window).min(axis=0) == 0`

This checks: "Is the minimum value in the last 6 months equal to 0?"

### Processing Steps

1. **Identify zeros**: `min(last_6_months) == 0`
2. **Calculate geometric mean**: Ignore zeros in calculation (replace with NaN)
3. **Apply zero guard**: Set prediction to 0 if any zero found
4. **Fill NaN**: Convert any remaining NaN (all zeros) to 0

---

## Missing Sector Handling

### What It Is

Sector 95 appears in test data but **never appears in training data**.

### Problem

```
Training data sectors: 1, 2, 3, ..., 94, 96  (missing 95!)
Test data sectors: 1, 2, 3, ..., 94, 95, 96  (has 95!)
```

If we don't handle this, predictions for sector 95 would fail.

### Solution

**Add sector 95 with all zeros** to the training matrix.

### Code Location

**File**: `src/utils.py` (lines 46-49)

```python
# Ensure all sectors 1..96 exist
amount = amount.fillna(0)
for sector in range(1, 97):
    if sector not in amount.columns:
        amount[sector] = 0  # ← Add missing sectors as zeros
amount = amount[[i for i in range(1, 97)]]
```

### Example

**Before:**

```
amount.columns = [1, 2, 3, ..., 94, 96]  # Missing 95
```

**After:**

```
amount.columns = [1, 2, 3, ..., 94, 95, 96]  # All sectors present
amount[95] = [0, 0, 0, ..., 0]  # All zeros
```

### Why Predict Zero?

- Sector 95 has **no historical data**
- Predicting zero is conservative (won't cause catastrophic APE)
- If actual is 0: Perfect (APE = 0%)
- If actual > 0: At worst APE = 100% (acceptable)

---

## Lags Explained

### What Are Lags?

**Lags** are past values of the target variable, shifted back in time.

### Concept

Think of it as "looking back" at previous months:

```
Time | Actual | lag_1  | lag_2  | lag_3
-----|--------|--------|--------|--------
t=0  | 25,000 | NaN    | NaN    | NaN    (no past data)
t=1  | 26,000 | 25,000 | NaN    | NaN    (lag_1 = value at t=0)
t=2  | 24,000 | 26,000 | 25,000 | NaN    (lag_1 = t=1, lag_2 = t=0)
t=3  | 27,000 | 24,000 | 26,000 | 25,000 (lag_1 = t=2, lag_2 = t=1, lag_3 = t=0)
```

### Example: Sector 10

**Original data:**

```
Time | Sector | Amount
-----|--------|--------
0    | 10     | 30,000
1    | 10     | 32,000
2    | 10     | 28,000
3    | 10     | 35,000
4    | 10     | 33,000
```

**After creating lags:**

```
Time | Sector | Amount  | lag_1  | lag_2  | lag_3  | lag_6  | lag_12
-----|--------|---------|--------|--------|--------|--------|--------
0    | 10     | 30,000  | NaN    | NaN    | NaN    | NaN    | NaN
1    | 10     | 32,000  | 30,000 | NaN    | NaN    | NaN    | NaN
2    | 10     | 28,000  | 32,000 | 30,000 | NaN    | NaN    | NaN
3    | 10     | 35,000  | 28,000 | 32,000 | 30,000 | NaN    | NaN
...
12   | 10     | 36,000  | 34,000 | 33,000 | 32,000 | 30,000 | 30,000
```

**Note**: `s.shift(lag)` produces `NaN` when there's not enough historical data (e.g., `lag_12` is NaN for times < 12).

### Code Location

**File**: `src/features.py` (lines 15-47)

```python
def build_time_lagged_features(nht, lags=[1, 2, 3, 6, 12], rolling_windows=[3, 6, 12]):
    # Group by sector
    for sector_id, g in df.groupby("sector_id"):
        s = g["amount_new_house_transactions"].fillna(0)

        # Create lags
        for lag in lags:
            feat[f"lag_{lag}"] = s.shift(lag)  # ← Shift backward by 'lag' periods
```

**Key operation**: `s.shift(lag)` shifts the series backward by `lag` periods.

### Handling Missing Lags (NaN)

**Important**: When a lag doesn't exist (not enough history), `shift()` produces `NaN`, not zero!

**During Training**:

- Rows with **any** missing lag are **dropped** (not used for training)
- Code: `df_model = df.dropna(subset=[c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')])`
- Why: Models train only on samples with complete feature history

**During Prediction**:

- Missing lags are **filled with 0**
- Code: `X_t = step_df[feature_cols].fillna(0)`
- Why: Can't drop test samples - must predict for all test cases

**Example:**

```python
# Training: Time 0-2 are dropped (incomplete lags)
# Only Time 3+ are used (all lags available)

# Prediction: Time 67 with missing lag_12
# lag_12 = NaN → filled with 0
# Model uses: [lag_1=33k, lag_2=32k, ..., lag_12=0]
```

### Why Use Lags?

- **Time series patterns**: Recent values often predict future values
- **Trends**: Can capture increasing/decreasing patterns
- **Seasonality**: `lag_12` captures year-over-year patterns

### Example: Predicting Time 67

To predict time 67 for sector 10, we use:

- `lag_1` = value at time 66 (last month)
- `lag_2` = value at time 65 (2 months ago)
- `lag_12` = value at time 55 (same month last year)

**Important**: Only use **past** data to avoid data leakage!

---

## How Lags Are Used as Features (Ridge & XGBoost)

### Yes, Lags Are Feature Vectors!

Lags are used as **columns in a feature matrix** where:

- **Each row** = one sample (a specific time × sector combination)
- **Each column** = one feature (lag_1, lag_2, lag_3, roll_mean_3, etc.)
- **The model** learns weights/coefficients for each feature column

### Example: Feature Matrix Construction

**Step 1: Create lag features** (as shown above)

**Step 2: Extract feature columns**

```python
# From scripts/baseline_ridge.py line 31-32
feature_cols = [c for c in df_model.columns if c.startswith('lag_') or c.startswith('roll_')]
X = df_model[feature_cols]  # ← Feature matrix
y = df_model['y']  # ← Target vector
```

**Step 3: Feature matrix structure**

For example, with 3 sectors and 3 time points:

```
Feature Matrix X (samples × features):
┌─────────┬──────────┬──────────┬──────────┬─────────────┬─────────────┐
│ Sample  │ lag_1    │ lag_2    │ lag_3    │ roll_mean_3 │ roll_mean_6 │
├─────────┼──────────┼──────────┼──────────┼─────────────┼─────────────┤
│ t=3, s=1│ 32,000   │ 31,000   │ 30,000   │ 31,000      │ 30,500      │
│ t=3, s=2│ 18,000   │ 19,000   │ 17,000   │ 18,000      │ 18,200      │
│ t=3, s=3│ 25,000   │ 24,000   | 26,000   │ 25,000      │ 25,100      │
│ t=4, s=1│ 33,000   │ 32,000   │ 31,000   │ 32,000      │ 31,200      │
│ ...     │ ...      │ ...      │ ...      │ ...         │ ...         │
└─────────┴──────────┴──────────┴──────────┴─────────────┴─────────────┘

Target Vector y:
┌─────────┬─────────┐
│ Sample  │ y       │
├─────────┼─────────┤
│ t=3, s=1│ 34,000  │
│ t=3, s=2│ 19,500  │
│ t=3, s=3│ 27,000  │
│ t=4, s=1│ 35,000  │
│ ...     │ ...     │
└─────────┴─────────┘
```

### Ridge Regression Usage

**Ridge learns linear coefficients:**

```python
# From scripts/baseline_ridge.py line 32-42
X = df_model[feature_cols]  # Feature matrix
y = df_model['y']           # Target vector

# Ridge learns: y ≈ w₁·lag_1 + w₂·lag_2 + w₃·lag_3 + ...
pipe = build_linear_pipeline(alpha=1.0, kind='ridge')
pipe.fit(X_tr, y_tr)        # Learn weights w₁, w₂, w₃, ...
yhat = pipe.predict(X_va)   # Predict using learned weights
```

**Mathematical form:**

$$\hat{y} = w_1 \cdot \text{lag}_1 + w_2 \cdot \text{lag}_2 + w_3 \cdot \text{lag}_3 + w_4 \cdot \text{roll\_mean}_3 + \ldots$$

Where $w_1, w_2, w_3, \ldots$ are learned coefficients.

**Example prediction:**

For a sample with:

- `lag_1 = 32,000`
- `lag_2 = 31,000`
- `lag_3 = 30,000`
- `roll_mean_3 = 31,000`

Ridge might learn: $w_1 = 0.3, w_2 = 0.2, w_3 = 0.1, w_4 = 0.4$

Prediction: $\hat{y} = 0.3 \times 32,000 + 0.2 \times 31,000 + 0.1 \times 30,000 + 0.4 \times 31,000 = 33,900$

### XGBoost Usage

**XGBoost uses lags as decision tree features:**

```python
# From scripts/tune_xgb_optuna.py line 42-44
X_tr, y_tr = df_model.loc[tr_mask, feature_cols], df_model.loc[tr_mask, 'y']
X_va, y_va = df_model.loc[va_mask, feature_cols], df_model.loc[va_mask, 'y']

model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, ...)
model.fit(X_tr, y_tr)  # Build trees using lag features
yhat = model.predict(X_va)  # Predict using trees
```

**How trees use lags:**

XGBoost builds decision trees with rules like:

```
Tree 1:
  if lag_1 > 30,000:
    if lag_2 > 29,000:
      return +5,000
    else:
      return +2,000
  else:
    if roll_mean_3 < 25,000:
      return -1,000
    else:
      return +500
```

Each tree splits on lag features to make predictions.

### Code Locations

**Missing lag handling:**

- **Training (drop rows)**: `scripts/baseline_ridge.py` line 30, `scripts/tune_xgb_optuna.py` line 24
- **Prediction (fill with 0)**: `scripts/baseline_ridge.py` line 64

**Feature extraction:**

- **Ridge**: `scripts/baseline_ridge.py` lines 31-32
- **XGBoost**: `scripts/tune_xgb_optuna.py` lines 25-26
- **Benchmark**: `scripts/benchmark.py` lines 42-43

**Model training:**

- **Ridge**: `scripts/baseline_ridge.py` lines 41-42
- **XGBoost**: `scripts/tune_xgb_optuna.py` lines 44-56
- **Benchmark**: `scripts/benchmark.py` lines 60-63 (Ridge), 124-125 (XGBoost)

### Complete Example: Predicting Time 12, Sector 10

**1. Feature values:**

```python
lag_1 = 34,000   # Value at time 11
lag_2 = 33,000   # Value at time 10
lag_3 = 32,000   # Value at time 9
lag_6 = 30,000   # Value at time 6
lag_12 = 28,000  # Value at time 0
roll_mean_3 = 33,000
roll_mean_6 = 31,500
```

**2. Feature vector (one row in X):**

```python
x = [34,000, 33,000, 32,000, 30,000, 28,000, 33,000, 31,500, ...]
```

**3. Ridge prediction:**

```python
prediction = model.predict([x])  # Input: 1D array (one sample)
# Returns: scalar prediction, e.g., 35,200
```

**4. XGBoost prediction:**

```python
prediction = model.predict([x])  # Same input format
# Returns: scalar prediction (may differ from Ridge)
```

### Key Points

1. **Lags are features**: Each lag becomes a column in the feature matrix
2. **Each sample is a row**: One row = one (time, sector) combination
3. **Ridge uses linear combination**: $\hat{y} = \sum w_i \cdot \text{feature}_i$
4. **XGBoost uses tree splits**: Trees split on lag values to make decisions
5. **Same input format**: Both Ridge and XGBoost receive the same feature matrix `X`

---

## Where Preprocessing Happens

### 1. Core Utility Functions

**File**: `src/utils.py`

- `build_amount_wide()`: Time encoding + wide format conversion + missing sector handling
- `geometric_mean_with_zero_guard()`: Zero handling

**File**: `src/data.py`

- `split_month_sector()`: Time encoding + sector ID parsing
- `prepare_train_target()`: Wide format conversion with zero filling

### 2. Feature Engineering

**File**: `src/features.py`

- `build_time_lagged_features()`: Creates lag and rolling features
- `join_static_sector_features()`: Joins POI features

### 3. Notebooks

**File**: `notebooks/01_eda.ipynb`

- **Cell 2**: Loads data and applies `split_month_sector()` (time encoding)
- **Cell 3**: Uses `build_amount_wide()` (format conversion)
- **Cell 4+**: Exploratory analysis on preprocessed data

**File**: `notebooks/02_baseline_seasonality.ipynb`

- Uses `build_amount_wide()` for preprocessing
- Applies zero guard and December boost

### 4. Scripts

**File**: `scripts/baseline_seasonality.py` (lines 27-38)

```python
# Load raw data
train_nht = train_data['new_house_transactions']

# Preprocessing happens here:
amount = build_amount_wide(train_nht)  # ← Time encoding + wide format + missing sectors

# Then apply zero guard
base_geo = geometric_mean_with_zero_guard(amount, ...)  # ← Zero handling
```

**File**: `scripts/ultimate_ensemble.py` (lines 97-122)

```python
# Feature engineering preprocessing
lag_features = build_time_lagged_features(nht, lags=[1, 2, 3, 4, 5, 6, 12])
features_df = features_df.merge(lag_features, ...)  # ← Lag features added
```

### 5. Execution Flow

```
Raw CSV Files
    ↓
load_all_training_tables()  [src/data.py]
    ↓
Long format DataFrame (month, sector, amount)
    ↓
split_month_sector()  [src/data.py]
    ↓
Adds: year, month_num, time, sector_id
    ↓
build_amount_wide()  [src/utils.py]
    ↓
Wide format matrix (time × sector)
    ↓
Feature engineering (if ML model)
    ↓
Ready for modeling
```

### Summary Table

| Preprocessing Step | Function                           | File              | Line  |
| ------------------ | ---------------------------------- | ----------------- | ----- |
| Time encoding      | `split_month_sector()`             | `src/data.py`     | 86    |
| Wide format        | `build_amount_wide()`              | `src/utils.py`    | 43-44 |
| Missing sectors    | `build_amount_wide()`              | `src/utils.py`    | 46-49 |
| Zero handling      | `geometric_mean_with_zero_guard()` | `src/utils.py`    | 65    |
| Lag features       | `build_time_lagged_features()`     | `src/features.py` | 34    |

---

## Quick Reference

### Time Encoding Formula

```python
time = (year - 2019) * 12 + month_num - 1
```

### Zero Guard Rule

```python
if min(last_6_months) == 0:
    prediction = 0
```

### Lag Creation

```python
lag_k = series.shift(k)  # Value from k periods ago
```

### Wide Format Access

```python
amount.loc[time, sector]  # Get value for specific time and sector
amount.tail(6)  # Last 6 months
```
