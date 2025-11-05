# Methods Overview: Mathematical Explanations

## Table of Contents

1. [Competition Metric](#competition-metric)
2. [Geometric Mean Baseline](#geometric-mean-baseline)
3. [Seasonality Enhancement](#seasonality-enhancement)
4. [Zero Guard Mechanism](#zero-guard-mechanism)
5. [Ridge Regression](#ridge-regression)
6. [Gradient Boosting Methods](#gradient-boosting-methods)
7. [Ensemble Methods](#ensemble-methods)
8. [Performance Summary](#performance-summary)

---

## Competition Metric

### Two-Stage Scaled MAPE

The competition uses a custom metric that severely penalizes poor predictions, especially for zero values.

#### Stage 1: Absolute Percentage Error (APE)

For each prediction $(i, j)$ where $i$ is time and $j$ is sector:

$$\text{APE}_{ij} = \frac{|y_{ij} - \hat{y}_{ij}|}{\max(y_{ij}, \epsilon)}$$

where $\epsilon = 10^{-12}$ prevents division by zero.

**Important**: Lower APE is better. APE = 0% means perfect prediction, APE = 100% means the error equals the actual value.

#### Stage 2: Two-Stage Scoring

Define the "good" prediction set (predictions with APE ≤ 100%):
$$\mathcal{G} = \{(i,j) : \text{APE}_{ij} \leq 1.0\}$$

**Bad predictions** are those with $\text{APE}_{ij} > 1.0$ (APE > 100%).

Good rate:
$$\text{good\_rate} = \frac{|\mathcal{G}|}{N} = \frac{\text{number of predictions with APE} \leq 100\%}{N}$$

where $N$ is the total number of predictions.

**Scoring function:**

$$
\text{score} = \begin{cases}
0 & \text{if } \text{good\_rate} < 0.3 \\
\text{good\_rate} \cdot \frac{1}{|\mathcal{G}|} \sum_{(i,j) \in \mathcal{G}} \frac{1}{1 + \text{APE}_{ij}} & \text{otherwise}
\end{cases}
$$

**Score interpretation**: The score formula $\frac{1}{1 + \text{APE}}$ confirms that **lower APE gives higher score**:

- APE = 0% → score contribution = 1.0 (perfect)
- APE = 50% → score contribution = 0.67
- APE = 100% → score contribution = 0.5 (threshold for "good" predictions)

#### Key Properties

1. **30% Filter (Cliff Edge)**:

   - **Threshold**: APE > 100% (APE > 1.0) marks a "bad" prediction
   - **Rule**: If **more than 70%** of predictions have APE > 100%, then score = 0
   - **Equivalently**: If **less than 30%** of predictions have APE ≤ 100%, then score = 0
   - **Example**: Out of 1000 predictions, if 701 have APE > 100%, the score becomes 0

2. **Zero sensitivity**: When $y_{ij} = 0$, any $\hat{y}_{ij} > 0$ gives $\text{APE}_{ij} = \hat{y}_{ij}$, which can be catastrophic

   - Example: Predicting 25,000 when actual = 0 gives APE = 25,000% (very bad!)

3. **Asymmetric penalty**: Overpredicting zeros is much worse than underpredicting large values

---

## Geometric Mean Baseline

### Mathematical Formulation

For sector $j$ at time $t$, let $X_{t-k:t-1}^{(j)} = \{x_{t-k}^{(j)}, x_{t-k+1}^{(j)}, \ldots, x_{t-1}^{(j)}\}$ be the last $k$ historical values.

**Geometric mean:**

$$\hat{y}_t^{(j)} = \left(\prod_{i=1}^{k} x_{t-i}^{(j)}\right)^{1/k}$$

**Log-space computation (numerically stable):**

$$\hat{y}_t^{(j)} = \exp\left(\frac{1}{k} \sum_{i=1}^{k} \log(x_{t-i}^{(j)})\right)$$

where $\log(0)$ is handled by replacing zeros with NaN and skipping them.

### Why Geometric Mean?

1. **Multiplicative processes**: Real estate transactions often follow multiplicative growth/decay patterns
2. **Outlier robustness**: Less sensitive to extreme values than arithmetic mean
3. **Skewed distributions**: Handles right-skewed distributions common in transaction amounts

**Comparison:**

For values $[10,000, 20,000, 50,000]$:

- Arithmetic mean: $\frac{80,000}{3} = 26,667$
- Geometric mean: $(10,000 \times 20,000 \times 50,000)^{1/3} = 21,544$

The geometric mean is closer to the smaller values, making it more conservative.

### Implementation

```python
# For each sector j:
last_k_months = X[t-k:t-1, j]  # Last k months of sector j
log_values = np.log(last_k_months.replace(0, np.nan))
geo_mean = np.exp(log_values.mean(skipna=True))
prediction = geo_mean if not np.isnan(geo_mean) else 0
```

**Parameters:**

- `lookback_months = 6`: Uses last 6 months
- Rationale: Balances recency with stability

---

## Seasonality Enhancement

### December Boost Factor

Historical analysis shows December has ~30% higher transaction volumes on average.

#### Per-Sector Boost Calculation

For sector $j$, define:

- $D_j = \{d_1, d_2, \ldots\}$: All December values (non-zero)
- $N_j = \{n_1, n_2, \ldots\}$: All non-December values (non-zero)

**Boost factor:**

$$\beta_j = \min\left(\frac{\bar{D}_j}{\bar{N}_j}, \text{cap}\right)$$

where:

- $\bar{D}_j = \frac{1}{|D_j|} \sum_{d \in D_j} d$ (mean December value)
- $\bar{N}_j = \frac{1}{|N_j|} \sum_{n \in N_j} n$ (mean non-December value)
- $\text{cap} = 2.0$ (upper bound, typically use 1.3 conservatively)

**Default fallback:**
If insufficient data ($|D_j| = 0$ or $|N_j| = 0$), use $\beta_j = 1.3$.

#### Prediction Formula

$$
\hat{y}_t^{(j)} = \begin{cases}
\hat{y}_{\text{base}}^{(j)} \cdot \beta_j & \text{if month}(t) = \text{December} \\
\hat{y}_{\text{base}}^{(j)} & \text{otherwise}
\end{cases}
$$

where $\hat{y}_{\text{base}}^{(j)}$ is the geometric mean prediction.

**Example:**

- Base prediction: 35,000 yuan
- December boost: $\beta_j = 1.3$
- December prediction: $35,000 \times 1.3 = 45,500$ yuan

---

## Zero Guard Mechanism

### Mathematical Definition

The zero guard is a conservative rule that prevents catastrophic errors on zero sectors.

**Zero guard predicate:**

$$\text{has\_zero}(j, w) = \min_{i=t-w}^{t-1} x_i^{(j)} = 0$$

where $w$ is the zero guard window (typically 6 months).

**Final prediction:**

$$
\hat{y}_t^{(j)} = \begin{cases}
0 & \text{if } \text{has\_zero}(j, w) \\
\hat{y}_{\text{geometric}}^{(j)} & \text{otherwise}
\end{cases}
$$

### Why This Works

1. **Metric-aware**: Prevents APE > 100% on zero sectors
2. **Conservative**: Better to predict 0 and get APE = 100% than predict non-zero and get APE = 1000%+
3. **Pattern recognition**: Sectors with recent zeros often remain zero

**Decision matrix:**

| Actual | Predicted | APE     | Status         |
| ------ | --------- | ------- | -------------- |
| 0      | 0         | 0%      | Perfect ✓      |
| 0      | 25,000    | 25,000% | Catastrophic ✗ |
| 5,000  | 0         | 100%    | Acceptable ✓   |
| 5,000  | 25,000    | 400%    | Poor ✗         |

---

## Ridge Regression

### Mathematical Formulation

Ridge regression minimizes the L2-regularized loss:

$$\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

where:

- $\mathbf{X} \in \mathbb{R}^{n \times p}$: Feature matrix (lags, rolling stats, etc.)
- $\mathbf{y} \in \mathbb{R}^{n}$: Target values
- $\mathbf{w} \in \mathbb{R}^{p}$: Regression coefficients
- $\alpha > 0$: Regularization strength

**Closed-form solution:**

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

### Features Used

1. **Lagged values**: $x_{t-k}^{(j)}$ for $k \in \{1, 2, 3, 6, 12\}$
2. **Rolling means**: $\bar{x}_{t-w:t-1}^{(j)} = \frac{1}{w}\sum_{i=t-w}^{t-1} x_i^{(j)}$ for $w \in \{3, 6, 12\}$
3. **Rolling standard deviations**: $\sigma_{t-w:t-1}^{(j)}$
4. **Sector indicators**: One-hot encoded sector IDs

### Why It Failed

1. **Non-linear metric**: Optimizes for RMSE, not the two-stage MAPE
2. **Zero handling**: Cannot learn the conservative zero-guard behavior
3. **Overfitting**: Learned training patterns that didn't generalize
4. **Score**: Local CV ~0.35-0.50, Public LB: 0.00000

---

## Gradient Boosting Methods

### XGBoost / CatBoost / LightGBM

All three use gradient boosting with different implementations.

#### Gradient Boosting Framework

**Initial prediction:**

$$\hat{y}_0^{(j)} = \bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$$

**Iterative refinement:**

For $m = 1, 2, \ldots, M$:

1. **Compute residuals:**
   $$r_{m-1}^{(i)} = y_i - \hat{y}_{m-1}^{(i)}$$

2. **Fit tree $h_m$ to residuals:**
   $$h_m = \arg\min_h \sum_{i=1}^{n} L(r_{m-1}^{(i)}, h(\mathbf{x}_i))$$

3. **Update prediction:**
   $$\hat{y}_m^{(i)} = \hat{y}_{m-1}^{(i)} + \eta \cdot h_m(\mathbf{x}_i)$$

where $\eta$ is the learning rate (typically 0.05-0.1).

**Final prediction:**

$$\hat{y}^{(i)} = \bar{y} + \eta \sum_{m=1}^{M} h_m(\mathbf{x}_i)$$

#### Loss Function

For regression, typically squared error:

$$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$

Gradient:
$$\frac{\partial L}{\partial \hat{y}} = -(y - \hat{y}) = -r$$

#### Key Hyperparameters

1. **n_estimators ($M$)**: Number of trees (typically 100-500)
2. **learning_rate ($\eta$)**: Step size (typically 0.05-0.1)
3. **max_depth**: Tree depth (typically 4-6)
4. **subsample**: Row sampling ratio (typically 0.6-0.9)
5. **colsample_bytree**: Column sampling ratio

#### Why Gradient Boosting Failed

1. **Limited data**: Only 57 months × 96 sectors = 5,472 training samples
2. **Metric mismatch**: Optimizes for squared error, not two-stage MAPE
3. **Zero sensitivity**: Cannot learn conservative zero predictions
4. **Overfitting**: Complex models learned noise, not signal
5. **Score**: Local CV ~0.45, Public LB: 0.00000

**Lesson**: Simple geometric mean (0 trees) outperformed 100+ tree ensembles!

---

## Ensemble Methods

### Weighted Ensemble

Combine multiple models with learned weights:

$$\hat{y}_{\text{ensemble}}^{(j)} = \sum_{k=1}^{K} w_k \cdot \hat{y}_k^{(j)}$$

where:

- $K$: Number of models
- $w_k$: Weight for model $k$ (constrained: $\sum_{k=1}^{K} w_k = 1$, $w_k \geq 0$)
- $\hat{y}_k^{(j)}$: Prediction from model $k$

### Zero Classification Ensemble

Two-stage approach:

1. **Zero classifier**: Predict probability $p(\text{zero}|\mathbf{x})$
2. **Regressor ensemble**: Predict amount for non-zero cases
3. **Final prediction:**

$$
\hat{y}^{(j)} = \begin{cases}
0 & \text{if } p(\text{zero}|\mathbf{x}) > \theta \\
\sum_{k=1}^{K} w_k \cdot \hat{y}_k^{(j)} & \text{otherwise}
\end{cases}
$$

where $\theta$ is the zero threshold (typically 0.3-0.6).

### Model Components

1. **Geometric baseline**: $\hat{y}_{\text{geo}}^{(j)}$
2. **CatBoost regressor**: $\hat{y}_{\text{cb}}^{(j)}$
3. **LightGBM regressor**: $\hat{y}_{\text{lgb}}^{(j)}$
4. **Zero classifier**: CatBoost binary classifier

### Weight Optimization

Grid search over validation set:

$$\mathbf{w}^*, \theta^* = \arg\max_{\mathbf{w}, \theta} \text{competition\_score}(\mathbf{y}_{\text{val}}, \hat{\mathbf{y}}_{\text{ensemble}}(\mathbf{w}, \theta))$$

**Typical optimal weights:**

- $w_{\text{geo}} \approx 0.4-0.6$ (geometric baseline)
- $w_{\text{cb}} \approx 0.2-0.3$ (CatBoost)
- $w_{\text{lgb}} \approx 0.2-0.3$ (LightGBM)
- $\theta \approx 0.4-0.5$ (zero threshold)

---

## Performance Summary

### Model Comparison

| Method                           | Local CV  | Public LB   | Notes                |
| -------------------------------- | --------- | ----------- | -------------------- |
| **Geometric Mean + Seasonality** | 0.53      | **0.56248** | Best single model    |
| Geometric Mean                   | 0.52      | 0.55528     | Baseline             |
| Ensemble (Geo + CB + LGB)        | 0.54-0.55 | ~0.56       | Marginal improvement |
| Ridge Regression                 | 0.38      | 0.00000     | Failed on public     |
| XGBoost                          | 0.45      | 0.00000     | Failed on public     |

### Key Insights

1. **Simplicity wins**: Geometric mean beats complex ML models
2. **Metric understanding**: Conservative zero-guard is critical
3. **Domain knowledge**: December boost provides measurable improvement
4. **Local-public gap**: Simple models generalize better (smaller gap)

### Mathematical Lessons

1. **Not all loss functions are equal**: RMSE optimization ≠ competition metric optimization
2. **Regularization matters**: Zero guard is a form of conservative regularization
3. **Feature engineering**: Sometimes less is more (geometric mean uses only historical values)
4. **Ensemble diversity**: Combining models helps, but baseline must be strong

---

## References

- Competition metric: Two-stage scaled MAPE (see `docs/METRICS.md`)
- Implementation: `src/utils.py`, `scripts/baseline_seasonality.py`
- Approach evolution: `docs/APPROACH.md`
- Concepts: `docs/CONCEPTS.md`
