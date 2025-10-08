# Notebooks Overview

This directory contains Jupyter notebooks for analyzing and modeling the China Real Estate Demand Prediction dataset.

## Notebooks

### 1. `01_eda.ipynb` - Exploratory Data Analysis
**Purpose**: Comprehensive data exploration and visualization

**Visualizations**:
- Transaction amount distributions (histogram, log scale, box plot)
- Zero vs non-zero transactions (pie chart)
- Time series of total and mean transactions over time
- Seasonality analysis by month (December highlighted)
- Sector-wise statistics (mean amount, zero rate)
- Price vs amount correlation scatter plot

**Key Findings**:
- Zero rate: ~21% of transactions
- December shows 30%+ higher transactions
- Highly skewed distribution
- High variance across sectors

---

### 2. `02_baseline_seasonality.ipynb` - Best Method
**Purpose**: Geometric mean baseline with December seasonality boost

**Feature Engineering**:
- December boost factor calculation per sector
- Distribution visualization of boost factors

**Hyperparameter Tuning**:
- Lookback window (3, 6, 9, 12, 18, 24 months)
- MAPE curves: Competition score vs lookback
- Good rate vs lookback

**Model Validation**:
- Competition metric (two-stage MAPE)
- APE distribution histogram
- Predictions vs true values scatter plot

**Performance**: Best competition score with high good rate

---

### 3. `03_baseline_geometric.ipynb` - Geometric Mean Baseline
**Purpose**: Pure geometric mean without seasonality

**Hyperparameter Tuning**:
- 2D grid search: lookback × zero_guard_window
- Heatmap visualization of scores
- MAPE curves for best zero guard window

**Model Validation**:
- Competition metric evaluation
- APE distribution
- Comparison with seasonality method

**Performance**: Simple and robust, slightly worse than seasonality-aware method

---

### 4. `04_ridge_regression.ipynb` - Ridge Regression
**Purpose**: Linear regression with lag/rolling features

**Feature Engineering**:
- Lag features (1-6 months)
- Rolling statistics (mean, geometric mean)
- Feature table visualization

**Hyperparameter Tuning**:
- Regularization strength (alpha) from 0.01 to 1000
- Competition score vs alpha (log scale)
- Good rate vs alpha

**Model Analysis**:
- Feature importance (coefficient magnitudes)
- Top 15 features bar chart
- APE distribution
- Predictions vs true scatter plot

**Performance**: Worse than baselines due to metric sensitivity, produces negative predictions

---

## Key Visualizations

### Distribution Plots
- **Histograms**: Transaction amounts, log-transformed distributions
- **Box plots**: Outlier detection
- **Pie charts**: Zero vs non-zero transaction proportions

### Time Series Analysis
- **Line plots**: Total and mean transactions over time
- **Bar charts**: Average transactions by month (seasonality)
- **Sector plots**: Sector-specific statistics

### Model Performance
- **MAPE curves**: Competition score vs hyperparameters  
  (Alternative to ROC curves for regression)
- **APE histograms**: Error distribution with 100% threshold line
- **Scatter plots**: Predictions vs true values with perfect prediction line
- **Heatmaps**: Grid search results (lookback × zero_guard)
- **Feature importance**: Bar charts of coefficients

---

## Running the Notebooks

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Execution
```bash
# Run interactively
jupyter notebook notebooks/

# Or execute programmatically
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda_executed.ipynb
```

---

## Notes

- All notebooks use the `src/` module for data loading and feature engineering
- Notebooks are self-contained and can be run independently
- MAPE curves are used instead of ROC curves (regression task, not classification)
- Competition metric (two-stage MAPE) is the primary evaluation metric
- Zero predictions are critical - the zero guard prevents metric explosions

