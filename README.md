# China Real Estate Demand Prediction

## Project Overview

This repository contains the solution for the Kaggle competition "China Real Estate Demand Prediction" that achieved a **public leaderboard score of 0.56248**, successfully meeting the target of >0.4-0.5 accuracy.

### Competition Details

- **Objective**: Predict monthly new house transaction amounts across 96 sectors in China
- **Evaluation**: Custom two-stage MAPE-based metric (highly sensitive to zero predictions)
- **Data**: 67 months of historical data (Jan 2019 - Jul 2024), predict next 12 months
- **Challenge**: Metric severely penalizes predictions with APE > 100%, especially on zero values

## Project Structure

```
ClassicML/
├── data/                    # Data directory (raw data in .gitignore)
│   └── raw/                # Kaggle competition data
├── docs/                   # Documentation
├── legacy/                 # Archived/old files
├── models/                 # Saved models
├── notebooks/
│   ├── eda.ipynb          # Exploratory Data Analysis
│   ├── train.ipynb        # Model training experiments
│   └── lab_final.ipynb    # Comprehensive lab report
├── reports/               # Generated analysis reports
│   ├── cv_ridge_results.csv
│   ├── cv_advanced_models.csv
│   └── xgb_optuna_best.json
├── scripts/
│   ├── baseline_geometric.py      # Geometric mean baseline
│   ├── baseline_seasonality.py    # Best approach (0.562 score)
│   ├── baseline_simple.py         # Conservative baseline
│   ├── baseline_ridge.py          # Ridge regression baseline
│   ├── benchmark.py               # Model benchmarking (writes CSVs and plots)
│   └── tune_xgb_optuna.py        # Bayesian hyperparameter tuning
├── src/
│   ├── data.py            # Data loading utilities
│   ├── features.py        # Feature engineering
│   ├── models.py          # Model definitions and metrics
│   └── utils.py           # Shared helpers (wide matrix, geo mean, seasonality, I/O)
├── requirements.txt       # Python dependencies
├── submissions/           # All generated submissions (standardized)
└── docker/                # Containerization to run best baseline
```

## Results Summary

| Method                           | Local CV | Public Score | Description                                     |
| -------------------------------- | -------- | ------------ | ----------------------------------------------- |
| **Geometric Mean + Seasonality** | 0.53     | **0.56248**  | Best score - Geometric mean with December boost |
| Geometric Mean Baseline          | 0.52     | 0.55528      | 6-month geometric mean with zero guard          |
| Simple Median                    | ~0.22    | 0.21591      | Conservative median-based approach              |
| XGBoost (Optuna-tuned)           | 0.45     | 0.00000      | Failed due to zero predictions                  |
| Ridge Regression                 | 0.38     | 0.00000      | Linear model couldn't handle zeros              |

**Note**: Only XGBoost was tuned with Optuna (Bayesian hyperparameter optimization). Other models used manual hyperparameter selection.

## Key Insights

### What Worked

1. **Geometric Mean**: More robust than arithmetic mean for skewed distributions
2. **Zero Guard**: Critical - if any of last 6 months was zero, predict zero
3. **Seasonality**: December typically shows 30% higher transactions (seasonality = monthly patterns, specifically December boost factor of ~1.3x)
4. **Simplicity**: Conservative approaches outperformed complex ML models

### What Didn't Work

- Complex models (XGBoost, Ridge) despite good local CV scores (~0.35-0.50)
- The metric's extreme sensitivity to zero predictions caused these to fail
- Recursive forecasting without proper zero handling

## Installation & Setup

### Prerequisites

- Python 3.10+
- Kaggle API credentials configured

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd ClassicML

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download competition data
kaggle competitions download -c china-real-estate-demand-prediction
unzip china-real-estate-demand-prediction.zip -d data/raw/
```

## Usage

### Run Winning Solution

```bash
# Python: generate submission with best approach (0.562 score)
python scripts/baseline_seasonality.py

# Submit to Kaggle
kaggle competitions submit -c china-real-estate-demand-prediction \
  -f submissions/baseline_seasonality.csv -m "Your message"
```

### Run Experiments and View Plots

```bash
# Run model benchmarking (writes CSVs and plots to reports/)
python scripts/benchmark.py

# Tune XGBoost with Optuna
python scripts/tune_xgb_optuna.py

# Run EDA notebook
jupyter notebook notebooks/eda.ipynb
```

### Run in Docker (best baseline)

```bash
# Build image
docker build -t classicml -f docker/Dockerfile .

# Run (mount data and submissions)
docker run --rm \
  -v "$(pwd)/data/raw:/app/data/raw" \
  -v "$(pwd)/submissions:/app/submissions" \
  classicml
```

## Methodology

### 1. Data Analysis

- Explored temporal patterns and seasonality
- Identified 21% of training samples have zero transactions
- Found December consistently shows higher transaction volumes

### 2. Feature Engineering & Processing

#### Data Processing

- **Time encoding**: Converted month/year to time index (Jan 2019 = 0, Feb 2019 = 1, ...)
- **Wide format conversion**: Pivoted data to time × sector matrix for efficient computation
- **Zero handling**: Special handling for sectors with zero transactions (21% of training data)
- **Missing sector handling**: Added sector 95 (missing in training, present in test) with zeros

#### Feature Engineering (ML Models Only)

**Note**: The winning approach (Geometric Mean + Seasonality) uses **NO traditional features** - only raw historical transaction amounts. However, ML models attempted extensive feature engineering:

##### Temporal Features (Used in Ridge/XGBoost/CatBoost)

- **Lagged values**: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` (transaction amounts from 1, 2, 3, 6, 12 months ago)
- **Rolling statistics**:
  - Rolling means: `roll_mean_3`, `roll_mean_6`, `roll_mean_12` (3, 6, 12-month windows)
  - Rolling standard deviations: `roll_std_3`, `roll_std_6`, `roll_std_12`
  - Rolling geometric means: `roll_geo_mean_3`, `roll_geo_mean_6`, `roll_geo_mean_12`
  - Rolling maximums: `roll_max_3`, `roll_max_6`, `roll_max_12`
- **Time-based features**:
  - `month_num`: Month number (1-12)
  - `quarter`: Quarter (1-4)
  - `is_december`: Binary indicator for December
  - `december_boost`: Per-sector December boost factor

##### External Data Features (Tried but Didn't Help)

- **POI (Points of Interest)**: Static sector features (schools, hospitals, metro stations, malls, parks)
- **Nearby sectors**: Aggregated transactions from neighboring sectors (mean, sum, std, max)
- **Land transactions**: Nearby land transaction amounts
- **Pre-owned house transactions**: Nearby pre-owned house transaction amounts
- **City-level features**: GDP, resident population (from city indexes)

##### Feature Selection

- **No explicit feature selection**: All engineered features were used in ML models
- **Missing value handling**: Filled with -999 or -1 (depending on model)
- **Leakage prevention**: All rolling features use `shift(1)` to avoid future data leakage

#### Winning Approach Features (Geometric Mean + Seasonality)

**Uses only historical transaction amounts - no feature engineering:**

1. **Raw historical values**: Last 6 months of transaction amounts per sector
2. **Geometric mean calculation**: Computed from raw values
3. **Zero guard**: Binary rule based on presence of zeros in last 6 months
4. **December boost**: Per-sector multiplier (1.0-2.0x) computed from historical December/non-December ratios

**Key Insight**: The best model uses **zero features** (only raw data), while ML models used **20-50+ engineered features** and performed worse. This demonstrates that feature engineering is not always beneficial - sometimes raw data + domain knowledge beats complex feature sets.

### 3. Model Development

#### Baseline Models

- Ridge Regression with cross-validation
- Simple median-based predictions

#### Advanced Models

- XGBoost with Bayesian optimization (Optuna) - only XGBoost was tuned with Optuna
- LightGBM and CatBoost experiments (manual hyperparameter selection)
- Time series cross-validation

#### Winning Approach

```python
# Geometric Mean with Seasonality
1. Calculate 6-month geometric mean for each sector
2. Apply zero guard (if any recent month = 0, predict 0)
3. Boost December predictions by 1.3x
4. Result: 0.56248 public score
```

### 4. Evaluation Strategy

- TimeSeriesSplit cross-validation
- Custom competition metric implementation
- Careful validation to avoid overfitting

## Documentation

### Core Documentation

- **`README.md`** (this file): Project overview, setup, usage, results summary
- **`docs/APPROACH.md`**: Technical approach, solution evolution, implementation details
- **`docs/METRICS.md`**: Deep dive into two-stage MAPE metric, why it's challenging
- **`docs/CONCEPTS.md`**: Technical concepts explained (lags, rolling mean, XGBoost, etc.)
- **`docs/METHODS_OVERVIEW.md`**: Mathematical explanations of all methods used
- **`docs/PREPROCESSING.md`**: Data preprocessing explained with examples (time encoding, format conversion, zero handling, lags)
- **`docs/LAB_SUMMARY.md`**: Lab completion checklist and achievements
- **`notebooks/README.md`**: Notebook descriptions and visualizations guide

### Jupyter Notebooks

- **`notebooks/01_eda.ipynb`**: Exploratory data analysis with visualizations
- **`notebooks/02_baseline_seasonality.ipynb`**: Best method (geometric mean + seasonality)
- **`notebooks/03_baseline_geometric.ipynb`**: Geometric mean baseline
- **`notebooks/04_ridge_regression.ipynb`**: Ridge regression with feature engineering

## Lessons Learned

1. **Metric Understanding is Crucial**: The two-stage MAPE metric's design made simple, conservative approaches more effective than complex models
2. **Zero Handling**: Proper treatment of zero values was the key differentiator
3. **Domain Knowledge**: Understanding real estate seasonality (December boost) improved scores
4. **Simplicity Wins**: In competitions with sensitive metrics, robust baselines often outperform sophisticated models

## Future Improvements

1. Ensemble of conservative models
2. Sector-specific modeling for high-variance sectors
3. External economic indicators integration
4. More sophisticated seasonality modeling

## Contributors

- Pavel (Project Lead)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for hosting the competition
- Competition organizers for the challenging metric design
- Community solutions that provided insights into effective approaches
