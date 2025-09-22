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
│   ├── baseline_geometric.py      # Winning approach (0.555 score)
│   ├── baseline_seasonality.py    # Best approach (0.562 score)
│   ├── benchmark.py               # Model benchmarking
│   ├── make_simple_submission.py  # Conservative baseline
│   ├── make_submission.py         # Ridge regression submission
│   └── tune_xgb_optuna.py        # Bayesian hyperparameter tuning
├── src/
│   ├── data.py            # Data loading utilities
│   ├── features.py        # Feature engineering
│   └── models.py          # Model definitions and metrics
├── requirements.txt       # Python dependencies
└── submission_*.csv       # Kaggle submission files
```

## Results Summary

| Method                           | Public Score | Description                                     |
| -------------------------------- | ------------ | ----------------------------------------------- |
| **Geometric Mean + Seasonality** | **0.56248**  | Best score - Geometric mean with December boost |
| Geometric Mean Baseline          | 0.55528      | 6-month geometric mean with zero guard          |
| Simple Median                    | 0.21591      | Conservative median-based approach              |
| XGBoost (Optuna-tuned)           | 0.00000      | Failed due to zero predictions                  |
| Ridge Regression                 | 0.00000      | Linear model couldn't handle zeros              |

## Key Insights

### What Worked

1. **Geometric Mean**: More robust than arithmetic mean for skewed distributions
2. **Zero Guard**: Critical - if any of last 6 months was zero, predict zero
3. **Seasonality**: December typically shows 30% higher transactions
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
# Generate submission with best approach (0.562 score)
python scripts/baseline_seasonality.py

# Submit to Kaggle
kaggle competitions submit -c china-real-estate-demand-prediction \
  -f submission_seasonality.csv -m "Your message"
```

### Run Experiments

```bash
# Run model benchmarking
python scripts/benchmark.py

# Tune XGBoost with Optuna
python scripts/tune_xgb_optuna.py

# Run EDA notebook
jupyter notebook notebooks/eda.ipynb
```

## Methodology

### 1. Data Analysis

- Explored temporal patterns and seasonality
- Identified 21% of training samples have zero transactions
- Found December consistently shows higher transaction volumes

### 2. Feature Engineering

- Time-lagged features (1-12 months)
- Rolling statistics (mean, std)
- Careful handling of data leakage in time series

### 3. Model Development

#### Baseline Models

- Ridge Regression with cross-validation
- Simple median-based predictions

#### Advanced Models

- XGBoost with Bayesian optimization (Optuna)
- LightGBM and CatBoost experiments
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

## Lab Work Documentation

The complete lab report is available in `notebooks/lab_final.ipynb`, which includes:

- Detailed competition analysis
- Data visualization and insights
- Model benchmarking results
- RMSE/MAPE curves for different methods
- Bayesian optimization results
- Final leaderboard performance

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
