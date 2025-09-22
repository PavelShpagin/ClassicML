## China Real Estate Demand Prediction - Classic ML Lab Report

### Data & Metric

- Target: `amount_new_house_transactions` per `month` x `sector`.
- Evaluation: two-stage metric based on absolute percentage errors; final score = 1 − (MAPE_good / good_rate). High good_rate and low MAPE on good samples are both critical.
- Leakage guard: all rolling features use shifted windows; time-series CV uses four 12-month holdouts.

### Features

- Lags: 1, 2, 3, 6, 12 months.
- Shifted rolling means and geometric means: 3, 6, 12 months.
- Future work: seasonal indicators (e.g., December), recency decay, POI/static joins, nearby-sector aggregates, zero-guard classification head.

### Cross-Validation

- Folds: (0..18→19..30), (0..30→31..42), (0..42→43..54), (0..54→55..66).
- Each fold simulates a 12-month forecasting horizon.

### Results: Ridge (TimeSeries CV)

|    alpha | score_mean | score_std | good_rate_mean | rmse_mean | mape_mean |
| -------: | ---------: | --------: | -------------: | --------: | --------: |
|  19.3069 |   0.262835 |  0.263099 |       0.697570 |           |           |
| 100.0000 |   0.260663 |  0.261011 |       0.692179 |           |           |
|   3.7276 |   0.260548 |  0.260628 |       0.702585 |           |           |
|   0.7197 |   0.251243 |  0.251321 |       0.699314 |           |           |
|   0.1389 |   0.237865 |  0.239249 |       0.690187 |           |           |

(Full CSV: `reports/cv_ridge_results.csv`)

### Results: Advanced Models (TimeSeries CV)

| model    | best_params              | score_mean | good_rate_mean | rmse_mean | mape_mean |
| -------- | ------------------------ | ---------: | -------------: | --------: | --------: |
| XGBoost  | {max_depth: 6, lr: 0.05} |   0.375390 |       0.734506 |           |           |
| CatBoost | {depth: 8}               |   0.267961 |       0.707470 |           |           |

(Full CSV: `reports/cv_advanced_models.csv`)

### Interpretation

- Local CV ≳ 0.55 is typically required to exceed 0.60 on the public leaderboard. Current best (XGB ≈ 0.38) suggests more feature engineering and tuning are needed.
- Good_rate should remain ≥ 0.85–0.90; otherwise the score collapses due to scaling by good_rate.

### Next Steps (to reach ≥0.60 LB)

- Expand features: seasonality (December peaks), recency decay, POI/static joins, nearby-sector aggregates, zero-guard module.
- Hyperparameter optimization: Optuna-based Bayesian search with TimeSeries CV across key grids for XGBoost/LightGBM/CatBoost.
- Ensembling: weighted blend of top 2–3 models.
- Submission: train best model on full training span, recursive roll-forward prediction, preserve row order.

### Reproducibility

- Environment: `.venv` with pinned `requirements.txt`.
- Scripts: `scripts/benchmark.py` (CV + reports), `scripts/make_submission.py` (submission generation).
- Outputs: `reports/cv_ridge_results.csv`, `reports/cv_advanced_models.csv`, `reports/benchmark_log.txt`.
