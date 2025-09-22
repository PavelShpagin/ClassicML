"""
Advanced submission script with zero handling and ensemble methods.
Targets a public leaderboard score of ~0.6.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id, split_month_sector
from src.features import build_time_lagged_features
from src.models import competition_score


def add_advanced_features(df):
    """Add advanced features including seasonality and trends."""
    df = df.copy()
    
    # Month features (seasonality)
    df['month_sin'] = np.sin(2 * np.pi * (df['time'] % 12) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['time'] % 12) / 12)
    
    # Year progress
    df['year_progress'] = df['time'] / df['time'].max()
    
    # Lag ratios (momentum indicators)
    for lag in [1, 3, 6]:
        lag_col = f'lag_{lag}'
        if lag_col in df.columns and 'lag_1' in df.columns:
            df[f'ratio_lag1_lag{lag}'] = df['lag_1'] / (df[lag_col] + 1)  # +1 to avoid division by zero
    
    # Zero indicators
    df['lag1_is_zero'] = (df['lag_1'] == 0).astype(int)
    df['lag3_is_zero'] = (df['lag_3'] == 0).astype(int)
    
    # Volatility measure (std of recent lags)
    lag_cols = [c for c in df.columns if c.startswith('lag_') and c.split('_')[1].isdigit()]
    if lag_cols:
        df['lag_std'] = df[lag_cols].std(axis=1)
        df['lag_cv'] = df['lag_std'] / (df[lag_cols].mean(axis=1) + 1)
    
    return df


def train_zero_classifier(df, feature_cols):
    """Train a classifier to predict zero vs non-zero."""
    df_train = df.dropna(subset=feature_cols + ['y']).copy()
    
    # Create binary target
    y_binary = (df_train['y'] == 0).astype(int)
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(df_train[feature_cols], y_binary)
    
    # Report performance
    zero_prob = clf.predict_proba(df_train[feature_cols])[:, 1]
    print(f"Zero classifier trained. Feature importance (top 5):")
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importances.head().items():
        print(f"  {feat}: {imp:.3f}")
    
    return clf


def train_ensemble_models(df, feature_cols, use_optuna=False):
    """Train ensemble of XGBoost, LightGBM, and CatBoost."""
    df_train = df.dropna(subset=feature_cols + ['y']).copy()
    
    # Remove zeros for regression training (we'll handle them separately)
    df_nonzero = df_train[df_train['y'] > 0].copy()
    
    X = df_nonzero[feature_cols]
    y = df_nonzero['y']
    
    models = {}
    
    # XGBoost
    if use_optuna:
        optuna_path = Path(ROOT) / "reports" / "xgb_optuna_best.json"
        if optuna_path.exists():
            with open(optuna_path, 'r') as f:
                params = json.load(f)['best_params']
        else:
            params = {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 800}
    else:
        params = {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 800}
    
    models['xgb'] = xgb.XGBRegressor(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist'
    )
    models['xgb'].fit(X, y)
    
    # LightGBM
    models['lgb'] = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    models['lgb'].fit(X, y)
    
    # CatBoost
    models['cat'] = cb.CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    models['cat'].fit(X, y)
    
    print("Ensemble models trained (XGBoost, LightGBM, CatBoost)")
    
    return models


def predict_with_ensemble(models, zero_clf, X, zero_threshold=0.7):
    """Make predictions using ensemble with zero handling."""
    # Predict zero probability
    zero_prob = zero_clf.predict_proba(X)[:, 1]
    
    # Ensemble predictions for non-zero values
    preds = []
    for name, model in models.items():
        pred = model.predict(X)
        preds.append(pred)
    
    # Average ensemble predictions
    ensemble_pred = np.mean(preds, axis=0)
    
    # Apply zero mask based on threshold
    final_pred = ensemble_pred.copy()
    final_pred[zero_prob > zero_threshold] = 0
    
    # Ensure non-negative
    final_pred = np.maximum(0, final_pred)
    
    return final_pred, zero_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(ROOT))
    parser.add_argument("--out", type=str, default=str(ROOT / "submission_advanced.csv"))
    parser.add_argument("--use_optuna", action="store_true", help="Use Optuna-tuned parameters")
    parser.add_argument("--zero_threshold", type=float, default=0.7, help="Threshold for zero classification")
    args = parser.parse_args()

    print("=== ADVANCED SUBMISSION PIPELINE ===")
    print(f"Zero threshold: {args.zero_threshold}")
    print(f"Use Optuna: {args.use_optuna}")
    
    # Load data
    paths = DatasetPaths(root_dir=args.root)
    train = load_all_training_tables(paths)
    target_wide, _ = prepare_train_target(train['new_house_transactions'])
    test_df = load_test(paths)
    test_exploded = explode_test_id(test_df)
    
    # Identify sectors
    train_sectors = set(train['new_house_transactions']['sector'].str.extract('(\d+)')[0].astype(int))
    test_sectors = set(test_df['id'].str.extract('sector (\d+)')[0].astype(int))
    all_sectors = sorted(train_sectors | test_sectors)
    missing_sectors = test_sectors - train_sectors
    
    print(f"\nData loaded. Missing sectors in train: {missing_sectors}")
    
    # Build features with advanced engineering
    print("\nBuilding features...")
    lag_feats = build_time_lagged_features(train['new_house_transactions']).sort_values(['time','sector_id'])
    lag_feats = add_advanced_features(lag_feats)
    
    # Prepare training data
    y_long = target_wide.unstack().reset_index(name='y').rename(columns={'level_0':'sector_id','time':'time'})
    df = lag_feats.merge(y_long, on=['time','sector_id'], how='left')
    
    # Get feature columns
    base_feature_cols = [c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]
    advanced_feature_cols = ['month_sin', 'month_cos', 'year_progress', 'lag1_is_zero', 'lag3_is_zero', 'lag_std', 'lag_cv']
    advanced_feature_cols += [c for c in df.columns if 'ratio_' in c]
    feature_cols = base_feature_cols + [c for c in advanced_feature_cols if c in df.columns]
    
    print(f"Features: {len(feature_cols)} total")
    
    # Train zero classifier
    print("\nTraining zero classifier...")
    zero_clf = train_zero_classifier(df, feature_cols)
    
    # Train ensemble models
    print("\nTraining ensemble models...")
    models = train_ensemble_models(df, feature_cols, args.use_optuna)
    
    # Handle missing sectors
    if missing_sectors:
        print(f"\nHandling missing sectors {missing_sectors}...")
        base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
        time_medians = base.groupby('time')['amount_new_house_transactions'].median()
        
        for sector in missing_sectors:
            synthetic_data = pd.DataFrame({
                'time': time_medians.index,
                'sector_id': [sector] * len(time_medians),
                'amount_new_house_transactions': time_medians.values * 0.7  # Dampen synthetic data
            })
            base = pd.concat([base, synthetic_data], ignore_index=True)
    else:
        base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
    
    current_series = base.copy()
    
    # Recursive forecasting
    preds = []
    test_times = sorted(test_exploded['time'].unique())
    print(f"\nPredicting for test times: {test_times}")
    
    for t in test_times:
        # Add placeholder rows
        placeholders = pd.DataFrame({
            'time': [t] * len(all_sectors),
            'sector_id': all_sectors,
            'amount_new_house_transactions': [np.nan] * len(all_sectors)
        })
        combined = pd.concat([current_series, placeholders], ignore_index=True)
        
        # Build features
        lag_features = build_time_lagged_features(combined)
        lag_features = add_advanced_features(lag_features)
        lag_t = lag_features[lag_features['time'] == t]
        
        # Merge with test IDs
        step_df = test_exploded[test_exploded['time'] == t][['id','sector_id','time']].merge(
            lag_t, on=['time','sector_id'], how='left'
        )
        
        # Prepare features
        X_t = step_df[feature_cols].copy()
        
        # Impute missing values
        for col in feature_cols:
            if col in df.columns and col in X_t.columns:
                X_t[col] = X_t[col].fillna(df[col].median() if col in df.columns else 0)
        
        # Predict with ensemble and zero handling
        yhat_t, zero_prob = predict_with_ensemble(models, zero_clf, X_t, args.zero_threshold)
        
        # Additional dampening for missing sectors
        for sector in missing_sectors:
            mask = step_df['sector_id'] == sector
            if mask.any():
                yhat_t[mask] *= 0.5
        
        # Store predictions
        out_t = step_df[['id','sector_id','time']].copy()
        out_t['new_house_transaction_amount'] = yhat_t
        preds.append(out_t[['id','new_house_transaction_amount']])
        
        # Update series
        update_t = out_t.rename(columns={'new_house_transaction_amount':'amount_new_house_transactions'})
        current_series = pd.concat([current_series, update_t[['time','sector_id','amount_new_house_transactions']]], ignore_index=True)
        
        # Stats
        zeros = (yhat_t == 0).sum()
        print(f"  Time {t}: mean={yhat_t[yhat_t>0].mean() if (yhat_t>0).any() else 0:.2f}, zeros={zeros}/{len(yhat_t)}, max={yhat_t.max():.2f}")
    
    # Combine predictions
    submission = pd.concat(preds, ignore_index=True)
    submission = test_df[['id']].merge(submission, on='id', how='left')
    
    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    
    print(f'\n=== SUBMISSION SAVED ===')
    print(f'File: {args.out}')
    print(f'Shape: {submission.shape}')
    print(f'Mean (non-zero): {submission[submission["new_house_transaction_amount"]>0]["new_house_transaction_amount"].mean():.2f}')
    print(f'Zeros: {(submission["new_house_transaction_amount"] == 0).sum()} / {len(submission)}')
    print(f'Max: {submission["new_house_transaction_amount"].max():.2f}')


if __name__ == '__main__':
    main()
