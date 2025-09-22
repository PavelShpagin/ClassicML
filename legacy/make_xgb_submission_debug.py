import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id, split_month_sector
from src.features import build_time_lagged_features
from src.models import competition_score


def main():
    paths = DatasetPaths(root_dir=str(ROOT))
    train = load_all_training_tables(paths)
    target_wide, _ = prepare_train_target(train['new_house_transactions'])

    # Feature matrix for training
    lag_feats = build_time_lagged_features(train['new_house_transactions']).sort_values(['time','sector_id'])
    y_long = target_wide.unstack().reset_index(name='y').rename(columns={'level_0':'sector_id','time':'time'})
    df = lag_feats.merge(y_long, on=['time','sector_id'], how='left')
    df_model = df.dropna(subset=[c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]).copy()
    feature_cols = [c for c in df_model.columns if c.startswith('lag_') or c.startswith('roll_')]
    
    print(f"Training features shape: {df_model.shape}")
    print(f"Feature columns: {feature_cols}")
    print(f"Training time range: {df_model['time'].min()} - {df_model['time'].max()}")

    # Simple model for testing
    print("\nTraining simple XGBoost...")
    final = xgb.XGBRegressor(
        max_depth=6, learning_rate=0.1, n_estimators=100,
        objective='reg:squarederror', random_state=42, tree_method='hist'
    )
    final.fit(df_model[feature_cols], df_model['y'])
    print("Model trained.")

    # Test data
    test_df = load_test(paths)
    test_exploded = explode_test_id(test_df)
    
    # Start with historical data
    base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
    current_series = base.copy()
    
    # Get all unique sectors
    all_sectors = sorted(current_series['sector_id'].unique())
    print(f"\nSectors in training: {len(all_sectors)}, list: {all_sectors[:5]}...{all_sectors[-5:]}")
    
    # Test for first test time only
    t = 67  # 2024 Aug
    print(f"\n=== Testing prediction for time {t} ===")
    
    # Method 1: Direct (broken)
    print("\nMethod 1: Direct call on current_series")
    lag_direct = build_time_lagged_features(current_series)
    lag_t_direct = lag_direct[lag_direct['time'] == t]
    print(f"  Features generated: {len(lag_t_direct)} rows")
    
    # Method 2: With placeholders
    print("\nMethod 2: With placeholder rows")
    placeholders = pd.DataFrame({
        'time': [t] * len(all_sectors),
        'sector_id': all_sectors,
        'amount_new_house_transactions': [np.nan] * len(all_sectors)
    })
    combined = pd.concat([current_series, placeholders], ignore_index=True)
    lag_combined = build_time_lagged_features(combined)
    lag_t_fixed = lag_combined[lag_combined['time'] == t]
    print(f"  Features generated: {len(lag_t_fixed)} rows")
    
    # Get test rows for this time
    test_t = test_exploded[test_exploded['time'] == t][['id','sector_id','time']]
    print(f"\nTest rows for time {t}: {len(test_t)} rows")
    print(f"Test sectors: {sorted(test_t['sector_id'].unique())[:5]}...{sorted(test_t['sector_id'].unique())[-5:]}")
    
    # Merge and check
    step_df = test_t.merge(lag_t_fixed, on=['time','sector_id'], how='left')
    print(f"\nAfter merge: {len(step_df)} rows")
    print(f"Feature columns in merged: {[c for c in step_df.columns if c.startswith('lag_') or c.startswith('roll_')]}")
    
    # Check for NaNs
    X_t = step_df[feature_cols]
    print(f"\nNaN counts in features:")
    for col in feature_cols[:5]:  # Show first 5
        nan_count = X_t[col].isna().sum()
        print(f"  {col}: {nan_count} NaNs")
    
    # Make prediction
    X_t_filled = X_t.fillna(0)
    yhat = final.predict(X_t_filled)
    print(f"\nPredictions: mean={yhat.mean():.2f}, std={yhat.std():.2f}, min={yhat.min():.2f}, max={yhat.max():.2f}")
    
    # Check if all predictions are the same
    unique_preds = np.unique(yhat)
    print(f"Unique prediction values: {len(unique_preds)}")
    if len(unique_preds) <= 5:
        print(f"  Values: {unique_preds}")


if __name__ == '__main__':
    main()
