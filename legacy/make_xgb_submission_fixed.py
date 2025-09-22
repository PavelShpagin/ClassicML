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


def build_features_for_time(df, target_time, target_sectors):
    """Build lag features for a specific time point using historical data."""
    # Create placeholder rows for the target time
    target_rows = pd.DataFrame({
        'time': [target_time] * len(target_sectors),
        'sector_id': target_sectors,
        'amount_new_house_transactions': [np.nan] * len(target_sectors)
    })
    
    # Combine with historical data
    combined = pd.concat([df, target_rows], ignore_index=True)
    
    # Build features
    features = build_time_lagged_features(combined)
    
    # Return only the features for the target time
    return features[features['time'] == target_time]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(ROOT))
    parser.add_argument("--out", type=str, default=str(ROOT / "submission.csv"))
    parser.add_argument("--t2", type=int, default=6, help="months to check nonzero in last window")
    args = parser.parse_args()

    paths = DatasetPaths(root_dir=args.root)
    train = load_all_training_tables(paths)
    target_wide, _ = prepare_train_target(train['new_house_transactions'])

    # Zero-guard sectors: if any of last t2 months == 0 → predict 0 for all test months
    last_window = target_wide.tail(args.t2)
    zero_guard_sectors = set(int(c) for c in last_window.columns[(last_window.min(axis=0) == 0)])
    print(f"Zero-guard sectors (had 0 in last {args.t2} months): {len(zero_guard_sectors)}")

    # Feature matrix for training
    lag_feats = build_time_lagged_features(train['new_house_transactions']).sort_values(['time','sector_id'])
    y_long = target_wide.unstack().reset_index(name='y').rename(columns={'level_0':'sector_id','time':'time'})
    df = lag_feats.merge(y_long, on=['time','sector_id'], how='left')
    df_model = df.dropna(subset=[c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]).copy()
    feature_cols = [c for c in df_model.columns if c.startswith('lag_') or c.startswith('roll_')]
    
    print(f"Training features shape: {df_model.shape}")
    print(f"Feature columns: {feature_cols}")

    # TimeSeries CV for quick hyperparams
    folds = [
        (0, 18, 19, 30),
        (0, 30, 31, 42),
        (0, 42, 43, 54),
        (0, 54, 55, 66),
    ]
    best = None
    for md in [6, 8]:
        for lr in [0.03, 0.05, 0.1]:
            scores = []
            for (tr_s, tr_e, va_s, va_e) in folds:
                tr_mask = (df_model['time'] >= tr_s) & (df_model['time'] <= tr_e)
                va_mask = (df_model['time'] >= va_s) & (df_model['time'] <= va_e)
                X_tr, y_tr = df_model.loc[tr_mask, feature_cols], df_model.loc[tr_mask, 'y']
                X_va, y_va = df_model.loc[va_mask, feature_cols], df_model.loc[va_mask, 'y']
                model = xgb.XGBRegressor(
                    max_depth=md, learning_rate=lr, n_estimators=800,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.0, reg_lambda=1.0,
                    objective='reg:squarederror', random_state=42, tree_method='hist'
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                yhat = model.predict(X_va)
                # non-negativity
                yhat = np.maximum(0, yhat)
                scores.append(competition_score(y_va.values, yhat)['score'])
            score_mean = float(np.mean(scores))
            if (best is None) or (score_mean > best[0]):
                best = (score_mean, {'max_depth': md, 'learning_rate': lr})
    
    print(f"Best CV score: {best[0]:.4f} with params: {best[1]}")

    # Train best on all df_model
    params = best[1]
    final = xgb.XGBRegressor(
        max_depth=params['max_depth'], learning_rate=params['learning_rate'], n_estimators=1000,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
        objective='reg:squarederror', random_state=42, tree_method='hist'
    )
    final.fit(df_model[feature_cols], df_model['y'])

    # Recursive forecasting for test
    test_df = load_test(paths)
    test_exploded = explode_test_id(test_df)
    
    # Start with historical data
    base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
    current_series = base.copy()
    
    # Get all unique sectors
    all_sectors = sorted(current_series['sector_id'].unique())
    
    preds = []
    test_times = sorted(test_exploded['time'].unique())
    print(f"\nPredicting for test times: {test_times}")
    
    for t in test_times:
        print(f"  Processing time {t}...")
        
        # Build features for this time using the helper function
        lag_t = build_features_for_time(current_series, t, all_sectors)
        
        # Merge with test IDs
        step_df = test_exploded[test_exploded['time'] == t][['id','sector_id','time']].merge(
            lag_t, on=['time','sector_id'], how='left'
        )
        
        # Check if features were generated
        if len(lag_t) == 0:
            print(f"    WARNING: No features generated for time {t}")
            X_t = step_df[feature_cols].fillna(0)
        else:
            X_t = step_df[feature_cols]
            # Check for NaNs
            nan_count = X_t.isna().sum().sum()
            if nan_count > 0:
                print(f"    WARNING: {nan_count} NaN values in features, filling with 0")
                X_t = X_t.fillna(0)
        
        # Predict
        yhat_t = final.predict(X_t)
        
        # Apply constraints: non-negativity + zero-guard sectors
        yhat_t = np.maximum(0, yhat_t)
        mask_zero_guard = step_df['sector_id'].astype(int).isin(zero_guard_sectors).values
        yhat_t[mask_zero_guard] = 0.0
        
        # Store predictions
        out_t = step_df[['id','sector_id','time']].copy()
        out_t['new_house_transaction_amount'] = yhat_t
        preds.append(out_t[['id','new_house_transaction_amount']])
        
        # Update series with predictions for next iteration
        update_t = out_t.rename(columns={'new_house_transaction_amount':'amount_new_house_transactions'})
        current_series = pd.concat([current_series, update_t[['time','sector_id','amount_new_house_transactions']]], ignore_index=True)
        
        print(f"    Predicted mean: {yhat_t.mean():.2f}, std: {yhat_t.std():.2f}, zeros: {(yhat_t == 0).sum()}")

    # Combine all predictions
    submission = pd.concat(preds, ignore_index=True)
    submission = test_df[['id']].merge(submission, on='id', how='left')
    
    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    
    print(f'\nSaved submission to {args.out}')
    print(f'Submission stats:')
    print(f'  Shape: {submission.shape}')
    print(f'  Mean: {submission["new_house_transaction_amount"].mean():.2f}')
    print(f'  Std: {submission["new_house_transaction_amount"].std():.2f}')
    print(f'  Min: {submission["new_house_transaction_amount"].min():.2f}')
    print(f'  Max: {submission["new_house_transaction_amount"].max():.2f}')
    print(f'  Zeros: {(submission["new_house_transaction_amount"] == 0).sum()}')


if __name__ == '__main__':
    main()
