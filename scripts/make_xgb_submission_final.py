import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import json

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id, split_month_sector
from src.features import build_time_lagged_features
from src.models import competition_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(ROOT))
    parser.add_argument("--out", type=str, default=str(ROOT / "submission.csv"))
    parser.add_argument("--use_optuna", action="store_true", help="Use Optuna-tuned parameters if available")
    args = parser.parse_args()

    paths = DatasetPaths(root_dir=args.root)
    train = load_all_training_tables(paths)
    target_wide, _ = prepare_train_target(train['new_house_transactions'])

    # Identify all sectors (including test-only sectors)
    train_sectors = set(train['new_house_transactions']['sector'].str.extract('(\d+)')[0].astype(int))
    test_df = load_test(paths)
    test_sectors = set(test_df['id'].str.extract('sector (\d+)')[0].astype(int))
    all_sectors = sorted(train_sectors | test_sectors)
    missing_sectors = test_sectors - train_sectors
    
    print(f"Train sectors: {len(train_sectors)}")
    print(f"Test sectors: {len(test_sectors)}")
    print(f"Sectors missing in train: {missing_sectors}")

    # Feature matrix for training
    lag_feats = build_time_lagged_features(train['new_house_transactions']).sort_values(['time','sector_id'])
    y_long = target_wide.unstack().reset_index(name='y').rename(columns={'level_0':'sector_id','time':'time'})
    df = lag_feats.merge(y_long, on=['time','sector_id'], how='left')
    df_model = df.dropna(subset=[c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]).copy()
    feature_cols = [c for c in df_model.columns if c.startswith('lag_') or c.startswith('roll_')]
    
    print(f"\nTraining features shape: {df_model.shape}")
    print(f"Feature columns: {feature_cols}")

    # Check for Optuna parameters
    best_params = None
    optuna_path = Path(args.root) / "reports" / "xgb_optuna_best.json"
    if args.use_optuna and optuna_path.exists():
        with open(optuna_path, 'r') as f:
            optuna_result = json.load(f)
            best_params = optuna_result['best_params']
            print(f"\nUsing Optuna-tuned parameters: {best_params}")
    else:
        # Quick CV for hyperparameters
        print("\nRunning quick CV for hyperparameters...")
        folds = [
            (0, 30, 31, 42),
            (0, 42, 43, 54),
            (0, 54, 55, 66),
        ]
        best = None
        for md in [6, 8]:
            for lr in [0.05, 0.1]:
                scores = []
                for (tr_s, tr_e, va_s, va_e) in folds:
                    tr_mask = (df_model['time'] >= tr_s) & (df_model['time'] <= tr_e)
                    va_mask = (df_model['time'] >= va_s) & (df_model['time'] <= va_e)
                    X_tr, y_tr = df_model.loc[tr_mask, feature_cols], df_model.loc[tr_mask, 'y']
                    X_va, y_va = df_model.loc[va_mask, feature_cols], df_model.loc[va_mask, 'y']
                    
                    model = xgb.XGBRegressor(
                        max_depth=md, learning_rate=lr, n_estimators=500,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.0, reg_lambda=1.0,
                        objective='reg:squarederror', random_state=42, tree_method='hist'
                    )
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                    yhat = model.predict(X_va)
                    yhat = np.maximum(0, yhat)  # non-negativity
                    scores.append(competition_score(y_va.values, yhat)['score'])
                
                score_mean = float(np.mean(scores))
                if (best is None) or (score_mean > best[0]):
                    best = (score_mean, {'max_depth': md, 'learning_rate': lr})
        
        best_params = best[1]
        print(f"Best CV score: {best[0]:.4f} with params: {best_params}")

    # Train final model on all training data
    print("\nTraining final model on all data...")
    final = xgb.XGBRegressor(
        max_depth=best_params.get('max_depth', 8),
        learning_rate=best_params.get('learning_rate', 0.1),
        n_estimators=best_params.get('n_estimators', 1000),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        reg_alpha=best_params.get('reg_alpha', 0.0),
        reg_lambda=best_params.get('reg_lambda', 1.0),
        gamma=best_params.get('gamma', 0.0),
        min_child_weight=best_params.get('min_child_weight', 1),
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist'
    )
    final.fit(df_model[feature_cols], df_model['y'])

    # Prepare for recursive forecasting
    test_exploded = explode_test_id(test_df)
    base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
    
    # For missing sectors, use median of similar sectors as initial values
    if missing_sectors:
        print(f"\nHandling missing sectors {missing_sectors}...")
        # Get median values for each time point across all sectors
        time_medians = base.groupby('time')['amount_new_house_transactions'].median()
        
        # Add synthetic history for missing sectors
        for sector in missing_sectors:
            synthetic_data = pd.DataFrame({
                'time': time_medians.index,
                'sector_id': [sector] * len(time_medians),
                'amount_new_house_transactions': time_medians.values
            })
            base = pd.concat([base, synthetic_data], ignore_index=True)
    
    current_series = base.copy()
    
    # Recursive forecasting
    preds = []
    test_times = sorted(test_exploded['time'].unique())
    print(f"\nPredicting for test times: {test_times}")
    
    for t in test_times:
        # Add placeholder rows for this time
        placeholders = pd.DataFrame({
            'time': [t] * len(all_sectors),
            'sector_id': all_sectors,
            'amount_new_house_transactions': [np.nan] * len(all_sectors)
        })
        combined = pd.concat([current_series, placeholders], ignore_index=True)
        
        # Build features
        lag_features = build_time_lagged_features(combined)
        lag_t = lag_features[lag_features['time'] == t]
        
        # Merge with test IDs
        step_df = test_exploded[test_exploded['time'] == t][['id','sector_id','time']].merge(
            lag_t, on=['time','sector_id'], how='left'
        )
        
        # Prepare features
        X_t = step_df[feature_cols]
        
        # For missing sectors, use median imputation for features
        if X_t.isna().any().any():
            # Fill with column medians from training
            for col in feature_cols:
                if col in df_model.columns:
                    X_t[col] = X_t[col].fillna(df_model[col].median())
                else:
                    X_t[col] = X_t[col].fillna(0)
        
        # Predict
        yhat_t = final.predict(X_t)
        
        # Apply constraints: non-negativity
        yhat_t = np.maximum(0, yhat_t)
        
        # For sectors with no history, apply dampening factor
        for sector in missing_sectors:
            mask = step_df['sector_id'] == sector
            if mask.any():
                # Use 50% of predicted value for sectors with synthetic history
                yhat_t[mask] *= 0.5
        
        # Store predictions
        out_t = step_df[['id','sector_id','time']].copy()
        out_t['new_house_transaction_amount'] = yhat_t
        preds.append(out_t[['id','new_house_transaction_amount']])
        
        # Update series with predictions for next iteration
        update_t = out_t.rename(columns={'new_house_transaction_amount':'amount_new_house_transactions'})
        current_series = pd.concat([current_series, update_t[['time','sector_id','amount_new_house_transactions']]], ignore_index=True)
        
        print(f"  Time {t}: mean={yhat_t.mean():.2f}, std={yhat_t.std():.2f}, min={yhat_t.min():.2f}, max={yhat_t.max():.2f}")

    # Combine all predictions
    submission = pd.concat(preds, ignore_index=True)
    submission = test_df[['id']].merge(submission, on='id', how='left')
    
    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    
    print(f'\n=== Submission saved to {args.out} ===')
    print(f'Shape: {submission.shape}')
    print(f'Mean: {submission["new_house_transaction_amount"].mean():.2f}')
    print(f'Std: {submission["new_house_transaction_amount"].std():.2f}')
    print(f'Min: {submission["new_house_transaction_amount"].min():.2f}')
    print(f'Max: {submission["new_house_transaction_amount"].max():.2f}')
    print(f'Zeros: {(submission["new_house_transaction_amount"] == 0).sum()}')
    print(f'Unique values: {submission["new_house_transaction_amount"].nunique()}')


if __name__ == '__main__':
    main()
