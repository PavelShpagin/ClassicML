"""
Ultimate Ensemble Model
=======================

Combines geometric baseline, CatBoost, and LightGBM with zero classification
Target: Beat 0.60+ on public leaderboard

Strategy:
1. Zero Classifier (CatBoost binary) - predicts which will be zero
2. Geometric Baseline - robust metric-aware predictions
3. CatBoost Regressor - tree model for non-zero amounts
4. LightGBM - fast gradient boosting
5. Weighted Blending - optimized on validation set
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import DatasetPaths, load_all_training_tables, load_test, split_month_sector
from src.features import build_time_lagged_features, join_static_sector_features
from src.models import competition_score
from src.utils import build_amount_wide, compute_december_boost, ensure_dir

def predict_geometric_baseline(train_data, test_times, december_boost_dict, lookback=6):
    """Predict using geometric mean with zero guard and December boost"""
    predictions = []
    sectors = sorted(train_data['sector_id'].unique())
    
    for test_time in test_times:
        for sector in sectors:
            sector_data = train_data[
                (train_data['sector_id'] == sector) & 
                (train_data['time'] < test_time)
            ].sort_values('time')
            
            if len(sector_data) < lookback:
                pred = 0
            else:
                recent = sector_data.tail(lookback)['amount_new_house_transactions'].values
                
                if (recent == 0).any():
                    pred = 0
                else:
                    pred = np.exp(np.log(recent + 1e-10).mean())
                    
                    # December boost
                    month = (test_time % 12) if (test_time % 12) != 0 else 12
                    if month == 12 and sector in december_boost_dict:
                        pred *= december_boost_dict[sector]
            
            predictions.append({
                'time': test_time,
                'sector_id': sector,
                'pred_geometric': pred
            })
    
    return pd.DataFrame(predictions)

def create_ensemble(geo, cb, lgb_preds, zero_probs, weights, zero_threshold=0.5):
    """Create ensemble with zero masking"""
    w_geo, w_cb, w_lgb = weights
    ensemble = w_geo * geo + w_cb * cb + w_lgb * lgb_preds
    ensemble = np.where(zero_probs > zero_threshold, 0, ensemble)
    return np.maximum(ensemble, 0)

def main():
    print("="*60)
    print("ULTIMATE ENSEMBLE MODEL")
    print("="*60)
    print()
    
    # 1. Load data
    print("1. Loading data...")
    paths = DatasetPaths(root_dir='.')
    tables = load_all_training_tables(paths)
    test_df = load_test(paths)
    
    nht = tables['new_house_transactions']
    nht_split = split_month_sector(nht)
    print(f"   Training: {nht_split.shape}")
    print(f"   Test: {test_df.shape}")
    
    # 2. Feature engineering
    print("\n2. Engineering features...")
    amount_wide = build_amount_wide(nht)
    
    # Start with split data (has month_num, target, etc.)
    features_df = nht_split.copy()
    
    # Add lag/rolling features
    lag_features = build_time_lagged_features(nht.copy(), lags=[1, 2, 3, 4, 5, 6, 12])
    features_df = features_df.merge(lag_features, on=['time', 'sector_id'], how='left')
    
    # Add POI features
    if 'sector_POI' in tables:
        features_df = join_static_sector_features(features_df, tables['sector_POI'])
        print("   ✓ Added POI features")
    
    # Add nearby sectors
    if 'new_house_transactions_nearby_sectors' in tables:
        nearby = tables['new_house_transactions_nearby_sectors']
        nearby_split = split_month_sector(nearby)
        nearby_agg = nearby_split.groupby(['time', 'sector_id']).agg({
            'amount_new_house_transactions_nearby_sectors': ['mean', 'sum', 'std', 'max']
        }).reset_index()
        nearby_agg.columns = ['time', 'sector_id', 'nearby_mean', 'nearby_sum', 'nearby_std', 'nearby_max']
        features_df = features_df.merge(nearby_agg, on=['time', 'sector_id'], how='left')
        print("   ✓ Added nearby sectors features")
    
    # Seasonality features
    features_df['month'] = features_df['month_num']
    features_df['is_december'] = (features_df['month_num'] == 12).astype(int)
    features_df['quarter'] = ((features_df['month_num'] - 1) // 3 + 1)
    
    december_boost_dict = compute_december_boost(amount_wide)
    december_boost = pd.DataFrame(list(december_boost_dict.items()), columns=['sector_id', 'december_boost'])
    features_df = features_df.merge(december_boost, on='sector_id', how='left')
    features_df['december_boost'].fillna(1.3, inplace=True)
    
    print(f"   Total features: {features_df.shape[1]}")
    
    # 3. Prepare train/val
    print("\n3. Preparing train/validation split...")
    exclude_cols = ['time', 'sector_id', 'amount_new_house_transactions', 'month_sector', 'month_num', 
                    'month', 'sector', 'year']  # Exclude all non-numeric columns
    feature_cols = [col for col in features_df.columns if col not in exclude_cols and features_df[col].dtype in ['int64', 'float64']]
    
    train_data = features_df[features_df['amount_new_house_transactions'].notna()].copy()
    train_data['is_zero'] = (train_data['amount_new_house_transactions'] == 0).astype(int)
    
    max_time = train_data['time'].max()
    val_start = max_time - 11
    
    train_mask = train_data['time'] < val_start
    val_mask = train_data['time'] >= val_start
    
    X_train = train_data.loc[train_mask, feature_cols].fillna(-999)
    y_train = train_data.loc[train_mask, 'amount_new_house_transactions']
    y_train_binary = train_data.loc[train_mask, 'is_zero']
    
    X_val = train_data.loc[val_mask, feature_cols].fillna(-999)
    y_val = train_data.loc[val_mask, 'amount_new_house_transactions']
    y_val_binary = train_data.loc[val_mask, 'is_zero']
    
    print(f"   Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"   Zero rate - Train: {y_train_binary.mean():.1%} | Val: {y_val_binary.mean():.1%}")
    
    # 4. Train geometric baseline
    print("\n4. Training Geometric Baseline...")
    val_times = train_data.loc[val_mask, 'time'].unique()
    geo_preds = predict_geometric_baseline(train_data[train_mask], val_times, december_boost_dict)
    
    val_data = train_data[val_mask][['time', 'sector_id', 'amount_new_house_transactions']].merge(
        geo_preds, on=['time', 'sector_id'], how='left'
    )
    geo_val = val_data['pred_geometric'].fillna(0).values
    
    geo_result = competition_score(y_val.values, geo_val)
    print(f"   Score: {geo_result['score']:.4f} | Good rate: {geo_result['good_rate']:.1%}")
    
    # 5. Train zero classifier
    print("\n5. Training Zero Classifier (CatBoost)...")
    
    # Check if we have both classes
    if y_train_binary.nunique() < 2:
        print("   ⚠ No zeros in training data - skipping zero classifier")
        print(f"   Using zero guard from geometric baseline only")
        zero_probs_val = np.zeros(len(X_val))  # Predict no zeros
        zero_clf = None
    else:
        zero_clf = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=4,
            verbose=False,
            random_seed=42
        )
        zero_clf.fit(X_train, y_train_binary, eval_set=(X_val, y_val_binary))
        zero_probs_val = zero_clf.predict_proba(X_val)[:, 1]
        print(f"   Mean zero prob: {zero_probs_val.mean():.3f} | Actual: {y_val_binary.mean():.3f}")
    
    # 6. Train CatBoost regressor
    print("\n6. Training CatBoost Regressor...")
    train_nonzero_mask = y_train > 0
    X_train_nz = X_train[train_nonzero_mask]
    y_train_nz = y_train[train_nonzero_mask]
    
    cb_reg = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        verbose=False,
        random_seed=42
    )
    cb_reg.fit(X_train_nz, y_train_nz)
    cb_preds_val = np.maximum(cb_reg.predict(X_val), 0)
    print(f"   Mean pred: {cb_preds_val.mean():.0f} | Max: {cb_preds_val.max():.0f}")
    
    # 7. Train LightGBM
    print("\n7. Training LightGBM...")
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data_lgb = lgb.Dataset(X_train_nz, y_train_nz)
    lgb_model = lgb.train(lgb_params, train_data_lgb, num_boost_round=300)
    lgb_preds_val = np.maximum(lgb_model.predict(X_val), 0)
    print(f"   Mean pred: {lgb_preds_val.mean():.0f} | Max: {lgb_preds_val.max():.0f}")
    
    # 8. Optimize ensemble weights
    print("\n8. Optimizing ensemble weights...")
    best_score = 0
    best_params = None
    
    for zero_thresh in [0.3, 0.4, 0.5, 0.6]:
        for w_geo in [0.3, 0.4, 0.5, 0.6]:
            for w_cb in [0.2, 0.3, 0.4]:
                w_lgb = 1.0 - w_geo - w_cb
                if w_lgb < 0.1 or w_lgb > 0.5:
                    continue
                
                preds = create_ensemble(
                    geo_val, cb_preds_val, lgb_preds_val,
                    zero_probs_val,
                    (w_geo, w_cb, w_lgb),
                    zero_thresh
                )
                
                result = competition_score(y_val.values, preds)
                
                if result['score'] > best_score:
                    best_score = result['score']
                    best_params = {
                        'w_geo': w_geo,
                        'w_cb': w_cb,
                        'w_lgb': w_lgb,
                        'zero_threshold': zero_thresh,
                        'good_rate': result['good_rate']
                    }
    
    print(f"\n   BEST ENSEMBLE:")
    print(f"   Score: {best_score:.4f}")
    print(f"   Weights: Geo={best_params['w_geo']:.2f}, CB={best_params['w_cb']:.2f}, LGB={best_params['w_lgb']:.2f}")
    print(f"   Zero threshold: {best_params['zero_threshold']:.2f}")
    print(f"   Good rate: {best_params['good_rate']:.1%}")
    
    # 9. Generate test predictions
    print("\n9. Generating test predictions...")
    
    # Prepare test data
    test_df['id_split'] = test_df['id'].str.split('_')
    test_df['month_str'] = test_df['id_split'].str[0]
    test_df['sector_str'] = test_df['id_split'].str[1]
    test_df['sector_id'] = test_df['sector_str'].str.extract('(\d+)').astype(int)
    
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    test_df['month_num'] = test_df['month_str'].map(month_map)
    test_df['time'] = max_time + test_df['month_num']
    
    # Geometric predictions
    test_times = test_df['time'].unique()
    geo_test = predict_geometric_baseline(train_data, test_times, december_boost_dict)
    
    # Build features for test
    # Need to keep month_num and merge with lags
    test_meta = test_df[['time', 'sector_id', 'month_num']].copy()
    
    # Build lag features (requires historical data too)
    all_data_for_lags = pd.concat([
        train_data[['time', 'sector_id', 'amount_new_house_transactions', 'month_num']],
        test_df[['time', 'sector_id', 'month_num']].assign(amount_new_house_transactions=np.nan)
    ])
    
    # Convert back to original format for build_time_lagged_features
    all_data_original = nht.copy()
    test_rows = []
    for _, row in test_df.iterrows():
        test_rows.append({
            'month': row['month_str'] + ' 2024',  # Fake year
            'sector': f"sector {row['sector_id']}",
            'amount_new_house_transactions': np.nan
        })
    all_data_original = pd.concat([all_data_original, pd.DataFrame(test_rows)], ignore_index=True)
    
    lag_features = build_time_lagged_features(all_data_original, lags=[1, 2, 3, 4, 5, 6, 12])
    
    # Filter lag features to only test times
    lag_features = lag_features[lag_features['time'].isin(test_times)].copy()
    
    # Merge with test metadata
    test_features_df = test_meta.merge(lag_features, on=['time', 'sector_id'], how='left')
    
    # Add static features
    if 'sector_POI' in tables:
        test_features_df = join_static_sector_features(test_features_df, tables['sector_POI'])
    
    # Add seasonality
    test_features_df['month'] = test_features_df['month_num']
    test_features_df['is_december'] = (test_features_df['month_num'] == 12).astype(int)
    test_features_df['quarter'] = ((test_features_df['month_num'] - 1) // 3 + 1)
    test_features_df = test_features_df.merge(december_boost, on='sector_id', how='left')
    test_features_df['december_boost'].fillna(1.3, inplace=True)
    
    # Ensure all feature columns
    for col in feature_cols:
        if col not in test_features_df.columns:
            test_features_df[col] = 0
    
    print(f"   Test features shape: {test_features_df.shape}")
    print(f"   Expected: {len(test_df)} rows")
    
    # Verify we have the right number of rows
    assert len(test_features_df) == len(test_df), f"Row count mismatch: {len(test_features_df)} vs {len(test_df)}"
    
    X_test = test_features_df[feature_cols].fillna(-999)
    
    # Predict with all models
    if zero_clf is not None:
        zero_probs_test = zero_clf.predict_proba(X_test)[:, 1]
    else:
        zero_probs_test = np.zeros(len(X_test))
    
    cb_preds_test = np.maximum(cb_reg.predict(X_test), 0)
    lgb_preds_test = np.maximum(lgb_model.predict(X_test), 0)
    
    # Merge geometric predictions
    test_features_df = test_features_df.merge(geo_test, on=['time', 'sector_id'], how='left')
    geo_test_vals = test_features_df['pred_geometric'].fillna(0).values
    
    # Final ensemble
    final_preds_test = create_ensemble(
        geo_test_vals, cb_preds_test, lgb_preds_test,
        zero_probs_test,
        (best_params['w_geo'], best_params['w_cb'], best_params['w_lgb']),
        best_params['zero_threshold']
    )
    
    print(f"   Mean: {final_preds_test.mean():.0f}")
    print(f"   Zero rate: {(final_preds_test == 0).mean():.1%}")
    
    # 10. Save submission
    print("\n10. Saving submission...")
    submission_df = test_df[['id']].copy()
    submission_df['amount_new_house_transactions'] = final_preds_test
    
    ensure_dir('submissions')
    submission_path = 'submissions/ultimate_ensemble.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n{'='*60}")
    print("SUBMISSION CREATED!")
    print(f"{'='*60}")
    print(f"File: {submission_path}")
    print(f"Validation score: {best_score:.4f}")
    print(f"Previous best: 0.56248")
    print(f"Improvement: {best_score - 0.56248:+.4f}")
    print(f"\nExpected public score: ~{best_score:.2f}+ (if validation generalizes)")
    print(f"\nLocal validation:")
    print(f"  Competition score: {best_score:.4f}")
    print(f"  Good rate: {best_params['good_rate']:.1%}")
    print(f"\nTo submit to Kaggle:")
    print(f"kaggle competitions submit -c china-real-estate-demand-prediction \\")
    print(f"  -f {submission_path} -m 'Ultimate ensemble'")

if __name__ == '__main__':
    main()

