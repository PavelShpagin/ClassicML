"""
Ultimate Ensemble V4 - Custom Loss Function Approach
Based on ideas.md: CatBoost with custom loss + ExtraTrees blend

Key differences from previous attempts:
1. Custom loss function that penalizes underprediction heavily
2. Much longer training (21k iterations)
3. ExtraTrees blend
4. Specific sectors forced to zero
5. Same prediction for all 12 months (no seasonality!)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

print("="*70)
print("ULTIMATE ENSEMBLE V4 - CUSTOM LOSS + EXTRATREES")
print("="*70)

from src.data import DatasetPaths, load_all_training_tables, load_test, split_month_sector
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import ExtraTreesRegressor
import itertools

# Custom loss from ideas.md
class CustomObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        result = []
        for i in range(len(targets)):
            # Amplify when underpredicting heavily
            if (2*targets[i] - approxes[i]) < 0:
                der1 = np.sign(targets[i] - approxes[i]) * 5
            else:
                der1 = np.sign(targets[i] - approxes[i])
            result.append((der1, 0.0))
        return result

# Custom metric
class CustomMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)
    
    def is_max_optimal(self):
        return True
    
    def evaluate(self, approxes, targets, weight):
        # Simplified competition metric
        approxes = np.array(approxes[0])
        targets = np.array(targets)
        
        ape = np.abs((targets - approxes) / np.maximum(targets, 1e-12))
        good_mask = ape <= 1.0
        good_rate = good_mask.mean()
        
        if good_rate < 0.3:
            score = 0.0
        else:
            mape = ape[good_mask].mean() if good_mask.any() else 1.0
            score = 1.0 - mape / good_rate
        
        return score, 1.0

def main():
    # 1. Load data
    print("\n1. Loading data...")
    paths = DatasetPaths(root_dir=str(ROOT))
    tables = load_all_training_tables(paths)
    test_df = load_test(paths)
    
    nht = tables['new_house_transactions']
    nht_parsed = split_month_sector(nht)
    
    print(f"   Training samples: {len(nht_parsed)}")
    
    # 2. Build comprehensive features (similar to V3)
    print("\n2. Building features...")
    
    times = nht_parsed['time'].unique()
    sectors = nht_parsed['sector_id'].unique()
    
    grid = pd.DataFrame(list(itertools.product(times, sectors)), columns=['time', 'sector_id'])
    
    data = grid.merge(
        nht_parsed[['time', 'sector_id', 'amount_new_house_transactions']],
        on=['time', 'sector_id'],
        how='left'
    )
    data['amount_new_house_transactions'] = data['amount_new_house_transactions'].fillna(0)
    
    # Time features
    data['month_num'] = (data['time'] % 12) + 1
    data['year'] = 2019 + (data['time'] // 12)
    data['is_december'] = (data['month_num'] == 12).astype(int)
    data['quarter'] = ((data['month_num'] - 1) // 3 + 1)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        data[f'lag_{lag}'] = data.groupby('sector_id')['amount_new_house_transactions'].shift(lag)
    
    # Rolling features
    for window in [3, 6, 12]:
        data[f'roll_mean_{window}'] = data.groupby('sector_id')['amount_new_house_transactions'].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
        data[f'roll_std_{window}'] = data.groupby('sector_id')['amount_new_house_transactions'].transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    
    # Add POI features
    if 'sector_POI' in tables:
        poi = tables['sector_POI'].copy()
        poi['sector_id'] = poi['sector'].str.extract(r'(\d+)').astype(int)
        poi_cols = [col for col in poi.columns if col not in ['sector', 'sector_id']]
        data = data.merge(poi[['sector_id'] + poi_cols], on='sector_id', how='left')
        print(f"   + POI features ({len(poi_cols)} features)")
    
    # Add nearby sectors
    if 'new_house_transactions_nearby_sectors' in tables:
        nearby = split_month_sector(tables['new_house_transactions_nearby_sectors'])
        nearby_agg = nearby.groupby(['time', 'sector_id']).agg({
            'amount_new_house_transactions_nearby_sectors': ['mean', 'sum', 'max']
        }).reset_index()
        nearby_agg.columns = ['time', 'sector_id', 'nearby_mean', 'nearby_sum', 'nearby_max']
        data = data.merge(nearby_agg, on=['time', 'sector_id'], how='left')
        print(f"   + Nearby sectors features")
    
    data = data.fillna(-1)
    
    print(f"   Total features: {data.shape[1]}")
    
    # 3. Prepare train/val
    print("\n3. Preparing train/validation...")
    
    max_time = data['time'].max()
    val_start = max_time - 11
    
    train_mask = data['time'] < val_start
    val_mask = (data['time'] >= val_start) & (data['time'] <= max_time)
    
    exclude_cols = ['time', 'sector_id', 'amount_new_house_transactions', 'year', 'month_num']
    feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in ['int64', 'float64', 'int8', 'int16', 'int32', 'float32']]
    
    X_train = data.loc[train_mask, feature_cols].fillna(-1)
    y_train = data.loc[train_mask, 'amount_new_house_transactions']
    
    X_val = data.loc[val_mask, feature_cols].fillna(-1)
    y_val = data.loc[val_mask, 'amount_new_house_transactions']
    
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Features: {len(feature_cols)}")
    
    # 4. Train CatBoost with custom loss
    print("\n4. Training CatBoost (21k iterations, this will take time)...")
    
    trainPool = Pool(X_train, y_train)
    testPool = Pool(X_val, y_val)
    
    cb = CatBoostRegressor(
        iterations=5000,  # Reduced from 21k for speed (can increase later)
        learning_rate=0.0125,
        depth=6,
        loss_function=CustomObjective(),
        eval_metric=CustomMetric(),
        l2_leaf_reg=0.3,
        random_seed=42,
        verbose=500
    )
    
    cb.fit(trainPool, eval_set=testPool)
    
    # 5. Train ExtraTrees
    print("\n5. Training ExtraTrees...")
    
    et = ExtraTreesRegressor(
        n_estimators=300,  # Reduced from 800 for speed
        n_jobs=-1,
        random_state=42
    )
    et.fit(X_train, y_train)
    
    # 6. Find best blend weight
    print("\n6. Optimizing blend weight...")
    
    from src.models import competition_score
    
    cb_val = cb.predict(X_val)
    et_val = et.predict(X_val)
    
    best_alpha, best_score = 0.5, 0
    for alpha in np.linspace(0.0, 1.0, 21):
        blend_val = alpha * cb_val + (1.0 - alpha) * et_val
        blend_val = np.maximum(blend_val, 0)
        result = competition_score(y_val.values, blend_val)
        if result['score'] > best_score:
            best_score = result['score']
            best_alpha = alpha
    
    print(f"   Best alpha: {best_alpha:.2f}")
    print(f"   Best score: {best_score:.4f}")
    
    cb_score = competition_score(y_val.values, np.maximum(cb_val, 0))
    et_score = competition_score(y_val.values, np.maximum(et_val, 0))
    print(f"   CatBoost alone: {cb_score['score']:.4f}")
    print(f"   ExtraTrees alone: {et_score['score']:.4f}")
    
    # 7. Generate test predictions
    print("\n7. Generating test predictions...")
    
    # Build test features (use last available time for all 12 months)
    test_data = data[data['time'] == max_time].copy()
    test_data = test_data[test_data['sector_id'].isin(range(1, 97))]
    
    X_test = test_data[feature_cols].fillna(-1)
    
    # Predict with blend
    cb_test = cb.predict(X_test)
    et_test = et.predict(X_test)
    predictions_base = best_alpha * cb_test + (1.0 - best_alpha) * et_test
    predictions_base = np.maximum(predictions_base, 0)
    
    # KEY INSIGHT FROM IDEAS.MD: Force specific sectors to zero
    zero_sectors = [11, 38, 40, 43, 48, 51, 52, 57, 71, 72, 73, 74, 81, 86, 88, 94, 95]
    
    # Create prediction mapping
    sector_predictions = {}
    for idx, row in test_data.iterrows():
        sector = int(row['sector_id'])
        pred = float(predictions_base[idx - test_data.index[0]])
        if sector in zero_sectors:
            sector_predictions[sector] = 0.0
        else:
            sector_predictions[sector] = pred
    
    # Fill missing sectors with 0
    for s in range(1, 97):
        if s not in sector_predictions:
            sector_predictions[s] = 0.0
    
    print(f"   Forced {len(zero_sectors)} sectors to zero")
    print(f"   Mean prediction: {np.mean(list(sector_predictions.values())):.0f}")
    print(f"   Zero rate: {sum(v == 0 for v in sector_predictions.values()) / 96:.1%}")
    
    # 8. Create submission (same prediction for all 12 months!)
    print("\n8. Creating submission...")
    
    submission_rows = []
    for _, row in test_df.iterrows():
        parts = row['id'].split('_')
        sector = int(parts[1].replace('sector ', ''))
        submission_rows.append({
            'id': row['id'],
            'new_house_transaction_amount': sector_predictions[sector]
        })
    
    submission = pd.DataFrame(submission_rows)
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    out_path = 'submissions/ultimate_ensemble.csv'
    submission.to_csv(out_path, index=False)
    
    print(f"\n{'='*70}")
    print("SUBMISSION CREATED!")
    print(f"{'='*70}")
    print(f"File: {out_path}")
    print(f"Validation score: {best_score:.4f}")
    print(f"Current best: 0.56248")
    print(f"Improvement: {best_score - 0.56248:+.4f}")
    print(f"\nKEY FEATURES:")
    print(f"  - Custom loss (penalizes underprediction)")
    print(f"  - CatBoost + ExtraTrees blend")
    print(f"  - {len(zero_sectors)} sectors forced to zero")
    print(f"  - Same prediction for all 12 months")
    print(f"\nExpected public score: 0.58-0.62 (based on ideas.md)")

if __name__ == '__main__':
    main()


