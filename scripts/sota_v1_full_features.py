"""
SOTA V1: ALL FEATURES FROM ALL TABLES
Join ALL available data: land transactions, pre-owned houses, nearby sectors, POI, city indexes
Use CatBoost with proper feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

print("="*70)
print("SOTA V1 - ALL FEATURES FROM ALL DATA TABLES")
print("="*70)

from src.data import DatasetPaths, load_all_training_tables, load_test, split_month_sector
from catboost import CatBoostRegressor
from src.models import competition_score
import itertools

def add_prefix(df, prefix, exclude=['sector', 'month', 'sector_id', 'time']):
    """Add prefix to columns except excluded ones"""
    rename_dict = {col: f"{prefix}_{col}" for col in df.columns if col not in exclude}
    return df.rename(columns=rename_dict)

def main():
    # 1. Load ALL tables
    print("\n1. Loading ALL data tables...")
    paths = DatasetPaths(root_dir=str(ROOT))
    tables = load_all_training_tables(paths)
    test_df = load_test(paths)
    
    # Parse all tables
    nht = split_month_sector(tables['new_house_transactions'])
    nhtns = split_month_sector(tables['new_house_transactions_nearby_sectors'])
    pht = split_month_sector(tables['pre_owned_house_transactions'])
    phtns = split_month_sector(tables['pre_owned_house_transactions_nearby_sectors'])
    lt = split_month_sector(tables['land_transactions'])
    ltns = split_month_sector(tables['land_transactions_nearby_sectors'])
    ci = split_month_sector(tables['city_indexes'])
    poi = tables['sector_POI'].copy()
    poi['sector_id'] = poi['sector'].str.extract(r'(\d+)').astype(int)
    
    print(f"   Tables loaded: {len(tables)}")
    
    # 2. Create full grid and JOIN ALL
    print("\n2. Joining ALL features...")
    times = nht['time'].unique()
    sectors = nht['sector_id'].unique()
    
    grid = pd.DataFrame(list(itertools.product(times, sectors)), columns=['time', 'sector_id'])
    
    # Start with main target
    data = grid.merge(
        nht[['time', 'sector_id', 'amount_new_house_transactions']],
        on=['time', 'sector_id'], how='left'
    )
    data['amount_new_house_transactions'] = data['amount_new_house_transactions'].fillna(0)
    
    # Add nearby new house
    nhtns_agg = nhtns.groupby(['time', 'sector_id']).agg({
        'amount_new_house_transactions_nearby_sectors': ['mean', 'sum', 'max', 'std']
    }).reset_index()
    nhtns_agg.columns = ['time', 'sector_id', 'nhtns_mean', 'nhtns_sum', 'nhtns_max', 'nhtns_std']
    data = data.merge(nhtns_agg, on=['time', 'sector_id'], how='left')
    
    # Add pre-owned houses
    pht_agg = pht.groupby(['time', 'sector_id']).agg({
        'amount_pre_owned_house_transactions': ['mean', 'sum', 'max']
    }).reset_index()
    pht_agg.columns = ['time', 'sector_id', 'pht_mean', 'pht_sum', 'pht_max']
    data = data.merge(pht_agg, on=['time', 'sector_id'], how='left')
    
    # Add pre-owned nearby
    phtns_agg = phtns.groupby(['time', 'sector_id']).agg({
        'amount_pre_owned_house_transactions_nearby_sectors': ['mean', 'sum']
    }).reset_index()
    phtns_agg.columns = ['time', 'sector_id', 'phtns_mean', 'phtns_sum']
    data = data.merge(phtns_agg, on=['time', 'sector_id'], how='left')
    
    # Add land transactions
    lt_agg = lt.groupby(['time', 'sector_id']).agg({
        'amount_land_transactions': ['mean', 'sum', 'max']
    }).reset_index()
    lt_agg.columns = ['time', 'sector_id', 'lt_mean', 'lt_sum', 'lt_max']
    data = data.merge(lt_agg, on=['time', 'sector_id'], how='left')
    
    # Add land nearby
    ltns_agg = ltns.groupby(['time', 'sector_id']).agg({
        'amount_land_transactions_nearby_sectors': ['mean', 'sum']
    }).reset_index()
    ltns_agg.columns = ['time', 'sector_id', 'ltns_mean', 'ltns_sum']
    data = data.merge(ltns_agg, on=['time', 'sector_id'], how='left')
    
    # Add POI (static)
    poi_cols = [col for col in poi.columns if col not in ['sector', 'sector_id']]
    data = data.merge(poi[['sector_id'] + poi_cols], on='sector_id', how='left')
    
    # Add city indexes
    ci_cols = [col for col in ci.columns if col not in ['time', 'sector_id', 'month', 'sector', 'year', 'month_num']]
    data = data.merge(ci[['time'] + ci_cols], on='time', how='left')
    
    print(f"   Total features before engineering: {data.shape[1]}")
    
    # 3. Feature engineering
    print("\n3. Engineering lag and rolling features...")
    
    # Time features
    data['month_num'] = (data['time'] % 12) + 1
    data['year'] = 2019 + (data['time'] // 12)
    data['is_december'] = (data['month_num'] == 12).astype(int)
    data['quarter'] = ((data['month_num'] - 1) // 3 + 1)
    data['season'] = data['month_num'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
    
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
        data[f'roll_max_{window}'] = data.groupby('sector_id')['amount_new_house_transactions'].transform(
            lambda x: x.shift(1).rolling(window).max()
        )
    
    # Exponential weighted mean
    data['ewm_6'] = data.groupby('sector_id')['amount_new_house_transactions'].transform(
        lambda x: x.shift(1).ewm(span=6).mean()
    )
    
    # Growth rate
    data['growth_1'] = data.groupby('sector_id')['amount_new_house_transactions'].pct_change(1)
    data['growth_3'] = data.groupby('sector_id')['amount_new_house_transactions'].pct_change(3)
    
    data = data.fillna(-1)
    
    print(f"   Final features: {data.shape[1]}")
    
    # 4. Train/val split
    print("\n4. Training CatBoost...")
    
    max_time = data['time'].max()
    val_start = max_time - 11
    
    train_mask = data['time'] < val_start
    val_mask = (data['time'] >= val_start) & (data['time'] <= max_time)
    
    exclude_cols = ['time', 'sector_id', 'amount_new_house_transactions', 'year', 'month_num']
    feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in ['int64', 'float64', 'int8', 'int16', 'int32', 'float32']]
    
    X_train = data.loc[train_mask, feature_cols]
    y_train = data.loc[train_mask, 'amount_new_house_transactions']
    X_val = data.loc[val_mask, feature_cols]
    y_val = data.loc[val_mask, 'amount_new_house_transactions']
    
    print(f"   Features: {len(feature_cols)}")
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        verbose=500,
        random_seed=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    
    y_pred = np.maximum(model.predict(X_val), 0)
    result = competition_score(y_val.values, y_pred)
    
    print(f"\n   Validation score: {result['score']:.4f}")
    
    # 5. Test predictions (per month with proper features)
    print("\n5. Generating test predictions...")
    
    test_df['id_split'] = test_df['id'].str.split('_')
    test_df['month_str'] = test_df['id_split'].str[0]
    test_df['sector_str'] = test_df['id_split'].str[1]
    test_df['sector_id'] = test_df['sector_str'].str.extract(r'(\d+)').astype(int)
    
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    test_df['month_str_only'] = test_df['month_str'].str.split().str[1]
    test_df['month_num'] = test_df['month_str_only'].map(month_map)
    test_df['test_time'] = max_time + test_df['month_num']
    
    # Build features for each test row
    predictions = []
    
    for _, test_row in test_df.iterrows():
        sector = test_row['sector_id']
        test_time = test_row['test_time']
        
        # Get historical data for this sector
        hist = data[(data['sector_id'] == sector) & (data['time'] <= max_time)].sort_values('time')
        
        if len(hist) == 0:
            predictions.append(0)
            continue
        
        # Use most recent row as template
        test_feat = hist.iloc[-1][feature_cols].copy()
        
        # Update time-dependent features
        test_feat['month_num'] = test_row['month_num']
        test_feat['is_december'] = int(test_row['month_num'] == 12)
        test_feat['quarter'] = ((test_row['month_num'] - 1) // 3 + 1)
        test_feat['season'] = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}.get(test_row['month_num'], 0)
        
        # Update lags with most recent values
        if len(hist) >= 1:
            test_feat['lag_1'] = hist.iloc[-1]['amount_new_house_transactions']
        if len(hist) >= 2:
            test_feat['lag_2'] = hist.iloc[-2]['amount_new_house_transactions']
        if len(hist) >= 3:
            test_feat['lag_3'] = hist.iloc[-3]['amount_new_house_transactions']
        
        pred = model.predict([test_feat.values])[0]
        predictions.append(max(pred, 0))
    
    # 6. Save
    submission = pd.DataFrame({
        'id': test_df['id'],
        'new_house_transaction_amount': predictions
    })
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    submission.to_csv('submissions/sota_v1.csv', index=False)
    
    print(f"\n{'='*70}")
    print("SOTA V1 COMPLETE!")
    print(f"{'='*70}")
    print(f"Validation: {result['score']:.4f}")
    print(f"Mean pred: {np.mean(predictions):.0f}")
    print(f"Zero rate: {(np.array(predictions) == 0).mean():.1%}")

if __name__ == '__main__':
    main()


