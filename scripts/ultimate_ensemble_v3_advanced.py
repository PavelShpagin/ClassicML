"""
Ultimate Ensemble V3 - Advanced ML Approach
Inspired by ideas.md: CatBoost with rich features from all data tables
Target: Beat 0.60+ on public leaderboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def main():
    print("="*70)
    print("ULTIMATE ENSEMBLE V3 - ADVANCED ML WITH ALL FEATURES")
    print("="*70)

    from src.data import DatasetPaths, load_all_training_tables, load_test, split_month_sector
    from catboost import CatBoostRegressor
    from sklearn.model_selection import TimeSeriesSplit
    
    paths = DatasetPaths(root_dir=str(ROOT))
    tables = load_all_training_tables(paths)
    test_df = load_test(paths)
    
    print("\n1. Loading all available data tables...")
    print(f"   Available tables: {list(tables.keys())}")
    
    # Extract main table
    nht = tables['new_house_transactions']
    print(f"   Main table shape: {nht.shape}")
    
    # Parse all tables using existing function
    nht_parsed = split_month_sector(nht)
    
    print("\n2. Building comprehensive feature set...")
    
    # Create base data frame with all time x sector combinations
    times = nht_parsed['time'].unique()
    sectors = nht_parsed['sector_id'].unique()
    
    # Create full grid
    import itertools
    grid = pd.DataFrame(list(itertools.product(times, sectors)), columns=['time', 'sector_id'])
    
    # Merge with main target
    data = grid.merge(
        nht_parsed[['time', 'sector_id', 'amount_new_house_transactions']],
        on=['time', 'sector_id'],
        how='left'
    )
    data['amount_new_house_transactions'].fillna(0, inplace=True)
    
    # Add time features
    data['month_num'] = (data['time'] % 12) + 1
    data['year'] = 2019 + (data['time'] // 12)
    data['is_december'] = (data['month_num'] == 12).astype(int)
    data['quarter'] = ((data['month_num'] - 1) // 3 + 1)
    
    print(f"   Base data shape: {data.shape}")
    
    # Add lag features (1, 2, 12 months)
    for lag in [1, 2, 12]:
        data[f'lag_{lag}'] = data.groupby('sector_id')['amount_new_house_transactions'].shift(lag)
    
    # Add rolling statistics
    for window in [3, 6, 12]:
        data[f'roll_mean_{window}'] = data.groupby('sector_id')['amount_new_house_transactions'].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
        data[f'roll_std_{window}'] = data.groupby('sector_id')['amount_new_house_transactions'].transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    
    # Add nearby sectors features
    if 'new_house_transactions_nearby_sectors' in tables:
        nearby = split_month_sector(tables['new_house_transactions_nearby_sectors'])
        nearby_agg = nearby.groupby(['time', 'sector_id']).agg({
            'amount_new_house_transactions_nearby_sectors': ['mean', 'sum', 'max']
        }).reset_index()
        nearby_agg.columns = ['time', 'sector_id', 'nearby_mean', 'nearby_sum', 'nearby_max']
        data = data.merge(nearby_agg, on=['time', 'sector_id'], how='left')
        print(f"   + Nearby sectors features")
    
    # Add POI features (static sector features)
    if 'sector_POI' in tables:
        poi = tables['sector_POI'].copy()
        poi['sector_id'] = poi['sector'].str.extract(r'(\d+)').astype(int)
        poi_cols = [col for col in poi.columns if col not in ['sector', 'sector_id']]
        data = data.merge(poi[['sector_id'] + poi_cols], on='sector_id', how='left')
        print(f"   + POI features ({len(poi_cols)} features)")
    
    # Add city-level features
    if 'city_indexes' in tables:
        city = split_month_sector(tables['city_indexes'])
        city_features = ['GDP_10k', 'resident_population_10k']
        city_cols = [col for col in city_features if col in city.columns]
        if city_cols:
            data = data.merge(city[['time'] + city_cols], on='time', how='left')
            print(f"   + City index features ({len(city_cols)} features)")
    
    # Fill missing values
    data.fillna(-1, inplace=True)
    
    print(f"   Final feature count: {data.shape[1] - 1} (excluding target)")
    
    print("\n3. Preparing train/validation split...")
    
    # Use last 12 months for validation
    max_time = data['time'].max()
    val_start = max_time - 11
    
    train_mask = data['time'] < val_start
    val_mask = (data['time'] >= val_start) & (data['time'] <= max_time)
    
    # Features to exclude
    exclude_cols = ['time', 'sector_id', 'amount_new_house_transactions', 'year', 'month_num']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X_train = data.loc[train_mask, feature_cols]
    y_train = data.loc[train_mask, 'amount_new_house_transactions']
    
    X_val = data.loc[val_mask, feature_cols]
    y_val = data.loc[val_mask, 'amount_new_house_transactions']
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Features: {len(feature_cols)}")
    
    print("\n4. Training CatBoost model...")
    
    # Train only on non-zero samples for better regression
    train_nonzero = y_train > 0
    X_train_nz = X_train[train_nonzero]
    y_train_nz = y_train[train_nonzero]
    
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        verbose=100,
        random_seed=42
    )
    
    model.fit(X_train_nz, y_train_nz)
    
    # Predict on validation
    y_pred_val = model.predict(X_val)
    y_pred_val = np.maximum(y_pred_val, 0)  # Clip negatives
    
    # Add zero guard: if any recent value is zero, predict zero
    for idx in X_val.index:
        sector = data.loc[idx, 'sector_id']
        if data.loc[idx, 'lag_1'] == 0 or data.loc[idx, 'lag_2'] == 0:
            y_pred_val[idx - X_val.index[0]] = 0
    
    # Calculate competition metric
    from src.models import competition_score
    result = competition_score(y_val.values, y_pred_val)
    
    print(f"\n   Validation Results:")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Good rate: {result['good_rate']:.1%}")
    
    print("\n5. Generating test predictions...")
    
    # Parse test IDs
    test_df['id_split'] = test_df['id'].str.split('_')
    test_df['month_str'] = test_df['id_split'].str[0]
    test_df['sector_str'] = test_df['id_split'].str[1]
    test_df['sector_id'] = test_df['sector_str'].str.extract(r'(\d+)').astype(int)
    
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    test_df['month_num'] = test_df['month_str'].str.split().str[1].map(month_map)
    test_df['time'] = max_time + test_df['month_num']
    
    # Build test features (use last available data for lags)
    test_features_list = []
    
    for _, test_row in test_df.iterrows():
        sector = test_row['sector_id']
        test_time = test_row['time']
        
        # Get historical data for this sector
        hist = data[data['sector_id'] == sector].sort_values('time')
        
        # Create features for this test row
        test_feat = {'sector_id': sector, 'time': test_time}
        test_feat['month_num'] = test_row['month_num']
        test_feat['year'] = 2019 + (test_time // 12)
        test_feat['is_december'] = int(test_feat['month_num'] == 12)
        test_feat['quarter'] = ((test_feat['month_num'] - 1) // 3 + 1)
        
        # Lag features (use most recent values)
        if len(hist) >= 1:
            test_feat['lag_1'] = hist.iloc[-1]['amount_new_house_transactions']
        if len(hist) >= 2:
            test_feat['lag_2'] = hist.iloc[-2]['amount_new_house_transactions']
        if len(hist) >= 12:
            test_feat['lag_12'] = hist.iloc[-12]['amount_new_house_transactions']
        
        # Rolling features
        if len(hist) >= 3:
            test_feat['roll_mean_3'] = hist.tail(3)['amount_new_house_transactions'].mean()
            test_feat['roll_std_3'] = hist.tail(3)['amount_new_house_transactions'].std()
        if len(hist) >= 6:
            test_feat['roll_mean_6'] = hist.tail(6)['amount_new_house_transactions'].mean()
            test_feat['roll_std_6'] = hist.tail(6)['amount_new_house_transactions'].std()
        if len(hist) >= 12:
            test_feat['roll_mean_12'] = hist.tail(12)['amount_new_house_transactions'].mean()
            test_feat['roll_std_12'] = hist.tail(12)['amount_new_house_transactions'].std()
        
        # Static features (POI, etc.) - just copy from training data for this sector
        sector_rows = data[data['sector_id'] == sector]
        if len(sector_rows) > 0:
            sector_static = sector_rows.iloc[-1]
            for col in feature_cols:
                if col not in test_feat:
                    test_feat[col] = sector_static.get(col, -1)
        else:
            # Sector not in training (e.g., sector 95) - fill with defaults
            for col in feature_cols:
                if col not in test_feat:
                    test_feat[col] = -1
        
        test_features_list.append(test_feat)
    
    test_features_df = pd.DataFrame(test_features_list)
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in test_features_df.columns:
            test_features_df[col] = -1
    
    X_test = test_features_df[feature_cols].fillna(-1)
    
    # Predict
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)
    
    # Apply zero guard
    for i, row in test_features_df.iterrows():
        if row.get('lag_1', 0) == 0 or row.get('lag_2', 0) == 0:
            predictions[i] = 0
    
    print(f"   Prediction stats:")
    print(f"     Mean: {predictions.mean():.0f}")
    print(f"     Median: {np.median(predictions):.0f}")
    print(f"     Zero rate: {(predictions == 0).mean():.1%}")
    
    print("\n6. Saving submission...")
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'new_house_transaction_amount': predictions
    })
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    out_path = 'submissions/ultimate_ensemble.csv'
    submission.to_csv(out_path, index=False)
    
    print(f"\n{'='*70}")
    print("SUBMISSION CREATED!")
    print(f"{'='*70}")
    print(f"File: {out_path}")
    print(f"Validation score: {result['score']:.4f}")
    print(f"Current best: 0.56248")
    print(f"Improvement: {result['score'] - 0.56248:+.4f}")
    
    if result['score'] > 0.56248:
        print("\n*** NEW BEST SCORE! Ready to submit! ***")
    else:
        print("\nScore not improved. Consider using baseline_seasonality instead.")

if __name__ == '__main__':
    main()

