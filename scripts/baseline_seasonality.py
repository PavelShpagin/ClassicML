"""
Geometric Mean with Seasonality - Should achieve ~0.50-0.55 score
Adds December boost based on historical patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test

def main():
    print("=== Geometric Mean + Seasonality Baseline ===")
    
    # Load data
    paths = DatasetPaths(root_dir=str(ROOT))
    train_data = load_all_training_tables(paths)
    train_nht = train_data['new_house_transactions']
    test = load_test(paths)
    
    # Parse month and sector
    month_codes = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    # Process training data
    train_nht['sector_id'] = train_nht['sector'].str.extract('(\d+)').astype(int)
    train_nht['year'] = train_nht['month'].str.split('-').str[0].astype(int)
    train_nht['month_name'] = train_nht['month'].str.split('-').str[1]
    train_nht['month_num'] = train_nht['month_name'].map(month_codes)
    train_nht['time'] = (train_nht['year'] - 2019) * 12 + train_nht['month_num'] - 1
    
    # Create wide format
    amount = train_nht.set_index(['time', 'sector_id'])['amount_new_house_transactions'].unstack()
    amount = amount.fillna(0)
    
    # Add missing sector 95
    if 95 not in amount.columns:
        amount[95] = 0
    amount = amount[[i for i in range(1, 97)]]
    
    print(f"Training data shape: {amount.shape}")
    
    # Calculate December boost factors
    december_times = [11, 23, 35, 47, 59]  # December indices in training
    december_boost = {}
    
    for sector in range(1, 97):
        sector_data = amount[sector].values
        
        # Get December and non-December values
        dec_vals = [sector_data[t] for t in december_times if t < len(sector_data)]
        non_dec_vals = [sector_data[t] for t in range(len(sector_data)) 
                        if t not in december_times and t % 12 != 11]  # Not December
        
        # Remove zeros for calculation
        dec_vals = [v for v in dec_vals if v > 0]
        non_dec_vals = [v for v in non_dec_vals if v > 0]
        
        if dec_vals and non_dec_vals:
            # Boost is ratio of December mean to non-December mean
            boost = np.mean(dec_vals) / np.mean(non_dec_vals)
            december_boost[sector] = min(boost, 2.0)  # Cap at 2x to avoid extreme values
        else:
            december_boost[sector] = 1.3  # Default boost
    
    print(f"Average December boost: {np.mean(list(december_boost.values())):.2f}x")
    
    # Base predictions using geometric mean
    t1 = 6  # months for geometric mean
    t2 = 6  # months to check for zeros
    
    last_months = amount.tail(t1)
    geo_mean = np.exp(np.log(last_months.replace(0, np.nan)).mean(axis=0, skipna=True))
    geo_mean = geo_mean.fillna(0)
    
    # Zero guard
    zero_mask = amount.tail(t2).min(axis=0) == 0
    geo_mean[zero_mask] = 0
    
    # Create predictions with seasonality
    predictions = []
    test_month_map = {
        '2024 Aug': (67, 8), '2024 Sep': (68, 9), '2024 Oct': (69, 10), 
        '2024 Nov': (70, 11), '2024 Dec': (71, 12),  # December!
        '2025 Jan': (72, 1), '2025 Feb': (73, 2), '2025 Mar': (74, 3), 
        '2025 Apr': (75, 4), '2025 May': (76, 5), '2025 Jun': (77, 6), '2025 Jul': (78, 7)
    }
    
    for _, row in test.iterrows():
        parts = row['id'].split('_')
        month_str = parts[0]
        sector = int(parts[1].replace('sector ', ''))
        
        # Base prediction
        pred = geo_mean[sector]
        
        # Apply December boost
        _, month_num = test_month_map[month_str]
        if month_num == 12 and sector in december_boost:
            pred *= december_boost[sector]
        
        predictions.append(pred)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'new_house_transaction_amount': predictions
    })
    
    submission.to_csv('submission_seasonality.csv', index=False)
    
    print(f"\n✅ Submission saved: submission_seasonality.csv")
    print(f"  Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()}/{len(submission)}")
    print(f"  December predictions boosted")
    print(f"  Expected score: ~0.50-0.55")
    
    return submission

if __name__ == '__main__':
    main()

