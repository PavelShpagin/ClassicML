"""
Geometric Mean Baseline - Proven to achieve ~0.45 score
Based on the competition's example solution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test

def main():
    print("=== Geometric Mean Baseline ===")
    
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
    
    # Create wide format (time x sectors)
    amount = train_nht.set_index(['time', 'sector_id'])['amount_new_house_transactions'].unstack()
    amount = amount.fillna(0)
    
    # Add missing sector 95
    if 95 not in amount.columns:
        amount[95] = 0
    amount = amount[[i for i in range(1, 97)]]
    
    print(f"Training data shape: {amount.shape}")
    
    # Parameters from the proven baseline
    t1 = 6  # months for geometric mean
    t2 = 6  # months to check for zeros
    
    # Calculate geometric mean of last t1 months
    last_months = amount.tail(t1)
    
    # Geometric mean (handle zeros by replacing with NaN for mean calculation)
    geo_mean = np.exp(np.log(last_months.replace(0, np.nan)).mean(axis=0, skipna=True))
    geo_mean = geo_mean.fillna(0)
    
    # Zero guard: if any of last t2 months was zero, predict zero
    zero_mask = amount.tail(t2).min(axis=0) == 0
    geo_mean[zero_mask] = 0
    
    print(f"Sectors with zero guard: {zero_mask.sum()}")
    print(f"Mean prediction (non-zero): {geo_mean[geo_mean > 0].mean():.2f}")
    
    # Create predictions for all test months (constant prediction)
    predictions = []
    test_months = [
        '2024 Aug', '2024 Sep', '2024 Oct', '2024 Nov', '2024 Dec',
        '2025 Jan', '2025 Feb', '2025 Mar', '2025 Apr', '2025 May', '2025 Jun', '2025 Jul'
    ]
    
    for _, row in test.iterrows():
        sector = int(row['id'].split('_')[1].replace('sector ', ''))
        pred = geo_mean[sector]
        predictions.append(pred)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'new_house_transaction_amount': predictions
    })
    
    submission.to_csv('submission_geometric.csv', index=False)
    
    print(f"\n✅ Submission saved: submission_geometric.csv")
    print(f"  Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()}/{len(submission)}")
    print(f"  Expected score: ~0.45 (based on competition examples)")
    
    return submission

if __name__ == '__main__':
    main()

