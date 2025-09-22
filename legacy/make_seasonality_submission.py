"""
Submission with December seasonality bump.
Based on the observation that December has peaks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target

# Load data
paths = DatasetPaths(root_dir=str(ROOT))
train = load_all_training_tables(paths)
target_wide, _ = prepare_train_target(train['new_house_transactions'])

# Add missing sector 95
target_wide[95] = 0
target_wide = target_wide[[i for i in range(1, 97)]]

# Calculate December boost factor for each sector
december_months = [11, 23, 35, 47, 59]  # December indices in training
december_boost = {}

for sector in range(1, 97):
    sector_data = target_wide[sector].values
    
    # Get December values and non-December values
    dec_values = [sector_data[i] for i in december_months if i < len(sector_data) and sector_data[i] > 0]
    non_dec_values = [sector_data[i] for i in range(len(sector_data)) 
                      if i not in december_months and sector_data[i] > 0]
    
    if dec_values and non_dec_values:
        # Calculate boost factor as ratio of December mean to non-December mean
        december_boost[sector] = np.mean(dec_values) / np.mean(non_dec_values)
    else:
        december_boost[sector] = 1.2  # Default boost

# Base predictions using geometric mean
t1 = 6
t2 = 6
a_tr = target_wide

# Geometric mean predictions
base_pred = pd.DataFrame(
    {time: np.exp(np.log(a_tr.tail(t1).replace(0, np.nan)).mean(axis=0, skipna=True)) 
     for time in range(67, 79)}
).T

# Zero guard
base_pred.loc[:, a_tr.tail(t2).min(axis=0) == 0] = 0
base_pred = base_pred.fillna(0)

# Apply December boost
december_test_month = 71  # December 2024 in test set
for sector in range(1, 97):
    if sector in december_boost:
        base_pred.loc[december_test_month, sector] *= december_boost[sector]

# Create submission
test_df = load_test(paths)
submission_values = []

month_map = {
    '2024 Aug': 67, '2024 Sep': 68, '2024 Oct': 69, '2024 Nov': 70, '2024 Dec': 71,
    '2025 Jan': 72, '2025 Feb': 73, '2025 Mar': 74, '2025 Apr': 75, 
    '2025 May': 76, '2025 Jun': 77, '2025 Jul': 78
}

for _, row in test_df.iterrows():
    parts = row['id'].split('_')
    month_str = parts[0]
    sector = int(parts[1].replace('sector ', ''))
    time_idx = month_map.get(month_str, 67)
    
    pred = base_pred.loc[time_idx, sector]
    submission_values.append(pred)

submission = pd.DataFrame({
    'id': test_df['id'],
    'new_house_transaction_amount': submission_values
})

submission.to_csv('submission_seasonality.csv', index=False)

print("Seasonality-aware submission created: submission_seasonality.csv")
print(f"  Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()} / {len(submission)}")
print(f"  Mean (non-zero): {submission[submission['new_house_transaction_amount'] > 0]['new_house_transaction_amount'].mean():.2f}")
print(f"  December boost applied to test month 71")

