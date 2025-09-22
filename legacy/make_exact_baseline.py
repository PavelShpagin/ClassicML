"""
Exact implementation of the baseline from README.
Should achieve ~0.45 score as shown in the example.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test

# Load data
print("Loading data...")
paths = DatasetPaths(root_dir=str(ROOT))
train_nht = load_all_training_tables(paths)['new_house_transactions']
test = load_test(paths)

# Parse month and sector
month_codes = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# Process training data
train_nht['sector_id'] = train_nht['sector'].str.slice(7, None).astype(int)
train_nht['year'] = train_nht['month'].str.slice(0, 4).astype(int)
train_nht['month_num'] = train_nht['month'].str.slice(5, None).map(month_codes)
train_nht['time'] = (train_nht['year'] - 2019) * 12 + train_nht['month_num'] - 1

# Create wide format
amount_new_house_transactions = train_nht.set_index(['time', 'sector_id'])['amount_new_house_transactions'].unstack()
amount_new_house_transactions = amount_new_house_transactions.fillna(0)

# Add sector 95 (missing in training)
amount_new_house_transactions[95] = 0
amount_new_house_transactions = amount_new_house_transactions[[i for i in range(1, 97)]]

print(f"Training data shape: {amount_new_house_transactions.shape}")

# Baseline prediction parameters
t1 = 6  # months for geometric mean
t2 = 6  # months which must be nonzero

# Create predictions
a_tr = amount_new_house_transactions
print("Creating predictions...")

# Calculate geometric mean of last t1 months
last_t1 = a_tr.tail(t1)
# Replace zeros with small value for log calculation
last_t1_safe = last_t1.replace(0, 1e-10)
geom_mean = np.exp(np.log(last_t1_safe).mean(axis=0))

# Create prediction dataframe for all test months
a_pred = pd.DataFrame(
    {time: geom_mean for time in range(67, 79)}
).T

# Apply zero guard: if any of last t2 months was zero, predict zero
zero_mask = a_tr.tail(t2).min(axis=0) == 0
a_pred.loc[:, zero_mask] = 0

print(f"Predictions shape: {a_pred.shape}")
print(f"Zero sectors: {zero_mask.sum()}")

# Create submission
test_id_split = test['id'].str.split('_', expand=True)
test['month'] = test_id_split[0]
test['sector'] = test_id_split[1]
test['sector_id'] = test['sector'].str.slice(7, None).astype(int)

# Map test months to time indices
test_month_map = {
    '2024 Aug': 67, '2024 Sep': 68, '2024 Oct': 69, '2024 Nov': 70, '2024 Dec': 71,
    '2025 Jan': 72, '2025 Feb': 73, '2025 Mar': 74, '2025 Apr': 75,
    '2025 May': 76, '2025 Jun': 77, '2025 Jul': 78
}

# Get predictions for each test row
predictions = []
for _, row in test.iterrows():
    time_idx = test_month_map[row['month']]
    sector_id = row['sector_id']
    pred = a_pred.loc[time_idx, sector_id]
    predictions.append(pred)

test['new_house_transaction_amount'] = predictions

# Save submission
submission = test[['id', 'new_house_transaction_amount']]
submission.to_csv('submission_baseline.csv', index=False)

print("\n✅ Baseline submission created: submission_baseline.csv")
print(f"  Mean: {submission['new_house_transaction_amount'].mean():.2f}")
print(f"  Non-zero: {(submission['new_house_transaction_amount'] > 0).sum()} / {len(submission)}")
print(f"  Expected CV score: ~0.45 (based on README example)")

