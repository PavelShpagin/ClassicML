"""
Simple conservative submission to get a non-zero score.
Uses median-based predictions with heavy zero detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id, split_month_sector

# Load data
paths = DatasetPaths(root_dir=str(ROOT))
train = load_all_training_tables(paths)
target_wide, _ = prepare_train_target(train['new_house_transactions'])
test_df = load_test(paths)
test_exploded = explode_test_id(test_df)

# Simple approach: use last known values with heavy dampening
base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]

# Get last 6 months for each sector
last_6_months = {}
for sector in range(1, 97):
    sector_data = base[base['sector_id'] == sector].sort_values('time').tail(6)
    if len(sector_data) > 0:
        values = sector_data['amount_new_house_transactions'].values
        # If any zero in last 6 months, predict zero
        if (values == 0).any() or values.mean() < 100:
            last_6_months[sector] = 0
        else:
            # Use dampened median
            last_6_months[sector] = np.median(values) * 0.3  # Heavy dampening
    else:
        # Missing sector - use overall median
        last_6_months[sector] = target_wide.median().median() * 0.2

# Create predictions
predictions = []
for _, row in test_df.iterrows():
    sector = int(row['id'].split('_')[1].replace('sector ', ''))
    pred = last_6_months.get(sector, 0)
    predictions.append(pred)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'new_house_transaction_amount': predictions
})

# Save
submission.to_csv('submission_simple.csv', index=False)

# Stats
print(f"Submission created: submission_simple.csv")
print(f"  Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()} / {len(submission)}")
print(f"  Mean (non-zero): {submission[submission['new_house_transaction_amount'] > 0]['new_house_transaction_amount'].mean():.2f}")
print(f"  Max: {submission['new_house_transaction_amount'].max():.2f}")
