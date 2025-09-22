"""
Baseline with geometric mean as shown in the example.
This should achieve ~0.45-0.55 score.
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

# Parameters from the example
t1 = 6  # months for geometric mean
t2 = 6  # months which must be nonzero

# Prepare training data (last part)
a_tr = target_wide

# Create predictions using geometric mean
a_pred = pd.DataFrame(
    {time: np.exp(np.log(a_tr.tail(t1).replace(0, np.nan)).mean(axis=0, skipna=True)) 
     for time in range(67, 79)}
).T

# Set to zero if any of last t2 months was zero
a_pred.loc[:, a_tr.tail(t2).min(axis=0) == 0] = 0

# Fill NaN with 0 (for sectors with all zeros)
a_pred = a_pred.fillna(0)

# Create submission
test_df = load_test(paths)
submission_values = []

for _, row in test_df.iterrows():
    # Parse test id
    parts = row['id'].split('_')
    month_str = parts[0]  # e.g., "2024 Aug"
    sector = int(parts[1].replace('sector ', ''))
    
    # Map month to time index
    month_map = {
        '2024 Aug': 67, '2024 Sep': 68, '2024 Oct': 69, '2024 Nov': 70, '2024 Dec': 71,
        '2025 Jan': 72, '2025 Feb': 73, '2025 Mar': 74, '2025 Apr': 75, 
        '2025 May': 76, '2025 Jun': 77, '2025 Jul': 78
    }
    time_idx = month_map.get(month_str, 67)
    
    # Get prediction
    pred = a_pred.loc[time_idx, sector]
    submission_values.append(pred)

submission = pd.DataFrame({
    'id': test_df['id'],
    'new_house_transaction_amount': submission_values
})

# Save
submission.to_csv('submission_geometric.csv', index=False)

print("Geometric mean baseline created: submission_geometric.csv")
print(f"  Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()} / {len(submission)}")
print(f"  Mean (non-zero): {submission[submission['new_house_transaction_amount'] > 0]['new_house_transaction_amount'].mean():.2f}")
print(f"  Max: {submission['new_house_transaction_amount'].max():.2f}")

