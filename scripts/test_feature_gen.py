import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, split_month_sector
from src.features import build_time_lagged_features

# Load data
paths = DatasetPaths(root_dir=str(ROOT))
train = load_all_training_tables(paths)

# Get base data
base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
print("Base data shape:", base.shape)
print("Time range:", base['time'].min(), "-", base['time'].max())
print("Sectors:", base['sector_id'].nunique())

# Method 1: What the broken script does
print("\n=== Method 1: Direct call (BROKEN) ===")
lag_direct = build_time_lagged_features(base)
lag_t67 = lag_direct[lag_direct['time'] == 67]
print(f"Features for time 67: {len(lag_t67)} rows")

# Method 2: Add placeholder rows first
print("\n=== Method 2: With placeholder rows (SHOULD WORK) ===")
placeholders = pd.DataFrame({
    'time': [67] * 96,
    'sector_id': list(range(1, 97)),
    'amount_new_house_transactions': [np.nan] * 96
})
combined = pd.concat([base, placeholders], ignore_index=True)
print(f"Combined shape: {combined.shape}")

lag_combined = build_time_lagged_features(combined)
lag_t67_fixed = lag_combined[lag_combined['time'] == 67]
print(f"Features for time 67: {len(lag_t67_fixed)} rows")

if len(lag_t67_fixed) > 0:
    print("\nSample features for sector 1, time 67:")
    row = lag_t67_fixed[lag_t67_fixed['sector_id'] == 1].iloc[0]
    for col in ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12']:
        if col in row:
            print(f"  {col}: {row[col]}")

# Check what values we're getting
print("\n=== Checking actual lag values ===")
# Get the actual values that should be lags
sector_1_history = base[base['sector_id'] == 1].sort_values('time')
print("Last 5 values for sector 1:")
print(sector_1_history.tail())

print("\nExpected lag_1 for time 67, sector 1:", 
      sector_1_history[sector_1_history['time'] == 66]['amount_new_house_transactions'].values[0] if len(sector_1_history[sector_1_history['time'] == 66]) > 0 else "N/A")
