import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id, split_month_sector
from src.features import build_time_lagged_features

# Load data
paths = DatasetPaths(root_dir=str(ROOT))
train = load_all_training_tables(paths)
target_wide, _ = prepare_train_target(train['new_house_transactions'])
test_df = load_test(paths)
test_exploded = explode_test_id(test_df)

# Build features for training data
lag_feats = build_time_lagged_features(train['new_house_transactions']).sort_values(['time','sector_id'])
print("Training lag features shape:", lag_feats.shape)
print("\nLast 5 rows of training lag features:")
print(lag_feats.tail())

# Check what happens for test time 67 (2024 Aug)
base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
print("\n\nBase data for recursive forecasting:")
print("Shape:", base.shape)
print("Time range:", base['time'].min(), "-", base['time'].max())

# Try to build features for time 67
current_series = base.copy()
lag_tmp = build_time_lagged_features(current_series)
print("\n\nLag features after build_time_lagged_features on base:")
print("Shape:", lag_tmp.shape)
print("Time range in lag_tmp:", lag_tmp['time'].min(), "-", lag_tmp['time'].max())

# Filter for time 67
lag_t67 = lag_tmp[lag_tmp['time'] == 67]
print(f"\n\nFeatures for time 67: shape={lag_t67.shape}")
if len(lag_t67) > 0:
    print("First 5 rows:")
    print(lag_t67.head())
    print("\nFeature columns:")
    feature_cols = [c for c in lag_t67.columns if c.startswith('lag_') or c.startswith('roll_')]
    for col in feature_cols:
        print(f"  {col}: {lag_t67[col].iloc[0]:.4f} (non-null: {lag_t67[col].notna().sum()}/{len(lag_t67)})")
else:
    print("NO FEATURES GENERATED FOR TIME 67!")
    
# Check if we can generate features by extending the series
print("\n\n=== Trying to extend series for time 67 ===")
# Add dummy rows for time 67
dummy_t67 = pd.DataFrame({
    'time': [67] * 96,
    'sector_id': list(range(1, 97)),
    'amount_new_house_transactions': [np.nan] * 96
})
extended = pd.concat([base, dummy_t67], ignore_index=True)
lag_extended = build_time_lagged_features(extended)
lag_t67_ext = lag_extended[lag_extended['time'] == 67]
print(f"After extending with dummy rows: shape={lag_t67_ext.shape}")
if len(lag_t67_ext) > 0:
    print("First row features:")
    feature_cols = [c for c in lag_t67_ext.columns if c.startswith('lag_') or c.startswith('roll_')]
    for col in feature_cols[:5]:  # Show first 5 features
        val = lag_t67_ext[col].iloc[0]
        print(f"  {col}: {val if pd.notna(val) else 'NaN'}")
