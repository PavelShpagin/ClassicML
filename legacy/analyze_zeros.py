import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, prepare_train_target

# Load data
paths = DatasetPaths(root_dir=str(ROOT))
train = load_all_training_tables(paths)
target_wide, _ = prepare_train_target(train['new_house_transactions'])

# Analyze zeros
print("=== ZERO ANALYSIS ===")
print(f"Target shape: {target_wide.shape}")

# Flatten all values
all_values = target_wide.values.flatten()
all_values = all_values[~np.isnan(all_values)]

print(f"\nTotal non-NaN values: {len(all_values)}")
print(f"Zero values: {(all_values == 0).sum()} ({100*(all_values == 0).sum()/len(all_values):.1f}%)")
print(f"Non-zero values: {(all_values > 0).sum()} ({100*(all_values > 0).sum()/len(all_values):.1f}%)")

# Distribution of non-zero values
non_zero = all_values[all_values > 0]
print(f"\nNon-zero value statistics:")
print(f"  Mean: {non_zero.mean():.2f}")
print(f"  Median: {np.median(non_zero):.2f}")
print(f"  Std: {non_zero.std():.2f}")
print(f"  Min: {non_zero.min():.2f}")
print(f"  Max: {non_zero.max():.2f}")

# Check by sector
print("\n=== ZEROS BY SECTOR ===")
for col in target_wide.columns[:10]:  # First 10 sectors
    sector_vals = target_wide[col].dropna()
    zero_pct = 100 * (sector_vals == 0).sum() / len(sector_vals)
    print(f"Sector {col}: {zero_pct:.1f}% zeros")

# Check last months (what we use for prediction)
print("\n=== LAST 12 MONTHS ===")
last_12 = target_wide.tail(12)
last_vals = last_12.values.flatten()
last_vals = last_vals[~np.isnan(last_vals)]
print(f"Zero values in last 12 months: {(last_vals == 0).sum()} / {len(last_vals)} ({100*(last_vals == 0).sum()/len(last_vals):.1f}%)")

# Sectors that are zero in last month
last_month = target_wide.iloc[-1]
zero_sectors = [col for col in target_wide.columns if last_month[col] == 0]
print(f"\nSectors with 0 in last month: {len(zero_sectors)} / {len(target_wide.columns)}")
print(f"Sectors: {zero_sectors[:10]}..." if len(zero_sectors) > 10 else f"Sectors: {zero_sectors}")
