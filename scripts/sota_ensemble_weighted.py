"""
SOTA: Weighted Ensemble of Best Methods
Combine baseline_seasonality (0.56248) with improved variants
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

print("="*70)
print("SOTA - WEIGHTED ENSEMBLE OF BEST METHODS")
print("="*70)

from src.data import DatasetPaths, load_all_training_tables, load_test
from src.utils import build_amount_wide, compute_december_boost, geometric_mean_with_zero_guard

def main():
    paths = DatasetPaths(root_dir=str(ROOT))
    train_data = load_all_training_tables(paths)
    train_nht = train_data['new_house_transactions']
    test = load_test(paths)
    
    amount = build_amount_wide(train_nht)
    december_boost_dict = compute_december_boost(amount, cap=2.0, default=1.3)
    
    # Method 1: Geometric mean 6-month
    base_geo_6 = geometric_mean_with_zero_guard(amount, lookback_months=6, zero_guard_window=6)
    
    # Method 2: Geometric mean 3-month (more responsive)
    base_geo_3 = geometric_mean_with_zero_guard(amount, lookback_months=3, zero_guard_window=3)
    
    # Method 3: Geometric mean 9-month (more stable)
    base_geo_9 = geometric_mean_with_zero_guard(amount, lookback_months=9, zero_guard_window=9)
    
    # Method 4: Exponential weighted mean
    base_ewm = amount.fillna(0).ewm(span=6, axis=0).mean().iloc[-1]
    base_ewm[base_ewm < 0] = 0
    
    print(f"\nMethod means: Geo6={base_geo_6.mean():.0f}, Geo3={base_geo_3.mean():.0f}, Geo9={base_geo_9.mean():.0f}, EWM={base_ewm.mean():.0f}")
    
    # Create ensemble predictions with month-specific weights
    predictions = []
    for _, row in test.iterrows():
        parts = row['id'].split('_')
        month_str = parts[0]
        month_name = month_str.split(' ')[1]
        sector = int(parts[1].replace('sector ', ''))
        
        # Get all method predictions
        geo6 = float(base_geo_6[sector])
        geo3 = float(base_geo_3[sector])
        geo9 = float(base_geo_9[sector])
        ewm = float(base_ewm[sector])
        
        # Weighted ensemble (empirically tuned)
        # Geo6 is proven best, give it most weight
        # Geo3 catches recent trends
        # Geo9 provides stability
        # EWM smooths volatility
        base_pred = 0.50 * geo6 + 0.25 * geo3 + 0.15 * geo9 + 0.10 * ewm
        
        # Apply December boost
        if month_name == 'Dec':
            boost = december_boost_dict.get(sector, 1.3)
            # Be more conservative with boost
            boost = min(boost, 1.8)
            base_pred *= boost
        
        # Month-specific adjustments (learned from patterns)
        month_multipliers = {
            'Jan': 0.95,  # Post-holiday slowdown
            'Feb': 0.90,  # Chinese New Year
            'Mar': 1.05,  # Spring recovery
            'Apr': 1.00,
            'May': 1.00,
            'Jun': 1.05,  # Summer boost
            'Jul': 1.00,
            'Aug': 1.00,
            'Sep': 1.00,
            'Oct': 1.00,
            'Nov': 1.05,  # Pre-holiday
            'Dec': 1.00   # Already boosted above
        }
        
        if month_name != 'Dec':  # Don't double-apply for December
            base_pred *= month_multipliers.get(month_name, 1.0)
        
        predictions.append(base_pred)
    
    # Save
    submission = pd.DataFrame({
        'id': test['id'],
        'new_house_transaction_amount': predictions
    })
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    submission.to_csv('submissions/sota_ensemble.csv', index=False)
    
    print(f"\n{'='*70}")
    print("SOTA ENSEMBLE COMPLETE!")
    print(f"{'='*70}")
    print(f"Mean prediction: {np.mean(predictions):.0f}")
    print(f"Median: {np.median(predictions):.0f}")
    print(f"Zero rate: {(np.array(predictions) == 0).mean():.1%}")
    print(f"\nStrategy: 50% Geo6 + 25% Geo3 + 15% Geo9 + 10% EWM")
    print(f"Plus: Month-specific multipliers + Conservative December boost")

if __name__ == '__main__':
    main()


