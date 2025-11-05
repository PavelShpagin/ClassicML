"""
SOTA #5: Optimized December Boost
Use multiple years of December data to predict December boost more accurately
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test
from src.utils import build_amount_wide, geometric_mean_with_zero_guard

def compute_smart_december_boost(amount_wide):
    """Calculate December boost using year-over-year growth patterns"""
    dec_mask = (amount_wide.index.to_series() % 12) == 11
    
    boost = {}
    for sector in range(1, 97):
        if sector not in amount_wide.columns:
            boost[sector] = 1.3
            continue
        
        series = amount_wide[sector].astype(float)
        dec_vals = series[dec_mask]
        non_dec_vals = series[~dec_mask]
        
        # Remove zeros
        dec_vals_nz = dec_vals[dec_vals > 0]
        non_dec_vals_nz = non_dec_vals[non_dec_vals > 0]
        
        if len(dec_vals_nz) >= 2 and len(non_dec_vals_nz) > 0:
            # Use median ratio to be more robust to outliers
            dec_median = dec_vals_nz.median()
            non_dec_median = non_dec_vals_nz.median()
            
            ratio = dec_median / non_dec_median if non_dec_median > 0 else 1.3
            
            # Cap boost more conservatively
            boost[sector] = min(max(ratio, 0.8), 1.7)
        else:
            boost[sector] = 1.3
    
    return boost

def main():
    paths = DatasetPaths(root_dir=str(ROOT))
    train_data = load_all_training_tables(paths)
    train_nht = train_data['new_house_transactions']
    test = load_test(paths)
    
    amount = build_amount_wide(train_nht)
    december_boost_dict = compute_smart_december_boost(amount)
    base_geo = geometric_mean_with_zero_guard(amount, lookback_months=6, zero_guard_window=6)
    
    # Also calculate a "recent trend" multiplier
    trend_multipliers = {}
    for sector in range(1, 97):
        if sector in amount.columns:
            recent = amount[sector].fillna(0).tail(12)
            if len(recent) >= 12:
                first_half = recent.head(6).mean()
                second_half = recent.tail(6).mean()
                if first_half > 0:
                    trend = second_half / first_half
                    trend_multipliers[sector] = np.clip(trend, 0.8, 1.3)
                else:
                    trend_multipliers[sector] = 1.0
            else:
                trend_multipliers[sector] = 1.0
        else:
            trend_multipliers[sector] = 1.0
    
    predictions = []
    for _, row in test.iterrows():
        parts = row['id'].split('_')
        month_str = parts[0]
        month_name = month_str.split(' ')[1]
        sector = int(parts[1].replace('sector ', ''))
        
        pred = float(base_geo[sector])
        
        # Apply trend multiplier for all months
        pred *= trend_multipliers.get(sector, 1.0)
        
        # Apply December boost
        if month_name == 'Dec':
            boost = december_boost_dict.get(sector, 1.3)
            pred *= boost
        
        predictions.append(pred)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'new_house_transaction_amount': predictions
    })
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    submission.to_csv('submissions/sota_december.csv', index=False)
    
    print(f"Mean: {np.mean(predictions):.0f}, Zero rate: {(np.array(predictions) == 0).mean():.1%}")
    print(f"December boost range: {min(december_boost_dict.values()):.2f} - {max(december_boost_dict.values()):.2f}")

if __name__ == '__main__':
    main()


