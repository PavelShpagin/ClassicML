"""
SOTA #4: Ultra Conservative Blend
Blend our best method (0.56248) with median to reduce risk
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test
from src.utils import build_amount_wide, compute_december_boost, geometric_mean_with_zero_guard

def main():
    paths = DatasetPaths(root_dir=str(ROOT))
    train_data = load_all_training_tables(paths)
    train_nht = train_data['new_house_transactions']
    test = load_test(paths)
    
    amount = build_amount_wide(train_nht)
    december_boost_dict = compute_december_boost(amount, cap=2.0, default=1.3)
    base_geo = geometric_mean_with_zero_guard(amount, lookback_months=6, zero_guard_window=6)
    
    # Also calculate median and mean as conservative anchors
    base_median = amount.fillna(0).tail(6).median(axis=0)
    base_mean = amount.fillna(0).tail(6).mean(axis=0)
    
    predictions = []
    for _, row in test.iterrows():
        parts = row['id'].split('_')
        month_str = parts[0]
        month_name = month_str.split(' ')[1]
        sector = int(parts[1].replace('sector ', ''))
        
        geo = float(base_geo[sector])
        median = float(base_median[sector]) if sector in base_median.index else 0
        mean = float(base_mean[sector]) if sector in base_mean.index else 0
        
        # Conservative blend: 70% geometric (proven), 20% median, 10% mean
        pred = 0.70 * geo + 0.20 * median + 0.10 * mean
        
        # December boost (but more conservative)
        if month_name == 'Dec':
            boost = december_boost_dict.get(sector, 1.3)
            # Very conservative boost cap
            boost = min(boost, 1.5)
            pred *= boost
        
        predictions.append(pred)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'new_house_transaction_amount': predictions
    })
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    submission.to_csv('submissions/sota_conservative.csv', index=False)
    
    print(f"Mean: {np.mean(predictions):.0f}, Zero rate: {(np.array(predictions) == 0).mean():.1%}")

if __name__ == '__main__':
    main()


