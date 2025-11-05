"""
SOTA #3: Trend-Adjusted Predictions
Detect growth/decline trends and extrapolate forward
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test
from src.utils import build_amount_wide, compute_december_boost

def main():
    paths = DatasetPaths(root_dir=str(ROOT))
    train_data = load_all_training_tables(paths)
    train_nht = train_data['new_house_transactions']
    test = load_test(paths)
    
    amount = build_amount_wide(train_nht)
    december_boost_dict = compute_december_boost(amount, cap=2.0, default=1.3)
    
    predictions = []
    for _, row in test.iterrows():
        parts = row['id'].split('_')
        month_str = parts[0]
        month_name = month_str.split(' ')[1]
        sector = int(parts[1].replace('sector ', ''))
        
        if sector not in amount.columns:
            predictions.append(0.0)
            continue
        
        recent_12 = amount[sector].fillna(0).tail(12)
        recent_6 = recent_12.tail(6)
        
        # Zero guard
        if (recent_6 == 0).any():
            predictions.append(0.0)
            continue
        
        # Base prediction: geometric mean of last 6 months
        base_pred = np.exp(np.log(recent_6 + 1e-10).mean())
        
        # Calculate trend: compare recent 3 months vs previous 3 months
        if len(recent_12) >= 6:
            recent_3 = recent_12.tail(3).mean()
            prev_3 = recent_12.tail(6).head(3).mean()
            
            if prev_3 > 0:
                trend_ratio = recent_3 / prev_3
                # Apply trend adjustment (capped to avoid extreme extrapolation)
                trend_ratio = np.clip(trend_ratio, 0.7, 1.5)
                base_pred *= trend_ratio
        
        # December boost
        if month_name == 'Dec':
            boost = december_boost_dict.get(sector, 1.3)
            base_pred *= min(boost, 1.8)
        
        predictions.append(base_pred)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'new_house_transaction_amount': predictions
    })
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    submission.to_csv('submissions/sota_trend.csv', index=False)
    
    print(f"Mean: {np.mean(predictions):.0f}, Zero rate: {(np.array(predictions) == 0).mean():.1%}")

if __name__ == '__main__':
    main()


