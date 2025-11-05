"""
SOTA #2: Adaptive Lookback Based on Sector Volatility
Use shorter lookback for stable sectors, longer for volatile ones
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
    
    # Calculate sector volatility to determine optimal lookback
    sector_volatility = {}
    sector_lookback = {}
    
    for sector in range(1, 97):
        if sector in amount.columns:
            values = amount[sector].replace(0, np.nan).dropna()
            if len(values) > 6:
                # Coefficient of variation
                cv = values.std() / (values.mean() + 1e-10)
                sector_volatility[sector] = cv
                
                # High volatility → shorter lookback (more responsive)
                # Low volatility → longer lookback (more stable)
                if cv > 1.5:
                    sector_lookback[sector] = 3
                elif cv > 0.8:
                    sector_lookback[sector] = 6
                else:
                    sector_lookback[sector] = 9
            else:
                sector_lookback[sector] = 6
        else:
            sector_lookback[sector] = 6
    
    print(f"Lookback distribution: 3mo={sum(v==3 for v in sector_lookback.values())}, 6mo={sum(v==6 for v in sector_lookback.values())}, 9mo={sum(v==9 for v in sector_lookback.values())}")
    
    # Generate predictions with adaptive lookback
    predictions = []
    for _, row in test.iterrows():
        parts = row['id'].split('_')
        month_str = parts[0]
        month_name = month_str.split(' ')[1]
        sector = int(parts[1].replace('sector ', ''))
        
        lookback = sector_lookback.get(sector, 6)
        
        if sector in amount.columns:
            recent = amount[sector].fillna(0).tail(lookback)
            
            # Zero guard
            if (recent == 0).any():
                pred = 0.0
            else:
                # Geometric mean
                pred = np.exp(np.log(recent + 1e-10).mean())
                
                # December boost
                if month_name == 'Dec':
                    boost = december_boost_dict.get(sector, 1.3)
                    pred *= min(boost, 1.8)
        else:
            pred = 0.0
        
        predictions.append(pred)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'new_house_transaction_amount': predictions
    })
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    submission.to_csv('submissions/sota_adaptive.csv', index=False)
    
    print(f"Mean: {np.mean(predictions):.0f}, Zero rate: {(np.array(predictions) == 0).mean():.1%}")

if __name__ == '__main__':
    main()


