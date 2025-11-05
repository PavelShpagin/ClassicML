"""
Ultimate Ensemble - Simplified Version
Uses only geometric baseline with optimized parameters
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import DatasetPaths, load_all_training_tables, load_test, split_month_sector
from src.models import competition_score
from src.utils import build_amount_wide, compute_december_boost, ensure_dir

def main():
    print("="*60)
    print("ULTIMATE ENSEMBLE (Simplified)")
    print("="*60)
    
    # Load data
    paths = DatasetPaths(root_dir='.')
    tables = load_all_training_tables(paths)
    test_df = load_test(paths)
    
    nht = tables['new_house_transactions']
    nht_split = split_month_sector(nht)
    
    # Build wide matrix
    amount_wide = build_amount_wide(nht)
    december_boost_dict = compute_december_boost(amount_wide)
    
    # Use validation split to optimize
    max_time = nht_split['time'].max()
    val_start = max_time - 11
    
    train_mask = nht_split['time'] < val_start
    val_mask = nht_split['time'] >= val_start
    
    train_data = nht_split[train_mask].copy()
    val_data = nht_split[val_mask].copy()
    
    # Use proven best parameters from baseline_seasonality
    print("\nUsing proven best parameters...")
    best_params = {'lookback': 6, 'boost_mult': 1.0}
    
    # Validate on last 12 months
    predictions = []
    for _, row in val_data.iterrows():
        sector = row['sector_id']
        test_time = row['time']
        
        hist = train_data[
            (train_data['sector_id'] == sector) &
            (train_data['time'] < test_time)
        ].sort_values('time')
        
        if len(hist) < best_params['lookback']:
            pred = 0
        else:
            recent = hist.tail(best_params['lookback'])['amount_new_house_transactions'].values
            
            if (recent == 0).any():
                pred = 0
            else:
                pred = np.exp(np.log(recent + 1e-10).mean())
                
                # December boost (applied AFTER geometric mean)
                month = (test_time % 12) if (test_time % 12) != 0 else 12
                if month == 12 and sector in december_boost_dict:
                    pred *= december_boost_dict[sector]
        
        predictions.append(pred)
    
    result = competition_score(val_data['amount_new_house_transactions'].values, np.array(predictions))
    best_score = result['score']
    
    print(f"\nBest parameters:")
    print(f"  Lookback: {best_params['lookback']} months")
    print(f"  December multiplier: {best_params['boost_mult']:.2f}")
    print(f"  Validation score: {best_score:.4f}")
    
    # Generate test predictions (use ALL training data now)
    print("\nGenerating test predictions...")
    test_df['id_split'] = test_df['id'].str.split('_')
    test_df['month_str'] = test_df['id_split'].str[0]
    test_df['sector_str'] = test_df['id_split'].str[1]
    test_df['sector_id'] = test_df['sector_str'].str.extract('(\d+)').astype(int)
    
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    test_df['month_num'] = test_df['month_str'].map(month_map)
    test_df['time'] = max_time + test_df['month_num']
    
    predictions = []
    for _, row in test_df.iterrows():
        sector = row['sector_id']
        test_time = row['time']
        
        # Use ALL training data (not just pre-validation)
        hist = nht_split[
            (nht_split['sector_id'] == sector) &
            (nht_split['time'] <= max_time)  # Changed from < test_time to <= max_time
        ].sort_values('time')
        
        if len(hist) < best_params['lookback']:
            pred = 0
        else:
            recent = hist.tail(best_params['lookback'])['amount_new_house_transactions'].values
            
            if (recent == 0).any():
                pred = 0
            else:
                pred = np.exp(np.log(recent + 1e-10).mean())
                
                # December boost (direct multiplication, no additional multiplier)
                month = row['month_num']
                if month == 12 and sector in december_boost_dict:
                    pred *= december_boost_dict[sector]
        
        predictions.append(pred)
    
    # Save submission
    submission_df = test_df[['id']].copy()
    submission_df['new_house_transaction_amount'] = predictions
    
    Path('submissions').mkdir(parents=True, exist_ok=True)
    submission_path = 'submissions/ultimate_ensemble.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n{'='*60}")
    print("SUBMISSION CREATED!")
    print(f"{'='*60}")
    print(f"File: {submission_path}")
    print(f"Validation score: {best_score:.4f}")
    print(f"Previous best: 0.56248")
    print(f"Improvement: {best_score - 0.56248:+.4f}")
    print(f"\nPrediction stats:")
    print(f"  Mean: {np.mean(predictions):.0f}")
    print(f"  Median: {np.median(predictions):.0f}")
    print(f"  Zero rate: {(np.array(predictions) == 0).mean():.1%}")

if __name__ == '__main__':
    main()

