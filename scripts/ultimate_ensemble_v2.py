"""
Ultimate Ensemble V2 - Exact copy of baseline_seasonality (our best method)
Score: 0.56248 on public leaderboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def main():
    print("="*60)
    print("ULTIMATE ENSEMBLE V2 - Best Known Method")
    print("="*60)

    # Load data
    from src.data import DatasetPaths, load_all_training_tables, load_test
    from src.utils import (
        build_amount_wide,
        geometric_mean_with_zero_guard,
        compute_december_boost,
        ensure_dir,
    )

    paths = DatasetPaths(root_dir=str(ROOT))
    train_data = load_all_training_tables(paths)
    train_nht = train_data['new_house_transactions']
    test = load_test(paths)

    # Wide matrix: time x sector (1..96)
    amount = build_amount_wide(train_nht)
    print(f"Training matrix shape (time x sector): {amount.shape}")

    # Seasonality and base signal
    december_boost = compute_december_boost(amount, cap=2.0, default=1.3)
    base_geo = geometric_mean_with_zero_guard(amount, lookback_months=6, zero_guard_window=6)

    print(f"\nBase geometric mean predictions:")
    print(f"  Mean: {base_geo.mean():.0f}")
    print(f"  Zero rate: {(base_geo == 0).mean():.1%}")

    # Create predictions with sector-specific December boost
    predictions = []
    for _, row in test.iterrows():
        parts = row['id'].split('_')
        month_str = parts[0]
        month_name = month_str.split(' ')[1]
        sector = int(parts[1].replace('sector ', ''))

        pred = float(base_geo[sector])
        if month_name == 'Dec':
            pred *= december_boost.get(sector, 1.3)
        predictions.append(pred)

    # Write submission
    out_dir = ROOT / 'submissions'
    ensure_dir(out_dir)
    out_path = out_dir / 'ultimate_ensemble.csv'
    submission = pd.DataFrame({
        'id': test['id'],
        'new_house_transaction_amount': predictions
    })
    submission.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("SUBMISSION CREATED!")
    print(f"{'='*60}")
    print(f"File: {out_path}")
    print(f"Rows: {len(submission)}")
    print(f"Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()}")
    print(f"\nPrediction stats:")
    print(f"  Mean: {np.mean(predictions):.0f}")
    print(f"  Median: {np.median(predictions):.0f}")
    print(f"  Zero rate: {(np.array(predictions) == 0).mean():.1%}")
    print(f"\nExpected public score: 0.56248 (proven on leaderboard)")

if __name__ == '__main__':
    main()


