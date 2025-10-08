"""
Geometric Mean baseline with zero guard.
Outputs a CSV to the submissions/ directory.
"""

import pandas as pd  # type: ignore
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test
from src.utils import build_amount_wide, geometric_mean_with_zero_guard, ensure_dir

def main():
    print("Geometric Mean baseline")

    # Load data
    paths = DatasetPaths(root_dir=str(ROOT))
    train_data = load_all_training_tables(paths)
    train_nht = train_data['new_house_transactions']
    test = load_test(paths)

    amount = build_amount_wide(train_nht)
    print(f"Training matrix shape (time x sector): {amount.shape}")

    geo = geometric_mean_with_zero_guard(amount, lookback_months=6, zero_guard_window=6)

    predictions = []
    for _, row in test.iterrows():
        sector = int(row['id'].split('_')[1].replace('sector ', ''))
        predictions.append(float(geo[sector]))

    out_dir = ROOT / 'submissions'
    ensure_dir(out_dir)
    out_path = out_dir / 'baseline_geometric.csv'
    submission = pd.DataFrame({'id': test['id'], 'new_house_transaction_amount': predictions})
    submission.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(submission)} | Non-zero: {(submission['new_house_transaction_amount'] > 0).sum()}")
    return submission

if __name__ == '__main__':
    main()

