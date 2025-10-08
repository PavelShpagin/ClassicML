"""
Geometric Mean with Seasonality baseline.
Uses geometric mean with a zero guard and a December boost.
Outputs a CSV to the submissions/ directory.
"""

import pandas as pd  # type: ignore
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def main():
    print("Geometric Mean + Seasonality baseline")

    # Load data
    from src.data import DatasetPaths, load_all_training_tables, load_test
    from src.utils import (
        build_amount_wide,
        geometric_mean_with_zero_guard,
        compute_december_boost,
        ensure_dir,
    )
    import numpy as np  # local import for metric calc

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

    # Local validation on last 12 months (approximate competition metric)
    try:
        times = amount.index.to_list()
        if len(times) >= 18:
            ape_all = []
            last_12_times = times[-12:]
            for t in last_12_times:
                # Build base from prior 6 months
                hist = amount[amount.index < t]
                last6 = hist.tail(6)
                if last6.shape[0] == 0:
                    continue
                geo_t = np.exp(np.log(last6.replace(0, np.nan)).mean(axis=0, skipna=True)).fillna(0)
                zero_mask_t = last6.min(axis=0) == 0
                geo_t[zero_mask_t] = 0
                # Seasonality
                if (t % 12) == 11:
                    for s in geo_t.index:
                        geo_t[s] = float(geo_t[s]) * float(december_boost.get(int(s), 1.3))
                # Actuals at t
                actual_t = amount.loc[t]
                ape_t = (actual_t - geo_t).abs() / actual_t.clip(lower=1)
                ape_all.append(ape_t.values)
            if ape_all:
                ape = np.concatenate(ape_all)
                good_mask = ape <= 1.0
                good_rate = float(good_mask.mean())
                if good_rate < 0.3:
                    local_score = 0.0
                else:
                    good_ape = ape[good_mask]
                    local_score = float(good_rate * (1.0 / (1.0 + good_ape)).mean())
                print(f"Local validation (last 12 months): score={local_score:.4f} good_rate={good_rate:.4f} n={ape.size}")
    except Exception as e:
        print(f"Local validation skipped: {e}")

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

    # Write submission to standardized directory
    out_dir = ROOT / 'submissions'
    ensure_dir(out_dir)
    out_path = out_dir / 'baseline_seasonality.csv'
    submission = pd.DataFrame({'id': test['id'], 'new_house_transaction_amount': predictions})
    submission.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(submission)} | Non-zero: {(submission['new_house_transaction_amount'] > 0).sum()}")
    return submission

if __name__ == '__main__':
    main()

