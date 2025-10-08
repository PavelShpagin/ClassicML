"""
Simple conservative baseline using dampened median with zero guard.
Outputs a CSV to the submissions/ directory.
"""

import pandas as pd  # type: ignore
import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id, split_month_sector
from src.utils import ensure_dir


def main() -> None:
	# Load data
	paths = DatasetPaths(root_dir=str(ROOT))
	train = load_all_training_tables(paths)
	target_wide, _ = prepare_train_target(train['new_house_transactions'])
	test_df = load_test(paths)
	_ = explode_test_id(test_df)

	# Simple approach: use last known values with heavy dampening
	base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]

	# Get last 6 months for each sector
	last_6_months: dict[int, float] = {}
	for sector in range(1, 97):
		sector_data = base[base['sector_id'] == sector].sort_values('time').tail(6)
		if len(sector_data) > 0:
			values = sector_data['amount_new_house_transactions'].values
			# If any zero in last 6 months, predict zero
			if (values == 0).any() or values.mean() < 100:
				last_6_months[sector] = 0.0
			else:
				# Use dampened median
				last_6_months[sector] = float(np.median(values) * 0.3)
		else:
			# Missing sector - use overall median
			last_6_months[sector] = float(target_wide.median().median() * 0.2)

	# Create predictions
	predictions: list[float] = []
	for _, row in test_df.iterrows():
		sector = int(row['id'].split('_')[1].replace('sector ', ''))
		pred = float(last_6_months.get(sector, 0.0))
		predictions.append(pred)

	out_dir = ROOT / 'submissions'
	ensure_dir(out_dir)
	out_path = out_dir / 'baseline_simple.csv'
	submission = pd.DataFrame({'id': test_df['id'], 'new_house_transaction_amount': predictions})
	submission.to_csv(out_path, index=False)

	print(f"Saved: {out_path}")
	print(f"Rows: {len(submission)} | Non-zero: {(submission['new_house_transaction_amount'] > 0).sum()}")


if __name__ == '__main__':
	main()



