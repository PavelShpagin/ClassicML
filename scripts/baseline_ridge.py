import argparse
from pathlib import Path
import numpy as np
import pandas as pd  # type: ignore

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id, split_month_sector
from src.features import build_time_lagged_features
from src.models import competition_score, build_linear_pipeline


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=str, default=str(ROOT))
	parser.add_argument("--out", type=str, default=str(ROOT / "submissions" / "baseline_ridge.csv"))
	args = parser.parse_args()

	paths = DatasetPaths(root_dir=args.root)
	train = load_all_training_tables(paths)
	target_wide, _ = prepare_train_target(train["new_house_transactions"])

	lag_feats = build_time_lagged_features(train['new_house_transactions'])
	lag_feats = lag_feats.sort_values(['time', 'sector_id'])

	y_long = target_wide.unstack().reset_index(name='y').rename(columns={'level_0':'sector_id','time':'time'})
	df = lag_feats.merge(y_long, on=['time','sector_id'], how='left')
	df_model = df.dropna(subset=[c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]).copy()
	feature_cols = [c for c in df_model.columns if c.startswith('lag_') or c.startswith('roll_')]
	X = df_model[feature_cols]
	y = df_model['y']

	alphas = np.logspace(-3, 2, 10)
	best = None
	mask_train = df_model['time'] <= 54
	X_tr, y_tr = X[mask_train], y[mask_train]
	X_va, y_va = X[~mask_train], y[~mask_train]
	for a in alphas:
		pipe = build_linear_pipeline(alpha=float(a), kind='ridge')
		pipe.fit(X_tr, y_tr)
		yhat = pipe.predict(X_va)
		sc = competition_score(y_va.values, yhat)['score']
		if best is None or sc > best[0]:
			best = (sc, a)

	best_alpha = float(best[1]) if best else 1.0
	pipe = build_linear_pipeline(alpha=best_alpha, kind='ridge')
	pipe.fit(X, y)

	# Recursive forecasting over the 12-month horizon using clean base series
	test_df = load_test(paths)
	test_exploded = explode_test_id(test_df)

	base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
	current_series = base.copy()

	predictions = []
	for t in sorted(test_exploded['time'].unique()):
		lag_tmp = build_time_lagged_features(current_series)
		lag_t = lag_tmp[lag_tmp['time'] == t]
		step_df = test_exploded[test_exploded['time'] == t][['id','sector_id','time']].merge(lag_t, on=['time','sector_id'], how='left')
		X_t = step_df[feature_cols].fillna(0)
		yhat_t = pipe.predict(X_t)
		out_t = step_df[['id','sector_id','time']].copy()
		out_t['new_house_transaction_amount'] = yhat_t
		predictions.append(out_t[['id','new_house_transaction_amount']])

	# Append predictions for next-step lag computation
		update_t = out_t.rename(columns={'new_house_transaction_amount':'amount_new_house_transactions'})
		current_series = pd.concat([current_series, update_t[['time','sector_id','amount_new_house_transactions']]], ignore_index=True)

	submission = pd.concat(predictions, ignore_index=True)
	submission = test_df[['id']].merge(submission, on='id', how='left')
	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	submission.to_csv(args.out, index=False)
	print(f"Saved submission to {args.out} with {len(submission)} rows")


if __name__ == "__main__":
	main()

