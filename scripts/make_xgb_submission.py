import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id, split_month_sector
from src.features import build_time_lagged_features
from src.models import competition_score


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=str, default=str(ROOT))
	parser.add_argument("--out", type=str, default=str(ROOT / "submission.csv"))
	parser.add_argument("--t2", type=int, default=6, help="months to check nonzero in last window")
	args = parser.parse_args()

	paths = DatasetPaths(root_dir=args.root)
	train = load_all_training_tables(paths)
	target_wide, _ = prepare_train_target(train['new_house_transactions'])

	# Zero-guard sectors: if any of last t2 months == 0 → predict 0 for all test months
	last_window = target_wide.tail(args.t2)
	zero_guard_sectors = set(int(c) for c in last_window.columns[(last_window.min(axis=0) == 0)])

	# Feature matrix
	lag_feats = build_time_lagged_features(train['new_house_transactions']).sort_values(['time','sector_id'])
	y_long = target_wide.unstack().reset_index(name='y').rename(columns={'level_0':'sector_id','time':'time'})
	df = lag_feats.merge(y_long, on=['time','sector_id'], how='left')
	df_model = df.dropna(subset=[c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]).copy()
	feature_cols = [c for c in df_model.columns if c.startswith('lag_') or c.startswith('roll_')]

	# TimeSeries CV for quick hyperparams
	folds = [
		(0, 18, 19, 30),
		(0, 30, 31, 42),
		(0, 42, 43, 54),
		(0, 54, 55, 66),
	]
	best = None
	for md in [6, 8]:
		for lr in [0.03, 0.05, 0.1]:
			scores = []
			for (tr_s, tr_e, va_s, va_e) in folds:
				tr_mask = (df_model['time'] >= tr_s) & (df_model['time'] <= tr_e)
				va_mask = (df_model['time'] >= va_s) & (df_model['time'] <= va_e)
				X_tr, y_tr = df_model.loc[tr_mask, feature_cols], df_model.loc[tr_mask, 'y']
				X_va, y_va = df_model.loc[va_mask, feature_cols], df_model.loc[va_mask, 'y']
				model = xgb.XGBRegressor(
					max_depth=md, learning_rate=lr, n_estimators=800,
					subsample=0.8, colsample_bytree=0.8,
					reg_alpha=0.0, reg_lambda=1.0,
					objective='reg:squarederror', random_state=42, tree_method='hist'
				)
				model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
				yhat = model.predict(X_va)
				# non-negativity
				yhat = np.maximum(0, yhat)
				scores.append(competition_score(y_va.values, yhat)['score'])
			score_mean = float(np.mean(scores))
			if (best is None) or (score_mean > best[0]):
				best = (score_mean, {'max_depth': md, 'learning_rate': lr})

	# Train best on all df_model
	params = best[1]
	final = xgb.XGBRegressor(
		max_depth=params['max_depth'], learning_rate=params['learning_rate'], n_estimators=1000,
		subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
		objective='reg:squarederror', random_state=42, tree_method='hist'
	)
	final.fit(df_model[feature_cols], df_model['y'])

	# Recursive forecasting for test
	test_df = load_test(paths)
	test_exploded = explode_test_id(test_df)
	base = split_month_sector(train['new_house_transactions'])[['time','sector_id','amount_new_house_transactions']]
	current_series = base.copy()

	preds = []
	for t in sorted(test_exploded['time'].unique()):
		lag_tmp = build_time_lagged_features(current_series)
		lag_t = lag_tmp[lag_tmp['time'] == t]
		step_df = test_exploded[test_exploded['time'] == t][['id','sector_id','time']].merge(lag_t, on=['time','sector_id'], how='left')
		X_t = step_df[feature_cols].fillna(0)
		yhat_t = final.predict(X_t)
		# non-negativity + zero-guard sectors
		yhat_t = np.maximum(0, yhat_t)
		mask_zero_guard = step_df['sector_id'].astype(int).isin(zero_guard_sectors).values
		yhat_t[mask_zero_guard] = 0.0
		out_t = step_df[['id','sector_id','time']].copy()
		out_t['new_house_transaction_amount'] = yhat_t
		preds.append(out_t[['id','new_house_transaction_amount']])
		# roll forward
		update_t = out_t.rename(columns={'new_house_transaction_amount':'amount_new_house_transactions'})
		current_series = pd.concat([current_series, update_t[['time','sector_id','amount_new_house_transactions']]], ignore_index=True)

	submission = pd.concat(preds, ignore_index=True)
	submission = test_df[['id']].merge(submission, on='id', how='left')
	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	submission.to_csv(args.out, index=False)
	print('Saved submission to', args.out, 'with zero-guard sectors:', len(zero_guard_sectors))


if __name__ == '__main__':
	main()
