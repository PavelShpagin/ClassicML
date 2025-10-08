import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, prepare_train_target
from src.features import build_time_lagged_features
from src.models import competition_score, build_linear_pipeline
from sklearn.metrics import mean_squared_error


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def main() -> None:
	print('Benchmark: start')
	paths = DatasetPaths(root_dir=str(ROOT))
	reports_dir = ROOT / 'reports'
	ensure_dir(reports_dir)
	(reports_dir / '_marker.txt').write_text('benchmark started')

	# Load data
	train = load_all_training_tables(paths)
	print('Benchmark: data loaded')

	# Build supervised dataset
	lag_feats = build_time_lagged_features(train['new_house_transactions']).sort_values(['time','sector_id'])
	print('Benchmark: features built', len(lag_feats))
	target_wide, _ = prepare_train_target(train['new_house_transactions'])
	y_long = target_wide.unstack().reset_index(name='y').rename(columns={'level_0':'sector_id','time':'time'})
	df = lag_feats.merge(y_long, on=['time','sector_id'], how='left')
	df_model = df.dropna(subset=[c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]).copy()
	print('Benchmark: model rows', len(df_model))
	feature_cols = [c for c in df_model.columns if c.startswith('lag_') or c.startswith('roll_')]
	X = df_model[feature_cols]
	y = df_model['y']

	# TimeSeries CV folds on unique times (mirrors competition period)
	unique_times = np.sort(df_model['time'].unique())
	folds = [
		(0, 18, 19, 30),
		(0, 30, 31, 42),
		(0, 42, 43, 54),
		(0, 54, 55, 66),
	]

	def cv_scores_for_model(mdl) -> dict:
		score_list, rmse_list, mape_list, good_list = [], [], [], []
		for tr_start, tr_end, va_start, va_end in folds:
			tr_mask = (df_model['time'] >= tr_start) & (df_model['time'] <= tr_end)
			va_mask = (df_model['time'] >= va_start) & (df_model['time'] <= va_end)
			X_tr, y_tr = X[tr_mask], y[tr_mask]
			X_va, y_va = X[va_mask], y[va_mask]
			mdl.fit(X_tr, y_tr)
			yhat = mdl.predict(X_va)
			sc = competition_score(y_va.values, yhat)
			score_list.append(sc['score'])
			good_list.append(sc['good_rate'])
			rmse_list.append(float(np.sqrt(mean_squared_error(y_va, yhat))))
			mape_list.append(float(np.mean(np.abs((y_va.values - yhat) / np.maximum(y_va.values, 1e-12)))))
		return {
			'score_mean': float(np.mean(score_list)),
			'score_std': float(np.std(score_list)),
			'good_rate_mean': float(np.mean(good_list)),
			'rmse_mean': float(np.mean(rmse_list)),
			'mape_mean': float(np.mean(mape_list)),
		}

	rows = []
	# Ridge curve with CV
	alphas = np.logspace(-3, 2, 8)
	for a in alphas:
		res = cv_scores_for_model(build_linear_pipeline(alpha=float(a), kind='ridge'))
		rows.append({'model':'ridge','alpha':float(a), **res})
	ridge_df = pd.DataFrame(rows)
	ridge_df.to_csv(reports_dir / 'cv_ridge_results.csv', index=False)
	print('Benchmark: wrote cv_ridge_results.csv with', len(ridge_df), 'rows')
	best_row = ridge_df.sort_values('score_mean', ascending=False).iloc[0]
	(reports_dir / 'best_ridge.txt').write_text(str(best_row.to_dict()))
	print('Best ridge (CV):', best_row.to_dict())

    # Optional tree models
	adv_rows = []
	try:
		import lightgbm as lgb
		for nl in [31, 63]:
			for lr in [0.05, 0.1]:
				p = {'num_leaves': nl, 'learning_rate': lr, 'min_data_in_leaf': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'objective': 'regression', 'metric': 'rmse', 'seed': 42}
				# CV for LightGBM
				score_list, rmse_list, mape_list, good_list = [], [], [], []
				for tr_start, tr_end, va_start, va_end in folds:
					tr_mask = (df_model['time'] >= tr_start) & (df_model['time'] <= tr_end)
					va_mask = (df_model['time'] >= va_start) & (df_model['time'] <= va_end)
					dtrain = lgb.Dataset(X[tr_mask], label=y[tr_mask])
					dvalid = lgb.Dataset(X[va_mask], label=y[va_mask], reference=dtrain)
					booster = lgb.train(p, dtrain, num_boost_round=400, valid_sets=[dvalid], verbose_eval=False, early_stopping_rounds=30)
					yhat = booster.predict(X[va_mask], num_iteration=booster.best_iteration)
					sc = competition_score(y[va_mask].values, yhat)
					score_list.append(sc['score'])
					good_list.append(sc['good_rate'])
					rmse_list.append(float(np.sqrt(mean_squared_error(y[va_mask], yhat))))
					mape_list.append(float(np.mean(np.abs((y[va_mask].values - yhat) / np.maximum(y[va_mask].values, 1e-12)))))
				adv_rows.append({'model':'lightgbm','params':str(p),'score_mean':float(np.mean(score_list)),'good_rate_mean':float(np.mean(good_list)),'rmse_mean':float(np.mean(rmse_list)),'mape_mean':float(np.mean(mape_list))})
	except Exception as e:
		print('LightGBM skipped:', e)

	try:
		import xgboost as xgb
		for md in [6, 8]:
			for lr in [0.05, 0.1]:
				# CV for XGBoost
				score_list, rmse_list, mape_list, good_list = [], [], [], []
				for tr_start, tr_end, va_start, va_end in folds:
					tr_mask = (df_model['time'] >= tr_start) & (df_model['time'] <= tr_end)
					va_mask = (df_model['time'] >= va_start) & (df_model['time'] <= va_end)
					model = xgb.XGBRegressor(max_depth=md, learning_rate=lr, n_estimators=400, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=42, tree_method='hist')
					model.fit(X[tr_mask], y[tr_mask], eval_set=[(X[va_mask], y[va_mask])], verbose=False)
					yhat = model.predict(X[va_mask])
					sc = competition_score(y[va_mask].values, yhat)
					score_list.append(sc['score'])
					good_list.append(sc['good_rate'])
					rmse_list.append(float(np.sqrt(mean_squared_error(y[va_mask], yhat))))
					mape_list.append(float(np.mean(np.abs((y[va_mask].values - yhat) / np.maximum(y[va_mask].values, 1e-12)))))
				adv_rows.append({'model':'xgboost','params':str({'max_depth':md,'lr':lr}),'score_mean':float(np.mean(score_list)),'good_rate_mean':float(np.mean(good_list)),'rmse_mean':float(np.mean(rmse_list)),'mape_mean':float(np.mean(mape_list))})
	except Exception as e:
		print('XGBoost skipped:', e)

	try:
		from catboost import CatBoostRegressor
		for depth in [6, 8]:
			# CV for CatBoost
			score_list, rmse_list, mape_list, good_list = [], [], [], []
			for tr_start, tr_end, va_start, va_end in folds:
				tr_mask = (df_model['time'] >= tr_start) & (df_model['time'] <= tr_end)
				va_mask = (df_model['time'] >= va_start) & (df_model['time'] <= va_end)
				model = CatBoostRegressor(depth=depth, learning_rate=0.1, loss_function='RMSE', random_seed=42, iterations=800, verbose=False)
				model.fit(X[tr_mask], y[tr_mask], eval_set=(X[va_mask], y[va_mask]))
				yhat = model.predict(X[va_mask])
				sc = competition_score(y[va_mask].values, yhat)
				score_list.append(sc['score'])
				good_list.append(sc['good_rate'])
				rmse_list.append(float(np.sqrt(mean_squared_error(y[va_mask], yhat))))
				mape_list.append(float(np.mean(np.abs((y[va_mask].values - yhat) / np.maximum(y[va_mask].values, 1e-12)))))
			adv_rows.append({'model':'catboost','params':str({'depth':depth}),'score_mean':float(np.mean(score_list)),'good_rate_mean':float(np.mean(good_list)),'rmse_mean':float(np.mean(rmse_list)),'mape_mean':float(np.mean(mape_list))})
	except Exception as e:
		print('CatBoost skipped:', e)

	if adv_rows:
		adv_df = pd.DataFrame(adv_rows)
		adv_df.to_csv(reports_dir / 'cv_advanced_models.csv', index=False)
		print('Advanced models written (CV):', len(adv_df))

	print('Benchmark: done')

	# Export ridge plots
	plots_dir = reports_dir / 'plots'
	ensure_dir(plots_dir)
	try:
		fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
		ridge_df_sorted = ridge_df.sort_values('alpha')
		axes[0].plot(ridge_df_sorted['alpha'], ridge_df_sorted['score_mean'], marker='o')
		axes[0].set_xscale('log')
		axes[0].set_ylabel('Score (mean)')
		axes[0].grid(True, linestyle='--', alpha=0.4)
		axes[1].plot(ridge_df_sorted['alpha'], ridge_df_sorted['rmse_mean'], marker='o', color='tab:orange')
		axes[1].set_xscale('log')
		axes[1].set_ylabel('RMSE (mean)')
		axes[1].grid(True, linestyle='--', alpha=0.4)
		axes[2].plot(ridge_df_sorted['alpha'], ridge_df_sorted['mape_mean'], marker='o', color='tab:green')
		axes[2].set_xscale('log')
		axes[2].set_xlabel('alpha (log)')
		axes[2].set_ylabel('MAPE (mean)')
		axes[2].grid(True, linestyle='--', alpha=0.4)
		fig.tight_layout()
		out_plot = plots_dir / 'ridge_curves.png'
		fig.savefig(out_plot, dpi=150)
		plt.close(fig)
		print(f'Benchmark: wrote plot {out_plot}')
	except Exception as e:
		print('Benchmark: plotting skipped:', e)


if __name__ == '__main__':
	main()


