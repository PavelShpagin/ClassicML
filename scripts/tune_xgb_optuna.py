import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, prepare_train_target
from src.features import build_time_lagged_features
from src.models import competition_score


def load_dataset():
	paths = DatasetPaths(root_dir=str(ROOT))
	train = load_all_training_tables(paths)
	target_wide, _ = prepare_train_target(train['new_house_transactions'])
	lag_feats = build_time_lagged_features(train['new_house_transactions']).sort_values(['time','sector_id'])
	y_long = target_wide.unstack().reset_index(name='y').rename(columns={'level_0':'sector_id','time':'time'})
	df = lag_feats.merge(y_long, on=['time','sector_id'], how='left')
	df_model = df.dropna(subset=[c for c in df.columns if c.startswith('lag_') or c.startswith('roll_')]).copy()
	feature_cols = [c for c in df_model.columns if c.startswith('lag_') or c.startswith('roll_')]
	return df_model, feature_cols


FOLDS = [
	(0, 18, 19, 30),
	(0, 30, 31, 42),
	(0, 42, 43, 54),
	(0, 54, 55, 66),
]


def cv_score(df_model: pd.DataFrame, feature_cols: list[str], params: dict) -> float:
	scores = []
	for (tr_s, tr_e, va_s, va_e) in FOLDS:
		tr_mask = (df_model['time'] >= tr_s) & (df_model['time'] <= tr_e)
		va_mask = (df_model['time'] >= va_s) & (df_model['time'] <= va_e)
		X_tr, y_tr = df_model.loc[tr_mask, feature_cols], df_model.loc[tr_mask, 'y']
		X_va, y_va = df_model.loc[va_mask, feature_cols], df_model.loc[va_mask, 'y']
		model = xgb.XGBRegressor(
			max_depth=int(params['max_depth']),
			min_child_weight=params['min_child_weight'],
			learning_rate=params['learning_rate'],
			n_estimators=int(params['n_estimators']),
			subsample=params['subsample'],
			colsample_bytree=params['colsample_bytree'],
			reg_lambda=params['reg_lambda'],
			reg_alpha=params['reg_alpha'],
			objective='reg:squarederror', random_state=42, tree_method='hist'
		)
		model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
		yhat = model.predict(X_va)
		scores.append(competition_score(y_va.values, yhat)['score'])
	return float(np.mean(scores))


def objective(trial: optuna.Trial, df_model: pd.DataFrame, feature_cols: list[str]) -> float:
	params = {
		'max_depth': trial.suggest_int('max_depth', 4, 10),
		'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 20.0),
		'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
		'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
		'subsample': trial.suggest_float('subsample', 0.5, 1.0),
		'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
		'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
		'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
	}
	return cv_score(df_model, feature_cols, params)


def main():
	df_model, feature_cols = load_dataset()
	study = optuna.create_study(direction='maximize')
	study.optimize(lambda t: objective(t, df_model, feature_cols), n_trials=40, show_progress_bar=False)
	best = study.best_trial
	print('Best score', best.value)
	print('Best params', best.params)
	out = {'best_score': best.value, 'best_params': best.params}
	(Path(ROOT) / 'reports' / 'xgb_optuna_best.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
	main()

