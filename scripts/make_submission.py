import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import DatasetPaths, load_all_training_tables, load_test, prepare_train_target, explode_test_id
from src.features import build_time_lagged_features
from src.models import competition_score, build_linear_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(ROOT))
    parser.add_argument("--out", type=str, default=str(ROOT / "submission.csv"))
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

    test_df = load_test(paths)
    test_exploded = explode_test_id(test_df)
    lag_full = build_time_lagged_features(train['new_house_transactions'])
    lag_full = lag_full.sort_values(['time','sector_id'])
    lag_test = lag_full[lag_full['time'].isin(test_exploded['time'].unique())]
    lag_test = lag_test.merge(test_exploded[['time','sector','sector_id','id']], on=['time','sector_id'], how='right')
    X_test = lag_test[feature_cols].fillna(0)
    y_pred_test = pipe.predict(X_test)
    submission = lag_test[['id']].copy()
    submission['new_house_transaction_amount'] = y_pred_test
    submission = test_df[['id']].merge(submission, on='id', how='left')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"Saved submission to {args.out} with {len(submission)} rows")


if __name__ == "__main__":
    main()


