from inspect import trace
from re import sub
import sys
import catboost
from catboost import CatBoostRegressor, Pool
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
import mlflow
import gc
import os

from preprocess import get_data
sys.path.append('../')
from utils import rmse, make_dirs, decorate, send_start_log, send_end_log, send_error_log

from path_info import SAVE_PATH, MLFLOW_PATH

mlflow.set_tracking_uri(MLFLOW_PATH)

cfg = {
    'name': 'exp014',
    'sub_name': 'none',
    'n_splits': 5,
    'seed': 55,
    'experiment_id': 0,
    'debug': False,
    'cb_params' : {
        'loss_function': 'RMSE', 
        'learning_rate': 0.05,
        'depth': 10,
        'random_seed': 55,
        'task_type': 'GPU',

        'iterations': 10000,
        'early_stopping_rounds': 100,
    }
}

def main():
    if cfg['debug']:
        cfg['n_splits'] = 2
        cfg['cb_params']['iterations'] = 10
        cfg['name'] = 'debug'
    send_start_log(cfg['name'])

    save_path = os.path.join(SAVE_PATH, cfg['name'], cfg['sub_name'])
    make_dirs(save_path)

    X_train, y_train, group, X_test, submit = get_data()

    oof = pd.Series(np.zeros(y_train.shape))
    models = []

    cv = X_train.pop('fold')
    categorical_features_indices = np.where(X_train.dtypes == "category")[0]
    print(categorical_features_indices)
    print(decorate('train start'))
    for i in range(cv.nunique()):
        trn_idx = X_train[cv != i].index
        val_idx = X_train[cv == i].index
        with mlflow.start_run(experiment_id=cfg['experiment_id'], nested=True):
            train_dataset = Pool(X_train.loc[trn_idx], y_train.loc[trn_idx], categorical_features_indices)
            valid_dataset = Pool(X_train.loc[val_idx], y_train.loc[val_idx], categorical_features_indices)
            model = CatBoostRegressor(**cfg['cb_params'])

            model.fit(
                train_dataset,
                eval_set = valid_dataset,
                use_best_model=True,
            )

            joblib.dump(model, os.path.join(save_path, f'catboost_{i}.pkl'))
            models.append(model)
            oof.iloc[val_idx] = model.predict(X_train.loc[val_idx])

            del train_dataset, valid_dataset, model
            gc.collect()
    
    oof[oof<0] = 0
    val_score = rmse(y_train, oof)
    mlflow.log_metric('val_rmse', val_score)
    print("val_rmse: ", val_score)
    oof.to_csv(os.path.join(save_path, 'oof.csv'), header=False, index=False)
    mlflow.log_artifact(os.path.join(save_path, 'oof.csv'))

    print(decorate('test start'))
    submit['pm25_mid'] = 0
    for model in models:
        submit['pm25_mid'] += model.predict(X_test)
    submit['pm25_mid'] /= cfg['n_splits']
    submit.loc[submit['pm25_mid']<0, 'pm25_mid'] = 0

    submit.to_csv(os.path.join(save_path, 'submit.csv'), header=False, index=False)
    mlflow.log_artifact(os.path.join(save_path, 'submit.csv'))
    
    send_end_log(cfg['name'])


if __name__ == "__main__":
    # mlflow.lightgbm.autolog()
    with mlflow.start_run(experiment_id=cfg['experiment_id']):
        lgbm_params = cfg.pop('cb_params')
        mlflow.log_params(cfg)
        cfg['cb_params'] = lgbm_params
        print(cfg['cb_params'])
        try:
            main()
        except:
            import traceback
            send_error_log(traceback.format_exc())

