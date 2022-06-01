from inspect import trace
from re import sub
import sys
import lightgbm as lgb
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
    'name': 'exp024',
    'sub_name': 'none',
    'n_splits': 5,
    'seed': 55,
    'experiment_id': 0,
    'debug': False,

    'lgbm_params' : {
        'objective': 'regression',
        'metric': 'rmse', 
        'boosting_type': 'gbdt',

        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 70,
        'min_data_in_leaf': 20,
        'max_bin': 255,
        
        'reg_lambda': 1.0,
        'reg_alpha': 1.,
    
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'subsample_freq': 1,
        
        'random_state': 55,
        'verbose': -1,
        'n_jobs': -1,

        'n_estimators': 10000,
        'early_stopping_rounds': 100,
        'verbose_eval': 100,
    }
}

def main():
    if cfg['debug']:
        cfg['n_splits'] = 2
        cfg['lgbm_params']['n_estimators'] = 10
        cfg['name'] = 'debug'
    send_start_log(cfg['name'])

    save_path = os.path.join(SAVE_PATH, cfg['name'], cfg['sub_name'])
    make_dirs(save_path)

    X_train, y_train, group, X_test, submit = get_data()

    oof = pd.Series(np.zeros(y_train.shape))
    models = []

    cv = X_train.pop('fold')
    print(decorate('train start'))
    for i in range(cv.nunique()):
        trn_idx = X_train[cv != i].index
        val_idx = X_train[cv == i].index
        with mlflow.start_run(experiment_id=cfg['experiment_id'], nested=True, ):
            train_dataset = lgb.Dataset(X_train.loc[trn_idx], y_train.loc[trn_idx])
            valid_dataset = lgb.Dataset(X_train.loc[val_idx], y_train.loc[val_idx])

            model = lgb.train(
                cfg['lgbm_params'],
                train_set = train_dataset,
                valid_sets = [train_dataset, valid_dataset],
                verbose_eval=50,
            )

            joblib.dump(model, os.path.join(save_path, f'lgbm_{i}.pkl'))
            models.append(model)
            oof.iloc[val_idx] = model.predict(X_train.loc[val_idx])

            del train_dataset, valid_dataset, model
            gc.collect()
    
    oof[oof<0] = 0
    val_score = rmse(y_train, oof)
    mlflow.log_metric('val_rmse', val_score)
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
    mlflow.lightgbm.autolog()
    with mlflow.start_run(experiment_id=cfg['experiment_id']):
        lgbm_params = cfg.pop('lgbm_params')
        mlflow.log_params(cfg)
        cfg['lgbm_params'] = lgbm_params
        print(cfg['lgbm_params'])
        try:
            main()
        except:
            import traceback
            send_error_log(traceback.format_exc())

