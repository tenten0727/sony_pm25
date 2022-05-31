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

sys.path.append('../')
from utils import rmse, make_dirs, decorate, send_start_log, send_end_log, send_error_log

from path_info import SAVE_PATH, MLFLOW_PATH, DATA_PATH

mlflow.set_tracking_uri(MLFLOW_PATH)

cfg = {
    'name': 'exp020',
    'sub_name': 'ensemble',
    'n_splits': 5,
    'seed': 55,
    'experiment_id': 0,
    'debug': False,

    'lgbm_params' : {
        'objective': 'regression',
        'metric': 'rmse', 
        'boosting_type': 'gbdt',

        'learning_rate': 0.05,
        'max_depth': -1,
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

    submit = pd.read_csv(os.path.join(DATA_PATH, 'submit_sample.csv'), header=None).rename(columns={0:'id', 1:'pm25_mid'})
    submit['pm25_mid'] = 0

    sub1 = pd.read_csv('/home/cggyoshikawa/work/signate/sony_pm25/src/mlruns/0/66fe71d5021545f791923fae46fcc4ff/artifacts/submit.csv', header=None)
    sub2 = pd.read_csv('/home/cggyoshikawa/work/signate/sony_pm25/src/mlruns/0/d76b35d4d21b4df69bb0df2d677a00db/artifacts/submit.csv', header=None)
    sub3 = pd.read_csv('/home/cggyoshikawa/work/signate/sony_pm25/src/mlruns/0/29553e1bc4fa4fef854cef074a708195/artifacts/submit.csv', header=None)
    sub4 = pd.read_csv('/home/cggyoshikawa/work/signate/sony_pm25/src/mlruns/0/16c8db5bdeec4feebeace19ddcb5101d/artifacts/submit.csv', header=None)
    submit['pm25_mid'] = (sub1.iloc[:, 1] + sub2.iloc[:, 1] + sub3.iloc[:, 1] + sub4.iloc[:, 1] ) / 4

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

