from inspect import trace
from re import sub
import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import mlflow
import gc
import os

from preprocess import get_data
from utils import rmse, make_dirs, decorate, send_start_log, send_end_log, send_error_log


SAVE_PATH = '../save/'

cfg = {
    'name': 'exp001',
    'n_splits': 5,
    'seed': 55,
    'experiment_id': 0,
    'debug': False,

    'lgbm_params' : {
        'objective': 'regression', 
        'metric': 'rmse', 
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'seed': 55,
        'max_depth': 8,
        'early_stopping_rounds': 50,
        'n_estimators': 500
    }
}

def main():
    save_path = os.path.join(SAVE_PATH, cfg['name'])
    make_dirs(save_path)

    if cfg['debug']:
        cfg['n_splits'] = 2
        cfg['lgbm_params']['n_estimators'] = 10

    X_train, y_train, X_test, submit = get_data()

    oof = pd.Series(np.zeros(y_train.shape))
    models = []

    kf = KFold(cfg['n_splits'], shuffle=True, random_state=cfg['seed'])
    print(decorate('train start'))
    for i, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        with mlflow.start_run(experiment_id=cfg['experiment_id'], nested=True):
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
    
    val_score = rmse(y_train, oof)
    mlflow.log_metric('val_rmse', val_score)

    print(decorate('test start'))
    submit['pm25_mid'] = 0
    for model in models:
        submit['pm25_mid'] += model.predict(X_test)
    submit['pm25_mid'] /= cfg['n_splits']

    submit.to_csv(os.path.join(save_path, 'submit.csv'), header=False, index=False)


if __name__ == "__main__":
    mlflow.lightgbm.autolog()
    with mlflow.start_run(experiment_id=cfg['experiment_id']):
        mlflow.log_params(cfg)
        try:
            send_start_log(cfg['name'])
            main()
            send_end_log(cfg['name'])
        except:
            import traceback
            send_error_log(traceback.format_exc())

