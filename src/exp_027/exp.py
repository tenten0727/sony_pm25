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

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import Callback

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

from preprocess import get_data
sys.path.append('../')
from utils import rmse, make_dirs, decorate, send_start_log, send_end_log, send_error_log

from path_info import SAVE_PATH, MLFLOW_PATH

mlflow.set_tracking_uri(MLFLOW_PATH)

cfg = {
    'name': 'exp027',
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

class TabnetMLFlowCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # send to MLFlow
        # mlflow.log_metric("train_rmse", logs['train_rmse'])
        mlflow.log_metric("valid_rmse", logs["valid_rmse"], step=epoch)
        mlflow.log_metric("loss", logs["loss"], step=epoch)

def main():
    if cfg['debug']:
        cfg['n_splits'] = 2
        cfg['lgbm_params']['n_estimators'] = 10
        cfg['name'] = 'debug'
    send_start_log(cfg['name'])

    save_path = os.path.join(SAVE_PATH, cfg['name'], cfg['sub_name'])
    make_dirs(save_path)

    X_train, y_train, group, X_test, submit = get_data()
    cv = X_train.pop('fold')

    for col in X_train.select_dtypes('category').columns:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    ss_num = StandardScaler()
    X_train[X_train.columns] = ss_num.fit_transform(X_train)
    X_test[X_test.columns] = ss_num.transform(X_test)

    oof = pd.Series(np.zeros(y_train.shape))
    models = []

    print(decorate('pre-training start'))

    tabnet_params = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                        n_independent=2, n_shared=2,
                        seed=cfg['seed'], lambda_sparse=1e-3, 
                        optimizer_fn=torch.optim.Adam, 
                        optimizer_params=dict(lr=2e-2),
                        mask_type="entmax",
                        scheduler_params=dict(mode="min",
                                            patience=5,
                                            min_lr=1e-5,
                                            factor=0.9,),
                        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                        verbose=10
                        )

    pretrainer = TabNetPretrainer(**tabnet_params)

    pretrainer.fit(
        X_train=X_train.values,
        eval_set=[X_train.values],
        max_epochs=10,
        patience=20, batch_size=256, virtual_batch_size=128,
        num_workers=1, drop_last=True)
    
    joblib.dump(pretrainer, save_path+'/pretrainer.pkl')

    tabnet_params = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3,
        n_independent=2, n_shared=2,
        seed=cfg['seed'], lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2,
                            weight_decay=1e-5
                            ),
        mask_type="entmax",
        scheduler_params=dict(max_lr=0.05,
                            steps_per_epoch=int(X_train.shape[0] / 256),
                            epochs=200,
                            is_batch_level=True
                            ),
        scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
        verbose=10,
    )

    tabnet_callback = TabnetMLFlowCallback()
    model = TabNetRegressor(**tabnet_params)

    print(decorate('training start'))
    for i in range(cv.nunique()):
        trn_idx = X_train[cv != i].index
        val_idx = X_train[cv == i].index
        with mlflow.start_run(experiment_id=cfg['experiment_id'], nested=True, ):
            X_trn, y_trn = X_train.loc[trn_idx].values, y_train.loc[trn_idx].values.reshape([-1, 1])
            X_val, y_val = X_train.loc[val_idx].values, y_train.loc[val_idx].values.reshape([-1, 1])

            model.fit(X_train=X_trn,
                    y_train=y_trn,
                    eval_set=[(X_val, y_val)],
                    eval_name = ["valid"],
                    eval_metric = ["rmse"],
                    max_epochs=10,
                    patience=20, batch_size=256, virtual_batch_size=128,
                    num_workers=0, drop_last=False,
                    from_unsupervised=pretrainer, # comment out when Unsupervised
                    callbacks=[tabnet_callback]
                    )

            joblib.dump(model, os.path.join(save_path, f'tabnet_{i}.pkl'))
            models.append(model)
            oof.iloc[val_idx] = model.predict(X_train.loc[val_idx].values).reshape(-1)

            gc.collect()
    
    oof[oof<0] = 0
    val_score = rmse(y_train, oof)
    mlflow.log_metric('val_rmse', val_score)
    oof.to_csv(os.path.join(save_path, 'oof.csv'), header=False, index=False)
    mlflow.log_artifact(os.path.join(save_path, 'oof.csv'))

    print(decorate('test start'))
    submit['pm25_mid'] = 0
    for model in models:
        submit['pm25_mid'] += model.predict(X_test).reshape(-1)
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

