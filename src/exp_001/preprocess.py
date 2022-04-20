from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('../')
from utils import AbstractBaseBlock, run_blocks, Timer
from path_info import DATA_PATH

class NumericBlock(AbstractBaseBlock):
    def transform(self, df):
        num_cols = df.select_dtypes(include=[int, float]).drop(['id'], axis=1).columns

        return df[num_cols].copy()

def read_data():
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    submit = pd.read_csv(os.path.join(DATA_PATH, 'submit_sample.csv'), header=None).rename(columns={0:'id', 1:'pm25_mid'})

    return train, test, submit

def create_feature(df_train, df_test):
    y = df_train['pm25_mid']

    feature_blocks = [
        NumericBlock()
    ]
    df_train_feat = run_blocks(df_train.drop('pm25_mid', axis=1), feature_blocks, y)
    df_test_feat = run_blocks(df_test, feature_blocks, y, test=True)

    return df_train_feat, df_test_feat

def get_data():
    df_train, df_test, submit = read_data()

    df_train_feat, df_test_feat = create_feature(df_train, df_test)
    y = df_train['pm25_mid']

    return df_train_feat, y, df_test_feat, submit
