from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import sys
sys.path.append('../src')
from sony_pm25_utils import read_data
from utils import AbstractBaseBlock, run_blocks, Timer

class NumericBlock(AbstractBaseBlock):
    def transform(self, df):
        num_cols = df.select_dtypes(include=[int, float]).drop(['id'], axis=1).columns

        return df[num_cols].copy()


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
