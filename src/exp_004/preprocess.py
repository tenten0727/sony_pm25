from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

sys.path.append('../')
from utils import AbstractBaseBlock, run_blocks, Timer
from path_info import DATA_PATH

class NumericBlock(AbstractBaseBlock):
    def transform(self, input_df):
        num_cols = input_df.select_dtypes(include=[int, float]).drop(['id'], axis=1).columns

        return input_df[num_cols].copy()

class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.test_TE = {}
    
    def fit(self, input_df, y):
        output_df = input_df[self.cols]
        for col in self.cols:
            data_tmp = pd.DataFrame({col: input_df[col], 'target': y})

            kf = GroupKFold(5)
            for trn, val in kf.split(input_df, groups=input_df['City']):
                target_mean = data_tmp.iloc[trn].groupby(col)['target'].mean()
                output_df.loc[val, col] = output_df.loc[val, col].map(target_mean)
            output_df[col] = output_df[col].astype(float)
            
            target_mean = data_tmp.groupby(col)['target'].mean()
            self.test_TE[col] = target_mean

        return output_df

    def transform(self, input_df):
        output_df = input_df[self.cols]
        for col in self.cols:
            output_df[col] = output_df[col].map(self.test_TE[col])
        return output_df

class CategoricalBlock(AbstractBaseBlock):
    def __init__(self, cols):
        self.cols = cols
        
    def transform(self, input_df):
        category_df = input_df[self.cols].astype('category')
        return category_df

class RangeMaxMinBlock(AbstractBaseBlock):
    def __init__(self, cols):
        self.cols = cols
        
    def transform(self, input_df):
        output_df = pd.DataFrame()
        for col in self.cols:
            output_df[col+'range'] = input_df[col+'_max'] - input_df[col+'_min']
        
        return output_df


def read_data():
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    submit = pd.read_csv(os.path.join(DATA_PATH, 'submit_sample.csv'), header=None).rename(columns={0:'id', 1:'pm25_mid'})

    return train, test, submit

def create_feature(df_train, df_test):
    y = df_train['pm25_mid']

    feature_blocks = [
        NumericBlock(),
        TargetEncodingBlock(['Country']),
        # CategoricalBlock(['Country']),
        RangeMaxMinBlock(['co', 'o3', 'so2', 'no2', 'temperature', 'humidity', 'pressure', 'ws', 'dew'])
    ]
    df_train_feat = run_blocks(df_train.drop('pm25_mid', axis=1), feature_blocks, y)
    df_test_feat = run_blocks(df_test, feature_blocks, test=True)

    return df_train_feat, df_test_feat

def get_data():
    df_train, df_test, submit = read_data()

    df_train_feat, df_test_feat = create_feature(df_train, df_test)
    y = df_train['pm25_mid']
    group = df_train['City']

    return df_train_feat, y, group, df_test_feat, submit
