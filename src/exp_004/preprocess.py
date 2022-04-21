from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from category_encoders import CountEncoder
sys.path.append('../')
from utils import AbstractBaseBlock, run_blocks, Timer
from path_info import DATA_PATH

class NumericBlock(AbstractBaseBlock):
    def transform(self, input_df):
        num_cols = input_df.select_dtypes(include=[int, float]).drop(['id', 'year', 'month', 'day'], axis=1).columns

        return input_df[num_cols]

class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, cols, cv):
        self.cols = cols
        self.test_TE = {}
        self.cv = cv
    
    def fit(self, input_df, y):
        output_df = input_df[self.cols].astype(str)
        for col in self.cols:
            data_tmp = pd.DataFrame({col: input_df[col].astype(str), 'target': y})

            for i in range(self.cv.nunique()):
                trn = input_df[self.cv != i].index
                val = input_df[self.cv == i].index
                target_mean = data_tmp.iloc[trn].groupby(col)['target'].mean()
                output_df.loc[val, col] = output_df.loc[val, col].map(target_mean)
            output_df[col] = output_df[col].astype(float)
            
            target_mean = data_tmp.groupby(col)['target'].mean()
            self.test_TE[col] = target_mean

        return output_df.add_prefix('TE_')

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[self.cols].astype(str)
        for col in self.cols:
            output_df[col] = output_df[col].map(self.test_TE[col])
        return output_df

class CountEncodingBlock(AbstractBaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = CountEncoder(cols)

    def fit(self, input_df, y=None):
        self.encoder.fit(input_df[self.cols])
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = self.encoder.transform(input_df[self.cols])
        return out_df.add_prefix('CE_')

class CategoricalBlock(AbstractBaseBlock):
    def __init__(self, cols):
        self.cols = cols
        
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        category_df = input_df[self.cols].astype('category')
        return category_df

class RangeMaxMinBlock(AbstractBaseBlock):
    def __init__(self, cols):
        self.cols = cols
        
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()
        for col in self.cols:
            output_df[col] = input_df[col+'_max'] - input_df[col+'_min']
        
        return output_df.add_suffix('_range')

class DatetimeBlock(AbstractBaseBlock):
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[['year', 'month', 'day']]
        output_df['date'] = output_df['year'].astype(str) + '-' + output_df['month'].astype(str) + '-' + output_df['day'].astype(str)
        output_df['date'] = pd.to_datetime(output_df['date'])
        output_df['dayofweek'] = output_df['date'].dt.dayofweek.astype('category')

        return output_df[['dayofweek']]

def read_data():
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    submit = pd.read_csv(os.path.join(DATA_PATH, 'submit_sample.csv'), header=None).rename(columns={0:'id', 1:'pm25_mid'})

    return train, test, submit

def create_feature(df_train, df_test):
    y = df_train['pm25_mid']

    kf = KFold(n_splits=5, shuffle=True, random_state=55)
    df_train['fold'] = -1
    group = pd.Series(df_train['City'].unique())
    for i, (_, val_group) in enumerate(kf.split(group.index)):
        val_city = group[val_group]
        
        val_idx = df_train[df_train['City'].isin(val_city)].index
        df_train.loc[val_idx, 'fold'] = i
    
    df_train_drop_y = df_train.drop('pm25_mid', axis=1)

    first_feature_blocks = [
        NumericBlock(),
        DatetimeBlock(),
        CategoricalBlock(['year', 'month', 'day', 'Country']),
        RangeMaxMinBlock(['co', 'o3', 'so2', 'no2', 'temperature', 'humidity', 'pressure', 'ws', 'dew']),
    ]

    df_train_feat = []
    df_test_feat = []
    for block in first_feature_blocks:
        df_train_feat.append(block.fit(df_train_drop_y))
        df_test_feat.append(block.transform(df_test))
        assert 'pm25_mid' not in df_train_feat[-1].columns, block
    
    df_train_feat = pd.concat(df_train_feat, axis=1)
    df_test_feat = pd.concat(df_test_feat, axis=1)

    second_feature_blocks = [
        TargetEncodingBlock(['year', 'month', 'day', 'Country', 'dayofweek'], df_train['fold']),
        CountEncodingBlock(['year', 'month', 'day', 'Country', 'dayofweek'])
    ]
    df_train_feat = pd.concat([df_train_feat, run_blocks(df_train_feat, second_feature_blocks, y)], axis=1)
    df_test_feat = pd.concat([df_test_feat, run_blocks(df_test_feat, second_feature_blocks, test=True)], axis=1)

    df_train_feat = df_train_feat
    df_test_feat = df_test_feat

    return df_train_feat, df_test_feat

def get_data():
    df_train, df_test, submit = read_data()

    df_train_feat, df_test_feat = create_feature(df_train, df_test)
    y = df_train['pm25_mid']
    group = df_train['City']

    return df_train_feat, y, group, df_test_feat, submit
