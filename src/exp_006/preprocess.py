from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import sys
import os
import itertools
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from category_encoders import CountEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
sys.path.append('../')
from utils import AbstractBaseBlock, run_blocks, Timer
from path_info import DATA_PATH

class NumericBlock(AbstractBaseBlock):
    def transform(self, input_df):
        num_cols = input_df.select_dtypes(include=[int, float]).drop(['id', 'year', 'month', 'day'], axis=1).columns

        return input_df[num_cols]

class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, cols, cv, agg):
        self.cols = cols
        self.test_TE = {}
        self.cv = cv
        self.agg = agg
    
    def fit(self, input_df, y):
        output_df = input_df[self.cols].astype(str)
        for col in self.cols:
            data_tmp = pd.DataFrame({col: input_df[col].astype(str), 'target': y})

            for i in range(self.cv.nunique()):
                trn = input_df[self.cv != i].index
                val = input_df[self.cv == i].index
                target_mean = data_tmp.iloc[trn].groupby(col)['target'].agg(self.agg)
                output_df.loc[val, col] = output_df.loc[val, col].map(target_mean)
            output_df[col] = output_df[col].astype(float)
            
            target_mean = data_tmp.groupby(col)['target'].agg(self.agg)
            self.test_TE[col] = target_mean

        return output_df.add_prefix('TE_'+self.agg)

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
        output_df['is_holiday'] = 0
        output_df.loc[output_df['dayofweek'].isin([5, 6]), 'is_holiday'] = 1

        def datetime_encode(df, col):
            # この方法だと場合によって最大値が変化するデータでは正確な値は出ない
            # 例：月の日数が30日や31日の場合がある
            df[col + '_cos'] = np.cos(2 * np.pi * (df[col].astype(float)) / df[col].astype(float).max())
            df[col + '_sin'] = np.sin(2 * np.pi * df[col].astype(float) / df[col].astype(float).max())
            return df
        encode_cols = ['month', 'day', 'dayofweek']
        for col in encode_cols:
            output_df = datetime_encode(output_df, col)
        
        output_df[['date', 'year', 'month', 'day']] = output_df[['date', 'year', 'month', 'day']].astype('category')
        return output_df

class KnnLocationBlock(AbstractBaseBlock):
    def __init__(self, cv, agg, n_neighbors=10):
        self.n_neighbors=n_neighbors
        self.cv = cv
        self.test_knn = None
        self.agg = agg
    
    def fit(self, input_df, y):
        output_df = input_df[['lat', 'lon']]
        output_df['y'] = y
        output_df['knn_location_'+self.agg] = 0
        for i in range(self.cv.nunique()):
            trn_city_mean = output_df[self.cv != i].groupby(['lat', 'lon'])['y'].agg(self.agg).reset_index()

            knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights='distance')
            knn.fit(trn_city_mean[['lat', 'lon']], trn_city_mean['y'])
            output_df.loc[self.cv == i, 'knn_location_'+self.agg] = knn.predict(output_df.loc[self.cv == i, ['lat', 'lon']])
        
        knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights='distance')
        city_mean = output_df.groupby(['lat', 'lon'])['y'].agg(self.agg).reset_index()
        knn.fit(city_mean[['lat', 'lon']], city_mean['y'])
        self.test_knn = knn

        return output_df[['knn_location_'+self.agg]]
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[['lat', 'lon']]
        output_df['knn_location_'+self.agg] = self.test_knn.predict(output_df[['lat', 'lon']])

        return output_df[['knn_location_'+self.agg]] 

class AggBlock(AbstractBaseBlock):
    def __init__(self, group, cols, aggs=['mean', 'std', 'max', 'min']):
        self.group = group
        self.cols = cols
        self.aggs = aggs
        self.test_agg = None
    
    def fit(self, input_df, y):
        self.test_agg = input_df.groupby(self.group)[self.cols].agg(self.aggs).reset_index()
        self.test_agg.columns = ['_'.join(col)+'_'+self.group if col[0] != self.group else self.group for col in self.test_agg.columns]

        return self.transform(input_df)
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df  = input_df[[self.group]]
        output_df = pd.merge(output_df, self.test_agg, how='left', on=self.group)

        return output_df.drop(self.group, axis=1)


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

    cols_prefix = ['co', 'o3', 'so2', 'no2', 'temperature', 'humidity', 'pressure', 'ws', 'dew']
    cols_suffix = ['cnt', 'min', 'mid', 'max', 'var']

    num_feat_cols = [pre+'_'+suf for pre, suf in itertools.product(cols_prefix, cols_suffix)]

    first_feature_blocks = [
        NumericBlock(),
        DatetimeBlock(),
        RangeMaxMinBlock(cols_prefix),
        CategoricalBlock(['Country']),
    ]

    df_train_feat = []
    df_test_feat = []
    for block in first_feature_blocks:
        df_train_feat.append(block.fit(df_train_drop_y, y))
        df_test_feat.append(block.transform(df_test))
        assert 'pm25_mid' not in df_train_feat[-1].columns, block
    
    df_train_feat = pd.concat(df_train_feat, axis=1)
    df_test_feat = pd.concat(df_test_feat, axis=1)

    second_feature_blocks = [
        *[KnnLocationBlock(df_train['fold'], agg) for agg in ['mean', 'std']],
        # *[AggBlock(group, num_feat_cols) for group in ['Country']],
        *[TargetEncodingBlock(['month', 'Country', 'dayofweek', 'is_holiday', 'date'], df_train['fold'], agg) for agg in ['mean', 'std']],
        CountEncodingBlock(['year', 'month', 'day', 'Country', 'dayofweek', 'date'])
    ]
    df_train_feat = pd.concat([df_train_feat, run_blocks(df_train_feat, second_feature_blocks, y)], axis=1)
    df_test_feat = pd.concat([df_test_feat, run_blocks(df_test_feat, second_feature_blocks, test=True)], axis=1)

    drop_cols = ['date']
    df_train_feat = df_train_feat.drop(drop_cols, axis=1)
    df_test_feat = df_test_feat.drop(drop_cols, axis=1)

    return df_train_feat, df_test_feat

def get_data():
    df_train, df_test, submit = read_data()

    df_train_feat, df_test_feat = create_feature(df_train, df_test)
    y = df_train['pm25_mid']
    group = df_train['City']

    return df_train_feat, y, group, df_test_feat, submit
