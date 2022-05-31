import pandas as pd
import numpy as np
import sys
import os
import itertools
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from category_encoders import CountEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import datetime


sys.path.append('../')
from utils import AbstractBaseBlock, run_blocks, Timer
from path_info import DATA_PATH, SAVE_PATH

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

        return output_df.add_prefix('TE_'+self.agg+'_')

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[self.cols].astype(str)
        for col in self.cols:
            output_df[col] = output_df[col].map(self.test_TE[col])
            output_df[col] = output_df[col].astype(float)

        return output_df.add_prefix('TE_'+self.agg+'_')

class MultiTargetEncodingBlock(TargetEncodingBlock):
    def __init__(self, col1, col2, cv, agg):
        self.col = col1+'_x_'+col2
        self.col1 = col1
        self.col2 = col2
        super().__init__([self.col], cv, agg)
    
    def fit(self, input_df, y):
        output_df = input_df.copy()
        output_df[self.col] = (input_df[self.col1].astype(str) + 'x' + input_df[self.col2].astype(str)).astype('category')
        return super().fit(output_df, y)
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df[self.col] = (input_df[self.col1].astype(str) + 'x' + input_df[self.col2].astype(str)).astype('category')
        return super().transform(output_df)

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
    def __init__(self) -> None:
        super().__init__()
        self.rounds = [3, 7]

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[['year', 'month', 'day']]
        output_df['date'] = output_df['year'].astype(str) + '-' + output_df['month'].astype(str) + '-' + output_df['day'].astype(str)
        output_df['date'] = pd.to_datetime(output_df['date'])
        output_df['dayofweek'] = output_df['date'].dt.dayofweek.astype('category')
        output_df['is_holiday'] = 0
        output_df.loc[output_df['dayofweek'].isin([5, 6]), 'is_holiday'] = 1

        date_map = pd.Series(output_df['date'].unique())
        date_map = {v: k for k, v in zip(date_map.index, date_map)}
        output_df['date_label'] = output_df['date'].map(date_map).astype(np.int16)

        for i in self.rounds:
            output_df['date_label_round_'+str(i)] = output_df['date_label'] // i

        return output_df

class LatLonCategoryBlock(AbstractBaseBlock):
    def __init__(self, rounds=[5, 10, 30]) -> None:
        self.rounds = rounds
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[['lon', 'lat']]
        for round in self.rounds:
            output_df['lon_round'] = output_df['lon'] // round
            output_df['lat_round'] = output_df['lat'] // round
            output_df['location_category_round_'+str(round)] = ((output_df['lon_round']).astype(str)+ '-' + output_df['lat_round'].astype(str)).astype('category')

        output_cols = [col for col in output_df.columns if 'location_category_round_' in col]
        return output_df[output_cols]

class KmeansBlock(AbstractBaseBlock):
    def __init__(self, cols, n_cluster=10) -> None:
        self.cols = cols
        self.n_cluster = n_cluster
        self.scaler = None
        self.kmeans_model = None
    
    def fit(self, input_df, y):
        output_df = input_df[self.cols]
        self.scaler = StandardScaler()
        output_df[self.cols] = self.scaler.fit_transform(output_df[self.cols])
        self.kmeans_model = KMeans(n_clusters=self.n_cluster, random_state=55)
        self.kmeans_model.fit(output_df)

        return self.transform(input_df)

    def transform(self, input_df):
        output_df = input_df[self.cols]
        output_df[self.cols] = self.scaler.transform(output_df[self.cols])
        output_df['kmeans_cluster'] = self.kmeans_model.predict(output_df)
        dists = pd.DataFrame(self.kmeans_model.transform(output_df[self.cols]))
        for i in range(self.n_cluster):
            output_df['kmeans_dist_'+str(i)] = dists.iloc[:, i]

        output_df['kmeans_cluster'] = output_df['kmeans_cluster'].astype('category')
        output_df = output_df.drop(self.cols, axis=1)

        return output_df

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

class YLagLeadBlock(AbstractBaseBlock):
    def __init__(self, groups, lag=1):
        self.groups = groups
        self.lag = lag
        self.df_lag_feature = None
    
    def fit(self, input_df, y):
        self.df_lag_feature = input_df[self.groups]
        self.df_lag_feature['y'] = y
        self.df_lag_feature = self.df_lag_feature.groupby(self.groups)['y'].agg('mean').reset_index()
    
        self.df_lag_feature['date'] = self.df_lag_feature['date'] + datetime.timedelta(days=self.lag)
        return self.transform(input_df)
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[self.groups]
        output_df = pd.merge(output_df, self.df_lag_feature, how='left', on=self.groups)

        output_df = output_df.rename(columns={'y': 'lag'+str(self.lag)+'_'.join(self.groups)+'_mean_pm25'})
        return output_df[['lag'+str(self.lag)+'_'.join(self.groups)+'_mean_pm25']]


class KnnLocationDateTargetBlock(AbstractBaseBlock):
    def __init__(self, agg, n_neighbors, time_extend):
        self.n_neighbors=n_neighbors
        self.time_extend = time_extend
        self.test_knn = None
        self.agg = agg
        self.num_ss = StandardScaler()
        self.col_name = f'knn_location_target_{self.agg}_{self.n_neighbors}_{self.time_extend}'
    
    def fit(self, input_df, y):
        output_df = input_df[['lat', 'lon', 'date_label', 'fold']]
        output_df[['lat', 'lon', 'date_label']] = self.num_ss.fit_transform(output_df[['lat', 'lon', 'date_label']])
        output_df['date_label'] = output_df['date_label'] * self.time_extend
        output_df['y'] = y
        
        for i in range(output_df['fold'].nunique()):
            trn = output_df[output_df['fold'] != i]
            knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights='distance')
            knn.fit(trn[['lat', 'lon', 'date_label']], trn['y'])
            output_df.loc[output_df['fold'] == i, self.col_name] = knn.predict(output_df.loc[output_df['fold'] == i, ['lat', 'lon', 'date_label']])
        
        knn.fit(output_df[['lat', 'lon', 'date_label']], output_df['y'])
        self.test_knn = knn

        return output_df[[self.col_name]]
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[['lat', 'lon', 'date_label']]
        output_df[['lat', 'lon', 'date_label']] = self.num_ss.transform(output_df[['lat', 'lon', 'date_label']])
        output_df['date_label'] = output_df['date_label'] * self.time_extend
        output_df[self.col_name] = self.test_knn.predict(output_df[['lat', 'lon', 'date_label']])

        return output_df[[self.col_name]]

class KnnLocationDateBlock(AbstractBaseBlock):
    def __init__(self, agg, n_neighbors, time_extend, col):
        self.n_neighbors=n_neighbors
        self.time_extend = time_extend
        self.test_knn = None
        self.agg = agg
        self.num_ss = StandardScaler()
        self.col = col
        self.col_name = f'knn_location_{self.col}_{self.agg}_{self.n_neighbors}_{self.time_extend}'

    def fit(self, input_df, y):
        output_df = input_df[['lat', 'lon', 'date_label']]
        output_df[['lat', 'lon', 'date_label']] = self.num_ss.fit_transform(output_df[['lat', 'lon', 'date_label']])
        output_df['date_label'] = output_df['date_label'] * self.time_extend
        output_df[self.col] = input_df[self.col]
        
        knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights='distance')
        knn.fit(output_df[['lat', 'lon', 'date_label']], output_df[self.col])
        output_df[self.col_name] = knn.predict(output_df[['lat', 'lon', 'date_label']])
        
        self.test_knn = knn

        return output_df[[self.col_name]]
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[['lat', 'lon', 'date_label']]
        output_df[['lat', 'lon', 'date_label']] = self.num_ss.transform(output_df[['lat', 'lon', 'date_label']])
        output_df['date_label'] = output_df['date_label'] * self.time_extend
        output_df[self.col_name] = self.test_knn.predict(output_df[['lat', 'lon', 'date_label']])

        return output_df[[self.col_name]]

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
        CategoricalBlock(['Country', 'City']),
        KmeansBlock(num_feat_cols),
        # LatLonCategoryBlock(),
    ]

    df_train_feat = []
    df_test_feat = []
    for block in first_feature_blocks:
        df_train_feat.append(block.fit(df_train_drop_y, y))
        df_test_feat.append(block.transform(df_test))
    
    df_train_feat = pd.concat(df_train_feat, axis=1)
    df_test_feat = pd.concat(df_test_feat, axis=1)

    second_feature_blocks = [
        *[KnnLocationDateBlock(agg, n, t, col) for agg in ['mean'] for n in [5, 10] for t in [10] for col in ['co_mid', 'no2_mid']],
        *[KnnLocationDateTargetBlock(agg, n, t) for agg in ['mean', 'std'] for n in [5, 10] for t in [10]],
        *[YLagLeadBlock(['Country', 'date'], lag=i) for i in [-10, -7, -3, -2, -1, 1, 2, 3, 7, 10]],
        *[TargetEncodingBlock(['month', 'Country', 'dayofweek', 'is_holiday'], df_train['fold'], agg) for agg in ['mean', 'std']],
        # *[MultiTargetEncodingBlock(col1, col2, df_train['fold'], agg) for agg in ['mean', 'std'] for col1 in ['date_label_round_'+str(i) for i in [3, 7]] for col2 in ['location_category_round_'+str(i) for i in [5, 10, 30]]],
        CountEncodingBlock(['month', 'day', 'Country', 'dayofweek']),
        *[AggBlock(group, ['co_mid']) for group in ['month', 'Country', 'date_label_round_3']],
    ]
    df_train_feat = pd.concat([df_train_feat, run_blocks(df_train_feat, second_feature_blocks, y)], axis=1)
    df_test_feat = pd.concat([df_test_feat, run_blocks(df_test_feat, second_feature_blocks, test=True)], axis=1)

    drop_cols = ['date', 'City', 'is_holiday']
    df_train_feat = df_train_feat.drop(drop_cols, axis=1)
    df_test_feat = df_test_feat.drop(drop_cols, axis=1)

    return df_train_feat, df_test_feat

def get_data():
    df_train, df_test, submit = read_data()
    df_train = df_train[~df_train['City'].isin(['Ürümqi', 'Novosibirsk', 'Darwin', 'Perth'])].reset_index(drop=True)

    df_train_feat, df_test_feat = create_feature(df_train, df_test)
    y = df_train['pm25_mid']
    group = df_train['City']

    return df_train_feat, y, group, df_test_feat, submit