import pandas as pd
import numpy as np
from sqlalchemy import column

DATA_PATH = '../data/'

def read_data():
    train = pd.read_csv(DATA_PATH+'train.csv')
    test = pd.read_csv(DATA_PATH+'test.csv')
    submit = pd.read_csv(DATA_PATH+'submit_sample.csv', header=None).rename(columns={0:'id', 1:'pm25_mid'})

    return train, test, submit