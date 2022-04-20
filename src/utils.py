import pandas as pd
from time import time
import numpy as np
import os
from sklearn.metrics import mean_squared_error

import requests
import json

from slack_info import WEB_HOOK

class AbstractBaseBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

def run_blocks(input_df, blocks, y=None, test=False):
    out_df = pd.DataFrame()

    print(decorate('start run blocks'))

    with Timer(prefix='run test = {}'.format(test)):
        for block in blocks:
            with Timer(prefix='\t- {}'.format(str(block))):
                if not test:
                    out_i = block.fit(input_df, y=y)
                else:
                    out_i = block.transform(input_df)

            assert len(input_df) == len(out_i), block
            name = block.__class__.__name__
            out_df = pd.concat([out_df, out_i.add_suffix(f'@{name}')], axis=1)

    return out_df


class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' ', verbose=0):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None
        self.verbose = verbose

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        if self.verbose is None:
            return
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


def decorate(s: str, decoration=None):
    if decoration is None:
        decoration = '=' * 20

    return ' '.join([decoration, str(s), decoration])


def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def make_dirs(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

# slackにメッセージ送信する
def send_start_log(name):
    requests.post(
        WEB_HOOK,
        headers={'content-type': 'application/json'},
        data=json.dumps({"text":"START - " + name})
    )

def send_end_log(name):
    requests.post(
        WEB_HOOK,
        headers={'content-type': 'application/json'},
        data=json.dumps({"text":"END - " + name})
    )

def send_error_log(message):
    requests.post(
        WEB_HOOK,
        headers={'content-type': 'application/json'},
        data=json.dumps({"text":":no_entry_sign:" + message})
    )
