from typing import Tuple
import os
import json
import functools
import s3fs

import pandas as pd

import sklearn.base as skbase


AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
S3_ENDPOINT_URL = os.environ["MLFLOW_S3_ENDPOINT_URL"]
S3_BUCKET = os.environ["S3_BUCKET"]


def get_s3fs():
    """s3 connection
    """
    s3_file_system = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID,
                                   secret=AWS_SECRET_ACCESS_KEY,
                                   client_kwargs=dict(endpoint_url=S3_ENDPOINT_URL),
                                   anon=False) 
    return s3_file_system


def df_read(filename: str, path: str, s3: bool) -> pd.DataFrame:
    """reads csv file from S3 or locally
    Arguments:
        filename: str
        local path: str
        s3: bool
            s3 flag

    Returns:
        preprocessed data: pd.DataFrame
    """
    if s3:
        s3_file_system = get_s3fs()
        with s3_file_system.open(S3_BUCKET + '/' + filename, 'r') as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(os.path.join(path, filename))
    return df  


class FunctionalTransformer(skbase.BaseEstimator, skbase.TransformerMixin):
    """base class for functional data transformers
    """
    def __init__(self, function, **params):
        self.function = functools.partial(function, **params)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def functional_transformer(function):
    def builder(**params):
        return FunctionalTransformer(function, **params)
    return builder


def json_read(name):
    with open(name, 'r') as json_data: 
        res = json.load(json_data)
    return res 


def json_save(ver, name):
    with open(name, 'w') as json_data: 
        json.dump(ver, json_data)
