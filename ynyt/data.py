import os
import urllib

from typing import List, Dict
from dateutil import relativedelta

import pandas as pd
import numpy as np
import sklearn.pipeline as skpipe
import datetime
import fastparquet

from .utils import functional_transformer


# ---------- downloading -------------

def download_raw_data_month(suffix: str, config: Dict) -> None:
    """Downloads raw data for one month.

    Parameters
    ----------
    suffix : str
        month / year index
    config : Dict
        config
    """
    head = config['data_url_head']
    path = config['raw_output_path']
    columns = set(config['raw_data_columns'])

    filename = 'yellow_tripdata_' + suffix + '.parquet'
    try:
        df = pd.read_parquet(head + filename, engine = 'fastparquet')
        if not columns.issubset(set(df.columns)):
            raise ValueError('{}: defferent feature names!!!'.format(suffix))
        print(filename + ' is downloaded')
        filename = os.path.join(path, 'yellow_tripdata_' + suffix + '.csv')
        df.to_csv('../' + filename, header=True, index=False)
    except urllib.error.HTTPError or TypeError:
        print(filename + ' is not found')


def get_suffix(year: int, month: int) -> str:
    # Makes year / month str index
    m_str = str(month) if month > 9 else '0' + str(month)
    return str(year) + '-' + m_str

def download_raw_data(config: Dict) -> None:
    """Downloads lacking data

    Parameters
    ----------
    config : Dict
        config
    """
    years = config['years']
    path = config['raw_output_path']
    downloaded_files = [f[-11:-4] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    for y in years:
        for m in range(1, 13): 
            suffix = get_suffix(y, m)
            if not suffix in downloaded_files:
                download_raw_data_month(suffix, config)
    
# ----------- preprocessing: functions ------------

def get_valid_preriod(suffix):
    # convert year / month index to time period of the corresponding month 
    s = suffix.split('-')
    return [datetime.datetime(int(s[0]), int(s[1]), 1),
            datetime.datetime(int(s[0]), int(s[1]), 1) + relativedelta.relativedelta(months=1)]


def data_cleaner(df: pd.DataFrame, suffix: str, format: str) -> pd.DataFrame:
    """Clean and check plausibility. Discards invalid rows.
    Parameters
    ----------
    df : pd.DataFrame
        input data
    suffix : str
        month / year index
    format: str
        datetime format

    Returns:
    ----------
        transformed df : pd.DataFrame      
    """
    for column in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
        df.loc[:, column]  = pd.to_datetime(df.loc[:, column], format=format)
    df.columns = df.columns.map(lambda x: x.replace(' ', ''))
    valid_preriod = get_valid_preriod(suffix)
    df = df.query('(passenger_count > 0) & (trip_distance > 0)')
    df = df.query('tpep_pickup_datetime < tpep_dropoff_datetime') 
    df = df.query('@valid_preriod[0] <= tpep_pickup_datetime < @valid_preriod[1]')
    end_plus = valid_preriod[1] + datetime.timedelta(days=1)
    df = df.query('@valid_preriod[0] < tpep_dropoff_datetime < @end_plus')
    return df


def daily_aggregation(df: pd.DataFrame, zones_in_use: List[int]) -> pd.DataFrame:
    """Aggregates drives data to hour / zone_id format
    
    Parameters
    ----------
    df : str
        data
    zones_in_use : NY zones for forecasting
        List[int]
    
    Returns:
    ----------
        transformed df : pd.DataFrame   
    """
    df = df.rename(columns={'PULocationID': 'zone_id'})
    df['datetime'] = df.loc[:, 'tpep_pickup_datetime'].map(lambda x: x.replace(minute=0, second=0))
    df['duration'] = df.apply(lambda x: (x['tpep_dropoff_datetime'] - 
                                         x['tpep_pickup_datetime']).total_seconds()/60, axis=1)
    df = df.drop(['tpep_dropoff_datetime', 'tpep_pickup_datetime'], axis=1)
    data = df.groupby(['datetime', 'zone_id'], as_index=False).apply(lambda x: pd.Series({
                        'y' : x['zone_id'].count(),
                        'distance'   : x['trip_distance'].mean(),
                        'duration'   : x['duration'].mean(),
                        'passengers' : x['passenger_count'].mean(),
                        'cost'       : x['total_amount'].mean(),
                        'tips'       : x['tip_amount'].mean(),
                        'vendor'     : x['VendorID'].value_counts().to_dict().get(1)
                        }))
    # drop location
    dol = df.groupby(['datetime', 'DOLocationID'], as_index=False).apply(lambda x: pd.Series({
                      'dol' : x['zone_id'].count()}))
    data = data.merge(dol, how='left', left_on=['datetime', 'zone_id'], 
                        right_on=['datetime', 'DOLocationID']).drop('DOLocationID', axis=1)

    data.loc[:, 'zone_id'] = data.loc[:, 'zone_id'].astype(np.int32())
    data = data[data.zone_id.isin(zones_in_use)]
    return data

DataCleanerTransformer = functional_transformer(data_cleaner)
DailyAggregationTransformer = functional_transformer(daily_aggregation)

# ---------- preprocessing: transformer -------------

class Preprocessor:
    """Raw data preprocessor
    """
    def __init__(self, config):
        self.format = config['format']
        
        years = sorted(config['years'])
        self.period = [datetime.datetime(years[0], 1, 1), 
                       datetime.datetime(years[-1], 12, 31)]
        self.path_row = config['raw_output_path']
        self.path_preprocessed = config['path_preprocessed']      
        self.config = config
        self.zones_in_use = config['zones_in_use']

    def clean(self):
        """Removes all preprocessed files
        """
        for file_name in os.listdir('../' + self.path_preprocessed):
            os.remove(os.path.join('..', self.path_preprocessed, file_name))
            
    def _transform(self, suffix):
        """Applies data transformers to a single month data
        """
        file_name = os.path.join(self.path_row, 'yellow_tripdata_' + suffix + '.csv')
        df = pd.read_csv('../' + file_name)

        transformers = [('cleaner', DataCleanerTransformer(suffix=suffix, format=self.format)),
                        ('daily_aggregation', DailyAggregationTransformer(zones_in_use=self.zones_in_use))
                        ]
        pipeline = skpipe.Pipeline(transformers)
        df = pipeline.transform(df)

        file_name_out = os.path.join(self.path_preprocessed, 'preprocessed_' + suffix + '.csv')
        df.to_csv('../' + file_name_out, header=True, index=False)
        print(suffix, 'preprocessed - ok', df.shape)

    def update(self):
        """Preprocess raw data files ()
        """
        downloaded_files_in = os.listdir('../' + self.path_row)
        preprocessed_files_in = os.listdir('../' + self.path_preprocessed)

        # checks whether there are all needed files
        downloaded_files = set([f[-11:-4] for f in downloaded_files_in 
                                if f[:6] == 'yellow'])
        preprocessed_files = set([f[-11:-4] for f in preprocessed_files_in 
                                  if f[-24:-12] == 'preprocessed'])

        y0, m0 = self.period[0].year, self.period[0].month
        y1, m1 = self.period[1].year, self.period[1].month
    
        for m in range(m0, 12*(y1 - y0) + m1 + 2):
            year = self.period[0].year  + (m-1)//12
            month = m%12 + 12*int(m%12==0)
            suffix = get_suffix(year, month)

            if suffix not in downloaded_files | preprocessed_files:
                download_raw_data_month(suffix, self.config)
            if suffix not in preprocessed_files:
                self._transform(suffix)
            else:
                print(suffix, 'is already preprocessed')


    @staticmethod
    def concatenate(config: Dict, save: str = True) -> None:
        """Concatenates preprocessed files and add time based features
        """
        path_preprocessed = config['path_preprocessed']  
        preprocessed_files = os.listdir('../' + path_preprocessed)

        output_df = pd.DataFrame([])
        for file_name in preprocessed_files:
            df = pd.read_csv(os.path.join('..', path_preprocessed, file_name))
            output_df = pd.concat([output_df, df], axis=0)

        output_df.sort_values(['datetime', 'zone_id'], inplace=True)
        if save:
            file_name_out = os.path.join(config['path_preprocessed'] , 'preprocessed_data.csv')
            output_df.to_csv('../' + file_name_out, header=True, index=False)
        else:
            return output_df
