from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import datetime
import holidays
from pickle import dump, load
from sklearn.preprocessing import StandardScaler
from pickle import dump, load


# ------ time based features ---------

def get_holidays(period_: List[datetime.datetime]) -> pd.DataFrame:
    """Assigns working / holiday / ... day index to a date range:
       - holiday or weekend after working
       - working day after holiday or weekend
       - working day before holiday or weekend
    
    Parameters
    ----------
        period_ : List[datetime.datetime]

    Returns:
    ----------
        mapper date to index : pd.DataFrame
    """
    def daytype(period_window: List[datetime.datetime]):
        # assigns index
        W = range(5)
        H = [5, 6, 7]
        if (period_window[1] in H) and (period_window[0] in W): res = 1   # holiday or weekend after working
        elif (period_window[1] in H) and (period_window[2] in W): res = 2 # working day after holiday or weekend
        elif (period_window[1] in W) and (period_window[2] in H): res = 3 # working day before holiday or weekend 
        else: res = np.nan
        return res

    period = [period_[0] - datetime.timedelta(days=1), 
              period_[1] + datetime.timedelta(days=1)]
    ny_holidays = holidays.country_holidays('US', subdiv='NY')[period[0].strftime(format='%Y-%m-%d'): 
                                                               period[1].strftime(format='%Y-%m-%d')]
    n = (period[1] - period[0]).days
    alldays = [(period[0] + datetime.timedelta(days=k)) for k in range(n)]
    alldays = pd.DataFrame(alldays, columns=['date'])
    alldays.loc[:, 'weekday'] = alldays.loc[:, 'date'].map(lambda x: x.weekday())

    indices = alldays[(alldays['weekday'].isin(range(5)))&(alldays['date'].isin(ny_holidays))].index
    alldays.loc[indices, 'weekday'] = 7 * np.ones(len(indices))
    alldays.loc[:, '-1'] = alldays.loc[:, 'weekday'].shift(1)
    alldays.loc[:, '1'] = alldays.loc[:, 'weekday'].shift(-1)
    alldays.loc[:, 'weekday'] = alldays.apply(lambda x: daytype([x['-1'], x['weekday'], x['1']]), axis=1)

    return alldays.drop(['-1', '1'], axis=1).iloc[1:-1, :]

def base_time_features(df: pd.DataFrame, features_on: Dict = None) -> pd.DataFrame:
    """Add time-based features

    Parameters
    ----------
    df : str
        data
    features_on : Dict
        optional features on / off
    
    Returns:
    ----------
        transformed df : pd.DataFrame  
    """
    df.loc[:, 'date'] = df.loc[:, 'datetime'].map(lambda h: h.date())
    df.loc[:, 'hours'] = df.loc[:, 'datetime'].map(lambda h: h.hour)

    if features_on is None:
        features_on = {'weekday': True, 'weekday_plus': True, 'weekhours': True}
    if features_on['weekday']:
        df.loc[:, 'weekday'] = df.loc[:, 'date'].map(lambda x: x.weekday())
    if features_on['weekday_plus']:
        period = [df.loc[:, 'date'].min(), df.loc[:, 'date'].max()]
        df = df.merge(get_holidays(period), on='date', how='left')
        df.loc[:, 'weekday_plus'] = df.loc[:, 'weekday_plus'].fillna(0)
    if features_on['weekhours']:
        df.loc[:, 'weekhours'] = df.apply(lambda x: x.hours + 24*x['weekday'], axis=1)
    df.drop(['date'], axis=1,inplace=True)
    return df

# ------- fourier harmonics  -------

def fourier_harmonics(time: np.array, K: Dict):
    """Transform time period to fourier harmonics

    Parameters
    ----------
    df : np.array
        full time period (in hours)
    K: Dict
        dimension settings for different periods 
    """
    T = time.shape[0]
    columns = []

    if K['year']:
        x_sin_week, x_cos_week = np.zeros((K['year'], T)), np.zeros((K['year'], T))
        for k in range(K['year']):
            x_sin_week[k] = np.vectorize(lambda x: np.sin(x*2*np.pi*(k + 1)/8766))(time)
            x_cos_week[k] = np.vectorize(lambda x: np.cos(x*2*np.pi*(k + 1)/8766))(time)
        columns += ['s_y' + str(k) for k in range(K['year'])] + ['c_y' + str(k) for k in 
                                                                range(K['year'])]
        X = np.concatenate((x_sin_week, x_cos_week), axis=0)

    if K['week']:
        x_sin_week, x_cos_week = np.zeros((K['week'], T)), np.zeros((K['week'], T))
        for k in range(K['week']):
            x_sin_week[k] = np.vectorize(lambda x: np.sin(x*2*np.pi*(k + 1)/168))(time)
            x_cos_week[k] = np.vectorize(lambda x: np.cos(x*2*np.pi*(k + 1)/168))(time)
        columns += ['s_w' + str(k) for k in range(K['week'])] + ['c_w' + str(k) for k in 
                                                                range(K['week'])]
        if K['year']:
            X = np.concatenate((X, x_sin_week, x_cos_week), axis=0)
        else:
            X = np.concatenate((x_sin_week, x_cos_week), axis=0)    

    if K['day']:
        x_sin_day, x_cos_day = np.zeros((K['day'], T)), np.zeros((K['day'], T))    
        for k in range(K['day']):
            x_sin_day[k] = np.vectorize(lambda x: np.sin(x*2*np.pi*(k + 1)/24))(time)
            x_cos_day[k] = np.vectorize(lambda x: np.cos(x*2*np.pi*(k + 1)/24))(time) 
        columns += ['s_d' + str(k) for k in range(K['day'])] + ['c_d' + str(k) for k in 
                                                                range(K['day'])]
        X = np.concatenate((X, x_sin_day, x_cos_day), axis=0)

    X = np.concatenate((time.reshape((1, -1)), X), axis=0)   
    return pd.DataFrame(X.T, columns=['t'] + columns)


# ------- rolling features -----

def rolling_features(
    df: pd.DataFrame,
    features: Dict[str, Tuple[str, int, str]],
) -> pd.DataFrame:
    """
    Add rolling features to the DataFrame based on the specified aggregations:
        - autoregressive values: "ar"
        - different aggegations: "sum", "avg"
    For each id, the features are computed as the aggregations of the last N-hours.
    Current hours is always included into rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add the feature to. Changes are applied inplace.
    features : Dict[str, Tuple[str, int, str, Optional[int]]]
        Dictionary with the following structure:
        {
            "feature_name": ("agg_col", "hours", "aggregation_function"),
            ...
        }
        where:
            - feature_name: name of the feature to add
            - agg_col: name of the column to aggregate
            - int: number of hours to include into rolling window
            - aggregation_function: one of the following: "ar", "sum", "avg"

    Raises
    ------
    ValueError
        If aggregation_function is not one of the following:
        "ar", "sum", "avg"
    """
    for feature_name, (agg_col, hours, agg_func) in features.items():
        if agg_func == "ar":
            df[feature_name] = (
                df.groupby("zone_id")[agg_col]
                .shift(hours)
                .reset_index(level=0, drop=True)
            )
        elif agg_func == "sum":
            df[feature_name] = (
                df.groupby("zone_id")[agg_col]
                .rolling(window=hours)
                .sum()
                .reset_index(level=0, drop=True)
            )
        elif agg_func == "avg":
            df[feature_name] = (
                df.groupby("zone_id")[agg_col]
                .rolling(window=hours)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
    return df


def add_targets(df: pd.DataFrame, target: Tuple[str, str, int]) -> pd.DataFrame:
    """
    Add targets to the DataFrame based on the specified aggregations.
    For each id, the targets is computed as the aggregations of the next N-days.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add the target to. Changes are applied inplace.
    targets : Tuple[str, str, int]
        Dictionary with the following structure:
            ("target_name", "agg_col", "hours"),
        where:
            - target_name: name of the target to add
            - agg_col: name of the column to aggregate
            - hours: number of next hour to include into rolling window
            (current hour is always excluded from the rolling window)
    """
    targets = {target[0] + '_' + str(i): (target[1], i) for i in range(1, target[2] + 1)}
    for feature_name, (agg_col, hours) in targets.items():
        df['zone_id_tmp'] = df['zone_id']
        df[feature_name] = (
                   df.iloc[::-1].groupby('zone_id_tmp')[["zone_id", agg_col]].shift(hours)
            .groupby("zone_id")[agg_col]
            .rolling(window=1)
            .sum()
            .reset_index(level=0, drop=True)
            )
        df.drop('zone_id_tmp', inplace=True, axis=1)
    return df


def dell_nan_rolling_features(
    df: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    """drops rows with Nan
    """
    indices = set()
    for feature_name in features:
        indices = indices.union(set(df[df[feature_name].isna()].index.to_list()))
    df.drop(indices, inplace=True)
    return df


def rolling_config(config: Tuple[str, Dict]) -> Dict[str, Dict]:
    """Build config dict for rolling features.

    Parameters:
    ----------
        eg. ["y", {"sum": [12, 24, 168], "ar_d": 4, "ar_D": 2}]
    
    Returns:
    ----------
        config dict for rolling features:
            {"feature_name": ("agg_col", hour, "agg_func")}
    """
    features = {}
    configs = {}
    for agg_col, aggregators in config:
        for agg_func, hours in aggregators.items():
            if isinstance(hours , int):
                hours = range(1, hours + 1)
            features[agg_func] = []
            for hour in hours:
                if agg_func == "ar_d":
                    feature_name = agg_col + '_' + str(hour)
                    configs [feature_name] = (agg_col, hour - 1, "ar")
                elif agg_func == "ar_D":
                    for j in range(-1, 6):
                        feature_name = agg_col + "_" + str(hour*24) + '_' + str(j + 1)
                        configs [feature_name] = (agg_col, 23 - j + (hour - 1)*24, "ar")
                else:
                    feature_name = agg_col + '_' + agg_func + '_' +  str(hour)
                    configs [feature_name] = (agg_col, hour, agg_func)
                features[agg_func].append(feature_name)
    return configs, features


# ------- normalizers -----

class FeatureNormalizer:
    """class for feature normalization
    """
    def __init__(self, config) -> None:
        self.scaler_path = config['normalizer']['path'] + '/' + 'feature_scaler.pkl'
        self.to_normalize = config['normalizer']['to_normalize']

    def fit(self, df: pd.DataFrame, mode: str):
        if mode == 'train':
            self.scaler = StandardScaler()
            self.scaler.fit(df.loc[:, self.to_normalize])
            with open(self.scaler_path, 'wb') as f:
                dump(self.scaler, f)
        else:
            with open(self.scaler_path, 'rb')  as f:
                self.scaler = load(f)

    def transform(self, df):
        not_to_normalize  = [c for c in df.columns if c not in self.to_normalize]
        df_sub_norm = pd.DataFrame(self.scaler.transform(df.loc[:, self.to_normalize]), 
                               columns=self.to_normalize, index=df.index)
        df_normalized = pd.concat([df.loc[:, not_to_normalize], 
                                    df_sub_norm], axis=1)
        return df_normalized.loc[:, df.columns]


class TargetNormalizer:
    """class for target forward nad back normalization
    """
    def __init__(self, config) -> None:
        self.filename_target_max = config['normalizer']['path'] + '/scaler_target_max.csv'
        self.filename_target_min = config['normalizer']['path'] + '/scaler_target_min.csv'
        self.target = config['normalizer']['target']

    def fit(self, df, mode='train'):
        if mode == 'train':
            Y = df.loc[:, ['t', 'zone_id', self.target]].pivot_table(values=self.target, 
                                                            index='t', columns='zone_id')
            self.Y_min = Y.min(axis=0)
            self.Y_max = Y.max(axis=0)

            self.Y_min.to_csv(self.filename_target_min, header=True, index=False)
            self.Y_max.to_csv(self.filename_target_max, header=True, index=False)
        else:
            self.Y_min = pd.read_csv(self.filename_target_min).loc[:, '0']
            self.Y_max = pd.read_csv(self.filename_target_max).loc[:, '0']
    
    def transform(self, df: pd.DataFrame, column: str ='y'):
        Y = df.loc[:, ['t', 'zone_id', column]].pivot_table(values=column, 
                                                        index='t', columns='zone_id')
        df.loc[:, column] = ((Y- self.Y_min.values) /
              (self.Y_max.values - self.Y_min.values)).unstack().values
        return df
    
    def transform_back(self, _df: pd.DataFrame, y: np.array):
        df = _df.loc[:, ['t', 'zone_id']]
        df['y'] = y
        Y = df.loc[:, ['t', 'zone_id', 'y']].pivot_table(values='y', 
                                                index='t', columns='zone_id')
        return (Y*(self.Y_max.values - self.Y_min.values) + self.Y_min.values).unstack().values
