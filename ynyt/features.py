import os
import datetime

from typing import Dict, Tuple, List
from itertools import product

import pandas as pd
import numpy as np

import sklearn.base as skbase
import sklearn.pipeline as skpipe

from scipy.sparse import csr_matrix, hstack

from .feature_utils import base_time_features, fourier_harmonics
from .feature_utils import rolling_features, rolling_config
from .feature_utils import add_targets, dell_nan_rolling_features
from .feature_utils import TargetNormalizer, FeatureNormalizer
from .utils import functional_transformer, df_read
from .data import get_suffix


# ----- registration of transformers --------

def register_transformer(transformer_class, name: str, transformer_reference) -> None:
    """transformer register
    """
    transformer_reference[name] = transformer_class

RollingFeaturesTransformer = functional_transformer(rolling_features)
TargetTransformer = functional_transformer(add_targets)
DellNanFeaturesTransformer = functional_transformer(dell_nan_rolling_features)
TimeFeaturesTransformer = functional_transformer(base_time_features)


TRANSFORMER_REFERENCE = {}
register_transformer(TimeFeaturesTransformer, 'time_based', TRANSFORMER_REFERENCE)
register_transformer(RollingFeaturesTransformer, 'rolling', TRANSFORMER_REFERENCE)
register_transformer(TargetTransformer, 'target', TRANSFORMER_REFERENCE)
register_transformer(DellNanFeaturesTransformer, 'dell_nan', TRANSFORMER_REFERENCE)


# ----- transformers pipeline --------

def create_transformer(name: str, transformer_reference=TRANSFORMER_REFERENCE, **kwargs
                    ) -> skbase.BaseEstimator:
    return transformer_reference[name](**kwargs)

class BaseFeatures:
    """class for not combined features:
         - autoregressive ts
         - drive information
         - fourier harmonics
         - time / date features

       Arguments:
        period: List[datetime.datetime]
          period of data extraction
        path_preprocessed: str
          path to preprocessed data
        s3: bool
         s3 flag
    """
    def __init__(self, period: List[datetime.datetime], config: str, path_preprocessed: str, s3: bool = False):
        self.period = period
        self.config = config
        self.path_preprocessed = path_preprocessed
        self.zero_hour = datetime.datetime.strptime(config['zero_hour'][0], 
                                           config['zero_hour'][1])
        self.target_normalizer = TargetNormalizer(config)
        self.normalizer = FeatureNormalizer(config)
        self.s3 = s3

    def _build_df(self):
        """data extractor
        """
        preprocessed_files = []
        y0, m0 = self.period[0].year, self.period[0].month
        y1, m1 = self.period[1].year, self.period[1].month
        for y in range(y0, y1 + 1):
            if y1 == y0:
                months = range(m0, m1 + 1)
            elif y == y0:
                months = range(m0, 13)
            elif y == y1:
                months = range(1, m1 + 1)
            else:
                months = range(1, 13)

            for m in months:
                suffix = get_suffix(y, m)
                preprocessed_files.append(f'preprocessed_{suffix}.csv')

        output_df = pd.DataFrame([])
        path = os.path.join('..', self.path_preprocessed)
        for file_name in preprocessed_files:
            df = df_read(file_name, path, self.s3)
            output_df = pd.concat([output_df, df], axis=0)
        output_df['datetime']  = pd.to_datetime(output_df['datetime'])
        output_df['t'] = output_df['datetime'].map(lambda x: int((x - 
                                                      self.zero_hour).total_seconds() / 3600))
        self.data = output_df

    def _build_grid(self):
        """built dataset grid as cartesian product of hours x zones
        """
        h0 = 24*(self.period[0].date() - self.zero_hour.date()).days
        h1 = 24*((self.period[1].date() - self.zero_hour.date()).days + 1)
        all_hours = range(h0, h1)

        combinations = product(all_hours, self.data.zone_id.unique())
        grid = pd.DataFrame(combinations, columns=['t', 'zone_id'])
        grid['datetime'] = grid['t'].map(lambda x: self.zero_hour + datetime.timedelta(hours=x))
        grid = grid.merge(self.data.loc[:, ['y', 't', 'zone_id']],
                                    on = ['t', 'zone_id'], how='left').fillna(0)
        grid.y = grid.y.astype(np.int32)
        self.grid = grid
        return grid

    def _add_features(self, grid):
        """fills grid with data
        """
        features = self.config["add_features"]
        self.data = grid.merge(self.data.loc[:, ['t', 'zone_id'] + features], 
                              on = ['t', 'zone_id'], how='left').fillna(0)
        self.data = self.data.sort_values(['zone_id', 't']).reset_index(drop=True)
        self.data.loc[:, 'h'] = self.data.loc[:, 't']
        self.data.loc[:, 'velocity'] = (self.data.loc[:, "distance"] / self.data.loc[:, "duration"]).fillna(0)
        self.target_normalizer.fit(grid, mode=self.mode)
        self.normalizer.fit(self.data, mode=self.mode)
        self.data = self.target_normalizer.transform(self.data, column ='y')
        self.data = self.target_normalizer.transform(self.data, column ='dol') 
        self.data = self.normalizer.transform(self.data)

    def _build_pipeline(self):
        """builds transformation pipeline
        """
        configs = []
        self.feature_groups = {}
        features_to_dell_nan = []
        for _type, config in self.config['base_features'].items():
            if _type == 'rolling':
                config, features = rolling_config(config)
                self.feature_groups.update(features)
                features_to_dell_nan += list(config.keys())
                configs.append((_type, {'features': config}))
            elif _type == "target" and self.mode != 'inference':
                features = [config[0] + '_' + str(i) 
                                            for i in range(1, config[2] + 1)]
                configs.append((_type, {"target": config}))
                features_to_dell_nan += features
                self.feature_groups['target'] = features
            else:
                configs.append((_type, config))

        transformers = []
        for i, config in enumerate(configs):
            if self.mode == 'inference' and config[0] == "target":
                continue
            transformer = create_transformer(config[0], **config[1])
            transformers.append((f'stage{i}', transformer))

        # drop nan s
        transformer = create_transformer('dell_nan', **{'features': features_to_dell_nan})
        transformers.append(('dell_nan', transformer))

        self.pipeline = skpipe.Pipeline(transformers)
        self.grid = pd.DataFrame([])

    def _build_harmonics(self):
        """builds fourier harmonics
        """
        zone_id = self.data.loc[:, 'zone_id'].values[0]
        time = self.data[self.data.zone_id==zone_id].loc[:, 't'].sort_values().values
        K = self.config["harmonics"]["K"]
        self.harmonics = fourier_harmonics(time, K)

    def fit(self, mode: str, *args, **kwargs) -> None:
        """
        Arguments:
           mode : str 
              define transformations: normalization / add / not add targets
              can only take values 'train', 'validation', 'inference'
        """
        self.mode = mode
        self._build_df()
        self._add_features(self._build_grid())
        self._build_pipeline()
        self.data = self.pipeline.transform(self.data)
        self.data.loc[:, 'b'] = 1
        if 'harmonics' in self.config.keys():
            self._build_harmonics()
        return self

    def transform(self, *args, **kwargs):
        return self


class FeatureCombiner:
    """Class for building feature combinations
       Arguments:
        base_features: BaseFeatures
          base features
    """
    def __init__(self, config: Dict, base_features: BaseFeatures) -> None:
        self.bf = base_features
        self.config = config
        self.combinations = config['combinations']
        self.n_zones = len(self.bf.data.zone_id.unique())

    def _cartesian_product(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """build cartesian product of two indices
           Arguments:
            df1, df2: pd.DataFrame
              ...
           Returns:
            combined indices: pd.DataFrame
        """
        rows = product(df1.T.iterrows(), df2.T.iterrows())
        df = pd.DataFrame(left*right for (_, left), (_, right) in rows)
        return df.reset_index(drop=True).T

    def _one_hot(self, feature: str) -> pd.DataFrame:
        """one hot encoder
           Arguments:
            feature: str
              feature for encoding
          Returns: 
            encoded feature: pd.DataFrame
        """
        res = pd.get_dummies(self.bf.data.loc[:, [feature]], prefix=[feature], columns = [feature], drop_first=False)
        if feature=='weekday':
            for i in [1, 2, 3]:
                if 'weekday_' + str(float(i)) not in res.columns:
                    res['weekday_' + str(float(i))] = 0      
        return res

    def _build_product(self, combination):
        """build cartesian product of two features
           Arguments:
            df1, df2: pd.DataFrame
              ...
           Returns:
            combined features: pd.DataFrame
        """

        if isinstance(combination[0], list):
            features = combination[0]
        elif combination[0] != 'harmonics':
            features = self.bf.feature_groups[combination[0]]

        if combination[0] == 'harmonics':
            res = pd.concat([self.bf.harmonics.iloc[:, 1:]]*self.n_zones, axis=0)
            res.index = self.bf.data.index 
        else:
            res = self.bf.data.loc[:, features] 

        if combination[1] != 'linear':
            res = self._cartesian_product(res, self._one_hot(combination[1]))

        res = res.values
        return csr_matrix(res)

    def fit(self, *args, **kwargs):
        res = []
        for combination in self.combinations:
            res.append(self._build_product(combination))
        self.X = hstack(res)
        return self

    def transform(self, horizon, mode: str ='train') -> Tuple[csr_matrix, np.array]:
        target = self.config["base_features"]["target"][0]
        agg_col = self.config["base_features"]["rolling"][0][0]
        hours =  self.config["base_features"]["rolling"][0][1]["ar_D"]
        combinations = self.config['D_combinations']
        if len(combinations) > 0:
            res = []
            for second in combinations:
                features = [agg_col + "_" + str(h*24) + '_' + str(horizon) for h in range(1, hours + 1)]
                res.append(self._cartesian_product(self.bf.data.loc[:, features], self._one_hot(second)))
        X = hstack([self.X] + res)
        if mode != 'inference':
            y = self.bf.data.loc[:, target + '_' + str(horizon)].values
            return X, y
        else:
            return X, None
