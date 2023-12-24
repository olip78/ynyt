import os
import sys

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

with open("./sttre/configs/config.env") as f:
    for line in f.readlines():
        if len(line) > 2:
            k, v = line[:-2].split('=')
            os.environ[k] = v

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from ynyt.features import BaseFeatures
from ynyt.utils import json_read

import warnings
warnings.filterwarnings('ignore')


class YNYT(Dataset):
    def __init__(self, period, mode, path_preprocessed, setting, seq_len=4, horizon=6):
        
        config = json_read('./sttre/configs/config.json')

        self.seq_len = seq_len
        base_features = BaseFeatures(period, config, path_preprocessed).fit(mode=mode)
        base_features.data.t = base_features.data.t - base_features.data.t.values[0]
        base_features.harmonics.t = base_features.harmonics.t - base_features.data.t.values[0]
        self.num_var = base_features.data.zone_id.unique().shape[0]

        self.X = []
        # AR features
        for c in [base_features.feature_groups['ar_d'][0]]:
            self.X.append(base_features.data.pivot_table(index='t', columns='zone_id', 
                                   values=c, aggfunc='first').values)

        # drives information
        if setting['features']:
            for c in base_features.config['add_features']:
                self.X.append(base_features.data.pivot_table(index='t', columns='zone_id', 
                                       values=c, aggfunc='first').values)
        
        if setting['D']:
            D = config["base_features"]["rolling"][0][1]['ar_D']
            D_columns = [f'y_{168*k}_0' for k in range(1, D//7 + 1)]
            for c in [base_features.feature_groups['ar_d'][0]] + base_features.config['add_features'] + D_columns:
                self.X.append(base_features.data.pivot_table(index='t', columns='zone_id', 
                                       values=c, aggfunc='first').values)
        # dummy 
        if setting['hours']:
            f = 'hours'
            df = pd.concat([base_features.data.loc[:, ['t', 'zone_id']], 
                            pd.get_dummies(base_features.data.loc[:, f], prefix=f)], axis=1)
            for c in df.columns[2:]:
                self.X.append(df.pivot_table(index='t', columns='zone_id', values=c, aggfunc='first').values)    

        if setting['weekday']:
            f = 'weekday'
            df = pd.concat([base_features.data.loc[:, ['t', 'zone_id']], 
                            pd.get_dummies(base_features.data.loc[:, f], prefix=f)], axis=1)
            for c in df.columns[2:]:
                self.X.append(df.pivot_table(index='t', columns='zone_id', values=c, aggfunc='first').values)    


        self.X = np.array(self.X)
        self.bf = base_features

        # additional features for final MLP
        self.regressors = []
        D = config["base_features"]["rolling"][0][1]['ar_D']
        D_columns = [f'y_{168*k}_{h}' for k in range(1, D//7 + 1) for h in range(1, horizon+1)]
        for c in D_columns:
            self.regressors.append(base_features.data.pivot_table(index='t', columns='zone_id', 
                                   values=c, aggfunc='first').values)
        self.regressors = np.hstack(self.regressors)
        
        if mode != 'inference':
            self.target = True
        else:
            self.target = False
        self.horizon = horizon
        #self.t_in = [self.seq_len, self.X.shape[1] - self.target * self.horizon]
        self.t_in = [D*168, self.X.shape[1] - self.target * self.horizon]
        self.zone_id = base_features.data.zone_id.unique()
        
        self.len = self.X.shape[1] - self.seq_len + 1 - self.target * self.horizon

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # history drives
        x = self.X[:, idx:idx+self.seq_len, :]
        r = self.regressors[idx+self.seq_len-1:idx+self.seq_len, :]
        if self.target:
            # targets
            y = self.X[0, idx+self.seq_len:idx+self.seq_len + self.horizon, :]
            return torch.tensor(x, dtype=torch.float).unsqueeze(-1), torch.tensor(r, dtype=torch.float), torch.tensor(y, dtype=torch.float)
        else:
            return torch.tensor(x, dtype=torch.float).unsqueeze(-1), torch.tensor(r, dtype=torch.float), None

    def output_to_df(self, outputs):
        """
        Arguments:
            outputs 
        """
        columns = [f'h_{i}' for i in range(1, 7)]
        res = []
        zeros = np.zeros(self.zone_id.shape[0])
        values = [zeros, self.zone_id] + 6*[zeros]
        res = []
        for output, t in zip(outputs, self.t_in):
            df_base = pd.DataFrame(values, index=['t', 'zone_id']+columns).T
            df_base.t = t
            df_base.loc[:, columns] = output
            res.append(df_base)
        res = pd.concat(res, axis=0)
        res.loc[:, ['t', 'zone_id']] = res.loc[:, ['t', 'zone_id']].astype(int)
        dt = self.bf.data.loc[:, ['t', 'zone_id', 'datetime']]
        dt = dt[dt.zone_id==self.zone_id[0]]
        dt = dt.drop('zone_id', axis=1)
        res = res.merge(dt, on='t', how='left')
        return res
            
    def normalize_back(self, Y): 
        drives = []
        for y in Y.T:
            drives.append(self.bf.target_normalizer.transform_back(self.bf.data, y))
        return np.hstack(drives)
