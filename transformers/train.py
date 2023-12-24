import sys

import datetime
import json

from sttre.sttre import train_val


filename = sys.argv[1]
with open(f'./settings/{filename}', 'r') as f:
    settings = json.load(f)

print(filename)

regression_head = settings['regression_head']
data_setting = settings['data_setting']
params = settings['params']

period = {'train': [datetime.datetime(2020, 10, 1), datetime.datetime(2022, 4, 30)], 
          'val': [datetime.datetime(2022, 5, 1), datetime.datetime(2022, 7, 31)]}

path_preprocessed = './../data/preprocessed'
horizon = 6

model = train_val(period=period, path_preprocessed=path_preprocessed,
                  regression_head=regression_head, data_setting=data_setting, 
                  verbose=True, horizon=horizon, **params)
