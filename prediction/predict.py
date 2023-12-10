from typing import Dict, Tuple, List
import os
import sys
import requests
import datetime
import mlflow

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

os.chdir('/usr/src/app/prediction')

sys.path.append("/usr/src/app")
from ynyt.features import BaseFeatures, FeatureCombiner


AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
S3_ENDPOINT_URL = os.environ["MLFLOW_S3_ENDPOINT_URL"]
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
ARTIFACT_URI = os.environ["ARTIFACT_URI"]

# dashboard address
api_ip = os.environ["APP_IP"]
api_port = 8081

def update_predictions(period: str, mode: str, model_paths: dict) -> pd.DataFrame:
    """batch inference
      Arguments:
        period: str
          batch period: day / month
        mode : str 
          can only take values 'train', 'validation', 'inference'
        model_paths : Dict[str]
          paths to models
      Returns:
        predictions : pd.DataFrame
          predictions in inference mode, fact / predictions otherwise
    """
    path_preprocessed = "./data/preprocessed" # local
    inference_base_features = BaseFeatures(period, 
                                           config, 
                                           path_preprocessed, s3=True).fit(mode=mode)

    def back_normalization(y, bf=inference_base_features):
        return bf.target_normalizer.transform_back(bf.data, y)

    fc_inference = FeatureCombiner(config, inference_base_features).fit()

    df_output = []
    for horizon in range(1, 7):
        model = mlflow.sklearn.load_model(model_paths[horizon])
        X, y = fc_inference.transform(horizon=horizon, mode=mode)
        y_pred = model.predict(X)
        df = inference_base_features.data.loc[:, ['t', 'datetime', 'zone_id']]
        df['h'] = horizon
        df['pred'] = back_normalization(y_pred)
        df['pred'] = np.maximum(0, df['pred'].values)
        if mode != 'inference':
            df['fact'] = back_normalization(y).astype(int)
        df_output.append(df)
    df_output = pd.concat(df_output)
    print(f'{datetime.datetime.now()}: predictions are updated, {period}')
    return df_output


def update_results(results: pd.DataFrame, period, api_ip: str = api_ip, api_port: str = api_port) -> None:
    """updates data in dashboard
    """
    results_json = results.to_json(orient='records')
    api_url = f'http://{api_ip}:{api_port}/update/{period}'
    print(api_url)
    response = requests.post(api_url, results_json)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")


# mlflow client
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# config
config = mlflow.artifacts.load_dict(ARTIFACT_URI + "/configs/features_conf.json")

# model paths
model_paths = dict()
for horizon in range(1, 7):
    model_paths[horizon ] = f'models:/fourie_plus_cartesian_ar_h{horizon}/Staging'

# current day / month imitation
year = 2022
month = 5
day = datetime.datetime.now().day
hour = datetime.datetime.now().hour

# daily update
h_start = datetime.datetime(year, month, day, hour, 0)
period = [h_start - datetime.timedelta(days=14), h_start + datetime.timedelta(days=1)]
results_day = update_predictions(period, 'inference', model_paths)
update_results(results_day, 'day')

# monthly update
api_url = f'http://{api_ip}:{api_port}/month/'
response = requests.get(api_url)
if response.status_code != 200:
   print(f"Error: {response.status_code}")

forced = bool(response.json()['result'])

if datetime.datetime.now().day == 1 or forced:
    period = [datetime.datetime(year, month, 1, 1, 0) - datetime.timedelta(days=14), 
              datetime.datetime(year, month, 31, 23, 0)]
    results_month = update_predictions(period, 'validation', model_paths)
    update_results(results_month, 'month')
