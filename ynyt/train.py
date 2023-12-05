import os
import sys
import datetime
import mlflow

import pandas as pd

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ynyt.features import BaseFeatures, FeatureCombiner
from ynyt.utils import json_read



# datasets
train_period = [datetime.datetime(2021, 10, 1), datetime.datetime(2022, 4, 30)]
test_period = [datetime.datetime(2022, 4, 17), datetime.datetime(2022, 5, 31)]

data_config = json_read('../configs/data_conf.json')
config = json_read('../configs/features_conf.json')
path_preprocessed = data_config['path_preprocessed']

train_base_features = BaseFeatures(train_period, config, path_preprocessed).fit(mode='train')
test_base_features = BaseFeatures(test_period, config, path_preprocessed).fit(mode='validation')

def n_back(y, bf_tr=train_base_features, bf_val=test_base_features, mode='t'):
    """warper on back normalizer
    """
    if mode == 't':
        res = bf_tr.target_normalizer.transform_back(bf_tr.data, y)
    else:
        res = bf_val.target_normalizer.transform_back(bf_val.data, y)
    return res


# mlflow experiment
experiment = client.get_experiment_by_name("fourie_plus_cartesian_ar")
print(experiment)

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='main') as run:
    config = json_read('../configs/features_conf.json')
    mlflow.log_artifact('../configs/features_conf.json', artifact_path="configs")
    fc_train = FeatureCombiner(config, train_base_features).fit()
    fc_test = FeatureCombiner(config, test_base_features).fit()
    models = {}

    # loop per forecasting hours
    for horizon in range(1, 7):
        with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True) as nested_run:
            # train
            model = linear_model.Ridge(alpha=1, fit_intercept=False)

            X, y = fc_train.transform(horizon=horizon, mode='train')
            model = model.fit(X, y)
            y_pred = model.predict(X)
            output = f', train: {r2_score(y, y_pred):.5f}, {r2_score(n_back_(y), n_back(y_pred)):.5f}'

            # eval
            X, y = fc_test.transform(horizon=horizon, mode='test')
            y_pred = model.predict(X)
            r2_test = round(r2_score(y, y_pred), 5)
            mae = round(mean_absolute_error(y, y_pred), 5)
            mse = round(mean_squared_error(y, y_pred), 5)
            r2_test_y = round(r2_score(n_back(y, mode='v'), n_back(y_pred, mode='v')), 5)

            print(f't={horizon}, mse: {mse}, mae: {mae}, r2: test: {r2_test}, {r2_test_y}' + output)

            model_name = f"fourie_plus_cartesian_ar_rige{horizon}"
            mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)
            mlflow.log_metric("r2", r2_test)
            mlflow.log_metric("mse", mse)
