import os
import datetime
import pandas as pd
import numpy as np
import json
import geopandas
from sklearn.metrics import r2_score

import plotly.express as px

from pydantic import BaseModel
from pydantic import Field

import time


PATH_DATA = os.environ['PATH_DATA']


# -- initialization ---

def init_predictions():
    day = True
    month = True
    while day or month:
        time.sleep(3)
        if os.path.isfile(os.path.join(PATH_DATA, 'results', f'results_month.csv')):
            month = False
            print('results_month.csv is there')
        if os.path.isfile(os.path.join(PATH_DATA, 'results', f'results_day.csv')):
            day = False
            print('results_day.csv is there')
    print('predictions are initialized')

# ------ filtering functions ------

def select(results, x, column='datetime'):
    df = results[results[column]==x]
    return df

def select_zone_h(zone_id, h, df):
    return df[(df['zone_id']==zone_id)&(df['h']==h)]

# ------ functions for output ------

def get_r_squared(results_month):
    """
    calculate r2_squared metric for all zine_id / h
    Args:
      None
    Return:
      r2(pd.DataFrame) r2 values for zine_id / h
    """
    def f_agg(x):
        return pd.Series({'r2_score': r2_score(x['fact'], x['pred'])})
    df = results_month.groupby(['zone_id', 'h'], as_index=False).apply(f_agg)
    return df

def current_hour_text(h, config):
    """
    Args:
      h(int): forecast hour 
    Returns:
      uptut_hour(str): simulation of current hour + h
    """
    h0 = datetime.datetime(datetime.datetime.now().year,
                             datetime.datetime.now().month, 
                             datetime.datetime.now().day, 
                             datetime.datetime.now().hour, 0)
    ouptut_hour = h0 + datetime.timedelta(seconds=3600*(h + config['output']['UTC correction']))
    return ouptut_hour.strftime("%d/%m %Hh")



# ------ map output functions ------

def plot_map(h, results_day, dbf, config):
    """Plots ny map with forecast results

    Args:
      h(int): forecsat horizont 
      dbf(geopandas.DataFrame): ny zones geo data
    Rerurn:
      None
    """
    # hour of prediction (in the data)
    prediction_hour = (datetime.datetime(2022, 5,
                                    datetime.datetime.now().day, 
                                    datetime.datetime.now().hour, 0)) 

    df = select(results_day, prediction_hour, column='datetime')
    df = select(df, h, column='h')
    df['drives'] = np.around(df['pred'], decimals=0).astype(int)
    
    fig = px.choropleth_mapbox(
        df,
        geojson=dbf,
        locations="zone_id",
        featureidkey='properties.LocationID',
        color="drives",
        mapbox_style="carto-positron",
        center={"lat": 40.76, "lon": -73.87}, 
        opacity=0.8, 
        color_continuous_scale = px.colors.sequential.Blues,
        title='Forecast for {}'.format(current_hour_text(h, config)),
        hover_name="zone_id"
    )
    return fig

def plot_map_r2(h, results_month, dbf):
    """Plots ny map with r2_score values 

    Args:
      h(int): forecsat horizont 
      dbf(geopandas.DataFrame): ny zones geo data
    Rerurn:
      None
    """   
    
    r2 = get_r_squared(results_month)

    fig = px.choropleth_mapbox(
        r2[r2['h']==h],
        geojson=dbf,
        locations="zone_id",
        featureidkey='properties.LocationID',
        color="r2_score",
        mapbox_style= "carto-positron",
        center={"lat": 40.761, "lon": -73.95}, 
        opacity=0.65, 
        color_continuous_scale = px.colors.sequential.BuPu, 
        title='Forecast quality, h={}'.format(h),
        hover_name="zone_id",
        labels={
            'r2_score': 'R<sup>2</sup>'
        }
    ).update_layout(mapbox={"zoom": 10.4})
    return fig


def exchange_update():
    """env flag for day / month updates
    """
    updater_exchange = json.loads(os.environ['UPDATER_EXCHANGE'])
    for period in ['day', 'month']:
        path = os.path.join(os.environ['PATH_DATA'], 'results', f'results_{period}.csv')
        ctime = os.path.getctime(path)
        if updater_exchange[f'{period}_ctime'] != ctime:
            updater_exchange['{period}'] = False
            updater_exchange[f'{period}_ctime'] = ctime
            print(f'{period} is updated')

    os.environ['UPDATER_EXCHANGE'] = json.dumps(updater_exchange)
