import os
import json
from datetime import datetime

import pandas as pd
import geopandas

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import plotly.graph_objs as go
import plotly.express as px

from utils import select_zone_h, get_r_squared, plot_map_r2
from utils import plot_map
from utils import current_hour_text
from utils import init_predictions, exchange_update


os.environ['DASH_DEBUG_MODE'] = 'False'
#os.environ['PATH_DATA'] = '../data/'
# #os.environ['PATH_CONFIG'] = '../'

PATH_DATA = os.environ['PATH_DATA']
PATH_CONFIG = os.environ['PATH_CONFIG']

path_day = os.path.join(PATH_DATA, 'results_day.csv')
path_month = os.path.join(PATH_DATA, 'results_month.csv')


# predictions initialization
init_predictions()

updater_exchange = {'day': True, 'month': True, 
                    'day_ctime': os.path.getctime(path_day), 'month_ctime': os.path.getctime(path_month)}
os.environ['UPDATER_EXCHANGE'] = json.dumps(updater_exchange)


with open(os.path.join(PATH_CONFIG, 'config.json')) as json_file:
    config = json.load(json_file)


def update_data() -> None:
    """Updates prediction and validations DataFrames
    """
    def to_datetime(df, column='datetime'):
        df[column]  = pd.to_datetime(df[column], unit='ms')
        return df

    global results_day, results_month, r2, r2_start, h0, id_start

    exchange_update()
    updater_exchange = json.loads(os.environ['UPDATER_EXCHANGE'])

    #  short run predictions
    if updater_exchange["day"]:
        results_day = pd.read_csv(path_day)
        results_day = to_datetime(results_day)

    # validations
    if updater_exchange["month"]:
        results_month = pd.read_csv(path_month)
        results_month = to_datetime(results_month)
        r2 = get_r_squared(results_month)
        r2.r2_score = r2.r2_score.map(lambda x: round(x, 3))
        r2_start = r2[(r2['h']==h0)&(r2['zone_id']==id_start)].r2_score.values[0]


# --- initial values ----
h0 = 1
id_start=163
update_data()

# ----- ny zones geo data -----  
dbf = geopandas.GeoDataFrame.from_file(os.path.join('data', 'taxi_zones', 'taxi_zones.dbf'))
dbf = dbf.to_crs(4326)


# ------ layouts ------

forecast_layout = html.Div([
    html.Div(
        dcc.Graph(id='ny-map')),
    
    html.Div(dcc.Slider(
        results_day['h'].min(),
        results_day['h'].max(),
        step=1,
        id='h-slider',
        value=h0
    ),
        className='w-30'
    ),
    html.Div(
        dcc.Interval(
        id='interval-component',
        interval=10*1000,
        n_intervals=0
    )
    ),
    html.Div(id='map_slider_text')
])

analytics_layout = html.Div(
    [html.Div([
        html.Div([
            dcc.Graph(
                id='crossfilter-ny-map',
                clickData={'points': [{'location': id_start, 'z': r2_start}]}
            )
        ],
            className='w-50 d-sm-inline-block'
        ),

        html.Div([
            dcc.Graph(
                id='crossfilter-plot',
            )
        ],
            className='w-50 d-sm-inline-block'
        )
    ]),
    
    html.Div(dcc.Slider(
        results_month['h'].min(),
        results_month['h'].max(),
        step=1,
        id='crossfilter-h-slider',
        value=h0
    ),
        className='w-30'
    ),
    html.Div(id='map_r2_slider_text')
])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
            #requests_pathname_prefix='/dashboard/'
            )
server = app.server

n_tabs = config['output']['n_tabs']
if n_tabs == 2:
    app.layout = dbc.Container(
        [ 
        html.Div(
                "Shortrun forecast for NY yellow taxi demand",
                className="heading",
            ),
        html.Div(config['output']['description'], className='description'),       

            dbc.Tabs(
                [
                    dbc.Tab(label='{}-hour forecast'.format(6), 
                            tab_id='map',
                            tabClassName='dash-tab',
                            labelClassName='.dash-tab-label',
                            activeTabClassName='dash-tab-active'),
                    dbc.Tab(label="Model analytics", 
                            tab_id='analytics',
                            tabClassName='dash-tab',
                            labelClassName='.dash-tab-label',
                            activeTabClassName='dash-tab-active'),
                ],
                id="tabs",
                active_tab="map",
            ),
            html.Div(id="tab-content"),
            html.Div(
                'Andrei Chekunov, 2020-2023',
                className='page-footer',
            )
        ]
    )
else:
    app.layout = dbc.Container(
    [ 
      html.Div(
            "Shortrun forecast for NY yellow taxi demand",
            className="heading",
        ),
      html.Div(config['output']['description'], className='description'),       
      html.Div(forecast_layout, className='one-tab-case')
    ]
)


# ---- tabs callback ----

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab_content(active_tab):
    update_data()
    if active_tab is not None:
        if active_tab == 'map':
            return forecast_layout
        elif active_tab == 'analytics':
            return analytics_layout
    return forecast_layout

# ---- forecast tab callbacks  ----
t0 = datetime.now().hour
h0 = 1
forced = True

@app.callback(
    Output('ny-map', 'figure'),
    [Input('h-slider', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_map_hourly(h, n_intervals):
    global t0, h0, forced
    if datetime.now().hour != t0 or h != h0 or forced:
        update_data()
        fig = plot_map(h, results_day, dbf, config)
        title = 'Forecast for {}'.format(current_hour_text(h, config))
        fig.update_layout(title=title, mapbox={"zoom": 10.4}, margin={"r":30,"t":80,"l":25,"b":30})
        h0 = h
        forced = False
        t0 = datetime.now().hour
        return fig
    else:
        return dash.no_update

# comment to slider
@app.callback(
    Output('map_slider_text', 'children'),
    Input('h-slider', 'value'))
def update_output_d1(value):
    return '{}-hour forecast'.format(value)

#  ---- analytics tab callbacks  ----

# map update
@app.callback(
    Output('crossfilter-ny-map', 'figure'),
    Input('crossfilter-h-slider', 'value'),
)
def update_map_d2(h):
    global forced
    update_data()
    fig = plot_map_r2(h, results_month, dbf)
    title = 'Forecast quality, h={}'.format(h)
    fig.update_layout(title=title, margin={"r":55,"t":78, "l":30,"b":50})
    forced = True
    return fig

@app.callback(
    Output('crossfilter-plot', 'figure'),
    Input('crossfilter-ny-map', 'clickData'),
    Input('crossfilter-h-slider', 'value'),
)

# graph update
def update_graph_d2(clickData, h):
    """Updates fact/predict graph for chosen zone_id and forecast hour

    clickData(dcc.Input): infornmation of the selected zone
    h(dcc.Input): forecast hour 
    """

    update_data()

    zone_id = clickData['points'][0]['location']
    id_ =  clickData['points'][0]['location']
    
    r2_scores = r2[(r2['h']==h)&(r2['zone_id']==id_)].r2_score.values[0]
    dff = pd.melt(select_zone_h(zone_id, h, results_month).loc[:, ['t', 'fact', 'pred']], id_vars='t')
    fig = px.line(dff, x='t', y='value', color='variable', template='plotly_white',
                  color_discrete_sequence = ['rgb(128, 177, 211)', 'rgb(133, 92, 117)'
                  ]
                  )

    fig.update_traces(opacity=0.7)
    title = 'Validation: zone id={}, h={}, R<sup>2</sup>={}'.format(zone_id, h, round(r2_scores, 2))
    
    fig.update_layout(title=title, xaxis = go.layout.XAxis(
        title = 'Hours',
        showticklabels=False), yaxis = go.layout.YAxis(
        title = 'Drives'),
        legend_title = '',
        margin={"r":75, "t":75, "l":0,"b":45}
        )
    return fig


#  slider comment
@app.callback(
    Output('map_r2_slider_text', 'children'),
    Input('crossfilter-h-slider', 'value'))
def update_output(value):
    return '{}-hour forecast'.format(value)

# healthcheck endpoint
@app.server.route("/healthcheck")
def ping():
  return "{status: ok}"


DASH_DEBUG_MODE = os.environ['DASH_DEBUG_MODE']

if __name__ == "__main__":
    app.run_server(debug=DASH_DEBUG_MODE, host='0.0.0.0', port=8080)
    #app.run_server(debug=DASH_DEBUG_MODE)
