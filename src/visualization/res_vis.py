import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas
import contextily as ctx
import views
import views.apps.plot.maps as maps
import pandas as pd
import os
import datetime as dt
import numpy as np
import csv

from sklearn.metrics import mean_squared_error


# Load map data for plot
mapdata = views.apps.plot.maps.MapData()
gdf_pgm = views.GEOMETRIES["GeomPriogrid"].gdf
df_pg_c = views.TABLES["skeleton.pg_c"].df.rename(
    columns={"country_id": "geo_country_id"}
)
gdf_pgm.join(df_pg_c)

col = 'ln_ged_best_sb'

df = views.DATASETS["pgm_africa_imp_0"].df[[col]]

df = df.sort_index()

tgt_file = './data/ViEWSpred_competition_Brandt_task1.csv'  
match_file = './ground_truth.csv'

main_df = pd.read_csv(tgt_file)
match_df = pd.read_csv(match_file)
main_frame = main_frame.filter(items=['pg_id', 'month_id', 'predict', 'dv']).set_index(['month_id', 'pg_id']).sort_index()

df.loc[488][col] = main_frame.loc[490]['predict']

maps.plot_map(
    s_patch=df[col],
    mapdata=mapdata,
    cmap='viridis', #seismic
    tick_values=[0.0, 0.2, .4, .6, .8, 1., 1.2, 1.4, 1.6],
    ymin=0.0,
    ymax=1.6,
    logodds=False,
    t=488,
    title=f'2020-10',
    textbox=f'Name: {col}\nDate: Oct 2020',
)

tgt_file = './data/ViEWSpred_competition_Brandt_task1.csv'
match_file = './ground_truth.csv'  


main_df = pd.read_csv(tgt_file)
match_df = pd.read_csv(match_file)
main_frame = main_frame.filter(items=['pg_id', 'month_id', 'predict', 'dv']).set_index(['month_id', 'pg_id']).sort_index()

df.loc[488][col] = main_frame.loc[495]['predict']

maps.plot_map(
    s_patch=df[col],
    mapdata=mapdata,
    cmap='viridis', #seismic
    tick_values=[0.0, 0.2, .4, .6, .8, 1., 1.2, 1.4, 1.6],
    ymin=0.0,
    ymax=1.6,
    logodds=False,
    t=488,
    title=f'2021-03',
    textbox=f'Name: {col}\nDate: Mar 2021',
)