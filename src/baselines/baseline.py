import sys
import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import openviews2
from openviews2.config import LOGFMT
from openviews2.utils.io import csv_to_df
from openviews2.utils.data import assign_into_df
from openviews2.apps.transforms import lib as translib
from openviews2.apps.model import api

logging.basicConfig(format=LOGFMT, stream=sys.stdout, level=logging.INFO)
paths = {'cm.csv': 'data/raw/cm.csv', 'pgm': 'data/raw/pgm.csv'}

df = csv_to_df(paths['cm.csv']).set_index(['month_id', 'country_id']).sort_index()
# for col in df:
#     print(col)

period_develop = api.Period(
    name='develop',
    train_start=121,
    train_end=444,
    predict_start=445,
    predict_end=480
)
periods = [period_develop]

steps = [1, 2, 3, 5, 6]

# Transformations. See views.apps.transforms.lib for more options
# Feel free to add any you like
df["ma_12_ged_best_sb"] = translib.moving_average(df["ged_best_sb"], time=12)
df["ma_12_ged_best_ns"] = translib.moving_average(df["ged_best_ns"], time=12)
df["ma_12_ged_best_os"] = translib.moving_average(df["ged_best_os"], time=12)
df["time_since_ged_dummy_sb"] = translib.time_since_previous_event(df["ged_dummy_sb"])
df["time_since_ged_dummy_ns"] = translib.time_since_previous_event(df["ged_dummy_ns"])
df["time_since_ged_dummy_os"] = translib.time_since_previous_event(df["ged_dummy_os"])

# Specify your wanted feature sets
cols_features_raw = [
    "ged_best_sb",
    "ged_best_ns",
    "ged_best_os",
]
cols_features_transforms = [
    "ma_12_ged_best_sb",
    "ma_12_ged_best_ns",
    "ma_12_ged_best_os",
    "time_since_ged_dummy_sb",
    "time_since_ged_dummy_ns",
    "time_since_ged_dummy_os",
]

# Specify an optional downsampling level
downsample_half = api.Downsampling(share_positive=0.5, share_negative=0.5, threshold=0)

# Define the models

model_raw = api.Model(
    name="raw",                      # A descripte name
    col_outcome="ln_ged_best_sb",    # The outcome column, log of state-based fatalities
    cols_features=cols_features_raw, # The list of features
    steps=steps,                     # The list of steps
    outcome_type="real",             # The outcome type, can be "real" or "prob"
    periods=periods,                 # The list of periods to work on
    estimator=RandomForestRegressor( # Defining the estimator to use
        criterion="mse",
    ),
    delta_outcome = True             # Specifies that the model should take the delta 
                                     # of the outcome column before training and when evaluation
)

model_transforms = api.Model(
    name="raw_and_transforms",
    col_outcome="ln_ged_best_sb",
    cols_features=cols_features_raw + cols_features_transforms,
    steps=steps,
    outcome_type="real",
    periods=periods,
    estimator=RandomForestRegressor(
        criterion="mse",
    ),
    delta_outcome = True,
    downsampling=downsample_half
)

models = [model_raw, model_transforms]

# Train all models
for model in models:
    model.fit_estimators(df)

# Store predictions for all models in our dataframe
for model in models:
    df_predictions = model.predict_steps(df)
    df = assign_into_df(df, df_predictions)

# Evaluate all models. Scores are stored in the model object
for model in models:
    model.evaluate(df)

# Show our scores, it looks like transforms did some good. 
for model in models:
    print(model.name)
    print(pd.DataFrame(model.scores))