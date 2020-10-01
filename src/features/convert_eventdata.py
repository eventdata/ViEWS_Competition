'''
Author: Yifan
Date: May 28, 2020
This code process the raw ICEWS/PHOENIXRT data to conduct the following two tasks:
    1. Assign an additional column 'prio-grid' to the data table;
    2. Aggregate the count of events groupby the following conditions:
        a. year
        b. month
        c. prio-grid id
        d. CAMOE root code
'''

import pandas as pd
import numpy as np
import argparse
import json
import math
from collections import defaultdict

def convert_eventdata(args):
    src_type = args.sourceType
    year_id = args.year
    opt_no = args.output

    print(' ')
    print('Generating 1st', src_type, 'file for year', year_id)
    
    # Configure the data sources
    _dir = './data/raw/'
    src_dir = _dir + 'Africa' + src_type + year_id + '.json'
    opt_dir = _dir + src_type + year_id + '_withPG.json'
    opt_dir_f2 = _dir + src_type + year_id + '_agg.json'

    # Load json file into memory
    with open(src_dir) as j:
        json_dict = json.load(j)
        data = json_dict['data'] # Exclude 'citation' and only select 'data'
    
    df = pd.json_normalize(data)
    pg = pd.read_csv('./data/raw/priogrid_AF.csv')

    # Change all column names to lower case:
    df.columns = map(str.lower, df.columns)
    # The column names of ICEWS and PHOENIX are not the same, even some of them are the same features
    # For example: CAMEO code (ICEWS) --> code (PHOENIXRT)
    # Also, date format in ICEWS follows 'yyyy-mm-dd', we are adding two more columns in here for
    #   the ease of processing: ['year', 'month']
    # Pre-process datasets to align the later procedure
    if src_type == 'ICEWS':
        df.rename(columns={'cameo code': 'cameo'}, inplace=True)
        year = []
        month = []
        for _, row in df.iterrows():
            year_tmp = row['event date'][:4]
            month_tmp = row['event date'][5:7]
            year.append(year_tmp)
            month.append(month_tmp)
        df['year'] = year
        df['month'] = month
    elif src_type == 'PHOENIXRT':
        df.rename(columns={'code': 'cameo'}, inplace=True)

    # Latitude ~ ycoord; Longitude ~ xcoord
    # pg_dict is a two layer dictionary that can be retreived by key: pg_dict[lat_box][long_box]
    # Value is the prio-grid id of a certain half-degree bounding box
    pg_dict = defaultdict(dict)
    for latitude in pg.ycoord.unique():
        lat_box = get_coord_box(latitude)
        pg_dict[lat_box] = defaultdict(int)
    for _, row in pg.iterrows():
        lat_box = get_coord_box(row.ycoord)
        long_box = get_coord_box(row.xcoord)
        pg_dict[lat_box][long_box] = row.gid


    # Iterate through the df to assign the prio-grid id
    df_pg_id = []
    for _, row in df.iterrows():
        lat_box = get_coord_box(row.latitude)
        long_box = get_coord_box(row.longitude)
        if long_box not in pg_dict[lat_box].keys():
            df_pg_id.append(0)
        else:
            pg_tmp = pg_dict[lat_box][long_box]
            df_pg_id.append(pg_tmp)

    df['pg-id'] = df_pg_id

    is_event = (df['pg-id'] != 0)
    df_africa = df[is_event].reset_index(drop=True)
    # The next line export the processed dataframe to json
    df_africa.to_json(opt_dir, orient='records')

    # Then we need to return addtional summary table if the opt_no equals 2, which means that we want
    #   to generate aggregation files as well.
    if opt_no == 2:
        print('Generating 2nd', src_type, 'file for year', year_id)
        CAMEO_root_code = []
        for _, row in df_africa.iterrows():
            if not row['cameo']:
                root_code_tmp = '--'
            else:
                root_code_tmp = row['cameo'][:2]
            CAMEO_root_code.append(root_code_tmp)
        df_africa['cameo rt'] = CAMEO_root_code
        df_tmp = df_africa[['year', 'month', 'cameo rt', 'pg-id']]
        # In PHOENIXRT, some cameo codes are missing, therefore we are filtering out those records for aggregation.
        is_not_null = (df_tmp['cameo rt'] != '--')
        df_tmp = df_tmp[is_not_null]

        # Generate aggregation dataframe and export
        df_count = df_tmp.groupby(['year', 'month', 'cameo rt', 'pg-id']).size().reset_index(name='counts')
        # The next line export aggregated info into json files
        df_count.to_json(opt_dir_f2, orient='records')
        # pass

    print(' ')
    return

# This function
def get_pg_box(latitude, longitude):
    pg_box = []

    # We handle the latitude first here.
    lat_box = get_coord_box(latitude)
    long_box = get_coord_box(longitude)
    pg_box.append((lat_box, long_box))
    return pg_box

# Given a coordinate (latitude or longitude), this function calculates for its bounding box.
def get_coord_box(coordinate):
    is_neg = 0
    if coordinate < 0:
        is_neg = 1
        coordinate = abs(coordinate)
    if (coordinate - math.floor(coordinate)) > 0.5:
        lower_bound = math.floor(coordinate) + 0.5
        upper_bound = math.ceil(coordinate)
    else:
        lower_bound = math.floor(coordinate)
        upper_bound = math.floor(coordinate) + 0.5
    if is_neg:
        coord_box = (-upper_bound, -lower_bound)
    else:
        coord_box = (lower_bound, upper_bound)
    return coord_box

if __name__ == '__main__':

    years = ['2017', '2018', '2019', '2020']
    for year in years:
        parser = argparse.ArgumentParser(description='UTD Event Data Converter', conflict_handler='resolve')
        parser.add_argument('--sourceType', type=str, default='ICEWS', help='type of source data tables')
        parser.add_argument('--year', type=str, default=year, help='year of data tables') # Four digit
        parser.add_argument('--output', type=int, default=2, help='number of output files')
        args = parser.parse_args()
        convert_eventdata(args)
