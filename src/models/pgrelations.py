'''
Author: Yifan
Date: May 15, 2020
This code convert the data format to follow the adjuncency matrix requirement
'''

import pandas as pd
import numpy as np
import bisect
from collections import defaultdict

def get_adj_matrix(path):

    pg_af_df = pd.read_csv(path, sep=',')

    # Loop 1: create dictionary to store the pg information: key: tuple(row, col), value: list(pgid)
    pg_dict = defaultdict()
    for idx, row in pg_af_df.iterrows():
        key = (row['row'], row['col'])
        val = row['gid']
        pg_dict[key] = val

    # Loop 2: create the connection dataframe: from, to, connection
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    pg_list = pg_af_df['gid'].tolist()
    pg_nums = pg_af_df.shape[0]
    adj_matrix = np.zeros((pg_nums, pg_nums))

    for idx, row in pg_af_df.iterrows():
        # from_pg_tmp = row['gid']
        # Adding self connection:
        row_center = row['row']
        col_center = row['col']
        adj_matrix[idx][idx] = 1

        for _dir in dirs:
            row_tmp, col_tmp = row_center + _dir[0], col_center + _dir[1]
            loc_tmp = (row_tmp, col_tmp)
            if loc_tmp in pg_dict:
                pg_tmp = pg_dict[loc_tmp]
                idx_tmp = bisect.bisect_left(pg_list, pg_tmp)
                adj_matrix[idx][idx_tmp] = 1
    return adj_matrix