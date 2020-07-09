# This file create the visualization for Africa each year

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

years = range(2000, 2017)

for year in years:
    _dir = './data/interim/'
    agg_data = _dir + 'ICEWS' + str(year) + '_agg.json'
    pg_data = _dir + 'priogrid_AF.csv'

    df_pg = pd.read_csv(pg_data)
    df_icews = pd.read_json(agg_data)

    df_icews.rename({'pg-id': 'gid'}, axis=1, inplace=True)
    df_conflict = df_icews[df_icews['cameo rt'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])]
    df_icews_year = df_conflict[['gid', 'counts']].groupby(['gid'], as_index=False).sum()

    df_join = df_pg.merge(df_icews_year, how='left', left_on='gid', right_on='gid')
    df_fill0 = df_join.fillna(value=0)
    # df_fill0.to_csv('counts.csv')
    df_fill0['log_counts'] = np.log(df_fill0['counts'] + 1)


    ax1 = df_fill0.plot.scatter(x='col', y='row', c='log_counts', colormap='viridis', s=1)
    plt.savefig('icews' + str(year) + '.png')

print(' ')