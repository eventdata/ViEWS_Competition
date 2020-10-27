import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

gt_path = 'data/raw/pgm.csv'
pg_path = 'data/processed/priogrid_AF.csv'
pred_path = 'data/results/views_pred.csv'

# gt_df = pd.read_csv(gt_path, index_col=[0, 1], skipinitialspace=True)
pg_df = pd.read_csv(pg_path)
pred_df = pd.read_csv(pred_path, index_col=[0], skipinitialspace=True)

for i in range(445, 481):
    idx = pd.IndexSlice

    pred_feature_list = ['pg_id','Brandt_STGCNTCN_s2']
    pred_fal = pred_df[pred_feature_list]
    pred_480_fal = pred_fal.loc[idx[i]]
    pred_480_fal.rename({'Brandt_STGCNTCN_s2': 'log_fatalities'}, axis=1, inplace=True)

    # gt_feature_list = ['ln_ged_best_sb']
    # gt_fal = gt_df[gt_feature_list]
    # gt_480_fal = gt_fal.loc[idx[i]]
    # gt_480_fal.rename({'ln_ged_best_sb': 'log_fatalities'}, axis=1, inplace=True)


    df_join = pg_df.merge(pred_480_fal, how='left', left_on='gid', right_on='pg_id')
    # df_join = pg_df.merge(gt_480_fal, how='left', left_on='gid', right_on='pg_id')
    # for idx, row in df_join.iterrows():
    #     if row['log_fatalities'] <= 0:
    #         df_join.loc[idx]['log_fatalities'] = 0
    df_join['log_fatalities'][df_join['log_fatalities'] < 0] = 0

    ax1 = df_join.plot.scatter(x='col', y='row', c='log_fatalities', colormap='magma', s=1)
    # ax1.axes.get_xaxis().set_ticks([])
    # ax1.axes.xaxis.set_ticklabels([])
    ax1.yaxis.set_visible(False)
    ax1.xaxis.set_visible(False)
    plt.savefig('pred_'+str(i)+'.png')


# print(gt_fal)

