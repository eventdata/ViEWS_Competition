'''
Author: Yi-Fan Li
Created: May 29, 2020
This program aims to process the data and generate features for STGCN
'''

import numpy as np
import pandas as pd
import torch as th
import pickle as pk
from collections import defaultdict
import bisect
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

def get_adj_matrix(path):
    '''Create the adjacency matrix of the graph
    Parameters
    ----------
    path : str
        The relative path to the file contains PRIO-grid information
    
    Returns
    -------
    adj_matrix : np.ndarray
        The adjacency matrix regarding PRIO-grid cells
    '''
    pg_af_df = pd.read_csv(path, sep=',')

    # Loop 1: create dictionary to store the pg information: 
    # key: tuple(row, col), value: list(pgid)
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


def get_feature_matrix(path_v, path_u, event_data=False, start_month=241, end_month=456):
    '''Get features as GCN input
    Parameters:
    -----------
    path_v : str
        The relative path to the file contains PRIO-grid level features of views data features
    path_u : str
        The relative path to the file contains PRIO-grid level features of utd event data features   
    event_data : bool
        Parameter that controls whether we want to use utd event data as additional feature inputs
    start_month : int
        ID of month that defines the first month of the data, month 0 refers to Jan, 1980
    end_month : int
        ID of month that defines the last month of the data

    Returns:
    --------
    views_data : np.ndarray
        The feature matrix with dimension of t x n x d, where t represents
        number of time steps, n represents number of nodes, and d represents
        number of features
    '''
    df_v = pd.read_parquet(path_v)
    df_u = pd.read_csv(path_u)
    views_features = ['acled_count_pr', 'acled_fat_pr', 'ged_best_ns',
                      'ged_best_os', 'ged_count_ns',
                      'ged_count_os', 'ged_count_sb', 'pgd_agri_gc',
                      'pgd_agri_ih', 'pgd_aquaveg_gc', 'pgd_barren_gc',
                      'pgd_barren_ih', 'pgd_bdist3', 'pgd_capdist',
                      'pgd_cmr_mean', 'pgd_diamprim', 'pgd_diamsec',
                      'pgd_drug_y', 'pgd_excluded', 'pgd_forest_gc',
                      'pgd_forest_ih', 'pgd_gcp_mer', 'pgd_gem',
                      'pgd_goldplacer', 'pgd_goldsurface', 'pgd_goldvein',
                      'pgd_grass_ih', 'pgd_gwarea', 'pgd_harvarea',
                      'pgd_herb_gc', 'pgd_imr_mean', 'pgd_landarea',
                      'pgd_maincrop', 'pgd_mountains_mean', 'pgd_nlights_calib_mean',
                      'pgd_pasture_ih', 'pgd_petroleum', 'pgd_pop_gpw_sum',
                      'pgd_savanna_ih', 'pgd_shrub_gc', 'pgd_shrub_ih',
                      'pgd_temp', 'pgd_ttime_mean', 'pgd_urban_gc',
                      'pgd_urban_ih', 'pgd_water_gc', 'pgd_water_ih', 'ln_ged_best_sb'] 
    utd_features_dict = {1: 'cameo_01', 2: 'cameo_02', 3: 'cameo_03', 4: 'cameo_04',
                         5: 'cameo_05', 6: 'cameo_06', 7: 'cameo_07', 8: 'cameo_08',
                         9: 'cameo_09', 10: 'cameo_10', 11: 'cameo_11', 12: 'cameo_12',
                         13: 'cameo_13', 14: 'cameo_14', 15: 'cameo_15', 16: 'cameo_16',
                         17: 'cameo_17', 18: 'cameo_18', 19: 'cameo_19', 20: 'cameo_20'}
    
    df_views_features = df_v[views_features]

    idx = pd.IndexSlice
    df_sliced = df_views_features.loc[idx[start_month:end_month]]
    df_sliced_utd = df_sliced.copy(deep=True)
    min_max_scaler = MinMaxScaler()
    max_abs_scaler = MaxAbsScaler()

    if event_data == False:
        # np_sliced = df_sliced.values
        cols_to_norm = views_features[:(len(views_features)-1)]
        df_sliced[cols_to_norm] = min_max_scaler.fit_transform(df_sliced[cols_to_norm])

        # Check if nan in data. If there is nan, raise error.
        if True in df_sliced.isna().any().tolist():
            raise Exception('Nan exists in the dataset, please double-check.')
        
        print(df_sliced.values)
        views_data = np.reshape(df_sliced.values, (end_month - start_month + 1, -1, 48))
        return views_data
    else:
        for feature in utd_features_dict.values():
            df_sliced_utd[feature] = 0

        for idx, row in df_u.iterrows():
            df_sliced_utd.loc[(row['month'], row['pg-id']), utd_features_dict[row['cameo rt']]] = row['counts']
            if int(idx) % 10000 == 0:
                print('processed:', idx)

        for feature in utd_features_dict.values():
            df_sliced_utd[feature] = df_sliced_utd[feature] / (df_sliced_utd['pgd_pop_gpw_sum'] + 1)

        cols = df_sliced_utd.columns.tolist()
        new_cols = cols[:47] + cols[48:] + [cols[47]]
        df_sliced_utd = df_sliced_utd[new_cols]
        feature_list_utd = list(df_sliced_utd.columns)
        cols_to_norm = feature_list_utd[:(len(feature_list_utd) - 1)]
        # df_sliced_utd[cols_to_norm] = min_max_scaler.fit_transform(df_sliced_utd[cols_to_norm])
        df_sliced_utd[cols_to_norm] = max_abs_scaler.fit_transform(df_sliced_utd[cols_to_norm])
        utd_data = np.reshape(df_sliced_utd.values, (end_month - start_month + 1, -1, 68))

        # Temporarily save the data to pickle
        pk.dump(utd_data, open('data/processed/pgm_utd_features.pkl', 'wb'))
        return utd_data


# This scaler needs to be further tested.
def get_feature_seqs(data, n_his, n_pred):
    '''Produce data seqs
    Parameters:
    -----------
    data : np.ndarray
        Sliced feature matrix for training, validation, and testing. The dimension
        of this matrix is t x n x d, where t is the number of time steps, n is
        the number of nodes and d is the number of features.
    n_his : int
        Number of time steps of history in order to predict fatalities in the
        future.
    n_pred : int
        Number of time steps in the future for predictions.
    
    Returns:
    --------
    X : th.Tensor
        The feature matrix with dimension of m x d x n_his x n, where m is the
        number of sequences.
    y : th.Tensor
        The label matrix with dimension of m x n_pred x n.
    '''
    n_samples, n_nodes, n_features = data.shape
    l = len(data)
    n_windows = l - n_his - n_pred
    # The last column in feature dimension is the output
    data = th.Tensor(data)
    X = th.zeros([n_windows, n_features - 1, n_his, n_nodes])
    y = th.zeros([n_windows, n_pred, n_nodes])

    cnt = 0

    for i in range(n_windows):
        head = i
        tail = i + n_his
        # Need permute overhere
        X[cnt, :, :, :] = data[head:tail, :, :-1].permute(2, 0, 1)
        y[cnt, :, :] = data[tail:tail + n_pred, :, -1]
        cnt += 1
    return X, y

def scale_features(train, val, test):
    '''Scale the train, val and test data
    Parameters:
    -----------
    train : np.ndarray
        The training matrix, with dimension of t x n x d.
    val : np.ndarray
        The validation matrix, with dimension of t x n x d.
    test : np.ndarray
        The testing matrix, with dimension of t x n x d.

    Returns:
    --------
    train : np.ndarray
        The scaled training matrix, with dimension of t x n x d.
    val : np.ndarray
        The scaled val matrix, with dimension of t x n x d.
    test : np.ndarray
        Tehe scaled testing matrix, with dimension of t x n x d.
    '''

    # The scaler is for each of the channels
    scalers = {}
    for i in range(train.shape[2]): 
        scalers[i] = StandardScaler()
        train[:, :, i] = scalers[i].fit_transform(train[:, :, i])
        val[:, :, i] = scalers[i].transform(val[:, :, i])
        test[:, :, i] = scalers[i].transform(test[:, :, i])
        mean_val = []
        for j in range(train.shape[1]):
            mean_val.append(np.mean(train[:, j, i]))
        print(i, 'max', max(mean_val))
        print(i, 'min', min(mean_val))
        print('finish')
    return train, val, test

if __name__ == '__main__':

    # # Test the get_adj_matrix function
    # path = 'data/processed/priogrid_AF.csv'
    # adj_matrix = get_adj_matrix(path)
    # print(adj_matrix)

    # Test the get_feature_matrix function
    path = 'data/processed/pgm_africa_imp_0.parquet'
    feature_matrix = get_feature_matrix(path)