'''
Author: Yi-Fan Li
Created: 2020-05-29
Note: This is the main program for running STGCN-CNN model and STGCN-LSTM model.
'''

import dgl
import random
import argparse
import sys
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import logging
import yaml
import datetime


cur_dir = os.getcwd()
sys.path.append(cur_dir)

import torch as th
import torch.nn as nn
# from torchsummary import summary

from src.features.build_features import *
from model import STGCNCNN
from utils import evaluate_model, evaluate_metric

def main(args):
    '''
    Setting up parameters of the model
    '''
    target_task = args.task
    rm_category = args.rmcategory
    use_event_data = args.event
    temporal_model = args.tmodel


    yaml_path = 'src/configs/' + target_task + '.yaml'
    with open(yaml_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        print(params)
    
    
    logging.basicConfig(filename='log.log',
                        format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S %p',
                        level=10)

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    # event_data = params['event_data']
    event_data = use_event_data

    logging.debug('constructing graphs')
    # Construct the graph
    adj_path = params.get('adj_path')
    adj_matrix = get_adj_matrix(adj_path)
    sp_matrix = sp.coo_matrix(adj_matrix)
    # g = dgl.DGLGraph()
    # g.from_scipy_sparse_matrix(sp_matrix)
    g = dgl.from_scipy(sp_matrix)

    logging.debug('converting features')
    # Convert features to tensors
    proc_pkl = params.get('proc_pkl')
    feature_path_v = 'data/raw/pgm.csv'
    feature_path_u = 'data/processed/pgm_africa_utd.csv'
    views_data = get_feature_matrix(feature_path_v, feature_path_u, rm_category=rm_category, end_month=486, event_data=event_data) # t x n x d
    n_samples, n_nodes, n_features = views_data.shape 

    logging.debug('loading params')
    # Define the dir of saving model
    _dir = params.get('fdir')
    _filename = params.get('fname')
    now = datetime.datetime.now()
    time = now.strftime('%H-%M-%S')
    save_path = _dir + '/' + time + '-' + _filename

    # Load all parameters
    n_his = params.get('window')
    n_pred = params.get('npred')
    p_drop = params.get('pdrop')

    channels = params.get('channels')
    control_str = params.get('cstring')
    n_layer = params.get('nlayer')
    batch_size = params.get('bsize')
    lr = params.get('lr')
    epochs = params.get('epochs')
    train_start, train_end = params.get('train_split')
    val_start, val_end = params.get('val_split')
    test_start, test_end = params.get('test_split')

    # Define the training, testing, validation set
    train_val_split = train_end - train_start + 1
    val_test_split = val_end - val_start + 1 + train_val_split
    train = views_data[:train_val_split, :, :]
    val = views_data[train_val_split - n_his:val_test_split, :, :]
    test = views_data[val_test_split - n_his:, :, :]

    X_train, y_train = get_feature_seqs(train, n_his, n_pred)
    X_val, y_val = get_feature_seqs(val, n_his, n_pred)
    X_test, y_test = get_feature_seqs(test, n_his, n_pred)

    X_train = X_train[:, 7:47, :, :]
    X_val = X_val[:, 7:47, :, :]
    X_test = X_test[:, 7:47, :, :]

    train_data = th.utils.data.TensorDataset(X_train, y_train)
    train_iter = th.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_data = th.utils.data.TensorDataset(X_val, y_val)
    val_iter = th.utils.data.DataLoader(val_data, batch_size, shuffle=True)
    test_data = th.utils.data.TensorDataset(X_test, y_test)
    test_iter = th.utils.data.DataLoader(test_data, batch_size, shuffle=False)

    print('No. of channels:', channels)
    loss = nn.MSELoss()
    g = g.to(device)
    model = STGCNCNN(channels, n_his, n_nodes, g, p_drop, n_layer, control_str).to(device)
    # optimizer = th.optim.RMSprop(model.parameters(), lr=lr)
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    logging.debug('start training...')
    min_val_loss = np.inf
    for epoch in range(1, epochs + 1):
        # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.debug('. Currently running Epoch {}...'.format(str(epoch)))
        l_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze()
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = evaluate_model(model, loss, val_iter, device=device)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            th.save(model.state_dict(), save_path)
        print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
    
    logging.debug('start evalutation')
    print('staring evaluation...')
    print(datetime.datetime.now())
    best_model = STGCNCNN(channels, n_his, n_nodes, g, p_drop, n_layer, control_str).to(device)
    best_model.load_state_dict(th.load(save_path))     
    MAE, MSE, CRPS, RMSE = evaluate_metric(best_model, test_iter, device=device)
    print('test loss:', l, '\nMAE', MAE, '\nMSE', MSE, '\nCRPS', CRPS, '\nRMSE', RMSE)
    print('finished...')
    print(datetime.datetime.now())
    logging.debug('finished evaluation')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(descriptioin='STGCN parameters')
    parser.add('--event', type=bool, default=True, help='determine if utd event data is used.')
    parser.add('--rmcategory', type=list, default=[0], help='determine if a certain category of feature needs to be removed.')
    parser.add('--tmodel', type=str, default='tcn', help='determine the model for temporal relationship modeling.')
    parser.add('--task', type=int, default=1, help='determine the prediction task of the model')
    args = parser.parse_args()
    main(args)