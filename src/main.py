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

cur_dir = os.getcwd()
sys.path.append(cur_dir)

import torch as th
import torch.nn as nn
# from torchsummary import summary

from src.features.build_features import *
from model import STGCNCNN
from utils import evaluate_model, evaluate_metric

def main():
    '''
    Setting up parameters of the model
    '''
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--bsize', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='epochs for training, default: 50')
    parser.add_argument('--window', default=54, type=int, help='window length')
    parser.add_argument('--fdir', default='models', type=str, help='directory to save trained models')
    parser.add_argument('--fname', default='stgcn_new.pt', type=str, help='name of model to be saved')
    parser.add_argument('--npred', default=6, type=int, help='number of steps/months to predict')
    parser.add_argument('--channels', default=[47, 32, 16, 32, 16, 6], type=int, nargs='+', help='model structure controller')
    parser.add_argument('--pdrop', default=0, type=float, help='probability setting for the dropout layer')
    parser.add_argument('--nlayer', default=9, type=int, help='number of layers')
    parser.add_argument('--cstring', default='TNTSTNTST', type=str, help='model architecture controller, T: Temporal Layer; S: Spatio Layer; N: Normalization Layer')
    args = parser.parse_args()

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    # Construct the graph
    adj_path = 'data/processed/priogrid_AF.csv'
    adj_matrix = get_adj_matrix(adj_path)
    sp_matrix = sp.coo_matrix(adj_matrix)
    g = dgl.DGLGraph()
    g.from_scipy_sparse_matrix(sp_matrix)

    # Convert features to tensors
    feature_path_v = 'data/processed/pgm_africa_imp_0.parquet'
    feature_path_u = 'data/processed/pgm_africa_utd.csv'
    views_data = get_feature_matrix(feature_path_v, feature_path_u) # t x n x d
    n_samples, n_nodes, n_features = views_data.shape 

    # Define the dir of saving model
    _dir = args.fdir
    _filename = args.fname
    save_path = _dir + '/' + _filename

    # Load all parameters
    n_his = args.window
    n_pred = args.npred
    p_drop = args.pdrop

    channels = args.channels
    control_str = args.cstring
    n_layer = args.nlayer
    batch_size = args.bsize
    lr = args.lr
    epochs = args.epochs

    # Define the training, testing, validation set
    train_val_split = 420 - 241 + 1
    val_test_split = 432 - 421 + 1 + train_val_split
    train = views_data[:train_val_split, :, :]
    val = views_data[train_val_split - n_his:val_test_split, :, :]
    test = views_data[val_test_split - n_his:, :, :]

    X_train, y_train = get_feature_seqs(train, n_his, n_pred)
    X_val, y_val = get_feature_seqs(val, n_his, n_pred)
    X_test, y_test = get_feature_seqs(test, n_his, n_pred)

    train_data = th.utils.data.TensorDataset(X_train, y_train)
    train_iter = th.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_data = th.utils.data.TensorDataset(X_val, y_val)
    val_iter = th.utils.data.DataLoader(val_data, batch_size, shuffle=True)
    test_data = th.utils.data.TensorDataset(X_test, y_test)
    test_iter = th.utils.data.DataLoader(test_data, batch_size, shuffle=True)

    loss = nn.MSELoss()
    g = g.to(device)
    model = STGCNCNN(channels, n_his, n_nodes, g, p_drop, n_layer, control_str).to(device)
    optimizer = th.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    # print(summary(model, (32, 48, 54, 10677)))

    min_val_loss = np.inf
    for epoch in range(1, epochs + 1):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.warning('. Currently running Epoch {}...'.format(str(epoch)))
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


    best_model = STGCNCNN(channels, n_his, n_nodes, g, p_drop, n_layer, control_str).to(device)
    best_model.load_state_dict(th.load(save_path))     


    l = evaluate_model(best_model, loss, test_iter, device=device)
    MAE, MSE, CRPS = evaluate_metric(best_model, test_iter, device=device)
    print('test loss:', l, '\nMAE', MAE, '\nMSE', MSE, '\nCRPS', CRPS)

if __name__ == '__main__':
    main()