'''
Author: Yi-Fan Li
Created: 2020-06-05
Note: This program contains evaluations for both traninig and testing
'''

import torch as th
import numpy as np
import properscoring as ps
from datetime import datetime

def evaluate_model(model, loss, data_iter, device='cpu'):
    model.eval()
    l_sum, n = 0.0, 0
    with th.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze()
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, device='cpu'):
    model.eval()
    with th.no_grad():
        mae, mse = [], []
        y_pred_hist, y_hist = [], []
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze()
            y, y_pred = y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            y_hist.append(y)
            y_pred_hist.append(y_pred)
            # now = datetime.now()
            # time = now.strftime('%H-%M-%S')
            # y_fname = 'y-actual-' + time
            # y_pred_fname = 'y-pred-' + time
            # np.save(y_fname, y)
            # np.save(y_pred_fname, y_pred)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mse += (d ** 2).tolist()
        
        now = datetime.now()
        time = now.strftime('%H-%M-%S')
        y_fname = 'y-actual-' + time
        y_pred_fname = 'y-pred-' + time
        np.save(y_fname, y_hist)
        np.save(y_pred_fname, y_pred_hist)
        # MAE = np.array(mae).mean()
        # MSE = np.array(mse).mean()

        MAE, MSE, RMSE = [], [], []
        for i in range(6):
            MAE.append(np.array(mae)[:, i, :].mean())
            MSE.append(np.array(mse)[:, i, :].mean())
            RMSE.append(np.sqrt(np.array(mse)[:, i, :].mean()))
    


    return MAE, MSE, RMSE



