'''
Author: Yi-Fan Li
Created: 2020-06-05
Note: This program contains evaluations for both traninig and testing
'''

import torch as th
import numpy as np
import properscoring as ps

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
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze()
            y, y_pred = y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mse += (d ** 2).tolist()
        
        # MAE = np.array(mae).mean()
        # MSE = np.array(mse).mean()

        MAE, MSE, CRPS = [], [], []
        for i in range(6):
            MAE.append(np.array(mae)[:, i, :].mean())
            MSE.append(np.array(mse)[:, i, :].mean())
            CRPS.append(ps.crps_ensemble(0, np.array(mae)[:, i, :].flatten()))

    return MAE, MSE, CRPS



