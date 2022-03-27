
__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'


import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import fmin
from matplotlib import pyplot as plt

"""Data Arrangement of Yields Data"""
df_all = pd.read_csv ('data/jgbcm.csv', index_col='基準日', encoding='cp932', parse_dates=True)
df_all.index.name = 'Date'
df_all.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40]
df_all = df_all.replace ('-', np.nan).astype (np.float32)
df_all = df_all.resample ('M').last ()
# df_all = df_all.drop([25,30,40],axis=1)
df_all = df_all[[2, 5, 10, 20]]
main_rate = df_all.dropna (axis=0)

"""Data Arrangement of Macro Data"""
macro_var = ['CPI', 'NK', 'CGPI', 'MBA', 'UP', 'CI', 'US10', 'FF']
macro_data = pd.read_csv ('data/macro_data2.csv', index_col=0, encoding='cp932', parse_dates=True)
macro_data = macro_data.replace ('ND', np.nan).astype (np.float32)
# macro_data = macro_data.dropna()
macro_data = macro_data.resample ('M').last ()
macro_data = macro_data.diff ()
macro_data.columns = macro_var
macro_data.index.name = 'Date'
macro_data = macro_data.dropna ()

"""Create the Available Data"""
full_period = macro_data.index & main_rate.index
macro_data = macro_data.reindex (full_period)
main_rate = main_rate.reindex (full_period)
maturity = 12 * main_rate.columns.values
rate = main_rate.values

lambda_hat = 0.06
x1 = np.divide (1. - np.exp(-maturity * lambda_hat), maturity * lambda_hat)
x2 = np.divide (1. - np.exp(-maturity * lambda_hat), maturity * lambda_hat) - np.exp (-maturity * lambda_hat)
v_one = np.ones(x1.shape)
W = np.c_[v_one, x1, x2]

X = np.random.random([3,376])
x0 = X.T[0]
Y = rate.T
macro = macro_data.values.T[:2]
N = macro.shape[0]

def log_likilihood(params, X, Y, m):
    u = params[0:3].reshape(3,1)
    F = params[3:12].reshape(3,3)
    G = params[12:12+3*N].reshape(3, N)
    sigma_ep = params[12+3*N:28+3*N].reshape(4, 4)
    sigma_xi = params[28+3*N:37+3*N].reshape(3, 3)

    X1 = (X.T[1:]).T
    X0 = (X.T[:-1]).T

    E1 = np.exp(-0.5 * np.trace(np.linalg.inv(sigma_ep) @ (Y - (W @ X1)) @ (Y - (W @ X1)).T)) \
    / np.linalg.det(sigma_ep) ** 0.5
    E2 = np.exp(-0.5 * np.trace((np.linalg.inv(sigma_xi) @ (X1 - u - F@X0 - G @ m)).T @ (X1 - u - F@X0 - G @ m))) \
    / np.linalg.det(sigma_xi) ** 0.5

    return np.log(E1 * E2)

def kalman_filter(u,F,G,sigma_ep,sigma_xi,Y,W,macro):
    x = x0
    P = np.eye(3) / (i+1)
    x_t = [x]
    P_t = []
    for m,y in zip(macro,Y):
        x_bar = u.T + F @ x + G @ m
        P_bar = F @ P @ F.T + sigma_xi
        S = W @ P_bar @ W.T + sigma_ep
        K = P_bar @ W.T @ np.linalg.inv(S.astype(np.float32))

        x = x_bar + (K @ (y.reshape(-1,1) - W @ x_bar.T)).T
        x = x[0]
        P = (1 - K @ W) @ P_bar
        x_t.append(x_bar)

    return np.vstack(x_t)

a = 0.001*np.random.rand(4,4)
b = 0.001*np.random.rand(3,3)

u0 = np.random.rand(3)
F0 = np.random.random([3,3]).ravel()
G0 = np.random.random([3,N]).ravel()
sigma_ep0 = (np.tril(a) + np.tril(a, -1).T).ravel()
sigma_xi0 = (np.tril(b) + np.tril(b, -1).T).ravel()
init_param = np.r_[u0,F0,G0,sigma_ep0,sigma_xi0]

for i in range(10):
    estimated = fmin(log_likilihood,init_param, args=(X, Y, macro), disp=False)
    u = estimated[0:3].reshape(3, 1)
    F = estimated[3:12].reshape(3, 3)
    G = estimated[12:12+3*N].reshape(3, N)
    sigma_ep = estimated[12+3*N:28+3*N].reshape(4, 4)
    sigma_xi = estimated[28+3*N:37+3*N].reshape(3, 3)
    init_param = estimated
    X = kalman_filter(u,F,G,sigma_ep,sigma_xi,Y.T,W, macro.T)
    x0 = X[0]
    X = X.T
