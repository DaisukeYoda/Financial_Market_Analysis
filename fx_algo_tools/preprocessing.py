# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 14:41:57 2017

@author: Daisuke Yoda
"""
import numpy as np

def normalized(data):
    vmax = np.nanmax(data)
    vmin = np.nanmin(data)
    
    return (data-vmin)/(vmax-vmin)
    
    
def probalized(data):
    data = np.array(data)
    Sum = np.sum(data)
    if Sum == 0:
        return data + 1./len(data)
    else:
        return data*(1./Sum)
        
        
def standard(data):
    u = np.nanmean(data)
    s = np.nanstd(data)
    
    return (data - u)/s
    
    
def Time_Series_Feature(rates):
    rate = normalized(rates)
    rate = np.round(rate,1)
    
    return rate
    
def pca(X,dim):
    cov = np.cov(X,rowvar=0,bias=1)#分散共分散行列
    eigen,vec = np.linalg.eig(cov)#固有値と固有ベクトル
    vec = vec.T
    N = np.argsort(-eigen)[:dim]
    feature_vec = vec[N]
    X_bar = X - np.mean(X,axis=0)
    X_pca = np.dot(X_bar,feature_vec.T)   
    
    return X_pca
    
    
