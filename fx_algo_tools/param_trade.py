# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:56:12 2016

@author: dy248
"""

import pickle
import talib as ta
import numpy as np
from fx_trade_system3 import Rate
from fx_trade_system3 import System_Trade 

with open('param2.dump','r') as f:
    param = pickle.load(f)
    
def normalized(data):
    std = np.nanstd(data)
    u = np.nanmean(data)
    normalized_data = []
    for X in data:
        normalized_data.append((X - u)/std)
        
    return normalized_data
    

class Signal:
    def __init__(self,open_,high,low,close,volume):
        self.open = np.array(open_)
        self.high = np.array(high)
        self.low = np.array(low)
        self.close = np.array(close)
        self.volume = np.array(volume,dtype=float)
        self.size = 10
        self.list = []
        
    def rsi(self):
        return ta.RSI(self.close)
    
    def stoch(self):
        return ta.STOCH(self.high,self.low,self.close)

    def adx(self):
        return ta.ADX(self.high,self.low,self.close)
        
    def mom(self):
        return ta.MOM(self.close)
        
    def obv(self):
        return ta.OBV(self.close,self.volume)
    
    def macd(self):
        return ta.MACD(self.close)
        
    def mfi(self):
        return ta.MFI(self.high,self.low,self.close,self.volume)
        
    def cci(self):
        return ta.CCI(self.high,self.low,self.close)
        
        
    def make_lists(self):
        s1 = normalized(self.rsi())
        s2 = normalized(self.stoch()[0])
        s3 = normalized(self.stoch()[1])
        s4 = normalized(self.adx())
        s5 = normalized(self.mom())
        s6 = normalized(self.obv())
        s7 = normalized(self.macd()[0])
        s8 = normalized(self.macd()[1])
        s9 = normalized(self.mfi())
        s10 = normalized(self.cci())
        self.list = np.dstack([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10])[0]
        
class Trading:
    def __init__(self,param):
        self.param = param
        
    def decision(self,time):
        value = 0
        for param,signal in zip(self.param,sig.list[time]):
            value += param*signal
            
        if value > 5.0:
            self.type = 'Long'
        elif value < -5.0:
            self.type = 'Short'
        else:
            self.type = 'None'
            
        #print self.type
            
    def order(self,time):
        self.decision(time)
      
  
        fts.close(time+48)
            
        if self.type == 'Long':
            fts.order('Long',time)
        elif self.type == 'Short':
            fts.order('Short',time)
        
            
        fts.position()
            
    def performance(self):
        return fts.total_return()
        
    def __del__(self):
        fts.close(69999)
        print self.performance()
        fts.reset()
        
       
r = Rate()
r.get_offline()
sig = Signal(r.open,r.high,r.low,r.close,r.volume)
sig.make_lists()
fts = System_Trade(rate=r.close,lot=1000)
T = Trading(param[40])
R = []
for time in np.arange(0,70000):
    T.order(time)
    R.append(fts.total_return())
    
del T

