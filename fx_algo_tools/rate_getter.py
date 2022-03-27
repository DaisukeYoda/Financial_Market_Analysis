# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 00:00:25 2016

@author: dy248
"""
import pickle
import numpy as np
import oandapy
import json

with open('./config.json', 'r') as f:
    api_key_information = json.load(f)
token = api_key_information['token']

oanda = oandapy.API(environment="practice", access_token=token)

def next_month(year,month):
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    return next_year,next_month



class Rate:
    def __init__(self):
        self.time = []
        self.open = []
        self.close = []
        self.high = []
        self.low = []
        self.volume = []
        
    def get_rate(self):

        response = oanda.get_history(
        instrument="USD_JPY",
        granularity="M5",
        start="2016-11_21T00:00:00Z",
        end="2016-12_11T00:00:00Z"
        )
        data = response.get("candles")
        l = np.shape(data)[0]
        print l
        
        for i in xrange(l):
            self.open.append(data[i].get('openAsk'))          
            self.close.append(data[i].get('closeAsk'))
            self.volume.append(data[i].get('volume'))
            self.high.append(data[i].get('highAsk'))
            self.low.append(data[i].get('lowAsk'))
            
def pickle_rate():
    r = Rate()
    r.get_rate()
    NEW_DATA = np.dstack([r.open,r.high,r.low,r.close,r.volume])[0]
    
    with open('M5DATA.dump', 'r') as f:
        PRESET = pickle.load(f)
        
    DATA = np.vstack([PRESET,NEW_DATA])
    
    with open('M5DATA.dump', 'w') as f:
        pickle.dump(DATA,f)

    return DATA
        
        
if __name__ == "__main__":
    DATA = pickle_rate()
    
    
            
