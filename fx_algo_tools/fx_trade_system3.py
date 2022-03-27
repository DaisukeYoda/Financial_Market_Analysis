# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:53:22 2016

@author: dy248
"""
import json
import oandapy

with open('./config.json', 'r') as f:
    api_key_information = json.load(f)
    
token = api_key_information['token']
oanda = oandapy.API(environment="practice", access_token=token)



class Position:
    def __init__(self,order_type,rate,lot):
        self.type = order_type
        self.rate = rate
        self.lot = lot 
        
    def visualize(self):
        print [self.type,self.rate,self.lot]
    
       
class Current_Position:
    def __init__(self):
        self.total_return = 0
        self.type = 'None'
        self.rate = 0
        self.lot = 0
    def order_mix(self,order):
        if self.type != order.type:
            _lot = abs(self.lot - order.lot)
        if self.type == order.type:
            _lot = self.lot + order.lot
        
        if self.type == 'None':
            _type = order.type
            _rate = order.rate
        elif self.lot > order.lot:
            _type = self.type
            _rate = self.rate
        elif self.lot < order.lot:
            _type = order.type
            _rate = order.rate

        if self.type == order.type:
            if self.lot == 0:
                _rate = order.rate
            else:
                _rate = (self.rate*self.lot + order.rate*order.lot)/(self.lot + order.lot)
            Return = 0
            if self.lot == order.lot:
                _type = order.type
            if _type == 'None':
                _rate = order.rate
                _lot = 0
        elif self.type != order.type:
            if self.lot == order.lot:
                _type = 'None'
                _rate = order.rate
            if self.type == 'Long':
                Return = (order.rate - self.rate - 0.003)*min(order.lot,self.lot)
            elif self.type == 'Short':
                Return = (self.rate - order.rate - 0.003)*min (order.lot,self.lot)
            else:
                Return = 0
        
        self.type = _type
        self.rate = round(_rate,3)
        self.lot = _lot
        
        self.total_return += Return
        return 
    
        
    def funding_constraint(self,assets=100000):
        if self.rate*self.lot > assets:
            print "Order Limited"
        

        
class Rate:
    def __init__(self):
        self.open = []
        self.close = []
        self.high = []
        self.low = []
        self.volume = []
        
    def get_rate(self,period=5000):

        response = oanda.get_history(
        instrument="USD_JPY",
        granularity="M1",
        
        #end="2016-12-20T20:00:00Z",
        count=period)
        data = response.get("candles")
        
        for i in xrange(period):
            self.open.append(data[i].get('openAsk'))          
            self.close.append(data[i].get('closeAsk'))
            self.volume.append(data[i].get('volume'))
            self.high.append(data[i].get('highAsk'))
            self.low.append(data[i].get('lowAsk')) 
     
    def get_offline(self):
        import pickle
        with open('M5DATA.dump','r') as f:
            rate = pickle.load(f)
        self.open = rate[:,0]
        self.high = rate[:,1]
        self.low = rate[:,2]
        self.close = rate[:,3]
        self.volume = rate[:,4]
        
    def to_array(self):
        import numpy as np
        self.open = np.array(self.open)
        self.high = np.array(self.high)
        self.low = np.array(self.low)
        self.close = np.array(self.close)
        self.volume = np.array(self.volume)
            
class System_Trade:
    def __init__(self,rate):
        self.rate = rate
        self.cp = Current_Position()
    
    def order(self,order_type,time,lot=1000):
        order_rate = self.rate[time]
        new_order = Position(order_type,order_rate,lot)
        self.cp.order_mix(new_order)
        
    def close(self,time):
        order_rate = self.rate[time]
        lot = self.cp.lot
        new_order = Position('None',order_rate,lot)
        self.cp.order_mix(new_order)
        
    def time_closure(self,order_type,time,lot=1000):
        self.order(order_type,time,lot)
        self.close(time+10)
    
    def price_closure(self,order_type,time,triger=0.3):
        self.order(order_type,time)
        i = 1
        while True:
            if abs(self.rate[time]-self.rate[time+i]) > triger:
                self.close(time+i)
                break
            else:
                i += 1
                continue
            

        
    def total_return(self):
        return self.cp.total_return
        
    def reset(self):
        self.cp.total_return = 0
        
    def position(self):
        rate = self.cp.rate
        lot = self.cp.lot
        type_ = self.cp.type
        position_ = Position(type_,rate,lot)
        position_.visualize()
        
        
if __name__ == "__main__":
    r = Rate()
    r.get_offline()
    st = System_Trade(r.close)
    st.time_closure('Long',100)