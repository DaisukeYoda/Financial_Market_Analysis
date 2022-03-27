# -*- coding: utf-8 -*-
"""
Created on Sat Jun 03 14:17:01 2017

@author: daisuke yoda
"""

import numpy as np
import random
from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd


def normalize(data):
    vmax = np.nanmax(data)
    vmin = np.nanmin(data)
    
    return (data-vmin)/(vmax-vmin)


def mutation(_param):
    new_param = []
    for param in _param:
        shape = np.shape(param)
        i = random.randrange(shape[0])
        j = random.randrange(shape[1])
             
        param[i,j] = param[i,j] + random.uniform(-1.0,1.0)
           
        new_param.append(param)
        
    return  new_param 
    
class Recurrent_Neural_Network:
    
    def __init__(self,structure,fnc='Relu',softmax=True):
        n_input = structure[0] + 1
        hidden_layers = structure[1:-1]
        output_layer = structure[-1]
        unit = []
        recurrent_unit = []
        self.softmax = softmax
        
        for n_layer in hidden_layers:
            n_input = n_input + n_layer
            param = np.random.random([n_layer,n_input]) - 0.5
            unit.append(param)
            recurrent_unit.append(np.zeros(n_layer))
            n_input = n_layer           
        else:
            param = np.random.random([output_layer,n_input])
            unit.append(param)
            
        self.unit = unit
        self.recurrent_unit = recurrent_unit
        self.fnc = fnc
        self.softmax = softmax
        
    def activate(self,input_data):
        input_x = np.append(input_data,1)
        recurrent_output = []
        rnn = self.recurrent_unit
        i = 0
        for layer in self.unit[:-1]:
            input_x = np.append(rnn[i],input_x)
            y = np.dot(layer,input_x)
            
            if self.fnc == 'tanh':
                z = np.tanh(y)
            elif self.fnc == 'sigmoid':
                z = (np.tanh(y)/2.0 + 1.0)/2.0
            elif self.fnc == 'softsign':
                z = (y + 1)/abs(y)
            elif self.fnc == 'Relu':
                z = np.maximum(y,0)
            elif self.fnc == 'LRelu':
                z = np.maximum(y,0.01*y)
            elif self.fnc == 'RRelu':
                a = 0.1*np.random.randint(0,4)
                z = np.maximum(y,a*y)
            elif self.fnc == 'Linear':
                z = y
            else:
                raise ValueError
                
            recurrent_output.append(z)
            input_x = z
            i += 1
        else:
            y = np.dot(self.unit[-1],input_x)
            
            if self.fnc == 'tanh':
                z = np.tanh(y)
            elif self.fnc == 'sigmoid':
                z = (np.tanh(y)/2.0 + 1.0)/2.0
            elif self.fnc == 'softsign':
                z = (y + 1)/abs(y)
            elif self.fnc == 'Relu':
                z = np.maximum(y,0)
            elif self.fnc == 'LRelu':
                z = np.maximum(y,0.01*y)
            elif self.fnc == 'RRelu':
                a = 0.1*np.random.randint(0,4)
                z = np.maximum(y,a*y)
            elif self.fnc == 'Linear':
                z = y
            else:
                raise ValueError
                
            input_x = z       
            
        if self.softmax:
            z = (z > 0.5).astype(np.int)
            
        self.recurrent_unit = recurrent_output
        
        return z    

class AI(Recurrent_Neural_Network):
    
    def __init__(self,structure):
        Recurrent_Neural_Network.__init__(self,structure)
        self.error = 0.0
        self.generation = 0
        
    def reset(self):
        for unit in self.recurrent_unit:
            unit *= 0
        
    def observe(self,data,answer):
        error = 0.0
        for X,Y in zip(data,answer):
            
            pred = self.activate(X)
            err = abs(pred - Y)
            error += err
            
        self.error = sum(error)
        
    def predict(self,data):     
        return self.activate(data)
        
        
class CrossOver:
    
    def __init__(self,parent1,parent2):
        self.param1 = parent1.unit
        self.param2 = parent2.unit
        self.length = len(self.param1)
        
    def one_point(self):
        [param1,param2] = random.sample([self.param1,self.param2],2)
        point = random.randint(0,self.length)
        new_param = np.r_[param1[:point],param2[point:]]
        self.new_param = new_param
      
    def two_point(self):
        [param1,param2] = random.sample([self.param1,self.param2],2)
        point1 = random.randint(0,self.length)
        point2 = random.randint(point1,self.length)
        new_param = np.r_[param1[:point1],param2[point1:point2],param1[point2:]]
        
        return new_param

    def uniform(self):
        new_param = []
        for param1,param2 in zip(self.param1,self.param2):  
            new_param_ = []
            for i,j in zip(param1,param2):
                new_param_.append(random.choice([i,j]))
            else:
                new_param.append(np.array(new_param_))
                
        return np.array(new_param)
        
    def brend(self):
        new_param = []
        for i,j in zip(self.param1,self.param2):
            new_param.append(round(random.uniform(1.1*i,1.1*j),2))
        self.new_param = np.array(new_param)

class Selection:
    def __init__(self,ai_group):
        self.ai_group = np.array(ai_group)
        self.parents = []

    def first_parents(self,rate=1):
        first_group = np.hsplit(self.ai_group,10)[-1]
        lest_group = self.ai_group[:]
        parents_group = []
        for parent1 in first_group:
            parent2_group = np.random.choice(lest_group,rate)
            for parent2 in parent2_group:
                parents_group.append([parent1,parent2])
           
        return parents_group
        
    def parents_with_grade(self,grade=2,rate=1):
        if grade==1: raise ValueError
        
        group = np.hsplit(self.ai_group,10)[-1*grade:-1*(grade-1)]
        lest_group = self.ai_group[:]
        parents_group = []
        for parent1 in group[0]:
            parent2_group = np.random.choice(lest_group,rate)
            for parent2 in parent2_group:
                parents_group.append([parent1,parent2])
           
        return parents_group
        
    def ranking_decision(self):
  
        self.parents.extend(self.first_parents(rate=5))
        self.parents.extend(self.parents_with_grade(grade=2,rate=3))
        self.parents.extend(self.parents_with_grade(grade=3,rate=2))        

class Population:
    def __init__(self,structure,n_pop):
        ai_group = [AI(structure) for i in xrange(n_pop)]
        self.ai_group = ai_group
        
    def measure_performance(self,X,Y):
        for ai in self.ai_group:
            ai.observe(X,Y)
        sorted_ai = sorted(self.ai_group,key=lambda ai:ai.error,reverse=True)
        self.ai_group = deepcopy(sorted_ai)
        self.elite = sorted_ai[-1]
        
        print sorted_ai[-1].error    
        
    def proceed_generation(self):
 
        s = Selection(self.ai_group)
        s.ranking_decision()
        elite_unit = deepcopy(self.ai_group[-1].unit)
        
        for ai,parents in zip(self.ai_group,s.parents):
            c = CrossOver(parents[0],parents[1])
            new_param = c.uniform()
            new_param = mutation(new_param)
            ai.unit = new_param
            ai.error = 0
            ai.generation += 1
            ai.reset()
            
        else:
            ai.unit = deepcopy(elite_unit)
                    
    def train(self,trainX,trainY,epoch=500):
        for i in xrange(epoch):
            self.measure_performance(trainX,trainY)
            self.proceed_generation()
            
        self.measure_performance(trainX,trainY)
        

    def predict(self,data):
        elite = self.ai_group[-1]
        
        return elite.predict(data)
        
    def reset(self):
        for ai in self.ai_group:
            ai.reset()
        

if __name__ == "__main__":
    structure = (2,10,1)
    pop = Population(structure,10)
    
    with open('Financial_Data1.csv') as f:
        Data = pd.read_csv(f)     
        
    RATE1 = np.array(normalize(Data['FX']))[::-1]
    RATE2 = np.array(normalize(Data['Nikkei']))[::-1]
    RATE3 = np.array(normalize(Data['Yield']))[::-1]    

    X = np.dstack([RATE2,RATE3])[0]
    trainX = []
    trainY = []
    for i in xrange(1000):
        trainX.append(X[i:i+1])        
        trainY.append(RATE1[i+1])
    
    pop.train(trainX,trainY,3000)
    
    pop.reset()
    result = np.hstack([pop.predict(x) for x in trainX])
    plt.plot(trainY[10:])
    plt.plot(result[10:])
    plt.show()
    plt.plot(result[10:]-trainY[10:])

