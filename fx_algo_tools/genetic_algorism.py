# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 04:16:32 2017

@author: dy248
"""
import numpy as np
from copy import deepcopy

def normalized(data):
    data = np.array(data,dtype=np.float)
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


class AI:
    def __init__(self,information_size):
        self.param = np.random.uniform(-1,1,(information_size,3))
        self.performance = 0
        
    def __call__(self,performance):
        self.performance = performance
        
    def valuation(self,X):
        y = np.dot(X,self.param)
        val = np.argmax(y)
        """
        val = 0 : Buy
        val = 1 : Sell
        val = 2 : Hold
        """
        return val
        
    
class Cross_Over:
    
    def __init__(self,ai_group):
        self.ai_group = ai_group
    
    def selection(self,elite=True):
        population = sorted(self.ai_group,key=lambda ai:ai.performance,reverse=True)
        self.elite = deepcopy(population[0])
        prf = normalized([ai.performance for ai in population])
        rprf = probalized(prf)
        n_pop = len(population)
        offspring = np.random.choice(population,size=n_pop,p=rprf)
        if elite:
            offspring[0] = self.elite
    
            
        self.ai_group = offspring
        return offspring
        
    def spx(self):
        ai_group = self.ai_group
        n_ind = len(ai_group)
        shape = ai_group[0].param.shape
        genom = [np.ravel(ai.param) for ai in ai_group]
        g = np.mean(genom,axis=1)
        ep = np.sqrt(n_ind + 2)
        s = [g[i] + ep*(genom[i] - g[i]) for i in xrange(n_ind)]
        C = []
        for i in xrange(n_ind):
            if i == 0:
                C.append(s[i])
            else:
                r = (np.random.uniform(0,1))**(1./i)
                c = s[i] + r*(s[i-1] - s[i] + C[-1])
                C.append(c)
                
        for ai,c in zip(ai_group,C):
            ai.param = c.reshape(shape)
            ai.param = self.mutation(ai.param)            
            ai.param = standard(ai.param)
            
        return True
        
    def mutation(self,param):
        n_param = len(param)
        for i in xrange(n_param):
            if np.random.randint(0,100) < 1:
                param[i] += 0.1
                
        return param
        
    
class Population(Cross_Over):
    
    def __init__(self,n_pop,information_size):
        self.ai_group = [AI(information_size) for i in xrange(n_pop)]
        self.generation = 0
        self.group_performance = []
        Cross_Over.__init__(self,self.ai_group)

    
    
        
if __name__ == '__main__':
    pop = Population(100,50)
    