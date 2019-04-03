# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 20:36:59 2019

@author: lkivi
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_demand(peak, total):
    
    print("Generating random demands...")

    T = 8760
#    max_diff_season = 4000
    max_diff_day = 2000
    max_diff_hour = 100

  
    # Create array of possible demand values
    values = {}
    for demand in ["heat", "cool"]:
        values[demand] = np.linspace(0,peak[demand],T)

    
    
    # demand arrays
    dem = {}
    for demand in ["heat", "cool"]:
        dem[demand] = np.zeros(T)
        
        
    # adjust values to get the correct total amounts
    d_max = {}
    sum_values = {}
    for demand in ["heat","cool"]:
        sum_values[demand] = np.sum(values[demand])
        # if sum of demands is to little: increase values
        if sum_values[demand] < total[demand]:
            d_max[demand] =  (2*total[demand])/T - peak[demand]
            for i in range(8760):
                values[demand][i] += d_max[demand]*( 1 - i/(T-1))
            sum_values[demand] = np.sum(values[demand])
        # else: set a certain amount of the lowest values to zero
        else:
            i = 0
            while sum_values[demand] > total[demand]:
                values[demand][i] = 0
                sum_values[demand] = np.sum(values[demand])
                i +=1
                
    
    # pick values
    series = {}
    for demand in ["heat", "cool"]:
        
        # introduce time series
        series[demand] = np.zeros(T)
        
        # generate first random index
        index = np.random.randint(0,T)
        
        for t in range(8760):
            
            # draw value from value list
            series[demand][t] = values[demand][index]
            
            # delete drawn value from value list
            values[demand] = np.delete(values[demand], index)
            n_values = np.size(values[demand])
            
            # set step width for next draw
            # if new day: allow big step
            if t/24 == int(t/24):
                max_diff = max_diff_day * (n_values/T)**1
            else: # during day: little steps
                max_diff = max_diff_hour * n_values/T

            
            # generate new index according to maximum allowed difference between two consecutive indices
            max_top = min(n_values-index, int(max_diff/2))
            max_bot = min(index,int(max_diff/2))
            
#            print(max_top)
#            print(max_bot)
            
            if max_top > -max_bot:
                index = index + np.random.randint(-max_bot, max_top)
            else:
                index = 0

    
    
    # PLot demands curves                
#    plt.plot(range(T), series["heat"]) 
    
#    # arrange cooling demand for worst doc
#    for demand in ["heat", "cool"]:
#        values[demand] = np.linspace(0,peak[demand],T)
#    for t in range(T):
#        series["cool"][t] = values["cool"][T-np.where(values["heat"] == series["heat"][t])[0][0]-1]
            
     
    # return demands
    dem = {}
    dem["heat"] = series["heat"]
    dem["cool"] = series["cool"]
            
    
    return dem
    
#

       
        
        
        
    
                    
        
    
    
