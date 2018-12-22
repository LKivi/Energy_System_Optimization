# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:46:17 2018

@author: lkivi
"""
import parameter as p
import pylab as plt
import numpy as np
import grid as gr
import os
import datetime
import time
from optim_model import run_optim


# Start time of procedure
start_time_procedure = time.time()

#lamb_list = np.arange(0,6)*0.02 +0.01
#t_list = np.arange(1,5)*0.025
#
#for t in t_list:
#    
#    anteile = []
#    
#    for lamb in lamb_list:
#
#        anteil,_,_,_ = p.load_params(lamb, t)
#        anteile.append(anteil)
#    
#    plt.plot(lamb_list, anteile, label = 't = ' + str(np.around(t, decimals = 3)))
#
#plt.legend()
#plt.show
#plt.grid()
#plt.xlabel('Wärmeleitfähigkeit Isolierung W/mK')
#plt.ylabel('Anteil Wärmeverluste Wärmebedarf')
#


#devs, param, dem = p.load_params()

tac = []
COP = []
dt = []




for twp in np.arange(21) + 0.0001 :


    devs, param, dem = p.load_params(twp)
    
    dt_hp = devs["HP"]["dT_cond"]
    COP_hp = devs["HP"]["COP"]
    
    dt.append(dt_hp)
    COP.append(COP_hp)
    
    # Create result directory
    dir_results = str(os.path.dirname(os.path.realpath(__file__))) + "\\Results\\k" + str(twp)
    
    print(twp)
    
    
    
    # tac optimization
#    tac_opt = run_optim("tac", "", "", str(dir_results), twp)["tac"]
#    
#    
#    tac.append(tac_opt)
#    
#    
#
#
###plt.plot(dt, COP)
###plt.plot(dt,tac)
##    
b = COP
#    
#plt.show()




#gr.plotGrid()


#path_weather = "input_data/weather.csv"
#T_amb = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(0))
#
#T_supply = np.zeros(8760)
#
#for i in range(8760):
#    T_supply[i] = gr.get_T_supply(T_amb[i])
#
#plt.scatter(T_amb, T_supply)
#plt.show()
#print(np.mean(T_supply))