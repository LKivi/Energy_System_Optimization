# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:46:17 2018

@author: lkivi
"""
import parameter as p
import pylab as plt
import numpy as np
import grid as gr

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


p.load_params()






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