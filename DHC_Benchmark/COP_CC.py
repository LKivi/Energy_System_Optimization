# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:51:14 2018

@author: lkivi
"""

import numpy as np
import pylab as plt

# Load weather data
path_weather = "input_data/weather.csv"

#t_air = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(0))               # Air temperatur °C
#r = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(2)) * 100             # relative humidity %

# wet bulb temperature °C
#t_wb = t_air*np.arctan(0.151977*(r + 8.313659)**0.5) + np.arctan(t_air + r) - np.arctan(r - 1.676331) + 0.00391838*r**1.5*np.arctan(0.023101*r)-4.686035

#plt.plot(t_air, t_wb,".")
#plt.show()
#print(max(t_wb))
#print()



# get parameter

dt_h = 8                                                                     # heat sink temperature difference
dt_c = np.arange(2,18)                                                                                    # heat sourve temperature difference
t_h_in = 27 + 273.15                                          # heat sink inlet temperature
t_c_in = 18 + 273.15                                            # heat source inlet temperature

                                                                    # heat sink temperature difference
#dt_h=5
#dt_c = np.arange(2,12)                                                                                 # heat sourve temperature difference
#t_h_in = 273.15 + 30                                            # heat sink inlet temperature
#t_c_in = 12 + 273.15  

#dt_h = 5                                                                     # heat sink temperature difference
#dt_c = np.arange(2,12)                                                                                    # heat sourve temperature difference
#t_h_in = 35 + 273.15                                            # heat sink inlet temperature
#t_c_in = 12 + 273.15  


dt_pp = 2                                                    # pinch point temperature difference
eta_is = 0.75                                                  # isentropic compression efficiency
f_Q = 0.1                                                    # heat loss rate during compression

# Entropic mean temperautures
t_h_s = dt_h/np.log((t_h_in + dt_h)/t_h_in)
t_c_s = dt_c/np.log(t_c_in/(t_c_in - dt_c))

#Lorentz-COP
COP_Lor = t_h_s/(t_h_s - t_c_s)

#print(min(COP_Lor))


# linear model equations
dt_r_H = 0.2*(t_h_in + dt_h - (t_c_in - dt_c) + 2*dt_pp) + 0.2*dt_h + 0.016        # mean entropic heat difference in condenser deducting dt_pp
w_is = 0.0014*(t_h_in + dt_h - (t_c_in - dt_c) + 2*dt_pp) - 0.0015*dt_h + 0.039    # ratio of isentropic expansion work and isentropic compression work


# help values
num = 1 + (dt_r_H + dt_pp)/t_h_s
denom = 1 + (dt_r_H + 0.5*dt_c + 2*dt_pp)/(t_h_s - t_c_s)

# COP
COP = COP_Lor * num/denom * eta_is * (1 - w_is) + 1 - eta_is - f_Q

COP = COP - 1
print(COP)

#for t in range(len(COP)):
#    if COP[t] > 7:
#        COP[t] = 7

a = t_c_in - dt_c -273.15
b = COP

c = dt_h/COP

d = dt_h/20*4.15677




plt.plot(a,b)
#plt.plot(a,d)
plt.show()



#print(t_h_in+dt_h-273.15)
#print(COP)




